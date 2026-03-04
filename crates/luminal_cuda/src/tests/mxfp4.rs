use luminal::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::utilities::{assert_close, get_cuda_stream, random_f32_vec};
use crate::runtime::CudaRuntime;

/// FP4 E2M1 lookup table (matches CUDA kernel)
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode E8M0 byte to f32: scale = 2^(byte - 127)
fn e8m0_to_float(byte: u8) -> f32 {
    if byte == 0xFF {
        return 0.0; // NaN → 0
    }
    2.0f32.powi(byte as i32 - 127)
}

/// Encode f32 to nearest E8M0 byte (closest power of 2)
fn float_to_e8m0(val: f32) -> u8 {
    if val <= 0.0 {
        return 0; // 2^-127 ≈ 0
    }
    // E8M0 represents 2^(byte-127), so byte = log2(val) + 127
    let log2_val = val.log2();
    let byte = (log2_val + 127.0).round() as i32;
    byte.clamp(0, 254) as u8
}

/// Quantize f32 to nearest FP4 E2M1 code (0..15)
fn float_to_fp4_e2m1(val: f32) -> u8 {
    let mut best_code = 0u8;
    let mut best_err = f32::MAX;
    for code in 0..16u8 {
        let err = (FP4_LUT[code as usize] - val).abs();
        if err < best_err {
            best_err = err;
            best_code = code;
        }
    }
    best_code
}

/// Pack FP32 weights [N, K] into separate MXFP4 buffers.
///
/// Returns (packed_data, block_scales) where:
/// - packed_data: N * K/2 bytes (F4E2M1, 2 values per byte, k-contiguous per column)
/// - block_scales: N * K/32 bytes (F8UE8M0, 1 scale per 32 elements per column)
fn pack_mxfp4(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(weights.len(), n * k);
    assert!(k.is_multiple_of(32), "K must be divisible by 32");

    let packed_per_col = k / 2;
    let scales_per_col = k / 32;
    let mut data_buf = vec![0u8; n * packed_per_col];
    let mut scales_buf = vec![0u8; n * scales_per_col];

    for col in 0..n {
        let col_weights = &weights[col * k..(col + 1) * k];

        for block in 0..(k / 32) {
            let block_start = block * 32;
            let block_vals = &col_weights[block_start..block_start + 32];
            let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            // block_scale chosen so max_abs / block_scale <= 6.0 (max FP4 value)
            let block_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 6.0 };
            let e8m0_scale = float_to_e8m0(block_scale);
            scales_buf[col * scales_per_col + block] = e8m0_scale;

            let block_scale_float = e8m0_to_float(e8m0_scale);

            for i in 0..32 {
                let val = col_weights[block_start + i];
                let scaled = if block_scale_float != 0.0 {
                    val / block_scale_float
                } else {
                    0.0
                };
                let fp4_code = float_to_fp4_e2m1(scaled);
                let k_idx = block_start + i;
                if k_idx & 1 == 0 {
                    data_buf[col * packed_per_col + k_idx / 2] |= fp4_code;
                } else {
                    data_buf[col * packed_per_col + k_idx / 2] |= fp4_code << 4;
                }
            }
        }
    }

    (data_buf, scales_buf)
}

/// Reference dequantized matmul: A [M,K] x dequant(B_data, B_scales) [K,N] -> C [M,N]
fn reference_mxfp4_matmul(
    a: &[f32],
    m: usize,
    k: usize,
    b_data: &[u8],
    b_scales: &[u8],
    n: usize,
) -> Vec<f32> {
    let packed_per_col = k / 2;
    let scales_per_col = k / 32;

    let mut result = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let packed = &b_data[col * packed_per_col..(col + 1) * packed_per_col];
            let scales = &b_scales[col * scales_per_col..(col + 1) * scales_per_col];
            let mut acc = 0.0f32;
            for ki in 0..k {
                let packed_byte = packed[ki / 2];
                let nibble = if ki & 1 == 1 {
                    packed_byte >> 4
                } else {
                    packed_byte & 0xF
                };
                let block_scale = e8m0_to_float(scales[ki / 32]);
                let w = FP4_LUT[nibble as usize] * block_scale;
                acc += a[row * k + ki] * w;
            }
            result[row * n + col] = acc;
        }
    }
    result
}

/// Build the explicit dequant graph pattern for MXFP4:
///   b_data.cast(F32) * b_scales.cast(F32) → matmul with A
///
/// Uses a 3D broadcast pattern so that the Mul has matching shapes on both
/// inputs, making the graph semantically valid for direct element-wise execution:
///   data  (n, k) -> split_dims -> (n, k/32, 32)  strides (k, 32, 1)
///   scales(n, k/32) -> unsqueeze+expand -> (n, k/32, 32)  strides (k/32, 1, 0)
///   dequant = data * scales in 3D, then merge_dims back to (n, k)
fn build_mxfp4_graph(
    cx: &mut Graph,
    m: usize,
    k: usize,
    n: usize,
) -> (GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let a = cx.tensor((m, k));
    let b_data = cx.tensor((n, k)).as_dtype(DType::F4E2M1);
    let b_scales = cx.tensor((n, k / 32)).as_dtype(DType::F8UE8M0);

    // Cast to F32
    let b_data_f32 = b_data.cast(DType::F32); // (n, k)
    let b_scales_f32 = b_scales.cast(DType::F32); // (n, k/32)

    // Reshape data: (n, k) -> (n, k/32, 32)
    let b_data_3d = b_data_f32.split_dims(1, 32); // strides (k, 32, 1)

    // Broadcast scales: (n, k/32) -> (n, k/32, 1) -> (n, k/32, 32)
    let mut b_scales_3d = b_scales_f32.unsqueeze(2); // strides (k/32, 1, 0)
    b_scales_3d.shape.expand((n, k / 32, 32)); // dim 2: 1 -> 32, stride stays 0

    // Valid element-wise multiply in 3D, then merge back to 2D
    let dequant = (b_data_3d * b_scales_3d).merge_dims(1, 2); // (n, k)
    let c = a.matmul(dequant.t()).output();

    (a, b_data, b_scales, c)
}

/// Minimal MXFP4 test: M=1, K=32, N=1, all ones activation, all-1.0 weight
#[test]
fn test_matmul_mxfp4_minimal() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 32;
    let n = 1;

    let a_data: Vec<f32> = vec![1.0; m * k];
    let b_fp32: Vec<f32> = vec![1.0; n * k];
    let (packed_data, scale_data) = pack_mxfp4(&b_fp32, n, k);
    let expected = reference_mxfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_mxfp4_graph(&mut cx, m, k, n);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b_data, packed_data);
    rt.set_data(b_scales, scale_data);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert_close(&result, &expected, 0.0, 0.5);
}

/// Test MXFP4 matmul with exact FP4 values (zero quantization error).
#[test]
fn test_matmul_mxfp4_exact() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 4;
    let k = 64;
    let n = 8;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    // Weights using exact FP4 values
    let mut rng = StdRng::seed_from_u64(42);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    let (packed_data, scale_data) = pack_mxfp4(&b_fp32, n, k);
    let expected = reference_mxfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_mxfp4_graph(&mut cx, m, k, n);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b_data, packed_data);
    rt.set_data(b_scales, scale_data);
    rt = cx.search(rt, 5);

    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(c);
    assert_close(&result, &expected, 0.0, 0.1);
}

/// Test MXFP4 matmul with random weights (includes quantization error).
#[test]
fn test_matmul_mxfp4_random() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 4;
    let k = 128;
    let n = 16;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    let mut rng = StdRng::seed_from_u64(99);
    let b_fp32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-3.0..3.0f32)).collect();

    let (packed_data, scale_data) = pack_mxfp4(&b_fp32, n, k);
    let expected = reference_mxfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_mxfp4_graph(&mut cx, m, k, n);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b_data, packed_data);
    rt.set_data(b_scales, scale_data);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert_close(&result, &expected, 0.0, 0.5);
}

/// Test MXFP4 matmul with M=1 (decode path in kernel).
#[test]
fn test_matmul_mxfp4_m1() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 128;
    let n = 64;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    let mut rng = StdRng::seed_from_u64(7);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    let (packed_data, scale_data) = pack_mxfp4(&b_fp32, n, k);
    let expected = reference_mxfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_mxfp4_graph(&mut cx, m, k, n);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b_data, packed_data);
    rt.set_data(b_scales, scale_data);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    assert_close(&result, &expected, 0.0, 0.1);
}
