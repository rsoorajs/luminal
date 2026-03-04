use luminal::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::utilities::{assert_close, get_cuda_stream, random_f32_vec};
use crate::runtime::CudaRuntime;

/// FP4 E2M1 lookup table (matches CUDA kernel)
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Convert FP8 E4M3 byte to f32
fn fp8_e4m3_to_float(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;
    let result = if exp == 0 {
        (mant as f32 / 8.0) * 2.0f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    };
    if sign == 1 { -result } else { result }
}

/// Convert f32 to nearest FP8 E4M3 byte
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    if abs_val == 0.0 {
        return 0;
    }
    let mut best_bits = 0u8;
    let mut best_err = f32::MAX;
    for bits in 0..=0x7Fu8 {
        let decoded = fp8_e4m3_to_float(bits);
        if decoded.is_nan() {
            continue;
        }
        let err = (decoded - abs_val).abs();
        if err < best_err {
            best_err = err;
            best_bits = bits;
        }
    }
    best_bits | (sign << 7)
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

/// Pack FP32 weights [N, K] into separate NvFp4 buffers.
///
/// Returns (packed_data, block_scales, tensor_scale) where:
/// - packed_data: N * K/2 bytes (F4E2M1, 2 values per byte, k-contiguous per column)
/// - block_scales: N * K/16 bytes (F8E4M3, 1 scale per 16 elements per column)
fn pack_nvfp4(weights: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert_eq!(weights.len(), n * k);
    assert!(k.is_multiple_of(16), "K must be divisible by 16");

    let tensor_scale = 1.0f32;
    let packed_per_col = k / 2;
    let scales_per_col = k / 16;
    let mut data_buf = vec![0u8; n * packed_per_col];
    let mut scales_buf = vec![0u8; n * scales_per_col];

    for col in 0..n {
        let col_weights = &weights[col * k..(col + 1) * k];

        for block in 0..(k / 16) {
            let block_start = block * 16;
            let block_vals = &col_weights[block_start..block_start + 16];
            let max_abs = block_vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            // block_scale chosen so max_abs / block_scale <= 6.0 (max FP4 value)
            let block_scale = if max_abs == 0.0 { 1.0 } else { max_abs / 6.0 };
            let fp8_scale = float_to_fp8_e4m3(block_scale);
            scales_buf[col * scales_per_col + block] = fp8_scale;

            let block_scale_float = fp8_e4m3_to_float(fp8_scale);
            let effective_scale = block_scale_float * tensor_scale;

            for i in 0..16 {
                let val = col_weights[block_start + i];
                let scaled = if effective_scale != 0.0 {
                    val / effective_scale
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

    (data_buf, scales_buf, tensor_scale)
}

/// Reference dequantized matmul: A [M,K] x dequant(B_data, B_scales) [K,N] -> C [M,N]
fn reference_nvfp4_matmul(
    a: &[f32],
    m: usize,
    k: usize,
    b_data: &[u8],
    b_scales: &[u8],
    n: usize,
    tensor_scale: f32,
) -> Vec<f32> {
    let packed_per_col = k / 2;
    let scales_per_col = k / 16;

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
                let block_scale = fp8_e4m3_to_float(scales[ki / 16]) * tensor_scale;
                let w = FP4_LUT[nibble as usize] * block_scale;
                acc += a[row * k + ki] * w;
            }
            result[row * n + col] = acc;
        }
    }
    result
}

/// Build the explicit dequant graph pattern for NvFp4:
///   b_data.cast(F32) * b_scales.cast(F32) → matmul with A
///
/// Uses a 3D broadcast pattern so that the Mul has matching shapes on both
/// inputs, making the graph semantically valid for direct element-wise execution:
///   data  (n, k) -> split_dims -> (n, k/16, 16)  strides (k, 16, 1)
///   scales(n, k/16) -> unsqueeze+expand -> (n, k/16, 16)  strides (k/16, 1, 0)
///   dequant = data * scales in 3D, then merge_dims back to (n, k)
fn build_nvfp4_graph(
    cx: &mut Graph,
    m: usize,
    k: usize,
    n: usize,
) -> (GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let a = cx.tensor((m, k));
    let b_data = cx.tensor((n, k)).as_dtype(DType::F4E2M1);
    let b_scales = cx.tensor((n, k / 16)).as_dtype(DType::F8E4M3);

    // Cast to F32
    let b_data_f32 = b_data.cast(DType::F32); // (n, k)
    let b_scales_f32 = b_scales.cast(DType::F32); // (n, k/16)

    // Reshape data: (n, k) -> (n, k/16, 16)
    let b_data_3d = b_data_f32.split_dims(1, 16); // strides (k, 16, 1)

    // Broadcast scales: (n, k/16) -> (n, k/16, 1) -> (n, k/16, 16)
    let mut b_scales_3d = b_scales_f32.unsqueeze(2); // strides (k/16, 1, 0)
    b_scales_3d.shape.expand((n, k / 16, 16)); // dim 2: 1 -> 16, stride stays 0

    // Valid element-wise multiply in 3D, then merge back to 2D
    let dequant = (b_data_3d * b_scales_3d).merge_dims(1, 2); // (n, k)
    let c = a.matmul(dequant.t()).output();

    (a, b_data, b_scales, c)
}

/// Minimal NvFp4 test: M=1, K=16, N=1, all ones activation, all-1.0 weight
#[test]
fn test_matmul_nvfp4_minimal() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 16;
    let n = 1;

    let a_data: Vec<f32> = vec![1.0; m * k];
    let b_fp32: Vec<f32> = vec![1.0; n * k];
    let (packed_data, scale_data, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n, tensor_scale);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_nvfp4_graph(&mut cx, m, k, n);

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

/// Test NvFp4 matmul with exact FP4 values (zero quantization error).
/// Uses weights that are exactly representable in FP4 E2M1 with block_scale=1.0.
#[test]
fn test_matmul_nvfp4_exact() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 4;
    let k = 32; // Must be divisible by 16
    let n = 8;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    // Weights using exact FP4 values (block_scale=1.0 means no quantization error)
    let mut rng = StdRng::seed_from_u64(42);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    let (packed_data, scale_data, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    assert_eq!(tensor_scale, 1.0);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n, tensor_scale);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_nvfp4_graph(&mut cx, m, k, n);

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

/// Test NvFp4 matmul with random weights (includes quantization error).
#[test]
fn test_matmul_nvfp4_random() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 8;
    let k = 64;
    let n = 16;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    // Random weights in a small range (will be quantized to FP4)
    let mut rng = StdRng::seed_from_u64(99);
    let b_fp32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-3.0..3.0f32)).collect();

    let (packed_data, scale_data, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n, tensor_scale);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_nvfp4_graph(&mut cx, m, k, n);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(a, a_data);
    rt.set_data(b_data, packed_data);
    rt.set_data(b_scales, scale_data);
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);
    // Wider tolerance since quantization introduces error
    assert_close(&result, &expected, 0.0, 0.5);
}

/// Test NvFp4 matmul with M=1 (decode path in kernel).
#[test]
fn test_matmul_nvfp4_m1() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let m = 1;
    let k = 48;
    let n = 32;

    let a_data = random_f32_vec(m * k, 0, -0.5, 0.5);

    let mut rng = StdRng::seed_from_u64(7);
    let b_fp32: Vec<f32> = (0..n * k)
        .map(|_| FP4_LUT[rng.random_range(0..16usize)])
        .collect();

    let (packed_data, scale_data, tensor_scale) = pack_nvfp4(&b_fp32, n, k);
    let expected = reference_nvfp4_matmul(&a_data, m, k, &packed_data, &scale_data, n, tensor_scale);

    let mut cx = Graph::default();
    let (a, b_data, b_scales, c) = build_nvfp4_graph(&mut cx, m, k, n);

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
