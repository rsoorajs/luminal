use candle_core::{Device, Tensor};
use cudarc::driver::CudaContext;
use luminal::prelude::*;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

use crate::cuda_bandwidth_gbps;
use crate::runtime::CudaRuntime;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}

fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

fn get_cuda_stream() -> Option<Arc<cudarc::driver::CudaStream>> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.bind_to_thread().ok()?;
    Some(ctx.default_stream())
}

/// Test a unary operation on CUDA against candle reference
pub fn test_unary(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let shape: Vec<usize> = shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let n_elements: usize = shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(shape.clone());
    let b = func(a).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_vec(n_elements);
    rt.set_data(a, input_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(b);

    // Reference using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(input_data, shape, &device).unwrap();
    let ref_b = ref_func(ref_a).flatten_all().unwrap();

    assert_close(&result, &ref_b.to_vec1::<f32>().unwrap());
}

/// Test a binary operation on CUDA against candle reference
pub fn test_binary(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let a_shape: Vec<usize> = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let a_elements: usize = a_shape.iter().product();
    let b_elements: usize = b_shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let a_data = random_vec(a_elements);
    let b_data = random_vec(b_elements);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);

    // Reference using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(a_data, a_shape, &device).unwrap();
    let ref_b = Tensor::from_vec(b_data, b_shape, &device).unwrap();
    let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();

    assert_close(&result, &ref_c.to_vec1::<f32>().unwrap());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_add(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
    }

    #[test]
    fn test_mul(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a * b, |a, b| (&a * &b).unwrap());
    }

    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
    }
}

// Note: TileMatmulSplitK is verified through the llama example (examples/llama)
// Isolated matmul tests have graph extraction issues due to how the egglog
// pattern matching requires specific stride configurations (contiguous K for B input)
// and additional operations to form a valid extractable graph.

/// Test that measures bandwidth utilization for a large element-wise add kernel.
/// This demonstrates that KernelAdd can achieve reasonable bandwidth with large tensors.
#[test]
pub fn kernel_add_bandwidth_test() {
    // 64M elements = 256MB per tensor, 768MB total memory traffic (2 reads + 1 write)
    let size = 64 * 1024 * 1024;

    let mut cx = Graph::default();
    let a = cx.tensor(size);
    let b = cx.tensor(size);
    let output = (a + b).output();

    // Generate test data
    let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size)
        .map(|i| ((i + 500) % 1000) as f32 * 0.001)
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    rt = cx.search(rt, 5);

    // Warm up
    rt.execute(&cx.dyn_map);

    // Run and measure
    rt.execute(&cx.dyn_map);

    // Print stats
    println!("\n=== Large KernelAdd Bandwidth Test ===");
    println!(
        "Tensor size: {} elements ({} MB per tensor)",
        size,
        size * 4 / 1024 / 1024
    );
    println!(
        "Total memory traffic: {} MB (2 reads + 1 write)",
        size * 4 * 3 / 1024 / 1024
    );
    rt.print_execution_stats();

    // Verify correctness (spot check)
    let result = rt.get_f32(output);
    for i in [0, size / 2, size - 1] {
        let expected = data_a[i] + data_b[i];
        let got = result[i];
        assert!(
            (got - expected).abs() < 1e-5,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    // Check bandwidth is reasonable (at least 50% of peak for large kernels)
    let stats = &rt.last_execution_stats;
    if let Some(peak_bw) = cuda_bandwidth_gbps(&ctx) {
        for stat in &stats.kernel_stats {
            let total_bytes = stat.bytes_loaded + stat.bytes_stored;
            if stat.name == "Add" && total_bytes > 0 {
                let utilization = stat.bandwidth_gbps / peak_bw as f64 * 100.0;
                println!(
                    "\nAdd kernel achieved {:.1} GB/s ({:.1}% of {:.0} GB/s peak)",
                    stat.bandwidth_gbps, utilization, peak_bw
                );
                println!(
                    "  Loaded: {} bytes, Stored: {} bytes",
                    stat.bytes_loaded, stat.bytes_stored
                );
                // Large adds should achieve decent bandwidth
                assert!(
                    utilization > 50.0,
                    "Bandwidth utilization too low: {:.1}%",
                    utilization
                );
            }
        }
    }
}
