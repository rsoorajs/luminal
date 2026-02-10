use candle_core::{Device, Tensor, WithDType};
use cudarc::driver::CudaContext;
use half::{bf16, f16};
use luminal::prelude::*;
use num_traits::Float;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use crate::runtime::{CudaRuntime, ToCudaInput};

/// Safety factor multiplied with epsilon for tolerance calculations
pub const TOLERANCE_SAFETY_FACTOR: f32 = 2.0;

/// Trait for test-compatible data types that can be used in generic test functions.
/// Bridges luminal's runtime types with candle's tensor types.
pub trait TestDType: Clone + Sized + WithDType + Float + std::fmt::Display + 'static
where
    Vec<Self>: ToCudaInput,
{
    /// The corresponding luminal DType
    const DTYPE: luminal::op::DType;

    /// Retrieve data from the runtime in this dtype
    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self>;
    /// Extract a Vec from a candle Tensor
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self>;
}

impl TestDType for f32 {
    const DTYPE: luminal::op::DType = luminal::op::DType::F32;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_f32(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<f32>().unwrap()
    }
}

impl TestDType for f16 {
    const DTYPE: luminal::op::DType = luminal::op::DType::F16;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_f16(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<f16>().unwrap()
    }
}

impl TestDType for bf16 {
    const DTYPE: luminal::op::DType = luminal::op::DType::Bf16;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_bf16(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<bf16>().unwrap()
    }
}

pub fn random_f32_vec(n: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(low..high)).collect()
}

/// Assert two vectors are close following NumPy/PyTorch conventions.
/// Formula: |a - b| <= atol + rtol * |b|
/// Generic version that works with any Float type (f32, f16, bf16).
pub fn assert_close<T: Float + std::fmt::Display>(a_vec: &[T], b_vec: &[T], rtol: T, atol: T) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let diff = (*a - *b).abs();
        let tolerance = atol + rtol * b.abs();

        if diff > tolerance {
            panic!("{a} is not close to {b}, index {i}, diff: {diff}, tolerance: {tolerance}");
        }
    }
}

pub fn get_cuda_stream() -> Option<Arc<cudarc::driver::CudaStream>> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.bind_to_thread().ok()?;
    Some(ctx.default_stream())
}

/// Machine epsilon for each dtype (approximate)
pub fn dtype_epsilon(dtype: luminal::op::DType) -> f32 {
    match dtype {
        luminal::op::DType::F32 => 1.19e-7,  // 2^-23
        luminal::op::DType::F16 => 9.77e-4,  // 2^-10
        luminal::op::DType::Bf16 => 7.81e-3, // 2^-7
        luminal::op::DType::Int => 0.0,
        luminal::op::DType::Bool => 0.0,
    }
}

/// Base unary test function with input generator (CUDA version)
/// Generic over dtype T - comparison happens in native precision.
pub fn test_unary_cuda<T: TestDType>(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
    generator: impl Fn(usize, u64) -> Vec<T>,
    seed: u64,
) where
    Vec<T>: ToCudaInput,
{
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

    let input_data = generator(n_elements, seed);
    rt.set_data(a, input_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = T::get_from_runtime(&rt, b.id);

    // Reference using candle on CUDA
    let device = Device::new_cuda(0).expect("Candle CUDA device required for test");
    let ref_a = Tensor::from_slice(&input_data, shape, &device).unwrap();
    let ref_b = ref_func(ref_a).flatten_all().unwrap();
    let ref_vec = T::candle_to_vec(&ref_b);

    // Tolerances derived from dtype epsilon
    let eps = dtype_epsilon(<T as TestDType>::DTYPE);
    let rtol = T::from(eps * TOLERANCE_SAFETY_FACTOR).unwrap();
    let atol = T::from(eps * TOLERANCE_SAFETY_FACTOR).unwrap();
    assert_close(&result, &ref_vec, rtol, atol);
}

/// Base binary test function with input generators
/// Generic over dtype T - comparison happens in native precision.
/// Requires explicit rtol and atol tolerances (as f32, converted to T internally).
pub fn test_binary_cuda<T: TestDType>(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
    a_generator: impl Fn(usize, u64) -> Vec<T>,
    b_generator: impl Fn(usize, u64) -> Vec<T>,
    seed: u64,
    rtol: f32,
    atol: f32,
) where
    Vec<T>: ToCudaInput,
{
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
    let a: GraphTensor = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let a_data = a_generator(a_elements, seed);
    let b_data = b_generator(b_elements, seed.wrapping_add(1));
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = T::get_from_runtime(&rt, c.id);

    // Reference using candle on CUDA
    let device = Device::new_cuda(0).expect("Candle CUDA device required for test");
    let ref_a = Tensor::from_slice(&a_data, a_shape, &device).unwrap();
    let ref_b = Tensor::from_slice(&b_data, b_shape, &device).unwrap();
    let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();
    let ref_vec = T::candle_to_vec(&ref_c);

    let rtol = T::from(rtol).unwrap();
    let atol = T::from(atol).unwrap();
    assert_close(&result, &ref_vec, rtol, atol);
}

/// Test mod operation with element-wise reference using Rust's % operator
pub fn test_mod(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    seed: u64,
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

    let a_data = random_f32_vec(a_elements, seed, -0.5, 0.5);
    // Generate divisor values away from zero (0.1 to 0.5) to avoid division issues
    let b_data = random_f32_vec(b_elements, seed.wrapping_add(1), 0.1, 0.5);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);

    // Reference: Rust's % operator matches CUDA's fmodf (IEEE 754 remainder)
    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x % y)
        .collect();

    let eps = dtype_epsilon(luminal::op::DType::F32);
    let rtol = eps * TOLERANCE_SAFETY_FACTOR;
    let atol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, rtol, atol);
}

/// Generate a slice range for an axis of given size.
/// If do_start is true, randomly choose a start offset (leaving at least 1 element).
/// If do_end is true, randomly choose an end before the axis end.
pub fn gen_slice_range(
    size: usize,
    do_start: bool,
    do_end: bool,
    rng: &mut impl Rng,
) -> (usize, usize) {
    let start = if do_start && size > 1 {
        rng.random_range(0..size)
    } else {
        0
    };
    let remaining = size - start;
    let end = if do_end && remaining > 1 {
        start + rng.random_range(1..remaining)
    } else {
        size
    };
    (start, end)
}
