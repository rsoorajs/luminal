use candle_core::{Device, Tensor};
use luminal::{prelude::*, visualization::ToDot};
use luminal_cuda::runtime::CudaRuntime;
use rand::Rng;
use std::fs;
use tracing::{debug, info, trace};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap())
        .with(fmt::layer())
        .init();

    let m = (2 as usize).pow(12);
    let n = (2 as usize).pow(12);
    let k = (2 as usize).pow(12);

    info!(m);
    info!(n);
    info!(k);

    // Create compute graph
    let mut cx = Graph::new();

    let a = cx.tensor(('m', 'k'));
    let b = cx.tensor(('k', 'n'));

    let c: GraphTensor = a.matmul(b).output();

    // Compile
    cx.build_search_space::<CudaRuntime>();

    let mut rt = CudaRuntime::new().unwrap();

    cx.set_dim('m', m);
    cx.set_dim('n', n);
    cx.set_dim('k', k);

    // Generate random input tensors based on dimensions
    let mut rng = rand::rng();
    let a_data: Vec<f32> = (0..(m * k)).map(|_| rng.random_range(0.0..1.0)).collect();
    let b_data_row_major: Vec<f32> = (0..(k * n)).map(|_| rng.random_range(0.0..1.0)).collect();

    // Create candle tensors on CPU for verification
    let device = Device::Cpu;

    // A is row-major (m x k)
    let a_candle = Tensor::from_vec(a_data.clone(), (m, k), &device).unwrap();
    debug!("a_candle: {:?}", a_candle);
    trace!("a_candle matrix:\n{}", a_candle);

    // B needs to be column-major for luminal, so create it as row-major (k x n)
    // then transpose to get column-major layout
    let b_candle_row_major = Tensor::from_vec(b_data_row_major.clone(), (k, n), &device).unwrap();
    debug!("b_candle: {:?}", b_candle_row_major);
    trace!("b_candle:\n{}", b_candle_row_major);

    // For luminal, we need B in column-major format, which means we store it as transposed
    // Convert row-major (k x n) to column-major by transposing to (n x k) then reading as flat
    let b_candle_transposed = b_candle_row_major.t().unwrap().contiguous().unwrap();
    debug!("b_candle.t(): {:?}", b_candle_transposed);
    trace!("b_candle.t():\n{}", b_candle_transposed);

    let b_data_col_major = b_candle_transposed
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Set input tensors for luminal
    // A is row-major as is
    rt.set_data(a, a_data.clone());
    // B is in column-major format
    rt.set_data(b, b_data_col_major.clone());

    // Use search limit of 1 to get HostMatmul like the test
    rt = cx.search(rt, 5);

    fs::write("llir.dot", (&rt.llir_graph).to_dot().unwrap()).unwrap();

    // Run
    cx.set_dim('m', m);
    cx.set_dim('n', n);
    cx.set_dim('k', k);

    rt.execute(&cx.dyn_map);

    // Get output tensor from luminal (row-major format)
    let luminal_result_data = rt.get_f32(c);

    let avg = luminal_result_data.iter().sum::<f32>() / luminal_result_data.len() as f32;
    debug!("Average value: {}", avg);

    // Compute matmul using candle for verification
    // a_candle is (m x k) row-major, b_candle_row_major is (k x n) row-major
    let candle_result = a_candle.matmul(&b_candle_row_major).unwrap();
    let candle_result_flat = candle_result
        .to_vec2::<f32>()
        .unwrap()
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<f32>>();

    // Compare results
    let diff = candle_result_flat
        .iter()
        .zip(luminal_result_data.iter())
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<f32>>();

    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);

    // Check if results match within tolerance
    // Use 0.1 tolerance for GPU vs CPU comparison (floating point precision differences)
    assert!(max_diff < 0.1, "max_diff = {}", max_diff);
}
