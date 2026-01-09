use env_logger;
use log::debug;
use luminal::{prelude::*, visualization::ToDot};
use luminal_cuda::runtime::CudaRuntime;
use ndarray::Array2;
use rand::Rng;
use std::fs;

fn main() {
    env_logger::init();

    let m = (2 as usize).pow(10);
    let n = (2 as usize).pow(10);
    let k = (2 as usize).pow(10);

    // Create compute graph
    let mut cx = Graph::new();

    let a = cx.tensor(('m', 'k'));
    let b = cx.tensor(('k', 'n'));

    let c: GraphTensor = a.matmul(b).output();

    // Compile
    cx.build_search_space::<CudaRuntime>();
    debug!("{:#?}", cx.ops);

    let mut rt = CudaRuntime::new().unwrap();

    cx.set_dyn_dim('m', m);
    cx.set_dyn_dim('n', n);
    cx.set_dyn_dim('k', k);

    // Generate random input tensors based on dimensions
    let mut rng = rand::rng();
    let a_data: Vec<f32> = (0..(m * k)).map(|_| rng.random_range(0.0..10.0)).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|_| rng.random_range(0.0..10.0)).collect();

    let a_matrix = Array2::from_shape_vec((m, k), a_data.clone()).unwrap();
    let b_matrix = Array2::from_shape_vec((k, n), b_data.clone()).unwrap();

   
    // Set input tensors
    rt.set_data(a, Box::new(a_data));
    rt.set_data(b, Box::new(b_data));

    rt.allocate_intermediate_buffers(&cx.dyn_map);

    rt = cx.search(rt, 1);
    fs::write("llir.dot", (&rt.llir_graph).to_dot().unwrap()).unwrap();

    // Run
    cx.set_dyn_dim('m', m);
    cx.set_dyn_dim('n', n);
    cx.set_dyn_dim('k', k);

    rt.execute(&cx.dyn_map);

    // Get output tensor
    let result_data = rt.get_f32(c);
    let result_matrix = Array2::from_shape_vec((m, n), result_data).unwrap();

    debug!("Matrix A:\n{:.2}", a_matrix);
    debug!("Matrix B:\n{:.2}", b_matrix);
    debug!("Result:\n{:.2}", result_matrix);
}
