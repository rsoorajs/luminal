use log::debug;
use luminal::{prelude::*, visualization::ToDot};
use luminal_cuda::runtime::CudaRuntime;
use luminal_tracing;
use ndarray::Array2;
use rand::Rng;
use std::fs;

fn main() {
    // Initialize tracing with Perfetto export
    let trace_session = luminal_tracing::subscriber()
        .perfetto("cuda_matmul_trace.pftrace")
        .env_filter("luminal_cuda=trace,luminal=trace")
        .init();

    let m = (2 as usize).pow(8);
    let n = (2 as usize).pow(12);
    let k = (2 as usize).pow(12);

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

    rt = cx.search(rt, 10);

    fs::write("llir.dot", (&rt.llir_graph).to_dot().unwrap()).unwrap();
    rt.allocate_intermediate_buffers(&cx.dyn_map); // if you remove this it all fails... 

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

    // Flush and stop the trace session
    trace_session.stop();
    if let Some(path) = trace_session.perfetto_path {
        println!("Trace saved to: {}", path.display());
        println!("View it at: https://ui.perfetto.dev/");
    }
}
