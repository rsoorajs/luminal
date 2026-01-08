use env_logger::{self, Builder};
use log::debug;
use luminal::{prelude::*, visualization::ToDot};
use luminal_cuda::runtime::CudaRuntime;
use std::fs::{self, OpenOptions};

fn main() {
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("app.log")
        .unwrap();

    Builder::from_default_env()
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();
    // Create compute graph
    let mut cx = Graph::new();

    let a = cx.tensor(('m', 'k'));
    let b = cx.tensor(('k', 'n')); 

    let c: GraphTensor = a.matmul(b).output();

    let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    ctx.set_flags(luminal_cuda::cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)
        .unwrap();
    let stream = ctx.default_stream();

    // Compile
    cx.build_search_space::<CudaRuntime>();
    debug!("{:#?}", cx.ops);

    let custom_state = FxHashMap::default();
    let mut rt = CudaRuntime::initialize((ctx.clone(), stream.clone(), custom_state));

    cx.set_dyn_dim('m', 3 as usize);
    cx.set_dyn_dim('n', 4 as usize); 
    cx.set_dyn_dim('k', 2 as usize);

    // Set input tensors
    rt.set_data(a, Box::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    rt.set_data(b, Box::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));

    rt.allocate_intermediate_buffers(&cx.dyn_map);

    rt = cx.search(rt, 1);
    fs::write("llir.dot", (&rt.llir_graph).to_dot().unwrap()).unwrap();

    // Run
    cx.set_dyn_dim('m', 3 as usize);
    cx.set_dyn_dim('n', 4 as usize); 
    cx.set_dyn_dim('k', 2 as usize);

    rt.execute(&cx.dyn_map);

    // Get output tensor
    println!("Result: {:?}", rt.get_f32(c));
}

