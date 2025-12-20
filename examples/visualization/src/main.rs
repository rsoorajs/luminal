use std::fs;

use luminal::{
    prelude::*,
    visualization::ToDot};
use luminal_nn::Linear;
use luminal_cuda::{
    self,
    runtime::CudaRuntime,
}; 



fn main() {
    // Create a new graph
    let mut cx = Graph::new();

    // Create a linear layer with input size 4 and output size 5
    let model = Linear::new(4, 5, false, &mut cx);

    // Create input tensor using constant values
    // Since we can't use .set() anymore, we'll construct a tensor from operations
    let a = cx.constant_float(1.0).expand((4,)) +
            cx.arange(4).cast(DType::F32);

    // Feed tensor through model
    let _b = model.forward(a);

    let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    let mut custom_state = FxHashMap::default();

    fs::write("HLIR.dot", cx.graph.to_dot().unwrap()).unwrap();

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    println!("Compiling...");
    let mut runtime = cx.search(CudaRuntime::initialize((ctx, stream, custom_state)), 10_000);
}