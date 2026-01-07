use cudarc::driver::CudaContext;
use luminal::prelude::*;

use crate::runtime::CudaRuntime;

#[test]
pub fn add_mul_test() {
    let mut cx = Graph::default();
    let input = cx.tensor(5);
    let output = (input + input * input).output();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
    rt.set_data(input, Box::new(vec![0., 1., 2., 3., 4.]));
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(output);
    assert_eq!(out, vec![0.0, 2.0, 6.0, 12.0, 20.0]);
}
