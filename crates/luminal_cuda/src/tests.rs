use luminal::prelude::*;

use crate::runtime::CudaRuntime;

#[test]
pub fn cuda_test() {
    let mut cx = Graph::default();
    let input = cx.tensor(5);
    let output = (input + input).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::new().unwrap();
    rt.set_data(input, Box::new(vec![0., 1., 2., 3., 4.]));
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let out = rt.get_f32(output);
    assert_eq!(out, vec![0., 2., 4., 6., 8.]);
}
