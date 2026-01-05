use cudarc::driver::CudaContext;
use luminal::prelude::*;
use proptest::prelude::*;

use crate::runtime::CudaRuntime;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]
    #[test]
    fn cuda_test(len in 1usize..32, values in proptest::collection::vec(-5.0f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return Ok(()),
        };
        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input + input).output();

        ctx.bind_to_thread().unwrap();
        let stream = ctx.default_stream();
        cx.build_search_space::<CudaRuntime>();
        let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
        let input_values = values.into_iter().take(len).collect::<Vec<f32>>();
        rt.set_data(input, Box::new(input_values.clone()));
        rt = cx.search(rt, 10);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);
        let out = rt.get_f32(output);
        let expected = input_values.into_iter().map(|v| v * 2.0).collect::<Vec<f32>>();
        assert_eq!(out, expected);
    }
}
