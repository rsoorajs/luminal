use cudarc::driver::CudaContext;
use luminal::prelude::*;
use luminal::graph::hlir_to_egglog;
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
        rt.set_data(input, input_values.clone());
        rt = cx.search(rt, 10);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);
        let out = rt.get_f32(output);
        let expected = input_values.into_iter().map(|v| v * 2.0).collect::<Vec<f32>>();
        assert_eq!(out, expected);
    }
}

#[test]
pub fn cuda_sum_reduce_test() {
    let mut cx = Graph::default();
    let input = cx.tensor((1000, 1000));
    let sum_dim0 = input.sum(0).output(); // row sum
    let sum_dim1 = input.sum(1).output(); // col sum

    let data: Vec<f32> = (0..1_000_000).map(|i| (i % 100) as f32 * 0.01).collect();

    let expected_dim0: Vec<f32> = (0..1000)
        .map(|col| (0..1000).map(|row| data[row * 1000 + col]).sum())
        .collect();
    let expected_dim1: Vec<f32> = (0..1000)
        .map(|row| (0..1000).map(|col| data[row * 1000 + col]).sum())
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
    rt.set_data(input, data);
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out_dim0 = rt.get_f32(sum_dim0);
    let out_dim1 = rt.get_f32(sum_dim1);

    for i in 0..1000 {
        let rel_err_0 = (out_dim0[i] - expected_dim0[i]).abs() / expected_dim0[i].abs().max(1.0);
        let rel_err_1 = (out_dim1[i] - expected_dim1[i]).abs() / expected_dim1[i].abs().max(1.0);
        assert!(
            rel_err_0 < 0.001,
            "dim0 mismatch at {i}: got {}, expected {}",
            out_dim0[i],
            expected_dim0[i]
        );
        assert!(
            rel_err_1 < 0.001,
            "dim1 mismatch at {i}: got {}, expected {}",
            out_dim1[i],
            expected_dim1[i]
        );
    }
}

#[test]
pub fn cuda_max_reduce_test() {
    let mut cx = Graph::default();
    let input = cx.tensor((1000, 1000));
    let max_dim0 = input.max(0).output(); // row max
    let max_dim1 = input.max(1).output(); // col max

    let data: Vec<f32> = (0..1_000_000).map(|i| (i % 100) as f32 * 0.01).collect();

    let expected_dim0: Vec<f32> = (0..1000)
        .map(|col| {
            (0..1000)
                .map(|row| data[row * 1000 + col])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    let expected_dim1: Vec<f32> = (0..1000)
        .map(|row| {
            (0..1000)
                .map(|col| data[row * 1000 + col])
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
    rt.set_data(input, data);
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out_dim0 = rt.get_f32(max_dim0);
    let out_dim1 = rt.get_f32(max_dim1);

    for i in 0..1000 {
        let rel_err_0 = (out_dim0[i] - expected_dim0[i]).abs() / expected_dim0[i].abs().max(1.0);
        let rel_err_1 = (out_dim1[i] - expected_dim1[i]).abs() / expected_dim1[i].abs().max(1.0);
        assert!(
            rel_err_0 < 0.001,
            "dim0 mismatch at {i}: got {}, expected {}",
            out_dim0[i],
            expected_dim0[i]
        );
        assert!(
            rel_err_1 < 0.001,
            "dim1 mismatch at {i}: got {}, expected {}",
            out_dim1[i],
            expected_dim1[i]
        );
    }
}

#[test]
pub fn cuda_mean_reduce_test() {
    let mut cx = Graph::default();
    let input = cx.tensor((1000, 1000));
    let mean_dim0 = input.mean(0).output(); // mean along rows
    let mean_dim1 = input.mean(1).output(); // mean along cols

    let data: Vec<f32> = (0..1_000_000).map(|i| (i % 100) as f32 * 0.01).collect();

    let expected_dim0: Vec<f32> = (0..1000)
        .map(|col| (0..1000).map(|row| data[row * 1000 + col]).sum::<f32>() / 1000.0)
        .collect();
    let expected_dim1: Vec<f32> = (0..1000)
        .map(|row| (0..1000).map(|col| data[row * 1000 + col]).sum::<f32>() / 1000.0)
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
    rt.set_data(input, data);
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out_dim0 = rt.get_f32(mean_dim0);
    let out_dim1 = rt.get_f32(mean_dim1);

    for i in 0..1000 {
        let rel_err_0 = (out_dim0[i] - expected_dim0[i]).abs() / expected_dim0[i].abs().max(1.0);
        let rel_err_1 = (out_dim1[i] - expected_dim1[i]).abs() / expected_dim1[i].abs().max(1.0);
        assert!(
            rel_err_0 < 0.001,
            "dim0 mismatch at {i}: got {}, expected {}",
            out_dim0[i],
            expected_dim0[i]
        );
        assert!(
            rel_err_1 < 0.001,
            "dim1 mismatch at {i}: got {}, expected {}",
            out_dim1[i],
            expected_dim1[i]
        );
    }
}

#[test]
pub fn cuda_argsort_test() {
    let rows = 10;   // shmem tet
    let cols = 5000; // no shmem test
    let total = rows * cols;

    let mut cx = Graph::default();
    let input = cx.tensor((rows, cols));
    let sorted_dim0 = input.argsort(0, true).output();   // descend
    let sorted_dim1 = input.argsort(1, false).output();  // ascend

    // random and unique data
    let data: Vec<f32> = (0..total).map(|i| ((i * 73 + 17) % total) as f32).collect();

    let sorted_cols: Vec<Vec<i32>> = (0..cols)
        .map(|col| {
            let mut indices: Vec<i32> = (0..rows as i32).collect();
            indices.sort_by(|&a, &b| {
                let va = data[(a as usize) * cols + col];
                let vb = data[(b as usize) * cols + col];
                vb.partial_cmp(&va).unwrap()
            });
            indices
        })
        .collect();

    let expected_dim0: Vec<i32> = (0..rows)
        .flat_map(|row| (0..cols).map(|col| sorted_cols[col][row]).collect::<Vec<_>>())
        .collect();

    let expected_dim1: Vec<i32> = (0..rows)
        .flat_map(|row| {
            let mut indices: Vec<i32> = (0..cols as i32).collect();
            indices.sort_by(|&a, &b| {
                let va = data[row * cols + (a as usize)];
                let vb = data[row * cols + (b as usize)];
                va.partial_cmp(&vb).unwrap()
            });
            indices
        })
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize((ctx, stream, FxHashMap::default()));
    rt.set_data(input, data);
    rt = cx.search(rt, 10);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out_dim0 = rt.get_i32(sorted_dim0.id).clone();
    let out_dim1 = rt.get_i32(sorted_dim1.id).clone();

    assert_eq!(out_dim0.len(), expected_dim0.len(), "dim0 length mismatch");
    assert_eq!(out_dim1.len(), expected_dim1.len(), "dim1 length mismatch");

    for i in 0..out_dim0.len() {
        assert_eq!(
            out_dim0[i], expected_dim0[i],
            "dim0 mismatch at {i}: got {}, expected {}",
            out_dim0[i], expected_dim0[i]
        );
    }

    for i in 0..out_dim1.len() {
        assert_eq!(
            out_dim1[i], expected_dim1[i],
            "dim1 mismatch at {i}: got {}, expected {}",
            out_dim1[i], expected_dim1[i]
        );
    }
}
