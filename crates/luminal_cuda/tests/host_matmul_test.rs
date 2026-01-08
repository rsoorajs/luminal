use luminal::prelude::*;
use luminal_cuda::runtime::CudaRuntime;
use ndarray::Array;

#[cfg(test)]
mod tests {
    use luminal::visualization::ToDot;

    use super::*;

    fn dynamic_matmul_graph_runtime() -> (Graph, CudaRuntime, GraphTensor, GraphTensor, GraphTensor)
    {
        let mut cx = Graph::new();
        let graph_a = cx.tensor(('m', 'k'));
        let graph_b = cx.tensor(('k', 'n'));

        let graph_c: GraphTensor = graph_a.matmul(graph_b).output();

        cx.build_search_space::<CudaRuntime>();

        let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();
        ctx.set_flags(luminal_cuda::cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)
            .unwrap();
        let stream = ctx.default_stream();

        let custom_state = FxHashMap::default();

        let mut rt = CudaRuntime::initialize((ctx.clone(), stream.clone(), custom_state));

        let example_m = 100;
        let example_n = 100;
        let example_k = 100;

        cx.set_dyn_dim('m', 100);
        cx.set_dyn_dim('n', 100);
        cx.set_dyn_dim('k', 100);

        let a = vec![1.0; example_m * example_k];
        let b = vec![1.0; example_k * example_n];

        rt.set_data(graph_a, Box::new(a.clone()));
        rt.set_data(graph_b, Box::new(b.clone()));

        rt = cx.search(rt, 1);
        assert!(rt.llir_graph.to_dot().unwrap().contains("HostMatmul"));

        return (cx, rt, graph_a, graph_b, graph_c);
    }

    fn cuda_matmul(
        mut cx: Graph,
        mut rt: CudaRuntime,
        graph_a: GraphTensor,
        graph_b: GraphTensor,
        graph_c: GraphTensor,
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Vec<f32> {
        assert!(m > 0);
        assert!(n > 0);
        assert!(k > 0);

        assert!(a.len() == ((m * k) as usize));
        assert!(b.len() == ((n * k) as usize));

        // Set input tensors
        rt.set_data(graph_a, Box::new(a.clone()));
        rt.set_data(graph_b, Box::new(b.clone()));

        cx.set_dyn_dim('m', m as usize);
        cx.set_dyn_dim('n', n as usize);
        cx.set_dyn_dim('k', k as usize);

        rt.allocate_intermediate_buffers(&cx.dyn_map);

        rt.execute(&cx.dyn_map);

        rt.get_f32(graph_c)
    }

    fn cpu_matmul(a: &Vec<f32>, b: &Vec<f32>, m: i32, n: i32, k: i32) -> Vec<f32> {
        assert!(m > 0);
        assert!(n > 0);
        assert!(k > 0);

        assert!(a.len() == ((m * k) as usize));
        assert!(b.len() == ((k * n) as usize));

        // Convert vectors to ndarray matrices
        let a_matrix = Array::from_shape_vec((m as usize, k as usize), a.clone()).unwrap();
        let b_matrix = Array::from_shape_vec((k as usize, n as usize), b.clone()).unwrap();

        // Perform matrix multiplication
        let c_matrix = a_matrix.dot(&b_matrix);

        // Convert result back to Vec<f32>
        c_matrix.iter().copied().collect()
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]
            #[test]
            fn test_cuda_vs_cpu_matmul(
                m in 1i32..=64,
                n in 1i32..=64,
                k in 1i32..=64,
                seed in any::<u64>(),
            ) {
                use rand::{Rng, SeedableRng};
                use rand::rngs::StdRng;

                // Create RNG from seed
                let mut rng = StdRng::seed_from_u64(seed);

                // Generate random input vectors
                let a: Vec<f32> = (0..(m * k))
                    .map(|_| rng.gen_range(-10.0..10.0))
                    .collect();
                let b: Vec<f32> = (0..(k * n))
                    .map(|_| rng.gen_range(-10.0..10.0))
                    .collect();

                let (cx, rt, graph_a, graph_b, graph_c) = dynamic_matmul_graph_runtime();

                // Run both implementations
                let cuda_result = cuda_matmul(cx, rt, graph_a, graph_b, graph_c, &a, &b, m, n, k);
                let cpu_result = cpu_matmul(&a, &b, m, n, k);

                // Compare results
                prop_assert_eq!(cuda_result.len(), cpu_result.len());
                for (i, (cuda_val, cpu_val)) in cuda_result.iter().zip(cpu_result.iter()).enumerate() {
                    let diff = (cuda_val - cpu_val).abs();
                    prop_assert!(
                        diff < 1e-3,
                        "Mismatch at index {}: cuda={}, cpu={}, diff={}",
                        i, cuda_val, cpu_val, diff
                    );
                }
            }
        }
    }
}
