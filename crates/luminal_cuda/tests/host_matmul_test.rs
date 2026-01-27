use candle_core::{Device, Tensor};
use luminal::prelude::*;
use luminal_cuda::runtime::CudaRuntime;
use luminal::visualization::ToDot;

#[cfg(test)]
mod tests {

    use super::*;


    /// Convert a row-major matrix B (k x n) to column-major layout using candle
    fn to_col_major(b_row_major: &[f32], k: usize, n: usize) -> Vec<f32> {
        let device = Device::Cpu;
        let b_tensor = Tensor::from_vec(b_row_major.to_vec(), (k, n), &device).unwrap();
        let b_transposed = b_tensor.t().unwrap().contiguous().unwrap();
        b_transposed
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn dynamic_matmul_graph_runtime() -> (Graph, CudaRuntime, GraphTensor, GraphTensor, GraphTensor)
    {
        let mut cx = Graph::new();
        let graph_a = cx.tensor(('m', 'k'));
        let graph_b = cx.tensor(('k', 'n'));

        let graph_c: GraphTensor = graph_a.matmul(graph_b).output();

        cx.build_search_space::<CudaRuntime>();

        let mut rt = CudaRuntime::new().unwrap();

        let example_m = 1000;
        let example_n = 1000;
        let example_k = 1000;

        cx.set_dim('m', example_m);
        cx.set_dim('n', example_n);
        cx.set_dim('k', example_k);

        let a = vec![1.0; example_m * example_k];
        let b_row_major = vec![1.0; example_k * example_n];
        let b_col_major = to_col_major(&b_row_major, example_k, example_n);

        rt.set_data(graph_a, a.clone());
        rt.set_data(graph_b, b_col_major.clone());

        cx.build_search_space::<CudaRuntime>();
        
        rt = cx.search(rt, 10);
        
        if !rt.llir_graph.to_dot().unwrap().contains("Cublas"){
            dbg!(rt.llir_graph.to_dot().unwrap());
            dbg!("NO CUBLAS");
        }
        // assert!(rt.llir_graph.to_dot().unwrap().contains("Cublas"));

        return (cx, rt, graph_a, graph_b, graph_c);
    }

    fn cuda_matmul(
        mut cx: Graph,
        mut rt: CudaRuntime,
        graph_a: GraphTensor,
        graph_b: GraphTensor,
        graph_c: GraphTensor,
        a: &Vec<f32>,
        b_row_major: &Vec<f32>,
        m: i32,
        n: i32,
        k: i32,
    ) -> Vec<f32> {
        assert!(m > 0);
        assert!(n > 0);
        assert!(k > 0);

        assert!(a.len() == ((m * k) as usize));
        assert!(b_row_major.len() == ((k * n) as usize));

        // Convert B from row-major to column-major for luminal
        let b_col_major = to_col_major(b_row_major, k as usize, n as usize);

        // Set input tensors
        rt.set_data(graph_a, a.clone());
        rt.set_data(graph_b, b_col_major);

        cx.set_dim('m', m as usize);
        cx.set_dim('n', n as usize);
        cx.set_dim('k', k as usize);

        rt.execute(&cx.dyn_map);

        rt.get_f32(graph_c)
    }

    fn cpu_matmul(a: &Vec<f32>, b: &Vec<f32>, m: i32, n: i32, k: i32) -> Vec<f32> {
        assert!(m > 0);
        assert!(n > 0);
        assert!(k > 0);

        assert!(a.len() == ((m * k) as usize));
        assert!(b.len() == ((k * n) as usize));

        // Use candle for CPU reference matmul
        let device = Device::Cpu;

        // A is row-major (m x k)
        let a_tensor = Tensor::from_vec(a.clone(), (m as usize, k as usize), &device).unwrap();

        // B is row-major (k x n)
        let b_tensor = Tensor::from_vec(b.clone(), (k as usize, n as usize), &device).unwrap();

        // Perform matrix multiplication
        let c_tensor = a_tensor.matmul(&b_tensor).unwrap();

        // Convert result to flat Vec<f32>
        c_tensor
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(32))]
            #[test]
            fn test_cuda_vs_cpu_matmul(
                m in 1i32..=32,
                n in 1i32..=32,
                k in 1i32..=32,
                seed in any::<u64>(),
            ) {
                use rand::{Rng, SeedableRng};
                use rand::rngs::StdRng;

                // Create RNG from seed
                let mut rng = StdRng::seed_from_u64(seed);

                // Generate random input vectors
                let a: Vec<f32> = (0..(m * k))
                    .map(|_| rng.random_range(-10.0..10.0))
                    .collect();
                let b: Vec<f32> = (0..(k * n))
                    .map(|_| rng.random_range(-10.0..10.0))
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
