use as_any::Downcast;
use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
use luminal::prelude::*;

use crate::kernel::KernelOp;
use crate::kernel::other_ops::{KernelFusedElementwise, UnaryFn};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{
    TOLERANCE_SAFETY_FACTOR, dtype_epsilon, random_f32_vec, test_binary_cuda, test_unary_cuda,
};

/// Return every distinct kernel_name that appears across many random extractions
/// of the search space. Used to check whether fusion produces a reachable
/// `KernelFusedElementwise` node (or, negatively, that it never does).
fn extract_all_kernel_names(cx: &mut Graph) -> Vec<String> {
    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut all_names = Vec::new();
    for _ in 0..50 {
        let choices = random_initial_choice(egraph, &mut rand::rng());
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        for op in llir.node_weights() {
            if let Some(k) = op.to_dialect::<dyn KernelOp>() {
                let name = k.kernel_name().to_string();
                if !all_names.contains(&name) {
                    all_names.push(name);
                }
            }
        }
    }
    all_names
}

/// Return every distinct `Vec<UnaryFn>` that appears inside a reachable
/// `KernelFusedElementwise` across many random extractions. Used to verify
/// that a specific fused configuration (e.g. a 3-op chain) is reachable.
fn extract_all_fused_configs(cx: &mut Graph) -> Vec<Vec<UnaryFn>> {
    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut all_configs: Vec<Vec<UnaryFn>> = Vec::new();
    for _ in 0..200 {
        let choices = random_initial_choice(egraph, &mut rand::rng());
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        for op in llir.node_weights() {
            if let Some(kop) = op.to_dialect::<dyn KernelOp>()
                && let Some(fused) = (***kop).downcast_ref::<KernelFusedElementwise>()
            {
                let cfg = fused.ops().to_vec();
                if !all_configs.contains(&cfg) {
                    all_configs.push(cfg);
                }
            }
        }
    }
    all_configs
}

#[test]
fn test_two_unary_ops_fuse() {
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let _b = a.sin().sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        names.iter().any(|n| n == "FusedElementwise"),
        "expected KernelSin→KernelSqrt on contiguous strides to be fusable into \
         a single FusedElementwise kernel, but reachable kernels were: {names:?}",
    );
}

#[test]
fn test_stride_mismatch_prevents_fusion() {
    // A permute between sin and sqrt gives sqrt a non-contiguous view of sin's
    // contiguous output, so sqrt's in_strides != its out_strides and the
    // non-linear `?strides` match in the fusion rule can't fire.
    let mut cx = Graph::new();
    let a = cx.tensor((3, 4));
    let _b = a.sin().permute((1, 0)).sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        !names.iter().any(|n| n == "FusedElementwise"),
        "a permute between sin and sqrt must prevent fusion, but \
         FusedElementwise appeared in reachable kernels: {names:?}",
    );
}

#[test]
fn test_reduction_prevents_unary_fusion() {
    // A reduction between two unaries is not elementwise, so the fusion rule
    // (which only matches unary+unary pairs) must not fire.
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let _b = a.sin().sum(1).sqrt().output();

    let names = extract_all_kernel_names(&mut cx);
    assert!(
        !names.iter().any(|n| n == "FusedElementwise"),
        "a reduction between sin and sqrt must prevent fusion, but \
         FusedElementwise appeared in reachable kernels: {names:?}",
    );
}

#[test]
fn test_unary_fusion_preserves_output() {
    // End-to-end numerical check: sqrt(sin(x)) must produce the same values
    // whether or not the fusion rule fired. Runs on GPU when available;
    // silently no-ops otherwise via get_cuda_stream().
    let seed = 0xC0FFEEu64;
    let gen_lambda = |n, s| random_f32_vec(n, s, 0.0, 1.0);
    test_unary_cuda::<f32>(
        8,
        |a| a.sin().sqrt(),
        |a| a.sin().unwrap().sqrt().unwrap(),
        gen_lambda,
        seed,
    );
}

#[test]
fn test_three_unary_ops_fuse() {
    // A chain of 3 pure-elementwise unaries with matching strides should be
    // reachable as a single FusedElementwise containing all three ops.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().output();

    let configs = extract_all_fused_configs(&mut cx);
    let expected = vec![UnaryFn::Sin, UnaryFn::Sqrt, UnaryFn::Exp2];
    assert!(
        configs.contains(&expected),
        "expected a Fused[Sin, Sqrt, Exp2] in reachable configs, got: {configs:?}",
    );
}

#[test]
fn test_four_unary_ops_fuse() {
    // 4-op chain should collapse into a single Fused containing all four ops.
    let mut cx = Graph::new();
    let a = cx.tensor(16);
    let _b = a.sin().sqrt().exp2().log2().output();

    let configs = extract_all_fused_configs(&mut cx);
    let expected = vec![UnaryFn::Sin, UnaryFn::Sqrt, UnaryFn::Exp2, UnaryFn::Log2];
    assert!(
        configs.contains(&expected),
        "expected a Fused[Sin, Sqrt, Exp2, Log2] in reachable configs, got: {configs:?}",
    );
}

#[test]
fn test_three_unary_chain_preserves_output() {
    // End-to-end numerical check for a 3-op chain.
    // Uses sin→sqrt→sin because candle lacks exp2/log2 and this still exercises
    // a 3-link chain. The structural tests above cover the distinct-ops shape.
    let seed = 0xBEEFu64;
    let gen_lambda = |n, s| random_f32_vec(n, s, 0.0, 1.0);
    test_unary_cuda::<f32>(
        16,
        |a| a.sin().sqrt().sin(),
        |a| a.sin().unwrap().sqrt().unwrap().sin().unwrap(),
        gen_lambda,
        seed,
    );
}

/// Isolated per-kernel microbenchmark: time two unfused kernels
/// (`sqrt_k` then `recip_k`) vs one fused kernel (`fused_k` that does
/// `1.0f / sqrtf(x)` in a single launch) on a fixed-size input, using
/// CUDA events for device-side timing.
///
/// Ignored by default — run with
/// `cargo test -p luminal_cuda_lite -- --ignored bench_fused_vs_unfused_sqrt_recip --nocapture`.
#[test]
#[ignore]
fn bench_fused_vs_unfused_sqrt_recip() {
    use crate::compile_module_image_for_current_device;
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

    const N: usize = 1 << 20; // 1M elements
    const WARMUP: usize = 100;
    const TRIALS: usize = 2000;

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => return, // no GPU available, skip
    };
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    // Prepare input (values in (0, 1] so sqrt/recip are well-defined).
    let host_input: Vec<f32> = (0..N).map(|i| (i as f32 + 1.0) / (N as f32)).collect();
    let d_in = stream.clone_htod(&host_input).unwrap();
    let mut d_scratch = stream.alloc_zeros::<f32>(N).unwrap();
    let mut d_out = stream.alloc_zeros::<f32>(N).unwrap();

    let compile = |src: &str, name: &str| {
        let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        module.load_function(name).unwrap()
    };

    let sqrt_k = compile(
        r#"
extern "C" __global__ void sqrt_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sqrtf(in[i]);
}
"#,
        "sqrt_k",
    );
    let recip_k = compile(
        r#"
extern "C" __global__ void recip_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / in[i];
}
"#,
        "recip_k",
    );
    let fused_k = compile(
        r#"
extern "C" __global__ void fused_k(float* out, const float* in, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    v = sqrtf(v);
    v = 1.0f / v;
    out[i] = v;
}
"#,
        "fused_k",
    );

    let cfg = LaunchConfig::for_num_elems(N as u32);
    let n_arg: i64 = N as i64;

    let launch_unfused = |d_out: &mut cudarc::driver::CudaSlice<f32>,
                          d_scratch: &mut cudarc::driver::CudaSlice<f32>| {
        let mut b = stream.launch_builder(&sqrt_k);
        b.arg(&mut *d_scratch).arg(&d_in).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
        let mut b = stream.launch_builder(&recip_k);
        b.arg(d_out).arg(&*d_scratch).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
    };
    let launch_fused = |d_out: &mut cudarc::driver::CudaSlice<f32>| {
        let mut b = stream.launch_builder(&fused_k);
        b.arg(d_out).arg(&d_in).arg(&n_arg);
        unsafe { b.launch(cfg) }.unwrap();
    };

    // Warmup
    for _ in 0..WARMUP {
        launch_unfused(&mut d_out, &mut d_scratch);
        launch_fused(&mut d_out);
    }
    stream.synchronize().unwrap();

    let start = ctx.new_event(None).unwrap();
    let end = ctx.new_event(None).unwrap();

    // Time unfused
    start.record(&stream).unwrap();
    for _ in 0..TRIALS {
        launch_unfused(&mut d_out, &mut d_scratch);
    }
    end.record(&stream).unwrap();
    end.synchronize().unwrap();
    let unfused_total_ms = start.elapsed_ms(&end).unwrap();

    // Time fused
    start.record(&stream).unwrap();
    for _ in 0..TRIALS {
        launch_fused(&mut d_out);
    }
    end.record(&stream).unwrap();
    end.synchronize().unwrap();
    let fused_total_ms = start.elapsed_ms(&end).unwrap();

    let unfused_us = unfused_total_ms as f64 * 1_000.0 / TRIALS as f64;
    let fused_us = fused_total_ms as f64 * 1_000.0 / TRIALS as f64;
    let speedup = unfused_us / fused_us;

    println!(
        "\n[fusion microbench, N={N}, trials={TRIALS}]\n\
         unfused (sqrt_k; recip_k):  {unfused_us:8.3} us/iter ({unfused_total_ms:.2} ms total)\n\
         fused   (sqrtf; 1.0f/):     {fused_us:8.3} us/iter ({fused_total_ms:.2} ms total)\n\
         speedup: {speedup:.2}x"
    );
}

// =========================================================================
// Binary-inclusive fusion tests (marker-based FusionStart / FusionEnd scheme).
//
// Detects fused regions by walking backward from each `FusionEnd`-tagged LLIR
// node through `Direction::Incoming` edges until a `FusionStart` is reached.
// The walker stops at FusionStarts (they mark the external-input boundary of
// the region). A region's summary is: the sorted set of internal op names,
// the count of distinct FusionStart nodes reached, and the count of FusionEnd
// nodes (invariant: always 1 per region).
// =========================================================================

/// A single fused region extracted from the LLIR graph after egglog.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FusedRegion {
    /// Sorted internal op `kernel_name()`s, excluding the `FusionStart` /
    /// `FusionEnd` markers. Sorted so DAG traversal order doesn't produce
    /// spurious "distinct" regions.
    internal_ops_sorted: Vec<String>,
    /// Number of distinct `FusionStart` nodes reached by the walk. Per design
    /// this equals the number of distinct external input tensors.
    start_count: usize,
    /// Number of `FusionEnd` nodes in the region. Per design this is always 1.
    end_count: usize,
}

/// Helper: collect every distinct fused region reachable across many random
/// extractions. Parallel to `extract_all_fused_configs` but for the marker
/// encoding.
fn extract_all_fused_regions(cx: &mut Graph) -> Vec<FusedRegion> {
    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().expect("egraph not built");
    let ops = cx.egglog_ops().expect("ops not built");
    let custom_ops = &cx.custom_ops;

    let mut seen: Vec<FusedRegion> = Vec::new();
    for _ in 0..50 {
        let choices = random_initial_choice(egraph, &mut rand::rng());
        let mut list_cache = Default::default();
        let mut expr_cache = Default::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );

        let name_of = |idx: NodeIndex| -> Option<String> {
            llir.node_weight(idx).and_then(|op| {
                op.to_dialect::<dyn KernelOp>()
                    .map(|k| k.kernel_name().to_string())
            })
        };

        let end_nodes: Vec<NodeIndex> = llir
            .node_indices()
            .filter(|&idx| name_of(idx).as_deref() == Some("FusionEnd"))
            .collect();

        for end in end_nodes {
            let mut internal: Vec<String> = Vec::new();
            let mut starts: FxHashSet<NodeIndex> = FxHashSet::default();
            let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
            visited.insert(end);
            let mut stack = vec![end];

            while let Some(node) = stack.pop() {
                for pred in llir.neighbors_directed(node, petgraph::Direction::Incoming) {
                    if !visited.insert(pred) {
                        continue;
                    }
                    match name_of(pred).as_deref() {
                        Some("FusionStart") => {
                            starts.insert(pred);
                            // Do not walk past a FusionStart — it's the boundary.
                        }
                        Some(other) => {
                            internal.push(other.to_string());
                            stack.push(pred);
                        }
                        None => {
                            // Non-KernelOp predecessor (shouldn't appear inside a
                            // fused region under the design). Stop walking this path.
                        }
                    }
                }
            }

            internal.sort();
            let region = FusedRegion {
                internal_ops_sorted: internal,
                start_count: starts.len(),
                end_count: 1,
            };
            if !seen.contains(&region) {
                seen.push(region);
            }
        }
    }
    seen
}

fn sorted_names(items: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(|s| (*s).to_string()).collect();
    v.sort();
    v
}

// ---- Structural tests: the expected fused shape is reachable ----

#[test]
fn test_single_binary_fuses() {
    // `a + b` with contiguous matching strides should be reachable as a single
    // fused region: one internal KernelAdd, two FusionStarts.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a + b).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["Add"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_chain_of_binaries_fuses() {
    // `(a + b) * c`: three external inputs collapse into one region with
    // internal [Add, Mul] and 3 FusionStarts.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let c = cx.tensor(8);
    let _d = ((a + b) * c).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["Add", "Mul"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 3),
        "expected a fused region of {expected:?} with 3 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_binary_then_unary_fuses() {
    // `sin(a + b)`: binary feeds a unary inside one fused region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a + b).sin().output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["Add", "Sin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_unary_then_binary_fuses() {
    // `sin(a) + b`: unary feeds a binary inside one fused region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let _c = (a.sin() + b).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["Add", "Sin"]);
    assert!(
        regions
            .iter()
            .any(|r| r.internal_ops_sorted == expected && r.start_count == 2),
        "expected a fused region of {expected:?} with 2 FusionStarts, got: {regions:#?}"
    );
}

#[test]
fn test_diamond_dag_fuses() {
    // The canonical diamond-DAG example agreed with the user:
    //   t = a + b; u = exp(t); v = sin(t); w = u * a; out = w + v
    // `a` is reused (feeds outer Add and Mul) and `t` is reused (feeds Exp and
    // Sin). Expected: one fused region with internal ops [Add, Add, Exp, Mul,
    // Sin], 2 FusionStarts (distinct tensors a, b), 1 FusionEnd.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let expected = sorted_names(&["Add", "Add", "Exp", "Mul", "Sin"]);
    assert!(
        regions.iter().any(|r| r.internal_ops_sorted == expected
            && r.start_count == 2
            && r.end_count == 1),
        "expected diamond DAG to fuse into one region with ops {expected:?}, \
         2 FusionStarts, 1 FusionEnd. Got: {regions:#?}"
    );
}

// ---- Negative tests: fusion must NOT happen across these blockers ----

#[test]
fn test_reduction_blocks_binary_fusion() {
    // A reduction between a binary and anything downstream is not elementwise,
    // so Add and SumReduce must never appear in the same fused region.
    let mut cx = Graph::new();
    let a = cx.tensor((4, 4));
    let b = cx.tensor((4, 4));
    let _c = (a + b).sum(1).output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        let has_add = r.internal_ops_sorted.iter().any(|n| n == "Add");
        let has_sum = r.internal_ops_sorted.iter().any(|n| n == "SumReduce");
        assert!(
            !(has_add && has_sum),
            "Add and SumReduce must not share a fused region, but got: {r:#?}"
        );
    }
}

#[test]
fn test_stride_mismatch_blocks_binary_fusion() {
    // A permute gives `b` a non-contiguous view whose strides do not match `a`'s,
    // so the binary fusion rule's stride-compatibility check must prevent the
    // Add from being absorbed into any fused region.
    let mut cx = Graph::new();
    let a = cx.tensor((3, 4));
    let b = cx.tensor((4, 3));
    let _c = (a + b.permute((1, 0))).output();

    let regions = extract_all_fused_regions(&mut cx);
    for r in &regions {
        assert!(
            !r.internal_ops_sorted.iter().any(|n| n == "Add"),
            "permuted binary must not fuse into a region, but found: {r:#?}"
        );
    }
}

// ---- Numerical parity tests: fused output matches candle reference ----

#[test]
fn test_simple_binary_fusion_preserves_output() {
    // End-to-end numerical check: `a + b` on GPU matches candle's add across
    // all reachable genomes (fused or unfused) via test_binary_cuda's fuzzer.
    let seed = 0xADDBEEFu64;
    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    test_binary_cuda::<f32>(
        16,
        16,
        |a, b| a + b,
        |a, b| (a + b).unwrap(),
        |n, s| random_f32_vec(n, s, 0.0, 1.0),
        |n, s| random_f32_vec(n, s, 0.0, 1.0),
        seed,
        tol,
        tol,
    );
}

#[test]
fn test_diamond_dag_preserves_output() {
    // Numerical parity for the diamond DAG: `(exp(a+b) * a) + sin(a+b)`
    // matches candle's equivalent across fused and unfused genomes.
    // Inputs are drawn from [-1, 1] so exp() doesn't overflow.
    let seed = 0xD1A_0D1Au64;
    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    // Five-op chain with exp + sin: allow ~5x safety to absorb accumulated
    // rounding vs candle's kernels.
    let tol = eps * TOLERANCE_SAFETY_FACTOR * 5.0;
    test_binary_cuda::<f32>(
        16,
        16,
        |a, b| {
            let t = a + b;
            let u = t.exp();
            let v = t.sin();
            let w = u * a;
            w + v
        },
        |a, b| {
            let t = (&a + &b).unwrap();
            let u = t.exp().unwrap();
            let v = t.sin().unwrap();
            let w = (&u * &a).unwrap();
            (&w + &v).unwrap()
        },
        |n, s| random_f32_vec(n, s, -1.0, 1.0),
        |n, s| random_f32_vec(n, s, -1.0, 1.0),
        seed,
        tol,
        tol,
    );
}

// ---- Marker invariant tests ----

#[test]
fn test_fused_region_has_exactly_one_end() {
    // Design invariant: a fused region always has exactly one FusionEnd.
    // Uses the diamond DAG so there's real fan-in/out inside the region.
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let full = regions
        .iter()
        .find(|r| r.internal_ops_sorted.len() == 5)
        .expect("expected at least one extraction to produce the full 5-op fused region");
    assert_eq!(
        full.end_count, 1,
        "fused region must have exactly one FusionEnd, got {}",
        full.end_count
    );
}

#[test]
fn test_fused_region_starts_match_distinct_external_tensors() {
    // Design invariant: FusionStart count == number of distinct external input
    // tensors, NOT number of edges crossing the boundary. In the diamond DAG
    // `a` is consumed inside the region by two ops (outer Add + Mul), so a
    // per-edge counting scheme would give 3; the correct per-distinct-tensor
    // count is 2 ({a, b}).
    let mut cx = Graph::new();
    let a = cx.tensor(8);
    let b = cx.tensor(8);
    let t = a + b;
    let u = t.exp();
    let v = t.sin();
    let w = u * a;
    let _out = (w + v).output();

    let regions = extract_all_fused_regions(&mut cx);
    let full = regions
        .iter()
        .find(|r| r.internal_ops_sorted.len() == 5)
        .expect("expected at least one extraction to produce the full 5-op fused region");
    assert_eq!(
        full.start_count, 2,
        "FusionStart count must equal distinct external tensors (expected 2 for {{a, b}}), \
         got {}. If this is 3, FusionStart is being counted per edge, not per tensor.",
        full.start_count
    );
}
