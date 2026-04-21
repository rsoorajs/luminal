use as_any::Downcast;
use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
use luminal::prelude::*;

use crate::kernel::KernelOp;
use crate::kernel::other_ops::{KernelFusedElementwise, UnaryFn};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::{random_f32_vec, test_unary_cuda};

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
