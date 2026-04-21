use luminal::egglog_utils::{egglog_to_llir, random_initial_choice};
use luminal::prelude::*;

use crate::kernel::KernelOp;
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
