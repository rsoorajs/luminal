use luminal::{dtype::DType, prelude::*, shape::Expression};

use super::utilities::get_cuda_stream;
use crate::{host::HostOp, runtime::CudaRuntime};

const SEQ: usize = 1;
const HIDDEN: usize = 16;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTERMEDIATE: usize = 6;

fn build_qwen_moe_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::default();
    let x = cx.tensor(('s', HIDDEN));
    let router = cx.tensor((NUM_EXPERTS, HIDDEN));
    let gate_up_weights = cx
        .tensor((NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = x.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let routing_weights = x.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);

    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);

    let gate_up_gathered = {
        let (_, d1, d2) = gate_up_weights.dims3();
        let io = d1 * d2;
        let base = top_k_indices * io;
        let within = x.graph().iota(Expression::from('z'), (d1, d2));
        let n_base = base.dims().len();
        let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }
        let expert_flat_idx = exp_base + exp_within;
        gate_up_weights.gather(expert_flat_idx).cast(DType::F32)
    };

    let x_exp = x.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gate.silu() * up;

    let down_gathered = {
        let (_, d1, d2) = down_weights.dims3();
        let io = d1 * d2;
        let base = top_k_indices * io;
        let within = x.graph().iota(Expression::from('z'), (d1, d2));
        let n_base = base.dims().len();
        let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }
        let expert_flat_idx = exp_base + exp_within;
        down_weights.gather(expert_flat_idx).cast(DType::F32)
    };

    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let out = (down_out * top_k_values.unsqueeze(top_k_values.dims().len())).sum(n - 1);
    out.output();

    (cx, x, router, gate_up_weights, down_weights)
}

#[test]
fn test_glumoe_matches_integer_index_pattern() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, router, gate_up_weights, down_weights) = build_qwen_moe_graph();
    cx.set_dim('s', SEQ);
    cx.build_search_space::<CudaRuntime>();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, vec![0.0f32; SEQ * HIDDEN]);
    rt.set_data(router, vec![0.0f32; NUM_EXPERTS * HIDDEN]);
    rt.set_data(
        gate_up_weights,
        vec![half::bf16::from_f32(0.0); NUM_EXPERTS * MOE_INTERMEDIATE * 2 * HIDDEN],
    );
    rt.set_data(
        down_weights,
        vec![half::bf16::from_f32(0.0); NUM_EXPERTS * HIDDEN * MOE_INTERMEDIATE],
    );
    rt = cx.search(rt, 10);

    let has_glumoe = rt.llir_graph().node_weights().any(|node| {
        node.to_dialect::<dyn HostOp>()
            .and_then(|op| op.stats_name())
            == Some("GLUMoE")
    });

    assert!(
        has_glumoe,
        "Expected GLUMoE host op to match the integer-only Qwen MoE gather pattern"
    );
}
