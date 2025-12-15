use luminal::{
    graph::{shape_to_egglog, strides_to_egglog, Graph},
    module::Module,
    op::Operator,
    prelude::GraphTensor,
    shape::{Expression, ShapeTracker},
};
use luminal_nn::LayerNorm;
use std::fmt::Debug;

pub struct Llama {
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Llama {
    pub fn init(
        cx: &mut Graph,
        batch: impl Into<Expression>,
        hidden: usize,
        intermediate: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        layers: usize,
    ) -> Self {
        let batch = batch.into();
        let mut w = vec![];
        let n_kv_groups = n_heads / n_kv_heads;
        for l in 0..layers {
            w.push(LlamaLayer {
                up: cx.named_tensor(
                    &format!("model.layers.{l}.mlp.up_proj.weight"),
                    (intermediate, hidden),
                ),
                gate: cx.named_tensor(
                    &format!("model.layers.{l}.mlp.gate_proj.weight"),
                    (intermediate, hidden),
                ),
                down: cx.named_tensor(
                    &format!("model.layers.{l}.mlp.down_proj.weight"),
                    (hidden, intermediate),
                ),
                q_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (hidden, hidden),
                ),
                k_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (hidden / n_kv_groups, hidden),
                ),
                v_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (hidden / n_kv_groups, hidden),
                ),
                o_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (hidden, hidden),
                ),
                attn_rms: LayerNorm::new(
                    hidden,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    hidden,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                batch,
                hidden,
                n_heads,
                n_kv_heads,
                layer: l,
            });
        }
        let lm_norm = LayerNorm::new(hidden, Some("model.norm.weight"), None, false, 1e-5, cx);
        let lm_head = cx.named_tensor("lm_head.weight", (vocab_size, hidden));
        Self {
            layers: w,
            lm_head,
            lm_norm,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(&self, input: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x, token_ids);
        }
        self.lm_norm.forward(x).matmul(self.lm_head.transpose(0, 1))
    }
}

struct LlamaLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
    batch: Expression,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    layer: usize,
}

impl LlamaLayer {
    pub fn forward(&self, input: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
        let cx = input.graph();
        let x_attn = self.attn_rms.forward(input);
        let q = x_attn.matmul(self.q_proj.transpose(0, 1));
        let k = x_attn.matmul(self.k_proj.transpose(0, 1));
        let v = x_attn.matmul(self.v_proj.transpose(0, 1));
        let q_rope = GraphTensor::from_id(
            cx.add_op(RopeFrontendOp {
                range: vec![Expression::from(self.batch)],
                stride: vec![Expression::from('z') * self.hidden],
                row_width: Expression::from(self.hidden),
            })
            .input(q.id, 0, q.shape)
            .input(token_ids.id, 0, token_ids.shape)
            .finish(),
            q.shape,
            cx,
        );
        let n_kv_groups = self.n_heads / self.n_kv_heads;
        let k_rope = GraphTensor::from_id(
            cx.add_op(RopeFrontendOp {
                range: vec![Expression::from(self.batch)],
                stride: vec![Expression::from('z') * (self.hidden / n_kv_groups)],
                row_width: Expression::from(self.hidden / n_kv_groups),
            })
            .input(k.id, 0, k.shape)
            .input(token_ids.id, 0, token_ids.shape)
            .finish(),
            k.shape,
            cx,
        );

        let attn_out = GraphTensor::from_id(
            cx.add_op(GQAAttentionFrontendOp {
                head_dim: 128,
                prev_seq: 'p'.into(),
                layer: self.layer,
            })
            .input(q_rope.id, 0, q_rope.shape)
            .input(k_rope.id, 0, k_rope.shape)
            .input(v.id, 0, v.shape)
            .finish(),
            ShapeTracker::new((self.batch, self.hidden)),
            cx,
        );
        let attn_out = attn_out.matmul(self.o_proj.transpose(0, 1));
        let resid1 = input + attn_out;
        let x_mlp = self.mlp_rms.forward(resid1);
        resid1
            + (x_mlp.matmul(self.gate.transpose(0, 1)).swish()
                * x_mlp.matmul(self.up.transpose(0, 1)))
            .matmul(self.down.transpose(0, 1))
    }
}

#[derive(Debug)]
pub struct RopeFrontendOp {
    pub range: Vec<Expression>,
    pub stride: Vec<Expression>,
    pub row_width: Expression,
}

impl Operator for RopeFrontendOp {
    fn process(
        &mut self,
        _inp: Vec<(
            luminal::prelude::InputTensor,
            luminal::prelude::ShapeTracker,
        )>,
    ) -> Vec<luminal::prelude::Tensor> {
        todo!()
    }

    fn to_egglog(
        &self,
        inputs: &Vec<(luminal::prelude::NodeIndex, String, ShapeTracker)>,
    ) -> String {
        format!(
            "(RowRope {} {} {} {} {})",
            shape_to_egglog(&self.range),
            inputs[0].1,
            strides_to_egglog(&self.stride),
            self.row_width.to_egglog(),
            inputs[1].1,
        )
    }
}

#[derive(Debug)]
pub struct PrintFrontendOp {
    pub path: String,
}

impl Operator for PrintFrontendOp {
    fn process(
        &mut self,
        _inp: Vec<(
            luminal::prelude::InputTensor,
            luminal::prelude::ShapeTracker,
        )>,
    ) -> Vec<luminal::prelude::Tensor> {
        todo!()
    }

    fn to_egglog(&self, _: &Vec<(luminal::prelude::NodeIndex, String, ShapeTracker)>) -> String {
        todo!()
    }
}

#[derive(Debug)]
pub struct GQAAttentionFrontendOp {
    pub head_dim: usize,
    pub prev_seq: Expression,
    pub layer: usize,
}

impl Operator for GQAAttentionFrontendOp {
    fn process(
        &mut self,
        _inp: Vec<(
            luminal::prelude::InputTensor,
            luminal::prelude::ShapeTracker,
        )>,
    ) -> Vec<luminal::prelude::Tensor> {
        todo!()
    }

    fn to_egglog(
        &self,
        inputs: &Vec<(luminal::prelude::NodeIndex, String, ShapeTracker)>,
    ) -> String {
        let seq = inputs[0].2.dims[0];
        let hidden = inputs[0].2.dims[1].to_usize().unwrap();
        let kv_hidden = inputs[1].2.dims[1].to_usize().unwrap();
        let kv_row_width = inputs[1].2.dims[1].to_usize().unwrap();
        let n_heads = hidden / self.head_dim;
        let n_kv_heads = kv_hidden / self.head_dim;
        let n_kv_groups = n_heads / n_kv_heads;
        format!(
            "(GQAAttention {} {} {} {} {} {} {} {} {} {} {} {} {})",
            shape_to_egglog(&[n_kv_heads.into(), n_kv_groups.into(), seq]),
            Expression::from(self.head_dim).to_egglog(),
            seq.to_egglog(),
            Expression::from(kv_row_width).to_egglog(),
            inputs[0].1,
            strides_to_egglog(&[
                Expression::from(self.head_dim * n_kv_groups) * 'z',
                Expression::from(self.head_dim) * 'z',
                Expression::from(hidden) * 'z'
            ]),
            inputs[1].1,
            strides_to_egglog(&[Expression::from(self.head_dim) * 'z', 0.into(), 0.into()]),
            inputs[2].1,
            strides_to_egglog(&[Expression::from(self.head_dim) * 'z', 0.into(), 0.into()]),
            strides_to_egglog(&[
                Expression::from(self.head_dim * n_kv_groups) * 'z',
                Expression::from(self.head_dim) * 'z',
                Expression::from(hidden) * 'z'
            ]),
            self.prev_seq.to_egglog(),
            self.layer,
        )
    }
}

#[allow(unused)]
fn print_tensor(tensor: GraphTensor, path: impl ToString) -> GraphTensor {
    let cx = tensor.graph();
    GraphTensor::from_id(
        cx.add_op(PrintFrontendOp {
            path: path.to_string(),
        })
        .input(tensor.id, 0, tensor.shape)
        .finish(),
        tensor.shape,
        cx,
    )
}
