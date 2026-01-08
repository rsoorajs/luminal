use luminal::{
    graph::{elist_to_egglog, Graph},
    op::{DType, HLIROp},
    prelude::{F32Pow, GraphTensor},
    shape::{Expression, ShapeTracker},
};
use luminal_nn::LayerNorm;
use std::fmt::Debug;

// Llama 7b hyperparams
pub const LAYERS: usize = 32;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 14336;
pub const HEAD_DIM: usize = 128;
pub const KV_GROUPS: usize = 4;
pub const VOCAB_SIZE: usize = 128256;

pub struct Llama {
    embedding: GraphTensor,
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Llama {
    pub fn init(cx: &mut Graph) -> Self {
        let mut w = vec![];
        for l in 0..LAYERS {
            w.push(LlamaLayer {
                up: cx.named_tensor(
                    format!("model.layers.{l}.mlp.up_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                gate: cx.named_tensor(
                    format!("model.layers.{l}.mlp.gate_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                down: cx.named_tensor(
                    format!("model.layers.{l}.mlp.down_proj.weight"),
                    (HIDDEN, INTERMEDIATE),
                ),
                q_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (HIDDEN, HIDDEN),
                ),
                k_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                ),
                v_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                ),
                o_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (HIDDEN, HIDDEN),
                ),
                attn_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                layer: l,
            });
        }
        let lm_norm = LayerNorm::new(HIDDEN, Some("model.norm.weight"), None, false, 1e-5, cx);
        let lm_head = cx.named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN));
        Self {
            embedding: cx.named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            layers: w,
            lm_head,
            lm_norm,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(&self, token_ids: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
        let batch = token_ids.dims1();
        let mut x = self.embedding.gather(
            (token_ids * HIDDEN).expand_dim(1, HIDDEN)
                + token_ids.graph().arange(HIDDEN).expand_dim(0, batch),
        );
        for layer in &self.layers {
            x = layer.forward(x, pos_ids);
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
    layer: usize,
}

fn apply_rotary_embeddings(mut input: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
    let orig_shape = input.shape;
    // Input: [seq, dim]
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // n_heads, seq, head_dim

    // Get freqs
    let freqs = input.graph().arange_options(0, HEAD_DIM, 2) / HEAD_DIM as f32;
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    // Split input into evens and odds
    let split = input.split_dims(2, 2);
    let x0 = split.slice((.., .., .., ..1));
    let x1 = split.slice((.., .., .., 1..));

    // Apply sin and cos embeddings
    let x0_out = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
    let x1_out = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

    // Combine back into output
    let mut s = x0_out.concat_along(x1_out, 3);
    s.shape = input.shape;
    s = s.transpose(0, 1) * 1.0;
    s.shape = orig_shape;
    s
}

impl LlamaLayer {
    pub fn forward(&self, input: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
        let cx = input.graph();
        let batch = input.dims()[0];
        let x_attn = self.attn_rms.forward(input);
        let q = x_attn.matmul(self.q_proj.transpose(0, 1));
        let k = x_attn.matmul(self.k_proj.transpose(0, 1));
        let v = x_attn.matmul(self.v_proj.transpose(0, 1));
        let q_rope = apply_rotary_embeddings(q, pos_ids);
        let k_rope = apply_rotary_embeddings(k, pos_ids);

        let attn_out = GraphTensor::from_id(
            cx.add_op(GQAAttentionFrontendOp {
                head_dim: HEAD_DIM,
                prev_seq: 'p'.into(),
                layer: self.layer,
            })
            .input(q_rope.id, 0, q_rope.shape)
            .input(k_rope.id, 0, k_rope.shape)
            .input(v.id, 0, v.shape)
            .finish(),
            ShapeTracker::new((batch, HIDDEN)),
            cx,
            DType::F32,
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
pub struct GQAAttentionFrontendOp {
    pub head_dim: usize,
    pub prev_seq: Expression,
    pub layer: usize,
}

impl HLIROp for GQAAttentionFrontendOp {
    fn to_egglog(&self, inputs: &[(luminal::prelude::NodeIndex, String, ShapeTracker)]) -> String {
        let seq = inputs[0].2.dims[0];
        let hidden = inputs[0].2.dims[1].to_usize().unwrap();
        let kv_hidden = inputs[1].2.dims[1].to_usize().unwrap();
        let kv_row_width = inputs[1].2.dims[1].to_usize().unwrap();
        let n_heads = hidden / self.head_dim;
        let n_kv_heads = kv_hidden / self.head_dim;
        let n_kv_groups = n_heads / n_kv_heads;
        format!(
            "(GQAAttention {} {} {} {} {} {} {} {} {} {} {} {} {})",
            elist_to_egglog(&[n_kv_heads.into(), n_kv_groups.into(), seq]),
            Expression::from(self.head_dim).to_egglog(),
            seq.to_egglog(),
            Expression::from(kv_row_width).to_egglog(),
            inputs[0].1,
            elist_to_egglog(&[
                Expression::from(self.head_dim * n_kv_groups),
                Expression::from(self.head_dim),
                Expression::from(hidden)
            ]),
            inputs[1].1,
            elist_to_egglog(&[Expression::from(self.head_dim), 0.into(), 0.into()]),
            inputs[2].1,
            elist_to_egglog(&[Expression::from(self.head_dim), 0.into(), 0.into()]),
            elist_to_egglog(&[
                Expression::from(self.head_dim * n_kv_groups),
                Expression::from(self.head_dim),
                Expression::from(hidden)
            ]),
            self.prev_seq.to_egglog(),
            self.layer,
        )
    }
}
