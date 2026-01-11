use luminal::{
    graph::{elist_to_egglog, Graph},
    op::{CustomOp, DType, HLIROp, LLIROp},
    prelude::{FxHashMap, GraphTensor},
    shape::{flatten_mul_strides, Expression, ShapeTracker},
};
use luminal_cuda::{
    block::{BlockOp, CStruct},
    cudarc::driver::{CudaStream, DevicePtr},
    runtime::CustomState,
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
                    &format!("model.layers.{l}.mlp.up_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                gate: cx.named_tensor(
                    &format!("model.layers.{l}.mlp.gate_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                down: cx.named_tensor(
                    &format!("model.layers.{l}.mlp.down_proj.weight"),
                    (HIDDEN, INTERMEDIATE),
                ),
                q_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (HIDDEN, HIDDEN),
                ),
                k_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                ),
                v_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                ),
                o_proj: cx.named_tensor(
                    &format!("model.layers.{l}.self_attn.o_proj.weight"),
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

impl LlamaLayer {
    pub fn forward(&self, input: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
        let cx = input.graph();
        let batch = input.dims()[0];
        let x_attn = self.attn_rms.forward(input);
        let q = x_attn.matmul(self.q_proj.transpose(0, 1));
        let k = x_attn.matmul(self.k_proj.transpose(0, 1));
        let v = x_attn.matmul(self.v_proj.transpose(0, 1));
        let q_rope = GraphTensor::from_id(
            cx.add_op(RopeFrontendOp {
                range: vec![Expression::from(batch)],
                stride: vec![HIDDEN.into()],
                row_width: Expression::from(HIDDEN),
            })
            .input(q.id, q.shape)
            .input(token_ids.id, token_ids.shape)
            .finish(),
            q.shape,
            cx,
            DType::F32,
        );
        let k_rope = GraphTensor::from_id(
            cx.add_op(RopeFrontendOp {
                range: vec![Expression::from(batch)],
                stride: vec![(HIDDEN / KV_GROUPS).into()],
                row_width: Expression::from(HIDDEN / KV_GROUPS),
            })
            .input(k.id, k.shape)
            .input(token_ids.id, token_ids.shape)
            .finish(),
            k.shape,
            cx,
            DType::F32,
        );

        let attn_out = cx.custom_op(
            GQAAttention::new(HEAD_DIM, 'p'.into(), self.layer, &[q_rope, k_rope, v]),
            &[q_rope, k_rope, v],
            q_rope.shape,
            q_rope.dtype,
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

impl HLIROp for RopeFrontendOp {
    fn to_egglog(&self, inputs: &[(luminal::prelude::NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(RowRope {} {} {} {} {})",
            elist_to_egglog(&self.range),
            inputs[0].1,
            elist_to_egglog(&self.stride),
            self.row_width.to_egglog(),
            inputs[1].1,
        )
    }
}

#[derive(Debug, Clone)]
pub struct GQAAttention {
    range: Vec<Expression>,
    head_dim: Expression,
    cur_seq: Expression,
    kv_row_stride: Expression,
    q_stride: Vec<Expression>,
    k_stride: Vec<Expression>,
    v_stride: Vec<Expression>,
    o_stride: Vec<Expression>,
    prev_seq: Expression,
    current_layer: usize,
}

impl GQAAttention {
    fn new(
        head_dim: usize,
        prev_seq: Expression,
        current_layer: usize,
        qkv: &[GraphTensor],
    ) -> Self {
        let seq = qkv[0].dims()[0];
        let hidden = qkv[0].dims()[1].to_usize().unwrap();
        let kv_hidden = qkv[1].dims()[1].to_usize().unwrap();
        let kv_row_width = qkv[1].dims()[1].to_usize().unwrap();
        let n_heads = hidden / head_dim;
        let n_kv_heads = kv_hidden / head_dim;
        let n_kv_groups = n_heads / n_kv_heads;
        Self {
            range: vec![n_kv_heads.into(), n_kv_groups.into(), seq],
            head_dim: head_dim.into(),
            cur_seq: seq,
            kv_row_stride: kv_row_width.into(),
            q_stride: vec![
                Expression::from(head_dim * n_kv_groups),
                Expression::from(head_dim),
                Expression::from(hidden),
            ],
            k_stride: vec![Expression::from(head_dim), 0.into(), 0.into()],
            v_stride: vec![Expression::from(head_dim), 0.into(), 0.into()],
            o_stride: vec![
                Expression::from(head_dim * n_kv_groups),
                Expression::from(head_dim),
                Expression::from(hidden),
            ],
            prev_seq,
            current_layer,
        }
    }
}

impl CustomOp for GQAAttention {
    fn to_llir_op(&self) -> luminal::op::LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for GQAAttention {
    fn op_name(&self) -> &'static str {
        "GQAAttention"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.head_dim
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        let mut q = vec![true; self.range.len()];
        q[self.range.len() - 1] = false;
        let mut k = vec![true; self.range.len()];
        k[self.range.len() - 1] = false;
        let mut v = vec![true; self.range.len()];
        v[self.range.len() - 1] = false;
        vec![q, k, v]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body = "
            int head_size;
            int cur_seq;
            int kv_row_stride;
            const int q;
            const int k;
            const int v;
            const int out;
            float* key_cache;
            float* val_cache;
            int prev_seq;
            int q_pos_stride;
            int group_pos_stride;
            int head_pos_stride;
        "
        .to_string();
        let function_body = "
            // shared buffer for block-wide reduction
            __shared__ float shared[32]; // max 32 warps per block

            // warp-level reduction
            auto warp_reduce_sum = [](float val) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                return val;
            };

            // block-level reduction (sum valid only in thread 0)
            auto block_reduce_sum = [&](float val) {
                int lane = threadIdx.x & 31;
                int wid  = threadIdx.x >> 5;

                val = warp_reduce_sum(val);       // each warp reduces to lane 0
                if (lane == 0) shared[wid] = val; // write warp result
                __syncthreads();

                // first warp reduces over warp results
                val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
                if (wid == 0) val = warp_reduce_sum(val);
                return val; // valid only in threadIdx.x == 0
            };
            // Current-chunk Q/K/V row pointers for this head
            const float* q   = source_ptrs[0] + eval_expression(payload.q, current);
            const float* k   = source_ptrs[1] + eval_expression(payload.k, current);
            const float* v   = source_ptrs[2] + eval_expression(payload.v, current);
            float*       out = out_ptr + eval_expression(payload.out, current);
            int q_pos_local = eval_expression(payload.q_pos_stride, current);
            const int group_pos_local = eval_expression(payload.group_pos_stride, current);
            const int head_pos_local = eval_expression(payload.head_pos_stride, current);

            // Cache pointers for this kv head (same layout/stride as k/v)
            const int d             = eval_expression(payload.head_size, 0);     // head_dim
            const float* __restrict__ K_cache = payload.key_cache + head_pos_local * d;
            const float* __restrict__ V_cache = payload.val_cache + head_pos_local * d;

            const int S             = eval_expression(payload.cur_seq, 0);        // number of tokens in this chunk
            const int kv_row_stride = eval_expression(payload.kv_row_stride, 0); // stride between rows (floats)
            const int prev          = eval_expression(payload.prev_seq, 0);      // number of cached tokens already present

            const float* __restrict__ K_cur = k;
            const float* __restrict__ V_cur = v;
            float*       __restrict__ O     = out;

            // Absolute causal index for this query: prev tokens + position in this chunk
            if (q_pos_local >= S) q_pos_local = S - 1;
            if (q_pos_local < 0)  q_pos_local = 0;

            const int q_pos_total = prev + q_pos_local; // index in [0 .. prev+S-1]

            const float scale = rsqrtf((float)d);       // 1 / sqrt(d)

            __shared__ float max_l_shared;
            __shared__ float inv_s_shared;
            __shared__ float w_shared;

            // --------------------------------------------------------------------
            // If we are the \"first head\" in this kv_group, copy current K/V rows
            // into the cache for future steps. This does NOT affect reads here.
            // --------------------------------------------------------------------
            if (group_pos_local == 0 && K_cache != nullptr && V_cache != nullptr) {
                // Copy S rows, this head's slice only, into positions [prev .. prev+S-1]
                for (int r = 0; r < S; ++r) {
                    const float* __restrict__ srcK = K_cur + r * kv_row_stride;
                    const float* __restrict__ srcV = V_cur + r * kv_row_stride;
                          float* __restrict__ dstK = const_cast<float*>(K_cache) + (prev + r) * kv_row_stride;
                          float* __restrict__ dstV = const_cast<float*>(V_cache) + (prev + r) * kv_row_stride;

                    // parallel over head_dim
                    for (int u = t; u < d; u += blockDim.x) {
                        dstK[u] = srcK[u];
                        dstV[u] = srcV[u];
                    }
                }
            }
            __syncthreads(); // only sync within this block; other heads are in other blocks

            // --------------------------------------------------------------------
            // Softmax over [0 .. q_pos_total], using:
            //   rows <  prev       -> cache
            //   rows >= prev       -> current chunk (index r - prev)
            // --------------------------------------------------------------------

            if (t == 0) max_l_shared = -__int_as_float(0x7f800000); // -INF
            __syncthreads();

            // -------- Pass 1: find row max over logits (parallel over u) --------
            for (int r = 0; r <= q_pos_total; ++r) {
                const float* __restrict__ k_row;
                if (r < prev) {
                    // from cache
                    k_row = K_cache + r * kv_row_stride;
                } else {
                    // from current chunk
                    int r_local = r - prev; // 0..S-1
                    k_row = K_cur + r_local * kv_row_stride;
                }

                // each thread does a stripe of the dot product
                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {
                    partial += q[u] * k_row[u];
                }
                float dot_qk = block_reduce_sum(partial);

                if (t == 0) {
                    float logit = dot_qk * scale;
                    max_l_shared = fmaxf(max_l_shared, logit);
                }
                __syncthreads(); // ensure all threads see updated max_l_shared each iter
            }

            __syncthreads();
            float max_l = max_l_shared;

            // -------- Pass 2: softmax weights + weighted sum (parallel over j) --------

            // init output in parallel
            for (int j = t; j < d; j += blockDim.x) {
                O[j] = 0.0f;
            }

            float s_local = 0.0f;  // sum of weights (thread 0 only)
            __syncthreads();

            for (int r = 0; r <= q_pos_total; ++r) {
                const float* __restrict__ k_row;
                const float* __restrict__ v_row;

                if (r < prev) {
                    k_row = K_cache + r * kv_row_stride;
                    v_row = V_cache + r * kv_row_stride;
                } else {
                    int r_local = r - prev;
                    k_row = K_cur + r_local * kv_row_stride;
                    v_row = V_cur + r_local * kv_row_stride;
                }

                // dot(q, k_row) in parallel over u
                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {
                    partial += q[u] * k_row[u];
                }
                float dot_qk = block_reduce_sum(partial);

                if (t == 0) {
                    float logit = dot_qk * scale;
                    float w     = __expf(logit - max_l);
                    s_local    += w;
                    w_shared    = w;
                }
                __syncthreads();

                float w = w_shared;

                // accumulate O[j] in parallel over j
                for (int j = t; j < d; j += blockDim.x) {
                    O[j] += w * v_row[j];
                }
                __syncthreads();
            }

            if (t == 0) {
                inv_s_shared = 1.0f / s_local;
            }
            __syncthreads();

            float inv_s = inv_s_shared;

            // -------- Normalize (parallel over j) --------
            for (int j = t; j < d; j += blockDim.x) {
                O[j] *= inv_s;
            }
        "
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        custom_state: &mut FxHashMap<String, CustomState>,
        stream: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        let CustomState::KVCache(kv_cache) = &custom_state["kv_cache"] else {
            unreachable!()
        };
        let (k_cache, v_cache) = &kv_cache[self.current_layer];
        let mut q_pos_stride = vec![0.into(); self.range.len()];
        q_pos_stride[self.range.len() - 1] = 1.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 1.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 1.into();
        CStruct::new()
            .int(expressions[&self.head_dim])
            .int(expressions[&self.cur_seq])
            .int(expressions[&self.kv_row_stride])
            .int(expressions[&flatten_mul_strides(&self.range, &self.q_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.k_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.v_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.o_stride)])
            .ptr_mut_f32(k_cache.device_ptr(stream).0 as *mut f32)
            .ptr_mut_f32(v_cache.device_ptr(stream).0 as *mut f32)
            .int(expressions[&self.prev_seq])
            .int(expressions[&flatten_mul_strides(&self.range, &q_pos_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &group_pos_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &head_pos_stride)])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        let mut q_pos_stride = vec![0.into(); self.range.len()];
        q_pos_stride[self.range.len() - 1] = 1.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 1.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 1.into();
        vec![
            flatten_mul_strides(&self.range, &self.q_stride),
            flatten_mul_strides(&self.range, &self.k_stride),
            flatten_mul_strides(&self.range, &self.v_stride),
            flatten_mul_strides(&self.range, &self.o_stride),
            self.head_dim,
            self.cur_seq,
            self.kv_row_stride,
            self.prev_seq,
            flatten_mul_strides(&self.range, &q_pos_stride),
            flatten_mul_strides(&self.range, &group_pos_stride),
            flatten_mul_strides(&self.range, &head_pos_stride),
        ]
    }
}

impl HLIROp for GQAAttention {
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
            self.current_layer,
        )
    }
}
