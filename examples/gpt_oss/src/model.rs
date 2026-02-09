use luminal::{
    graph::Graph,
    op::{CustomOp, LLIROp},
    prelude::GraphTensor,
    shape::{flatten_mul_strides, Expression, ToShape},
};
use luminal_cuda::{
    block::{cstruct::CStruct, BlockOp},
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr},
};
use luminal_nn::LayerNorm;
use std::{f64::consts::PI, fmt::Debug, mem::size_of, sync::Arc};

// GPT-OSS 120B hyperparams
pub const LAYERS: usize = 36;
pub const HIDDEN: usize = 2880;
pub const INTERMEDIATE: usize = 2880;
pub const HEAD_DIM: usize = 64;
pub const N_HEADS: usize = 64;
pub const N_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // 8
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 512
pub const VOCAB_SIZE: usize = 201088;
pub const NUM_EXPERTS: usize = 128;
pub const EXPERTS_PER_TOKEN: usize = 4;
pub const SLIDING_WINDOW: usize = 128;
pub const ROPE_THETA: f32 = 150000.0;
pub const SWIGLU_LIMIT: f32 = 7.0;
pub const RMS_NORM_EPS: f32 = 1e-5;
pub const FUSED_INTERMEDIATE: usize = 2 * INTERMEDIATE; // gate + up fused = 5760

// YaRN RoPE constants
const YARN_FACTOR: f64 = 32.0;
const YARN_BETA_FAST: f64 = 32.0;
const YARN_BETA_SLOW: f64 = 1.0;
const YARN_ORIG_MAX_POS: f64 = 4096.0;

/// Precompute YaRN-modified inverse frequencies for RoPE.
/// Returns (inv_freq[HEAD_DIM/2], attention_factor).
fn compute_yarn_inv_freq() -> ([f32; 32], f32) {
    let base = ROPE_THETA as f64;
    let dim = HEAD_DIM as f64;
    let half = HEAD_DIM / 2; // 32

    let find_correction_dim = |num_rotations: f64| -> f64 {
        (dim * (YARN_ORIG_MAX_POS / (num_rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
    };

    let low = find_correction_dim(YARN_BETA_FAST);
    let high = find_correction_dim(YARN_BETA_SLOW);

    let mut inv_freq = [0.0f32; 32];
    for i in 0..half {
        let pos_freq = base.powf(2.0 * i as f64 / dim);
        let inv_extrap = 1.0 / pos_freq;
        let inv_interp = 1.0 / (YARN_FACTOR * pos_freq);

        let diff = high - low;
        let t = if diff.abs() > 1e-10 {
            (i as f64 - low) / diff
        } else {
            0.0
        };
        let ramp = t.clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;

        inv_freq[i] =
            (inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor) as f32;
    }

    // attention_factor = 0.1 * ln(factor) + 1.0 (get_mscale with default mscale=1)
    let attention_factor = (0.1 * YARN_FACTOR.ln() + 1.0) as f32;

    (inv_freq, attention_factor)
}

// Layer types: alternating sliding_attention / full_attention
pub const LAYER_IS_SLIDING: [bool; LAYERS] = [
    true, false, true, false, true, false, true, false,
    true, false, true, false, true, false, true, false,
    true, false, true, false, true, false, true, false,
    true, false, true, false, true, false, true, false,
    true, false, true, false,
];

pub struct GptOss {
    pub embedding: GraphTensor,
    pub layers: Vec<GptOssLayer>,
    pub lm_norm: LayerNorm,
    pub lm_head: GraphTensor,
}

impl GptOss {
    pub fn init(cx: &mut Graph) -> Self {
        let mut layers = vec![];
        for l in 0..LAYERS {
            layers.push(GptOssLayer {
                q_proj_w: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (Q_DIM, HIDDEN),
                ),
                q_proj_b: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.q_proj.bias"),
                    Q_DIM,
                ),
                k_proj_w: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (KV_DIM, HIDDEN),
                ),
                k_proj_b: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.bias"),
                    KV_DIM,
                ),
                v_proj_w: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (KV_DIM, HIDDEN),
                ),
                v_proj_b: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.bias"),
                    KV_DIM,
                ),
                o_proj_w: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (HIDDEN, Q_DIM),
                ),
                o_proj_b: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.bias"),
                    HIDDEN,
                ),
                input_layernorm: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
                post_attention_layernorm: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
                router_w: cx.named_tensor(
                    format!("model.layers.{l}.mlp.router.weight"),
                    (NUM_EXPERTS, HIDDEN),
                ),
                router_b: cx.named_tensor(
                    format!("model.layers.{l}.mlp.router.bias"),
                    NUM_EXPERTS,
                ),
                is_sliding: LAYER_IS_SLIDING[l],
            });
        }
        let lm_norm = LayerNorm::new(HIDDEN, Some("model.norm.weight"), None, false, RMS_NORM_EPS, cx);
        let lm_head = cx.named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN));
        Self {
            embedding: cx.named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            layers,
            lm_head,
            lm_norm,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(
        &self,
        token_ids: GraphTensor,
        kv_cache: &KVCache,
        expert_weights: &ExpertWeightBuffers,
        scratchpad: &MoeScratchpad,
        sink_buffers: &SinkBuffers,
    ) -> GraphTensor {
        let seq = token_ids.dims1();
        let mut x = self.embedding.gather(
            (token_ids * HIDDEN).expand_dim(1, HIDDEN)
                + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
        );
        for (l, (layer, (k_cache, v_cache))) in
            self.layers.iter().zip(&kv_cache.layers).enumerate()
        {
            x = layer
                .forward(
                    x,
                    k_cache,
                    v_cache,
                    &expert_weights,
                    scratchpad,
                    &sink_buffers.layers[l],
                    l,
                )
                .graph_break();
        }
        self.lm_norm.forward(x).matmul(self.lm_head.t())
    }
}

pub struct GptOssLayer {
    q_proj_w: GraphTensor,
    q_proj_b: GraphTensor,
    k_proj_w: GraphTensor,
    k_proj_b: GraphTensor,
    v_proj_w: GraphTensor,
    v_proj_b: GraphTensor,
    o_proj_w: GraphTensor,
    o_proj_b: GraphTensor,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    router_w: GraphTensor,
    router_b: GraphTensor,
    is_sliding: bool,
}

impl GptOssLayer {
    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: &CudaSlice<u8>,
        v_cache: &CudaSlice<u8>,
        expert_weights: &ExpertWeightBuffers,
        scratchpad: &MoeScratchpad,
        sinks: &CudaSlice<u8>,
        layer_idx: usize,
    ) -> GraphTensor {
        let x_attn = self.input_layernorm.forward(x);
        let q = x_attn.matmul(self.q_proj_w.t()) + self.q_proj_b.expand_lhs(&x_attn.dims()[..x_attn.dims().len() - 1]);
        let k = x_attn.matmul(self.k_proj_w.t()) + self.k_proj_b.expand_lhs(&x_attn.dims()[..x_attn.dims().len() - 1]);
        let v = x_attn.matmul(self.v_proj_w.t()) + self.v_proj_b.expand_lhs(&x_attn.dims()[..x_attn.dims().len() - 1]);

        let attn_out = x.graph().custom_op(
            GptOssAttention::new(
                k_cache.device_ptr(v_cache.stream()).0,
                v_cache.device_ptr(k_cache.stream()).0,
                q.dims()[0],
                'p'.into(),
                self.is_sliding,
                sinks.device_ptr(sinks.stream()).0,
            ),
            (q, k, v),
            q.shape,
            q.dtype,
        );

        let h = attn_out.matmul(self.o_proj_w.t()) + self.o_proj_b.expand_lhs(&x_attn.dims()[..x_attn.dims().len() - 1]);
        let x = (x + h).graph_break(); // Split attention chunk from MoE chunk

        let x_ff = self.post_attention_layernorm.forward(x) * 1.0; // Materialize RMSNorm output
        let router_logits = x_ff.matmul(self.router_w.t()) + self.router_b.expand_lhs(&x_ff.dims()[..x_ff.dims().len() - 1]);

        let moe_out = x.graph().custom_op(
            MoeExperts::new(expert_weights, scratchpad, layer_idx),
            (x_ff, router_logits),
            x_ff.shape,
            x_ff.dtype,
        );

        x + moe_out
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KVCache {
    pub layers: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
}

impl KVCache {
    pub fn new(stream: &Arc<CudaStream>, capacity: usize) -> Self {
        Self {
            layers: (0..LAYERS)
                .map(|_| {
                    (
                        stream
                            .alloc_zeros(N_KV_HEADS * HEAD_DIM * capacity * size_of::<f32>())
                            .unwrap(),
                        stream
                            .alloc_zeros(N_KV_HEADS * HEAD_DIM * capacity * size_of::<f32>())
                            .unwrap(),
                    )
                })
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for (k_cache, v_cache) in &mut self.layers {
            v_cache.stream().memset_zeros(k_cache).unwrap();
            k_cache.stream().memset_zeros(v_cache).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Sink Buffers (learnable per-head scalar logits for attention)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SinkBuffers {
    /// Per-layer: [N_HEADS] FP32 sink logits (one per attention head)
    pub layers: Vec<CudaSlice<u8>>,
}

impl SinkBuffers {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            layers: (0..LAYERS)
                .map(|_| stream.alloc_zeros(N_HEADS * size_of::<f32>()).unwrap())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Expert Weight Buffers (loaded directly into GPU, passed via payload)
// ---------------------------------------------------------------------------

/// MXFP4 buffer layout per expert per projection:
/// Interleaved: [K/2 packed bytes | K/32 E8M0 scale bytes] per column
const MXFP4_COL_STRIDE_GATE_UP: usize = HIDDEN / 2 + HIDDEN / 32; // 1440 + 90 = 1530
const MXFP4_COL_STRIDE_DOWN: usize = INTERMEDIATE / 2 + INTERMEDIATE / 32; // same = 1530

#[derive(Debug)]
pub struct ExpertWeightBuffers {
    /// Per layer: interleaved gate_up weights for all 128 experts
    /// Shape: [NUM_EXPERTS * FUSED_INTERMEDIATE * MXFP4_COL_STRIDE_GATE_UP] bytes
    pub gate_up: Vec<CudaSlice<u8>>,
    /// Per layer: interleaved down weights for all 128 experts
    /// Shape: [NUM_EXPERTS * HIDDEN * MXFP4_COL_STRIDE_DOWN] bytes
    pub down: Vec<CudaSlice<u8>>,
    /// Per layer: gate_up bias [NUM_EXPERTS * FUSED_INTERMEDIATE] f32
    pub gate_up_bias: Vec<CudaSlice<u8>>,
    /// Per layer: down bias [NUM_EXPERTS * HIDDEN] f32
    pub down_bias: Vec<CudaSlice<u8>>,
}

impl ExpertWeightBuffers {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        let gate_up_size = NUM_EXPERTS * FUSED_INTERMEDIATE * MXFP4_COL_STRIDE_GATE_UP;
        let down_size = NUM_EXPERTS * HIDDEN * MXFP4_COL_STRIDE_DOWN;
        let gate_up_bias_size = NUM_EXPERTS * FUSED_INTERMEDIATE * size_of::<f32>();
        let down_bias_size = NUM_EXPERTS * HIDDEN * size_of::<f32>();

        Self {
            gate_up: (0..LAYERS)
                .map(|_| stream.alloc_zeros(gate_up_size).unwrap())
                .collect(),
            down: (0..LAYERS)
                .map(|_| stream.alloc_zeros(down_size).unwrap())
                .collect(),
            gate_up_bias: (0..LAYERS)
                .map(|_| stream.alloc_zeros(gate_up_bias_size).unwrap())
                .collect(),
            down_bias: (0..LAYERS)
                .map(|_| stream.alloc_zeros(down_bias_size).unwrap())
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// MoE Scratchpad (intermediate buffers for two-phase computation)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MoeScratchpad {
    /// Intermediate buffer: gate_up output [TOP_K * FUSED_INTERMEDIATE] + swiglu output [TOP_K * INTERMEDIATE]
    pub intermediate: CudaSlice<u8>,
}

impl MoeScratchpad {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        let gate_up_size = EXPERTS_PER_TOKEN * FUSED_INTERMEDIATE;
        let swiglu_size = EXPERTS_PER_TOKEN * INTERMEDIATE;
        Self {
            intermediate: stream
                .alloc_zeros((gate_up_size + swiglu_size) * size_of::<f32>())
                .unwrap(),
        }
    }
}

// ---------------------------------------------------------------------------
// GptOssAttention: RoPE + Causal Attention with KV Cache + Sliding Window + Sinks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GptOssAttention {
    range: Vec<Expression>,
    head_dim: Expression,
    cur_seq: Expression,
    kv_row_stride: Expression,
    q_stride: Vec<Expression>,
    k_stride: Vec<Expression>,
    v_stride: Vec<Expression>,
    o_stride: Vec<Expression>,
    prev_seq: Expression,
    k_cache: u64,
    v_cache: u64,
    sliding_window: usize,
    sinks_ptr: u64,
}

impl GptOssAttention {
    fn new(
        k_cache: u64,
        v_cache: u64,
        seq: Expression,
        prev_seq: Expression,
        is_sliding: bool,
        sinks_ptr: u64,
    ) -> Self {
        let sliding_window = if is_sliding { SLIDING_WINDOW } else { 0 };
        Self {
            range: (N_KV_HEADS, KV_GROUPS, seq).to_shape(),
            head_dim: HEAD_DIM.into(),
            cur_seq: seq,
            kv_row_stride: KV_DIM.into(),
            q_stride: (HEAD_DIM * KV_GROUPS, HEAD_DIM, Q_DIM).to_shape(),
            k_stride: (HEAD_DIM, 0, 0).to_shape(),
            v_stride: (HEAD_DIM, 0, 0).to_shape(),
            o_stride: (HEAD_DIM * KV_GROUPS, HEAD_DIM, Q_DIM).to_shape(),
            prev_seq,
            k_cache,
            v_cache,
            sliding_window,
            sinks_ptr,
        }
    }
}

impl CustomOp for GptOssAttention {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for GptOssAttention {
    fn op_name(&self) -> &'static str {
        "GptOssAttention"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.head_dim
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
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

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        let mut q_pos_stride = vec![0.into(); self.range.len()];
        q_pos_stride[self.range.len() - 1] = 1.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 1.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 1.into();
        payload
            .expr("head_size", self.head_dim)
            .expr("cur_seq", self.cur_seq)
            .expr("kv_row_stride", self.kv_row_stride)
            .expr("q", flatten_mul_strides(&self.range, &self.q_stride))
            .expr("k", flatten_mul_strides(&self.range, &self.k_stride))
            .expr("v", flatten_mul_strides(&self.range, &self.v_stride))
            .expr("out", flatten_mul_strides(&self.range, &self.o_stride))
            .ptr_mut_f32("key_cache", self.k_cache as *mut f32)
            .ptr_mut_f32("val_cache", self.v_cache as *mut f32)
            .expr("prev_seq", self.prev_seq)
            .expr(
                "q_pos_stride",
                flatten_mul_strides(&self.range, &q_pos_stride),
            )
            .expr(
                "group_pos_stride",
                flatten_mul_strides(&self.range, &group_pos_stride),
            )
            .expr(
                "head_pos_stride",
                flatten_mul_strides(&self.range, &head_pos_stride),
            )
            .int("sliding_window", self.sliding_window as i32)
            .ptr_const_f32("sinks", self.sinks_ptr as *const f32)
    }

    fn cuda_function(&self) -> String {
        // Precompute YaRN inv_freq values and embed in kernel
        let (inv_freq, attention_factor) = compute_yarn_inv_freq();
        let inv_freq_str = inv_freq
            .iter()
            .map(|v| format!("{:.10e}f", v))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"
            const float INV_FREQ[32] = {{ {inv_freq_str} }};
            const float ATTN_FACTOR = {attention_factor:.8}f;
            const int KV_GROUPS = {kv_groups};

            __shared__ float shared[32];
            __shared__ float q_buf[64];
            __shared__ float k_buf[64];

            auto warp_reduce_sum = [](float val) {{
                for (int offset = 16; offset > 0; offset >>= 1) {{
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }}
                return val;
            }};

            auto block_reduce_sum = [&](float val) {{
                int lane = threadIdx.x & 31;
                int wid  = threadIdx.x >> 5;
                val = warp_reduce_sum(val);
                if (lane == 0) shared[wid] = val;
                __syncthreads();
                val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
                if (wid == 0) val = warp_reduce_sum(val);
                return val;
            }};

            // Apply YaRN RoPE with attention scaling to a buffer in-place
            auto apply_rope = [&](float* buf, int pos) {{
                const int d = 64;
                const int half = 32;
                for (int i = t; i < half; i += blockDim.x) {{
                    float theta = (float)pos * INV_FREQ[i];
                    float cos_t, sin_t;
                    __sincosf(theta, &sin_t, &cos_t);
                    cos_t *= ATTN_FACTOR;
                    sin_t *= ATTN_FACTOR;
                    float x0 = buf[i];
                    float x1 = buf[i + half];
                    buf[i]        = x0 * cos_t - x1 * sin_t;
                    buf[i + half] = x1 * cos_t + x0 * sin_t;
                }}
                __syncthreads();
            }};

            // 3 graph inputs: Q, K, V
            const float* q_raw = source_ptrs[0] + eval_expression(payload.q, current);
            const float* k_base = source_ptrs[1] + eval_expression(payload.k, current);
            const float* v_base = source_ptrs[2] + eval_expression(payload.v, current);

            float* out = out_ptr + eval_expression(payload.out, current);
            int q_pos_local = eval_expression(payload.q_pos_stride, current);
            const int group_pos_local = eval_expression(payload.group_pos_stride, current);
            const int head_pos_local = eval_expression(payload.head_pos_stride, current);

            const int d             = eval_expression(payload.head_size, 0);
            float* __restrict__ K_cache = payload.key_cache + head_pos_local * d;
            float* __restrict__ V_cache = payload.val_cache + head_pos_local * d;

            const int S             = eval_expression(payload.cur_seq, 0);
            const int kv_row_stride = eval_expression(payload.kv_row_stride, 0);
            const int prev          = eval_expression(payload.prev_seq, 0);
            const int sliding_window = payload.sliding_window;
            const float* sink_values = (const float*)payload.sinks;

            const float* __restrict__ K_cur = k_base;
            const float* __restrict__ V_cur = v_base;
            float* __restrict__ O = out;

            if (q_pos_local >= S) q_pos_local = S - 1;
            if (q_pos_local < 0)  q_pos_local = 0;

            const int q_pos_total = prev + q_pos_local;
            const float scale = rsqrtf((float)d);

            const int half = d / 2;

            // ================================================================
            // Step 1: YaRN RoPE for this Q row
            // ================================================================
            {{
                for (int i = t; i < d; i += blockDim.x) {{
                    q_buf[i] = q_raw[i];
                }}
                __syncthreads();
                apply_rope(q_buf, q_pos_total);
            }}

            // ================================================================
            // Step 2: First group writes K (YaRN RoPE) + V to cache
            // ================================================================
            if (group_pos_local == 0) {{
                for (int r = 0; r < S; ++r) {{
                    const float* __restrict__ srcK = K_cur + r * kv_row_stride;
                    const float* __restrict__ srcV = V_cur + r * kv_row_stride;
                    float* __restrict__ dstK = K_cache + (prev + r) * kv_row_stride;
                    float* __restrict__ dstV = V_cache + (prev + r) * kv_row_stride;

                    for (int u = t; u < d; u += blockDim.x) {{
                        dstV[u] = srcV[u];
                    }}

                    for (int u = t; u < d; u += blockDim.x) {{
                        k_buf[u] = srcK[u];
                    }}
                    __syncthreads();
                    apply_rope(k_buf, prev + r);

                    for (int u = t; u < d; u += blockDim.x) {{
                        dstK[u] = k_buf[u];
                    }}
                    __syncthreads();
                }}
            }}
            __syncthreads();

            // ================================================================
            // Step 3: Online softmax attention with sliding window
            // ================================================================

            __shared__ float att_m;
            __shared__ float att_corr;
            __shared__ float att_w;
            float att_d = 0.0f;

            for (int j = t; j < d; j += blockDim.x) {{
                O[j] = 0.0f;
            }}
            if (t == 0) att_m = -__int_as_float(0x7f800000);
            __syncthreads();

            int attn_start = 0;
            if (sliding_window > 0 && q_pos_total >= sliding_window) {{
                attn_start = q_pos_total - sliding_window + 1;
            }}

            auto attend_row = [&](int r) {{
                const float* __restrict__ k_row;
                const float* __restrict__ v_row;

                if (r < prev) {{
                    k_row = K_cache + r * kv_row_stride;
                    v_row = V_cache + r * kv_row_stride;
                }} else {{
                    int r_local = r - prev;
                    v_row = V_cur + r_local * kv_row_stride;

                    const float* __restrict__ srcK = K_cur + r_local * kv_row_stride;
                    for (int u = t; u < d; u += blockDim.x) {{
                        k_buf[u] = srcK[u];
                    }}
                    __syncthreads();
                    apply_rope(k_buf, r);
                    k_row = k_buf;
                }}

                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {{
                    partial += q_buf[u] * k_row[u];
                }}
                float dot_qk = block_reduce_sum(partial);

                if (t == 0) {{
                    float logit = dot_qk * scale;
                    float m_old = att_m;
                    float m_new = fmaxf(m_old, logit);
                    float corr = __expf(m_old - m_new);
                    float w = __expf(logit - m_new);
                    att_d = att_d * corr + w;
                    att_m = m_new;
                    att_corr = corr;
                    att_w = w;
                }}
                __syncthreads();

                float corr = att_corr;
                float w = att_w;

                for (int j = t; j < d; j += blockDim.x) {{
                    O[j] = O[j] * corr + w * v_row[j];
                }}
                __syncthreads();
            }};

            if (sliding_window > 0) {{
                for (int r = max(attn_start, 0); r <= q_pos_total; ++r) {{
                    attend_row(r);
                }}
            }} else {{
                for (int r = 0; r <= q_pos_total; ++r) {{
                    attend_row(r);
                }}
            }}

            // ================================================================
            // Step 4: Learnable sink — add per-head scalar logit to softmax
            // ================================================================
            {{
                int head_idx = head_pos_local * KV_GROUPS + group_pos_local;
                float sink_logit = sink_values[head_idx];

                if (t == 0) {{
                    float m_old = att_m;
                    float m_new = fmaxf(m_old, sink_logit);
                    float corr = __expf(m_old - m_new);
                    float sink_w = __expf(sink_logit - m_new);
                    att_d = att_d * corr + sink_w;
                    att_corr = corr;
                }}
                __syncthreads();

                // Correct output for new max (sink has no V vector)
                float sink_corr = att_corr;
                for (int j = t; j < d; j += blockDim.x) {{
                    O[j] *= sink_corr;
                }}
                __syncthreads();
            }}

            // ================================================================
            // Step 5: Final normalization (sink included in denominator)
            // ================================================================
            if (t == 0) att_w = 1.0f / att_d;
            __syncthreads();
            float inv_d = att_w;

            for (int j = t; j < d; j += blockDim.x) {{
                O[j] *= inv_d;
            }}
        "#,
            inv_freq_str = inv_freq_str,
            attention_factor = attention_factor,
            kv_groups = KV_GROUPS,
        )
    }
}

// ---------------------------------------------------------------------------
// MoeExperts: Top-k routing + MXFP4 expert computation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MoeExperts {
    range: Vec<Expression>,
    gate_up_ptr: u64,
    down_ptr: u64,
    gate_up_bias_ptr: u64,
    down_bias_ptr: u64,
    intermediate_ptr: u64,
}

impl MoeExperts {
    fn new(
        expert_weights: &ExpertWeightBuffers,
        _scratchpad: &MoeScratchpad,
        layer_idx: usize,
    ) -> Self {
        Self {
            range: vec![1.into()], // Single block — no inter-SM sync needed
            gate_up_ptr: expert_weights.gate_up[layer_idx]
                .device_ptr(expert_weights.gate_up[layer_idx].stream())
                .0,
            down_ptr: expert_weights.down[layer_idx]
                .device_ptr(expert_weights.down[layer_idx].stream())
                .0,
            gate_up_bias_ptr: expert_weights.gate_up_bias[layer_idx]
                .device_ptr(expert_weights.gate_up_bias[layer_idx].stream())
                .0,
            down_bias_ptr: expert_weights.down_bias[layer_idx]
                .device_ptr(expert_weights.down_bias[layer_idx].stream())
                .0,
            intermediate_ptr: _scratchpad
                .intermediate
                .device_ptr(_scratchpad.intermediate.stream())
                .0,
        }
    }
}

impl CustomOp for MoeExperts {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for MoeExperts {
    fn op_name(&self) -> &'static str {
        "MoeExperts"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        HIDDEN.into()
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![false]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        // 2 inputs: x, router_logits
        vec![vec![false], vec![false]]
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .ptr_const_f32("gate_up_weights", self.gate_up_ptr as *const f32)
            .ptr_const_f32("down_weights", self.down_ptr as *const f32)
            .ptr_const_f32("gate_up_bias", self.gate_up_bias_ptr as *const f32)
            .ptr_const_f32("down_bias", self.down_bias_ptr as *const f32)
            .ptr_mut_f32("intermediate", self.intermediate_ptr as *mut f32)
    }

    fn cuda_function(&self) -> String {
        format!(
            r#"
        // MoeExperts: Single-block implementation
        // Top-k routing + MXFP4 expert gate_up → SwiGLU → down
        // Inputs: source_ptrs[0] = x [1, HIDDEN], source_ptrs[1] = router_logits [1, NUM_EXPERTS]

        __shared__ float fp4_lut[16];
        __shared__ float shared_buf[{hidden_or_experts}]; // max(HIDDEN, NUM_EXPERTS)
        __shared__ int top_indices[{top_k}];
        __shared__ float top_weights[{top_k}];

        if (t < 16) {{
            const float table[16] = {{
                0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
            }};
            fp4_lut[t] = table[t];
        }}
        __syncthreads();

        const float* x = source_ptrs[0];
        const float* router_logits = source_ptrs[1];
        float* output = out_ptr;

        const unsigned char* gate_up_weights = (const unsigned char*)payload.gate_up_weights;
        const unsigned char* down_weights = (const unsigned char*)payload.down_weights;
        const float* gate_up_bias = (const float*)payload.gate_up_bias;
        const float* down_bias = (const float*)payload.down_bias;
        float* intermediate = payload.intermediate;

        constexpr int HIDDEN = {hidden};
        constexpr int INTERMEDIATE = {intermediate};
        constexpr int FUSED_INTERMEDIATE = {fused_intermediate};
        constexpr int NUM_EXPERTS = {num_experts};
        constexpr int TOP_K = {top_k};
        constexpr int MXFP4_COL_STRIDE_GU = {col_stride_gu};
        constexpr int MXFP4_COL_STRIDE_D = {col_stride_d};

        const int threads = blockDim.x;
        const int lane = t & 31;
        const int warp_id = t >> 5;
        const int num_warps = threads >> 5;

        auto e8m0_decode = [](unsigned char s) -> float {{
            if (s == 0xFF) return 0.0f;
            return ldexpf(1.0f, (int)s - 127);
        }};

        // ======== PHASE 0: Top-K Selection ========
        for (int i = t; i < NUM_EXPERTS; i += threads) {{
            shared_buf[i] = router_logits[i];
        }}
        __syncthreads();

        if (t == 0) {{
            for (int ki = 0; ki < TOP_K; ki++) {{
                float best_val = -1e30f;
                int best_idx = 0;
                for (int e = 0; e < NUM_EXPERTS; e++) {{
                    if (shared_buf[e] > best_val) {{
                        best_val = shared_buf[e];
                        best_idx = e;
                    }}
                }}
                top_indices[ki] = best_idx;
                top_weights[ki] = best_val;
                shared_buf[best_idx] = -1e30f;
            }}
            // Softmax over top-K
            float max_val = top_weights[0];
            for (int ki = 1; ki < TOP_K; ki++) max_val = fmaxf(max_val, top_weights[ki]);
            float sum_exp = 0.0f;
            for (int ki = 0; ki < TOP_K; ki++) {{
                top_weights[ki] = __expf(top_weights[ki] - max_val);
                sum_exp += top_weights[ki];
            }}
            float inv_sum = 1.0f / sum_exp;
            for (int ki = 0; ki < TOP_K; ki++) top_weights[ki] *= inv_sum;
        }}
        __syncthreads();

        // ======== PHASE 1: Gate+Up matmul for each expert ========
        for (int global_col = warp_id; global_col < TOP_K * FUSED_INTERMEDIATE; global_col += num_warps) {{
            const int expert_local = global_col / FUSED_INTERMEDIATE;
            const int col_in_expert = global_col % FUSED_INTERMEDIATE;
            const int expert_idx = top_indices[expert_local];

            const unsigned char* col_data = gate_up_weights
                + (size_t)expert_idx * FUSED_INTERMEDIATE * MXFP4_COL_STRIDE_GU
                + col_in_expert * MXFP4_COL_STRIDE_GU;
            const unsigned char* packed = col_data;
            const unsigned char* scales = col_data + HIDDEN / 2;

            float acc = 0.0f;
            for (int block_start = lane * 32; block_start < HIDDEN; block_start += 32 * 32) {{
                float block_scale = e8m0_decode(scales[block_start / 32]);
                const int byte_start = block_start / 2;
                for (int bi = 0; bi < 16; bi++) {{
                    const int k0 = block_start + bi * 2;
                    if (k0 + 1 >= HIDDEN) break;
                    unsigned char pb = packed[byte_start + bi];
                    float w0 = fp4_lut[pb & 0xF] * block_scale;
                    float w1 = fp4_lut[pb >> 4] * block_scale;
                    acc += x[k0] * w0 + x[k0 + 1] * w1;
                }}
            }}
            for (int offset = 16; offset > 0; offset >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, offset);

            if (lane == 0) {{
                acc += gate_up_bias[expert_idx * FUSED_INTERMEDIATE + col_in_expert];
                intermediate[expert_local * FUSED_INTERMEDIATE + col_in_expert] = acc;
            }}
        }}
        __syncthreads();

        // ======== PHASE 2: Gated activation ========
        // Gate and up are INTERLEAVED in the gate_up output: gate=even indices, up=odd indices.
        // Activation: glu = clamp(gate, max=7) * sigmoid(clamp(gate, max=7) * 1.702)
        // Output: (clamp(up, -7, 7) + 1) * glu
        // Write to separate region to avoid races: swiglu_out starts at TOP_K * FUSED_INTERMEDIATE
        constexpr int SWIGLU_OFFSET = TOP_K * FUSED_INTERMEDIATE;
        for (int e = 0; e < TOP_K; e++) {{
            for (int i = t; i < INTERMEDIATE; i += threads) {{
                float gate_val = intermediate[e * FUSED_INTERMEDIATE + 2 * i];
                float up_val = intermediate[e * FUSED_INTERMEDIATE + 2 * i + 1];
                gate_val = fminf(gate_val, {swiglu_limit:.1}f);
                up_val = fmaxf(fminf(up_val, {swiglu_limit:.1}f), -{swiglu_limit:.1}f);
                float glu = gate_val * (1.0f / (1.0f + __expf(-gate_val * 1.702f)));
                intermediate[SWIGLU_OFFSET + e * INTERMEDIATE + i] = (up_val + 1.0f) * glu;
            }}
            __syncthreads();
        }}

        // ======== PHASE 3: Zero output ========
        for (int i = t; i < HIDDEN; i += threads) output[i] = 0.0f;
        __syncthreads();

        // ======== PHASE 4: Down matmul + weighted accumulation ========
        for (int global_col = warp_id; global_col < TOP_K * HIDDEN; global_col += num_warps) {{
            const int expert_local = global_col / HIDDEN;
            const int col_in_expert = global_col % HIDDEN;
            const int expert_idx = top_indices[expert_local];

            const unsigned char* col_data = down_weights
                + (size_t)expert_idx * HIDDEN * MXFP4_COL_STRIDE_D
                + col_in_expert * MXFP4_COL_STRIDE_D;
            const unsigned char* packed = col_data;
            const unsigned char* scales = col_data + INTERMEDIATE / 2;

            const float* expert_input = intermediate + SWIGLU_OFFSET + expert_local * INTERMEDIATE;

            float acc = 0.0f;
            for (int block_start = lane * 32; block_start < INTERMEDIATE; block_start += 32 * 32) {{
                float block_scale = e8m0_decode(scales[block_start / 32]);
                const int byte_start = block_start / 2;
                for (int bi = 0; bi < 16; bi++) {{
                    const int k0 = block_start + bi * 2;
                    if (k0 + 1 >= INTERMEDIATE) break;
                    unsigned char pb = packed[byte_start + bi];
                    float w0 = fp4_lut[pb & 0xF] * block_scale;
                    float w1 = fp4_lut[pb >> 4] * block_scale;
                    acc += expert_input[k0] * w0 + expert_input[k0 + 1] * w1;
                }}
            }}
            for (int offset = 16; offset > 0; offset >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, offset);

            if (lane == 0) {{
                acc += down_bias[expert_idx * HIDDEN + col_in_expert];
                acc *= top_weights[expert_local];
                // Single block, so no atomicAdd needed - direct accumulation
                output[col_in_expert] += acc;
            }}
        }}
        "#,
            hidden = HIDDEN,
            intermediate = INTERMEDIATE,
            fused_intermediate = FUSED_INTERMEDIATE,
            num_experts = NUM_EXPERTS,
            top_k = EXPERTS_PER_TOKEN,
            col_stride_gu = MXFP4_COL_STRIDE_GATE_UP,
            col_stride_d = MXFP4_COL_STRIDE_DOWN,
            swiglu_limit = SWIGLU_LIMIT,
            hidden_or_experts = std::cmp::max(HIDDEN, NUM_EXPERTS),
        )
    }
}
