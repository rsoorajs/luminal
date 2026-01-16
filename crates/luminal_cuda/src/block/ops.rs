use std::fmt::Debug;

use cudarc::driver::CudaStream;
use itertools::Itertools;
use luminal::{
    graph::{extract_expr, extract_expr_list},
    op::OpParam::*,
    op::*,
    prelude::*,
};

use crate::block::BlockOp;

pub type Ops = (RowAdd, RowSwishMul, RowRMSNorm, RowRope, TileMatmulSplitK);

#[derive(Debug, Default)]
pub struct RowAdd {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                ; get add
                (= ?sa (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                (= ?row_width (nth_from_end ?shape 0))
                (= (MNum ?row_width_num) ?row_width)
                (<= ?row_width_num 4096) ; currently load full row to sram, should instead load chunks in up to capacity and stream rest in
                ; assert the row is contiguous
                (= (MNum 1) (nth_from_end ?a_stride 0))
                (= (MNum 1) (nth_from_end ?b_stride 0))
                (= (MNum 1) (nth_from_end ?out_stride 0))
                ;(= (F32) (dtype ?a))
            )
            (
                (let ?new_shape (RemoveNthFromEnd ?shape 0))
                (let ?new_a_stride (RemoveNthFromEnd ?a_stride 0))
                (let ?new_b_stride (RemoveNthFromEnd ?b_stride 0))
                (let ?new_out_stride (RemoveNthFromEnd ?out_stride 0))
                (let ?ra (RowAdd ?new_shape ?a ?new_a_stride ?b ?new_b_stride ?new_out_stride ?row_width))
                (union ?sa ?ra)
                (set (dtype ?ra) (F32))
            )
            :name \"row add\"
        )".to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[6], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl BlockOp for RowAdd {
    fn op_name(&self) -> &'static str {
        "RowAdd"
    }

    fn launch_range(&self) -> Vec<Expression> {
        if self.range.is_empty() {
            vec![1.into()]
        } else {
            self.range.clone()
        }
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // 1 add per element
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width
    }

    fn cuda_struct(&self) -> String {
        "const int a_strides; const int b_strides; const int out_strides; int row_width;"
            .to_string()
    }

    fn bytes_loaded(&self) -> Expression {
        // Load 2 input rows (a + b) per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 2 * 4
    }

    fn cuda_function(&self) -> String {
        "
        const float* a = source_ptrs[0] + eval_expression(payload.a_strides, current);
        const float* b = source_ptrs[1] + eval_expression(payload.b_strides, current);
        float* out = out_ptr + eval_expression(payload.out_strides, current);
        int row_width = eval_expression(payload.row_width, 0);
        for (int idx = t; idx < row_width; idx += blockDim.x) {
            out[idx] = a[idx] + b[idx];
        }
        "
        .to_string()
    }

    fn schedule_op(&self, _: &CudaStream, expressions: &FxHashMap<Expression, i32>) -> Vec<u8> {
        CStruct::new()
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.b_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.out_stride)])
            .int(expressions[&self.row_width])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_mul_strides(&self.range, &self.a_stride),
            flatten_mul_strides(&self.range, &self.b_stride),
            flatten_mul_strides(&self.range, &self.out_stride),
            self.row_width,
        ]
    }
}

#[derive(Debug, Default)]
pub struct RowSwishMul {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    row_width: Expression,
    n_chunks: Expression,
}

impl EgglogOp for RowSwishMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowSwishMul".to_string(),
            vec![EList, Input, EList, Input, EList, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                (= ?sigmoid (Sigmoid
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                (= ?swish (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    ?sigmoid
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                (= ?swishmul (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?swish
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    ?other
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                ;(= (F32) (dtype ?self))
            )
            (
                (let ?n_chunks (MDiv (MAdd ?width (MNum 127)) (MNum 128)))
                (let ?rsm (RowSwishMul
                    (ECons ?batch (ENil))
                    ?self
                    (ECons ?width (ENil))
                    ?other
                    (ECons ?width (ENil))
                    ?width
                    ?n_chunks
                ))
                (union ?swishmul ?rsm)
                (set (dtype ?rsm) (F32))
            )
            :name \"row swish mul\"
        )"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[5], expr_cache).unwrap(),
                n_chunks: extract_expr(egraph, children[6], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl BlockOp for RowSwishMul {
    fn op_name(&self) -> &'static str {
        "RowSwishMul"
    }

    fn launch_range(&self) -> Vec<Expression> {
        // Add n_chunks as inner dimension: [batch..., n_chunks]
        let mut range = self.range.clone();
        range.push(self.n_chunks);
        range
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.launch_range().len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        // All batch dimensions separate barriers, plus chunk dimension
        let launch_dims = self.range.len() + 1;
        vec![vec![true; launch_dims], vec![true; launch_dims]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load 2 input rows (a + b) per launch (total across all chunks)
        self.range.iter().copied().product::<Expression>() * self.row_width * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch (total across all chunks)
        self.range.iter().copied().product::<Expression>() * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // swish(x) * b[idx] = x / (1 + exp(-x)) * b
        // ~5 ops per element: neg, exp, add, div, mul
        self.range.iter().copied().product::<Expression>() * self.row_width * 5
    }

    fn cuda_struct(&self) -> String {
        "const int a; const int b; const int out; int row_width; int chunk_idx;".to_string()
    }

    fn cuda_function(&self) -> String {
        "
        const int chunk_idx = eval_expression(payload.chunk_idx, current);
        const int row_width = eval_expression(payload.row_width, 0);
        const int chunk_start = chunk_idx * 128;
        const int chunk_end = min(chunk_start + 128, row_width);

        const float* a = source_ptrs[0] + eval_expression(payload.a, current) + chunk_start;
        const float* b = source_ptrs[1] + eval_expression(payload.b, current) + chunk_start;
        float* out = out_ptr + eval_expression(payload.out, current) + chunk_start;

        const int chunk_size = chunk_end - chunk_start;
        for (int idx = t; idx < chunk_size; idx += blockDim.x) {
            float x = a[idx];
            float sw = x / (1.0f + __expf(-x)); // swish(x)
            out[idx] = sw * b[idx];
        }
        "
        .to_string()
    }

    fn schedule_op(&self, _: &CudaStream, expressions: &FxHashMap<Expression, i32>) -> Vec<u8> {
        // Extend a_stride and b_stride with 0 for the chunk dimension
        // so batch offset doesn't change when chunk changes
        let mut a_stride_ext = self.a_stride.clone();
        a_stride_ext.push(0.into());
        let mut b_stride_ext = self.b_stride.clone();
        b_stride_ext.push(0.into());

        // Stride for extracting chunk index: [0, 0, ..., 1]
        let mut chunk_stride = vec![0.into(); self.range.len() + 1];
        chunk_stride[self.range.len()] = 1.into();

        let launch_range = self.launch_range();
        CStruct::new()
            .int(expressions[&flatten_mul_strides(&launch_range, &a_stride_ext)])
            .int(expressions[&flatten_mul_strides(&launch_range, &b_stride_ext)])
            .int(expressions[&flatten_mul_strides(&launch_range, &a_stride_ext)])
            .int(expressions[&self.row_width])
            .int(expressions[&flatten_mul_strides(&launch_range, &chunk_stride)])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        // Extend strides with 0 for the chunk dimension
        let mut a_stride_ext = self.a_stride.clone();
        a_stride_ext.push(0.into());
        let mut b_stride_ext = self.b_stride.clone();
        b_stride_ext.push(0.into());

        let mut chunk_stride = vec![0.into(); self.range.len() + 1];
        chunk_stride[self.range.len()] = 1.into();

        let launch_range = self.launch_range();
        vec![
            flatten_mul_strides(&launch_range, &a_stride_ext),
            flatten_mul_strides(&launch_range, &b_stride_ext),
            self.row_width,
            flatten_mul_strides(&launch_range, &chunk_stride),
        ]
    }
}

#[derive(Debug, Default)]
pub struct RowRMSNorm {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowRMSNorm {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowRMSNorm".to_string(),
            vec![EList, Input, EList, Expr, Input],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                (= ?square (Mul ?inp_range ?x ?inp_stride ?x ?inp_stride ?square_stride))
                (= ?width (nth_from_end ?inp_range 0))
                (= ?batch (nth_from_end ?inp_range 1))
                (= ?square_summed
                    (Sum
                        (ECons ?batch (ENil))
                        ?width
                        ?square
                        (ECons ?width (ENil))
                        (MNum 1)
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?inv_div_factor
                    (Recip (ECons ?batch (ENil)) (Cast (Iota ?width (MNum 1)) (F32))
                                    (ECons (MNum 0) (ENil))  ; broadcast the constant
                                    (ECons (MNum 1) (ENil)))) ; produce per-batch vector

                (= ?mean
                    (Mul (ECons ?batch (ENil))
                                ?square_summed (ECons (MNum 1) (ENil))
                                ?inv_div_factor (ECons (MNum 1) (ENil))
                                (ECons (MNum 1) (ENil))))
                (= ?eps_add
                    (Add
                        (ECons ?batch (ENil))
                        ?mean
                        (ECons (MNum 1) (ENil))
                        (Constant ?eps)
                        (ECons (MNum 0) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?sqrt
                    (Sqrt
                        (ECons ?batch (ENil))
                        ?eps_add
                        (ECons (MNum 1) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?recip
                    (Recip
                        (ECons ?batch (ENil))
                        ?sqrt
                        (ECons (MNum 1) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?std_normed
                    (Mul
                        ?inp_range
                        ?recip
                        (ECons (MNum 1) (ECons (MNum 0) (ENil)))
                        ?x
                        ?inp_stride
                        ?inp_stride
                    )
                )
                (= ?final
                    (Mul
                        ?inp_range
                        ?std_normed
                        ?inp_stride
                        ?weight
                        (ECons (MNum 0) (ECons (MNum 1) (ENil)))
                        ?inp_stride
                    )
                )
                ;(= (F32) (dtype ?x))
            )
            (
                (let ?new
                    (RowRMSNorm
                        (ECons ?batch (ENil))
                        ?x
                        (ECons ?width (ENil))
                        ?width
                        ?weight
                    )
                )
                (union ?final ?new)
                (set (dtype ?new) (F32))
            )
            :name \"row rms norm\"
        )"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[3], expr_cache).unwrap(),
            })),
            vec![children[1], children[4]],
        )
    }
}

impl BlockOp for RowRMSNorm {
    fn op_name(&self) -> &'static str {
        "RowRMSNorm"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load input row + weight row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // Per row: d squares, d-1 adds for sum, div by d, add eps, sqrt, recip, then 2d muls (inp * inv_rms * weight)
        // Approximate: 5*d ops per row
        self.range.iter().copied().product::<Expression>() * self.row_width * 5
    }

    fn cuda_struct(&self) -> String {
        "const int inp; const int out; int row_width;".to_string()
    }

    fn cuda_function(&self) -> String {
        "
        const float* inp = source_ptrs[0] + eval_expression(payload.inp, current);
        float*       out = out_ptr + eval_expression(payload.out, current);

        const int d       = eval_expression(payload.row_width, 0);
        const float eps   = 1e-5f;
        const int nthreads = blockDim.x;

        // Shared partial sums (double for accuracy)
        __shared__ double s_partials[1024];  // assumes blockDim.x <= 1024
        __shared__ float  s_inv_rms;

        // 1) Each thread computes a partial sum of squares over its stripe
        double ss_local = 0.0;
        for (int j = t; j < d; j += nthreads) {
            float x = inp[j];
            ss_local += (double)x * (double)x;
        }

        s_partials[t] = ss_local;
        __syncthreads();

        // 2) Parallel reduction in shared memory to get total sum of squares
        for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
            if (t < offset) {
                s_partials[t] += s_partials[t + offset];
            }
            __syncthreads();
        }

        // 3) Thread 0 computes inv_rms and broadcasts it
        if (t == 0) {
            double ss_total = s_partials[0];
            float denom     = sqrtf((float)(ss_total / (double)d) + eps);
            s_inv_rms       = 1.0f / denom;
        }
        __syncthreads();

        float inv_rms = s_inv_rms;

        // 4) All threads normalize their stripe
        for (int j = t; j < d; j += nthreads) {
            out[j] = inp[j] * inv_rms * source_ptrs[1][j];
        }
        "
        .to_string()
    }

    fn schedule_op(&self, _: &CudaStream, expressions: &FxHashMap<Expression, i32>) -> Vec<u8> {
        CStruct::new()
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&self.row_width])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_mul_strides(&self.range, &self.a_stride),
            self.row_width,
        ]
    }
}

#[derive(Debug, Default)]
pub struct RowRope {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowRope {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowRope".to_string(),
            vec![EList, Input, EList, Expr, Input],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[3], expr_cache).unwrap(),
            })),
            vec![children[1], children[4]],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
           (
                (= ?e (RowRope ?shape ?inp ?stride ?row_width ?pos_ids))
                (= (F32) (dtype ?inp))
            )
           ((set (dtype ?e) (F32)))
        )"
        .to_string()]
    }

    fn early_rewrites(&self) -> Vec<String> {
        vec![
        r#"
            (rule
              (
                ;; Bind the head-count and hidden-dim directly from the places they appear.
                ;; This matches graphs where these are literals (e.g. 32, 4096) *or* already-simplified expressions.
                (= ?inp_strides (ECons (MNum 128) (ECons ?hidden_dim (ECons (MNum 2) (ECons (MNum 1) (ENil))))))

                ;; -----------------------------
                ;; inv_freq construction (exact literals as in dump)
                ;; -----------------------------
                (= ?freq_indices        (Cast (Iota (MMul (MIter) (MNum 2)) (MNum 64)) (F32)))
                (= ?c_inv_head_dim      (Constant 0.007812))
                (= ?freq_scaled         (Mul (ECons (MNum 64) (ENil)) ?freq_indices
                                             (ECons (MNum 1) (ENil)) ?c_inv_head_dim
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?c_ln_theta          (Constant 13.122363))
                (= ?log_arg             (Mul (ECons (MNum 64) (ENil)) ?freq_scaled
                                             (ECons (MNum 1) (ENil)) ?c_ln_theta
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?c_log2e             (Constant 1.442695))
                (= ?exp2_arg            (Mul (ECons (MNum 64) (ENil)) ?log_arg
                                             (ECons (MNum 1) (ENil)) ?c_log2e
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?pow_theta           (Exp2 (ECons (MNum 64) (ENil)) ?exp2_arg
                                              (ECons (MNum 1) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?inv_freq            (Recip (ECons (MNum 64) (ENil)) ?pow_theta
                                               (ECons (MNum 1) (ENil)) (ECons (MNum 1) (ENil))))

                ;; -----------------------------
                ;; emb = pos_ids @ inv_freq
                ;; -----------------------------
                (= ?pos_f32             (Cast ?pos_ids (F32)))
                (= ?pos_times_invfreq_bcast
                   (Mul (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil))))
                        ?pos_f32
                        (ECons (MNum 1) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                        ?inv_freq
                        (ECons (MNum 0) (ECons (MNum 1) (ECons (MNum 0) (ENil))))
                        (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil))))))
                (= ?emb
                   (Sum (ECons (MVar "s") (ECons (MNum 64) (ENil)))
                        (MNum 1)
                        ?pos_times_invfreq_bcast
                        (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                        (MNum 1)
                        (ECons (MNum 64) (ECons (MNum 1) (ENil)))))

                ;; -----------------------------
                ;; Gather odd lane (x1) from inp (structure preserved; 32 -> ?n_heads, 4096 -> ?hidden_dim)
                ;; -----------------------------
                (= ?odd_lane_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MNum 1)
                           (MMul (MMod (MIter) (MNum 64)) (MNum 2)))
                         (MMul (MMod (MDiv (MIter) (MNum 64)) (MVar "s")) (MNum 128)))
                       (MMul (MDiv (MIter) (MMul (MNum 64) (MVar "s")))
                             (MMul (MNum 128) (MVar "s"))))
                     (MMul (MMul ?n_heads (MVar "s")) (MNum 64))))
                (= ?odd_lane
                   (Gather
                     ?odd_lane_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                     ?inp
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     ?inp_strides))

                ;; -----------------------------
                ;; cos(emb) = sin(-emb + pi/2), sin(emb)
                ;; -----------------------------
                (= ?c_neg1    (Constant -1.000000))
                (= ?neg_emb   (Mul (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil))) ?c_neg1
                                   (ECons (MNum 0) (ECons (MNum 0) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?c_pihalf  (Constant 1.570796))
                (= ?cos_phase (Add (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?neg_emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil))) ?c_pihalf
                                   (ECons (MNum 0) (ECons (MNum 0) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?cos_emb   (Sin (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?cos_phase
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?sin_emb   (Sin (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))

                ;; -----------------------------
                ;; even_lane_rot = x0*cos - x1*sin
                ;; -----------------------------
                (= ?x0_cos
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?inp
                        ?inp_strides
                        ?cos_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?x1_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?odd_lane
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?sin_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?c_neg1b (Constant -1.000000))
                (= ?neg_x1_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x1_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?c_neg1b
                        (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?even_lane_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x0_cos
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?neg_x1_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))

                ;; -----------------------------
                ;; odd_lane_rot = x0*sin + x1*cos
                ;; -----------------------------
                (= ?x0_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?inp
                        ?inp_strides
                        ?sin_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?x1_cos
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?odd_lane
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?cos_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?odd_lane_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x0_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?x1_cos
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))

                ;; -----------------------------
                ;; Scatter + masks (keep the same MMul nesting as original)
                ;; -----------------------------
                (= ?scatter_even_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MMin (MMod (MIter) (MNum 2)) (MNum 0))
                           (MMod (MDiv (MIter) (MNum 2)) (MNum 64)))
                         (MMul (MMod (MDiv (MIter) (MNum 128)) (MVar "s")) (MNum 64)))
                       (MMul (MDiv (MIter) (MMul (MNum 128) (MVar "s"))) (MMul (MNum 64) (MVar "s"))))
                     (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?scattered_even
                   (Gather
                     ?scatter_even_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                     ?even_lane_rot
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?even_mask
                   (Iota (MLt (MMod (MIter) (MNum 2)) (MNum 1))
                         (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?even_masked
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?scattered_even
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?even_mask
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                (= ?scatter_odd_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MMax (MSub (MMod (MIter) (MNum 2)) (MNum 1)) (MNum 0))
                           (MMod (MDiv (MIter) (MNum 2)) (MNum 64)))
                         (MMul (MMod (MDiv (MIter) (MNum 128)) (MVar "s")) (MNum 64)))
                       (MMul (MDiv (MIter) (MMul (MNum 128) (MVar "s"))) (MMul (MNum 64) (MVar "s"))))
                     (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?scattered_odd
                   (Gather
                     ?scatter_odd_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                     ?odd_lane_rot
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?odd_mask
                   (Iota (MGte (MMod (MIter) (MNum 2)) (MNum 1))
                         (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?odd_masked
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?scattered_odd
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?odd_mask
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                (= ?interleaved_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?even_masked
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?odd_masked
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                ;; Final identity mul "* 1.0" with output shape/strides (4096 -> ?hidden_dim)
                (= ?c_one (Constant 1.000000))
                (= ?rope_out
                   (Mul (ECons (MVar "s") (ECons ?n_heads (ECons (MNum 128) (ENil))))
                        ?interleaved_rot
                        (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil))))
                        ?c_one
                        (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                        (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil)))))
              )
              )
              (
                (union ?rope_out
                  (RowRope
                    (ECons (MVar "s") (ENil))
                    ?inp
                    (ECons ?hidden_dim (ENil))
                    ?hidden_dim
                    ?pos_ids))
                ; we want to subsume all terms up to ?inp and ?pos_ids. don't know how to do this.
                (delete (Mul (ECons (MVar "s") (ECons ?n_heads (ECons (MNum 128) (ENil))))
                     ?interleaved_rot
                     (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil))))
                     ?c_one
                     (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                     (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil))))))
              )
              :name "row rope"
            )
        "#.to_string()]
    }
}

impl BlockOp for RowRope {
    fn op_name(&self) -> &'static str {
        "RowRope"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load input row (row_width floats) + token_ids (1 int per row)
        self.range.iter().copied().product::<Expression>() * (self.row_width * 4 + 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // Per pair of elements: pow, sincos, 4 muls, 2 adds ≈ 10 ops
        // row_width/2 pairs per row
        self.range.iter().copied().product::<Expression>() * self.row_width * 5
    }

    fn cuda_struct(&self) -> String {
        "const int inp; const int out; int row_width; const int token_ids;".to_string()
    }

    fn cuda_function(&self) -> String {
        "
        const float* inp = source_ptrs[0] + eval_expression(payload.inp, current);
        float*       out = out_ptr + eval_expression(payload.out, current);
        const int* token_ids = (const int*)source_ptrs[1] + eval_expression(payload.token_ids, current);

        const int D_total = eval_expression(payload.row_width, 0);    // = n_heads * d_head
        const int d_head  = 128;            // head_dim
        const int n_heads = D_total / d_head;

        const int   pos  = token_ids[0];   // must match position_ids[batch, seq]
        const float base = 500000.0f;

        const int half = d_head / 2;            // 64 when d_head = 128

        for (int h = 0; h < n_heads; ++h) {
            const float* head_in  = inp + h * d_head;
            float*       head_out = out + h * d_head;

            // k indexes within the first half [0 .. half-1]
            for (int k = t; k < half; k += blockDim.x) {
                const int j0 = k;           // first half index
                const int j1 = k + half;    // corresponding second-half index

                // exponent = -(2*k / d_head) to match inv_freq = base^{-(arange(0,dim,2)/dim)}
                const float exponent = -(2.0f * (float)k) / (float)d_head;
                const float theta    = (float)pos * __powf(base, exponent);

                float s, c;
                __sincosf(theta, &s, &c);

                const float x0 = head_in[j0];
                const float x1 = head_in[j1];

                head_out[j0] = x0 * c - x1 * s;
                head_out[j1] = x1 * c + x0 * s;
            }
        }
        "
        .to_string()
    }

    fn schedule_op(&self, _: &CudaStream, expressions: &FxHashMap<Expression, i32>) -> Vec<u8> {
        CStruct::new()
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&self.row_width])
            .int(expressions[&'z'.into()])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_mul_strides(&self.range, &self.a_stride),
            self.row_width,
            'z'.into(),
        ]
    }
}

use crate::TILE_SIZE;
const K_CHUNK_SIZE: usize = 4096;

#[derive(Debug, Default)]
pub struct TileMatmulSplitK {
    range: Vec<Expression>,         // [batch..., tiled_m, tiled_n, k_chunks]
    untiled_range: Vec<Expression>, // [M, N]
    total_k: Expression,
    a_stride: Vec<Expression>,
    a_m_stride: Expression,
    b_stride: Vec<Expression>,
    b_n_stride: Expression,
    out_stride: Vec<Expression>,
    out_m_stride: Expression,
}

impl EgglogOp for TileMatmulSplitK {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "TileMatmulSplitK".to_string(),
            vec![
                EList, EList, Expr, Input, EList, Expr, Expr, Input, EList, Expr, Expr, EList,
                Expr, Expr,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            // Direct Mul -> Sum -> TileMatmulSplitK (A row-major, B col-major, C row-major)
            format!(
                "
        (rule
            (
                ; Match Mul node
                (= ?mul (Mul ?mul_shape ?a ?a_stride ?b ?b_stride ?mul_out_stride))

                ; Match Sum that reduces the Mul (k dimension)
                (= ?sum (Sum ?out_shape ?k ?mul ?sum_in_stride ?k_stride ?sum_out_stride))

                ; Get dimensions from output shape
                (= ?m (nth_from_end ?out_shape 1))
                (= ?n (nth_from_end ?out_shape 0))
                (!= ?m (MNum 0))
                (!= ?n (MNum 0))

                ; Get output strides
                (= ?sum_out_m_stride (nth_from_end ?sum_out_stride 1))
                (= ?sum_out_n_stride (nth_from_end ?sum_out_stride 0))

                ; Get A strides
                (= ?a_m_stride (nth_from_end ?a_stride 2))
                (= ?a_n_stride (nth_from_end ?a_stride 1))
                (= ?a_k_stride (nth_from_end ?a_stride 0))

                ; Get B strides
                (= ?b_m_stride (nth_from_end ?b_stride 2))
                (= ?b_n_stride (nth_from_end ?b_stride 1))
                (= ?b_k_stride (nth_from_end ?b_stride 0))

                ; Assert contiguous k stride on output (required for reduction)
                (= ?k_stride (MNum 1))

                ; Assert A has contiguous k (row-major A)
                (= ?a_k_stride (MNum 1))

                ; Assert B has contiguous k (col-major B / transposed)
                (= ?b_k_stride (MNum 1))

                (= (F32) (dtype ?a))
            )
            (
                ; Create tiled shape with K chunks
                (let ?tiled_m (MCeilDiv ?m (MNum {ts})))
                (let ?tiled_n (MCeilDiv ?n (MNum {ts})))
                (let ?k_chunks (MCeilDiv ?k (MNum {kc})))
                (let ?tiled_shape
                    (ECons ?k_chunks
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd ?out_shape ?tiled_n 0)
                        ?tiled_m 1)))

                ; Create tiled strides for A: scale m and n strides, remove k
                (let ?scaled_a_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?a_stride
                            (MMul ?a_n_stride (MNum {ts})) 1)
                        (MMul ?a_m_stride (MNum {ts})) 2))
                (let ?tiled_a_stride (ECons (MNum 0) (RemoveNthFromEnd ?scaled_a_stride 0)))

                ; Create tiled strides for B: scale m and n strides, remove k
                (let ?scaled_b_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?b_stride
                            (MMul ?b_n_stride (MNum {ts})) 1)
                        (MMul ?b_m_stride (MNum {ts})) 2))
                (let ?tiled_b_stride (ECons (MNum 0) (RemoveNthFromEnd ?scaled_b_stride 0)))

                ; Create tiled output strides (k_chunk dimension has 0 stride since all chunks write to same output)
                (let ?tiled_out_stride
                    (ECons (MNum 0)
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd ?sum_out_stride (MMul ?sum_out_n_stride (MNum {ts})) 0)
                        (MMul ?sum_out_m_stride (MNum {ts})) 1)))

                (let ?tm (TileMatmulSplitK
                    ?tiled_shape ?out_shape ?k
                    ?a ?tiled_a_stride ?a_m_stride (MNum 1)
                    ?b ?tiled_b_stride (MNum 1) ?b_n_stride
                    ?tiled_out_stride ?sum_out_m_stride (MNum 1)))
                (union ?sum ?tm)
                (set (dtype ?tm) (F32))
            )
            :name \"tile matmul split k\"
        )",
                ts = TILE_SIZE,
                kc = K_CHUNK_SIZE
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                untiled_range: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                total_k: extract_expr(egraph, children[2], expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                a_m_stride: extract_expr(egraph, children[5], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[8], list_cache, expr_cache).unwrap(),
                b_n_stride: extract_expr(egraph, children[10], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[11], list_cache, expr_cache)
                    .unwrap(),
                out_m_stride: extract_expr(egraph, children[12], expr_cache).unwrap(),
            })),
            vec![children[3], children[7]],
        )
    }
}

impl BlockOp for TileMatmulSplitK {
    fn op_name(&self) -> &'static str {
        "TileMatmulSplitK"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.untiled_range.iter().copied().product::<Expression>()
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        // All dimensions are separable except k_chunks (at index 0)
        // since multiple k_chunks write to the same output tile via atomicAdd
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        let mut sep = vec![true; self.range.len()];
        sep[0] = false; // k_chunk dimension at index 0 is NOT separable
        sep
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        // For input A: all dims except n (at index len-1)
        let mut a = vec![true; self.range.len()];
        a[self.range.len() - 1] = false; // n dimension
                                         // For input B: all dims except m (at index len-2)
        let mut b = vec![true; self.range.len()];
        b[self.range.len() - 2] = false; // m dimension
        vec![a, b]
    }

    fn bytes_stored(&self) -> Expression {
        // Store C (M * N) floats - each k_chunk atomically adds
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        batch * m * n * 4
    }

    fn flops(&self) -> Expression {
        // Matmul FLOPs: 2 * M * N * K (one mul + one add per output element per K iteration)
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        batch * m * n * k * 2
    }

    fn cuda_struct(&self) -> String {
        "const int untiled_range[2]; const int a; const int b; const int c; int total_k; int a_width; int b_width; int c_width; int m_pos_stride; int n_pos_stride; int k_chunk_stride;".to_string()
    }

    fn bytes_loaded(&self) -> Expression {
        // Load A (M * K) + B (K * N) per batch
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        batch * (m * k + k * n) * 4
    }

    fn cuda_function(&self) -> String {
        format!("
        auto warp_reduce_sum = [](float val) {{
            for (int offset = 16; offset > 0; offset >>= 1) {{
                val += __shfl_down_sync(0xffffffff, val, offset);
            }}
            return val;
        }};
        const int k_chunk = eval_expression(payload.k_chunk_stride, current);
        const int total_K = eval_expression(payload.total_k, 0);
        const int k_start = k_chunk * {kc};
        const int k_end = min(k_start + {kc}, total_K);
        const int K = k_end - k_start;

        if (K <= 0) return;

        const float* a = source_ptrs[0] + eval_expression(payload.a, current) + k_start;
        const float* b = source_ptrs[1] + eval_expression(payload.b, current) + k_start;
        float*       c = out_ptr + eval_expression(payload.c, current);
        const int m_pos = eval_expression(payload.m_pos_stride, current);
        const int n_pos = eval_expression(payload.n_pos_stride, current);

        const int threads   = blockDim.x;
        const int lane      = t & 31;
        const int warp_id   = t >> 5;
        const int num_warps = threads >> 5;

        constexpr int TILE_SIZE = {ts};

        const int global_m0 = m_pos * TILE_SIZE;
        const int global_n0 = n_pos * TILE_SIZE;
        const int M = eval_expression(payload.untiled_range[0], 0);
        const int N = eval_expression(payload.untiled_range[1], 0);

        const int rows_left = M - global_m0;
        const int cols_left = N - global_n0;
        if (rows_left <= 0 || cols_left <= 0) return;

        const int tile_m = min(rows_left, TILE_SIZE);
        const int tile_n = min(cols_left, TILE_SIZE);

        const int b_width = eval_expression(payload.b_width, 0);

        // Fast path for M=1 decode: warps parallelize over columns with K reduction
        if (tile_m == 1 && num_warps > 0) {{
            constexpr int COLS_PER_WARP = 4;
            for (int col_base = warp_id * COLS_PER_WARP; col_base < tile_n; col_base += num_warps * COLS_PER_WARP) {{
                float partial[COLS_PER_WARP] = {{0.0f, 0.0f, 0.0f, 0.0f}};

                // Stream K elements from this chunk
                for (int k = lane; k < K; k += 32) {{
                    float a_val = a[k];
                    #pragma unroll
                    for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                        if (col_base + ci < tile_n) {{
                            partial[ci] += a_val * b[(col_base + ci) * b_width + k];
                        }}
                    }}
                }}

                // Warp reduction for each column
                #pragma unroll
                for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                    partial[ci] = warp_reduce_sum(partial[ci]);
                }}
                // Lane 0 atomically adds results
                if (lane == 0) {{
                    #pragma unroll
                    for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                        if (col_base + ci < tile_n) {{
                            atomicAdd(&c[col_base + ci], partial[ci]);
                        }}
                    }}
                }}
            }}
        }} else {{
            // Generic path: handle any M, N tile
            const int tile_elems = tile_m * tile_n;
            const int c_width = eval_expression(payload.c_width, 0);
            const int a_width = eval_expression(payload.a_width, 0);

            for (int idx = t; idx < tile_elems; idx += threads) {{
                int ty = idx / tile_n;
                int tx = idx % tile_n;

                const float* A0 = a + ty * a_width;
                const float* B0 = b + tx * b_width;
                float*       C0 = c + ty * c_width + tx;

                float acc = 0.f;
                for (int k = 0; k < K; ++k) {{
                    acc += A0[k] * B0[k];
                }}

                atomicAdd(C0, acc);
            }}
        }}
        ", ts = TILE_SIZE, kc = K_CHUNK_SIZE)
    }

    fn schedule_op(&self, _: &CudaStream, expressions: &FxHashMap<Expression, i32>) -> Vec<u8> {
        assert_eq!(self.untiled_range.len(), 2);
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        // k_chunk is at index 0
        let mut k_chunk_stride = vec![0.into(); self.range.len()];
        k_chunk_stride[0] = 1.into();
        // m_pos (tiled_m) is at index len-2
        let mut m_pos_stride = vec![0.into(); self.range.len()];
        m_pos_stride[self.range.len() - 2] = 1.into();
        // n_pos (tiled_n) is at index len-1
        let mut n_pos_stride = vec![0.into(); self.range.len()];
        n_pos_stride[self.range.len() - 1] = 1.into();
        CStruct::new()
            .ints(
                &self
                    .untiled_range
                    .iter()
                    .map(|e| expressions[e])
                    .collect_vec(),
            )
            .int(expressions[&flatten_mul_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.b_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &self.out_stride)])
            .int(expressions[&self.total_k])
            .int(expressions[&self.a_m_stride])
            .int(expressions[&self.b_n_stride])
            .int(expressions[&self.out_m_stride])
            .int(expressions[&flatten_mul_strides(&self.range, &m_pos_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &n_pos_stride)])
            .int(expressions[&flatten_mul_strides(&self.range, &k_chunk_stride)])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        let mut k_chunk_stride = vec![0.into(); self.range.len()];
        k_chunk_stride[0] = 1.into();
        let mut m_pos_stride = vec![0.into(); self.range.len()];
        m_pos_stride[self.range.len() - 2] = 1.into();
        let mut n_pos_stride = vec![0.into(); self.range.len()];
        n_pos_stride[self.range.len() - 1] = 1.into();
        vec![
            self.untiled_range[0],
            self.untiled_range[1],
            flatten_mul_strides(&self.range, &self.a_stride),
            flatten_mul_strides(&self.range, &self.b_stride),
            flatten_mul_strides(&self.range, &self.out_stride),
            self.total_k,
            self.a_m_stride,
            self.b_n_stride,
            self.out_m_stride,
            flatten_mul_strides(&self.range, &m_pos_stride),
            flatten_mul_strides(&self.range, &n_pos_stride),
            flatten_mul_strides(&self.range, &k_chunk_stride),
        ]
    }
}

#[derive(Debug)]
pub struct CStruct {
    buf: Vec<u8>,
    max_align: usize,
}

impl Default for CStruct {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            max_align: 1,
        }
    }
}

impl CStruct {
    pub fn new() -> Self {
        Self::default()
    }

    fn align_to(&mut self, align: usize) {
        self.max_align = self.max_align.max(align);

        let len = self.buf.len();
        let rem = len % align;
        if rem != 0 {
            let pad = align - rem;
            self.buf.extend(std::iter::repeat_n(0u8, pad));
        }
    }

    pub fn int(mut self, v: i32) -> Self {
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn ints(mut self, vs: &[i32]) -> Self {
        self.align_to(4);
        for &v in vs {
            self.buf.extend_from_slice(&v.to_ne_bytes());
        }
        self
    }

    pub fn float(mut self, v: f32) -> Self {
        self.align_to(4);
        self.buf.extend_from_slice(&v.to_ne_bytes());
        self
    }

    pub fn floats(mut self, vs: &[f32]) -> Self {
        self.align_to(4);
        for &v in vs {
            self.buf.extend_from_slice(&v.to_ne_bytes());
        }
        self
    }

    pub fn bool(mut self, v: bool) -> Self {
        self.align_to(1);
        self.buf.push(if v { 1 } else { 0 });
        self
    }

    pub fn ptr_const_f32(mut self, p: *const f32) -> Self {
        let ptr_size = std::mem::size_of::<usize>(); // usually 8
        let ptr_align = ptr_size;
        self.align_to(ptr_align);

        let addr = p as usize;
        let bytes = addr.to_ne_bytes();

        self.buf.extend_from_slice(&bytes[..ptr_size]);
        self
    }

    pub fn ptr_mut_f32(self, p: *mut f32) -> Self {
        self.ptr_const_f32(p as *const f32)
    }

    /// Returns the current size of the buffer after alignment for a pointer field.
    /// Useful for computing field offsets.
    pub fn current_size(&self) -> usize {
        let ptr_align = std::mem::size_of::<usize>();
        let len = self.buf.len();
        let rem = len % ptr_align;
        if rem != 0 {
            len + (ptr_align - rem)
        } else {
            len
        }
    }

    /// Pad the struct size to a multiple of max_align.
    pub fn finish_struct(mut self) -> Vec<u8> {
        let align = self.max_align;
        if align > 1 {
            let len = self.buf.len();
            let rem = len % align;
            if rem != 0 {
                let pad = align - rem;
                self.buf.extend(std::iter::repeat_n(0u8, pad));
            }
        }
        self.buf
    }

    /// Insert a raw byte field (e.g., another struct).
    /// `align` must be the alignment of the nested struct.
    pub fn bytes(mut self, align: usize, data: &[u8]) -> Self {
        self.align_to(align);
        self.buf.extend_from_slice(data);
        self
    }
}
