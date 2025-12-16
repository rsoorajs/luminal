use std::fmt::Debug;

use super::CustomState;
use cudarc::driver::{CudaStream, DevicePtr};
use egraph_serialize::NodeId;
use itertools::Itertools;
use luminal::{
    graph::{extract_expr, extract_expr_list, SerializedEGraph},
    shape::Expression,
    utils::{
        flatten_strides, CStructBuilder, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
};
use rustc_hash::FxHashMap;

use crate::block::BlockOp;

pub type MKOps = (
    RowAdd,
    RowSwishMul,
    RowRMSNorm,
    RowRope,
    TileMatmul,
    GQAAttention,
);

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
                ; get  add
                (= ?sa (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                (= ?row_width (nth_from_end ?shape 0))
                ; assert the row is contiguous
                (= (MIter) (nth_from_end ?a_stride 0))
                (= (MIter) (nth_from_end ?b_stride 0))
                (= (MIter) (nth_from_end ?out_stride 0))
            )
            (
                (let ?new_shape (RemoveNthFromEnd ?shape 0))
                (let ?new_a_stride (RemoveNthFromEnd ?a_stride 0))
                (let ?new_b_stride (RemoveNthFromEnd ?b_stride 0))
                (let ?new_out_stride (RemoveNthFromEnd ?out_stride 0))
                (union ?sa (RowAdd ?new_shape ?a ?new_a_stride ?b ?new_b_stride ?new_out_stride ?row_width))
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
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
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
    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body =
            "const int a_strides; const int b_strides; const int out_strides; int row_width;"
                .to_string();
        let function_body = "
            const float* a = source_ptrs[0] + eval_expression(payload.a_strides, current);
            const float* b = source_ptrs[1] + eval_expression(payload.b_strides, current);
            float* out = out_ptr + eval_expression(payload.out_strides, current);

            for (int idx = t; idx < eval_expression(payload.row_width, 0); idx += blockDim.x) {
                out[idx] = a[idx] + b[idx];
            }
        "
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        _: &mut FxHashMap<String, CustomState>,
        _: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        CStructBuilder::new()
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.b_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.out_stride)])
            .int(expressions[&self.row_width])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_strides(&self.range, &self.a_stride),
            flatten_strides(&self.range, &self.b_stride),
            flatten_strides(&self.range, &self.out_stride),
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
}

impl EgglogOp for RowSwishMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowSwishMul".to_string(),
            vec![EList, Input, EList, Input, EList, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                (= ?sigmoid (Sigmoid
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                ))
                (= ?swish (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                    ?sigmoid
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                ))
                (= ?swishmul (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?swish
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                    ?other
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                    (ECons (MMul (MIter) ?width) (ECons (MIter) (ENil)))
                ))
            )
            (
                (union ?swishmul
                    (RowSwishMul
                        (ECons ?batch (ENil))
                        ?self
                        (ECons (MMul (MIter) ?width) (ENil))
                        ?other
                        (ECons (MMul (MIter) ?width) (ENil))
                        ?width
                    )
                )
            )
        )"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[5], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl BlockOp for RowSwishMul {
    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body = "
            const int a;
            const int b;
            const int out;
            int row_width;
        "
        .to_string();
        let function_body = "
            const float* a = source_ptrs[0] + eval_expression(payload.a, current);
            const float* b = source_ptrs[1] + eval_expression(payload.b, current);
            float* out = out_ptr + eval_expression(payload.out, current);

            for (int idx = t; idx < eval_expression(payload.row_width, 0); idx += blockDim.x) {
                float x = a[idx];
                float sw = x / (1.0f + __expf(-x)); // swish(x)
                out[idx] = sw * b[idx];
            }
        "
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        _: &mut FxHashMap<String, CustomState>,
        _: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        CStructBuilder::new()
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.b_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&self.row_width])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_strides(&self.range, &self.a_stride),
            flatten_strides(&self.range, &self.b_stride),
            self.row_width,
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
                        (ECons (MMul (MIter) ?width) (ENil))
                        (MIter)
                        (ECons (MIter) (ENil))
                    )
                )
                (= ?inv_div_factor
                    (Recip (ECons ?batch (ENil)) (Iota ?width (MNum 1))
                                    (ECons (MNum 0) (ENil))  ; broadcast the constant
                                    (ECons (MIter) (ENil)))) ; produce per-batch vector

                (= ?mean
                    (Mul (ECons ?batch (ENil))
                                ?square_summed (ECons (MIter) (ENil))
                                ?inv_div_factor (ECons (MIter) (ENil))
                                (ECons (MIter) (ENil))))
                (= ?eps_add
                    (Add
                        (ECons ?batch (ENil))
                        ?mean
                        (ECons (MIter) (ENil))
                        (Constant ?eps)
                        (ECons (MNum 0) (ENil))
                        (ECons (MIter) (ENil))
                    )
                )
                (= ?sqrt
                    (Sqrt
                        (ECons ?batch (ENil))
                        ?eps_add
                        (ECons (MIter) (ENil))
                        (ECons (MIter) (ENil))
                    )
                )
                (= ?recip
                    (Recip
                        (ECons ?batch (ENil))
                        ?sqrt
                        (ECons (MIter) (ENil))
                        (ECons (MIter) (ENil))
                    )
                )
                (= ?std_normed
                    (Mul
                        ?inp_range
                        ?recip
                        (ECons (MIter) (ECons (MNum 0) (ENil)))
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
                        (ECons (MNum 0) (ECons (MIter) (ENil)))
                        ?inp_stride
                    )
                )
            )
            (
                (let ?new
                    (RowRMSNorm
                        (ECons ?batch (ENil))
                        ?x
                        (ECons (MMul (MIter) ?width) (ENil))
                        ?width
                        ?weight
                    )
                )
                (union ?final ?new)
            )
        )"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
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
    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body = "
            const int inp;
            const int out;
            int row_width;
        "
        .to_string();
        let function_body = "
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
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        _: &mut FxHashMap<String, CustomState>,
        _: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        CStructBuilder::new()
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&self.row_width])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![flatten_strides(&self.range, &self.a_stride), self.row_width]
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
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
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

impl BlockOp for RowRope {
    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body = "
            const int inp;
            const int out;
            int row_width;
            const int token_ids;
        "
        .to_string();
        let function_body = "
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
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        _: &mut FxHashMap<String, CustomState>,
        _: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        CStructBuilder::new()
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&self.row_width])
            .int(expressions[&'z'.into()])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        vec![
            flatten_strides(&self.range, &self.a_stride),
            self.row_width,
            'z'.into(),
        ]
    }
}

#[derive(Debug, Default)]
pub struct TileMatmul {
    range: Vec<Expression>,
    untiled_range: Vec<Expression>,
    iters: Expression,
    a_stride: Vec<Expression>,
    a_m_stride: Expression,
    b_stride: Vec<Expression>,
    b_n_stride: Expression,
    out_stride: Vec<Expression>,
    out_m_stride: Expression,
}

impl EgglogOp for TileMatmul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "TileMatmul".to_string(),
            vec![
                EList, EList, Expr, Input, EList, Expr, Expr, Input, EList, Expr, Expr, EList,
                Expr, Expr,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
        ; Cube mul - Tile sum -> TileMatmul (row major)
        (rule
            (
                ; get cube mul
                (= ?cm (CubeMul ?mul_shape ?untiled_mul_shape ?a ?a_stride ?a_m_stride ?a_n_stride ?a_k_stride ?b ?b_stride ?b_m_stride ?b_n_stride ?b_k_stride ?out_stride ?out_m_stride ?out_n_stride ?out_k_stride))
                ; get tile sum
                (= ?ts (TileSum ?sum_shape ?untiled_sum_shape ?iters ?cm ?sum_in_stride ?sum_in_m_stride ?sum_in_n_stride ?sum_in_k_stride ?sum_out_stride ?sum_out_m_stride ?sum_out_n_stride))
                ; assert k stride on the intermediate is 1
                (= ?out_k_stride (MIter))
                (= ?sum_in_k_stride (MIter))
                ; assert matmul strides
                (= ?b_n_stride (MIter))
                ; get dimensions
                (= ?t_n (nth_from_end ?mul_shape 1))
                (= ?t_k (nth_from_end ?mul_shape 0))
            )
            (
                ; input strides are same as cube mul but without last element
                (let ?new_a_stride (RemoveNthFromEnd ?a_stride 0))
                (let ?new_b_stride (RemoveNthFromEnd ?b_stride 0))
                (union ?ts (TileMatmul ?sum_shape ?untiled_sum_shape ?iters ?a ?new_a_stride (MMul ?t_k (MNum 32)) (MNum 1) ?b ?new_b_stride (MNum 1) (MMul ?t_n (MNum 32)) ?sum_out_stride (MMul ?t_n (MNum 32)) (MNum 1)))
            )
        )".to_string(),
        "
        ; Cube mul - Tile sum -> TileMatmul (A row-major, B col-major, C row-major)
        (rule
            (
                ; get cube mul
                (= ?cm (CubeMul ?mul_shape ?untiled_mul_shape
                                ?a ?a_stride ?a_m_stride ?a_n_stride ?a_k_stride
                                ?b ?b_stride ?b_m_stride ?b_n_stride ?b_k_stride
                                ?out_stride ?out_m_stride ?out_n_stride ?out_k_stride))
                ; get tile sum
                (= ?ts (TileSum ?sum_shape ?untiled_sum_shape ?iters ?cm
                                ?sum_in_stride ?sum_in_m_stride ?sum_in_n_stride ?sum_in_k_stride
                                ?sum_out_stride ?sum_out_m_stride ?sum_out_n_stride))

                ; assert k stride on the intermediate is 1 (contiguous)
                (= ?out_k_stride (MIter))
                (= ?sum_in_k_stride (MIter))

                ; A row-major (contiguous in its last dim k)
                (= ?a_k_stride (MIter))

                ; B col-major (contiguous in its first dim k)
                (= ?b_k_stride (MIter))

                ; get tile dims
                (= ?t_n (nth_from_end ?mul_shape 1))
                (= ?t_k (nth_from_end ?mul_shape 0))
            )
            (
                ; input strides are same as cube mul but without last element
                (let ?new_a_stride (RemoveNthFromEnd ?a_stride 0))
                (let ?new_b_stride (RemoveNthFromEnd ?b_stride 0))

                ; Emit TileMatmul:
                ;  - A row-major tile strides: m -> t_k*32, k -> 1
                ;  - B col-major tile strides: k -> 1, n -> t_k*32
                ;  - C row-major tile strides: m -> t_n*32, n -> 1
                (union ?ts
                    (TileMatmul ?sum_shape ?untiled_sum_shape ?iters
                                ?a ?new_a_stride (MMul ?t_k (MNum 32)) (MNum 1)
                                ?b ?new_b_stride (MReplace ?b_k_stride (MIter) (MNum 1)) (MMul ?t_k (MNum 32))
                                ?sum_out_stride (MMul ?t_n (MNum 32)) (MNum 1)))
            )
        )
        ".to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                untiled_range: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, children[2], expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                a_m_stride: extract_expr(egraph, children[5], expr_cache).unwrap(),
                // a_n_stride: extract_expr(egraph, children[6], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[8], list_cache, expr_cache).unwrap(),
                // b_m_stride: extract_expr(egraph, children[9], expr_cache).unwrap(),
                b_n_stride: extract_expr(egraph, children[10], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[11], list_cache, expr_cache)
                    .unwrap(),
                out_m_stride: extract_expr(egraph, children[12], expr_cache).unwrap(),
                // out_n_stride: extract_expr(egraph, children[13], expr_cache).unwrap(),
            })),
            vec![children[3], children[7]],
        )
    }
}

impl BlockOp for TileMatmul {
    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.untiled_range.iter().copied().product::<Expression>()
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        let mut a = vec![true; self.range.len()];
        a[self.range.len() - 1] = false;
        let mut b = vec![true; self.range.len()];
        b[self.range.len() - 2] = false;
        vec![a, b]
    }

    fn cuda_op(&self) -> (String, String) {
        let struct_body = "
            const int untiled_range[2];
            const int a;
            const int b;
            const int c;
            int iters;
            int a_width;
            int b_width;
            int c_width;
            int m_pos_stride;
            int n_pos_stride;
        "
        .to_string();
        let function_body = "
            auto warp_reduce_sum = [](float val) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                return val;
            };
            extern __shared__ __align__(16) char smem[];
            const float* a = source_ptrs[0] + eval_expression(payload.a, current);
            const float* b = source_ptrs[1] + eval_expression(payload.b, current);
            float*       c = out_ptr + eval_expression(payload.c, current);
            const int m_pos = eval_expression(payload.m_pos_stride, current);
            const int n_pos = eval_expression(payload.n_pos_stride, current);

            const int threads   = blockDim.x;
            const int lane      = t & 31;
            const int warp_id   = t >> 5;
            const int num_warps = threads >> 5;  // assume threads is a multiple of 32
            const int iters = eval_expression(payload.iters, 0);

            // Fast path: 1 x 32 output row, full 32 columns, K multiple of 128
            if (eval_expression(payload.untiled_range[0], 0) == 1 &&
                n_pos * 32 + 32 <= eval_expression(payload.untiled_range[1], 0) &&
                iters % 128 == 0 &&
                num_warps > 0) {

                const int K      = iters;
                constexpr int tileK = 128;                 // tune as desired
                float* sh_a      = reinterpret_cast<float*>(smem); // >= tileK floats

                // Each warp may handle multiple columns (stride over 32 columns)
                constexpr int MAX_COLS_PER_WARP = 32;      // upper bound
                float acc[MAX_COLS_PER_WARP];
                #pragma unroll
                for (int i = 0; i < MAX_COLS_PER_WARP; ++i) {
                    acc[i] = 0.0f;
                }

                // Loop over K in tiles of tileK
                for (int k0 = 0; k0 < K; k0 += tileK) {
                    const int this_tile = min(tileK, K - k0);

                    // All threads cooperatively load A tile into shared memory
                    for (int kk = t; kk < this_tile; kk += threads) {
                        sh_a[kk] = a[k0 + kk];
                    }
                    __syncthreads();

                    if (warp_id < num_warps) {
                        int col_idx_in_warp = 0;
                        // Each warp covers columns j = warp_id, warp_id+num_warps, ...
                        for (int j = warp_id; j < 32; j += num_warps, ++col_idx_in_warp) {
                            const float* Brow = b + j * eval_expression(payload.b_width, 0);
                            float partial = 0.0f;

                            // Each lane handles kk = lane, lane+32, ...
                            #pragma unroll
                            for (int kk = lane; kk < this_tile; kk += 32) {
                                float a_k  = sh_a[kk];
                                float b_kj = Brow[k0 + kk];
                                partial = fmaf(a_k, b_kj, partial);
                            }

                            // Reduce within warp for this column
                            partial = warp_reduce_sum(partial);
                            if (lane == 0) {
                                acc[col_idx_in_warp] += partial;
                            }
                        }
                    }

                    __syncthreads();
                }

                // Final write: each warp writes its columns
                if (warp_id < num_warps && lane == 0) {
                    int col_idx_in_warp = 0;
                    for (int j = warp_id; j < 32; j += num_warps, ++col_idx_in_warp) {
                        c[j] = acc[col_idx_in_warp];
                    }
                }

            } else {
                // Generic / predicated path: handle M<=32, N<=32 with any number of threads
                const int global_m0 = m_pos * 32;
                const int global_n0 = n_pos * 32;

                int rows_left = eval_expression(payload.untiled_range[0], 0) - global_m0;
                int cols_left = eval_expression(payload.untiled_range[1], 0) - global_n0;

                if (rows_left <= 0 || cols_left <= 0) return;

                const int tile_m = rows_left > 32 ? 32 : rows_left;
                const int tile_n = cols_left > 32 ? 32 : cols_left;

                const int tile_elems = tile_m * tile_n;

                const int a_width = eval_expression(payload.a_width, 0);
                const int b_width = eval_expression(payload.b_width, 0);
                const int c_width = eval_expression(payload.c_width, 0);

                // Stride over all valid (ty, tx) in this tile
                for (int idx = t; idx < tile_elems; idx += threads) {
                    int ty = idx / tile_n;  // 0 .. tile_m-1
                    int tx = idx % tile_n;  // 0 .. tile_n-1

                    const float* A0 = a + ty * a_width;
                    const float* B0 = b + tx * b_width;
                    float*       C0 = c + ty * c_width + tx;

                    float acc = 0.f;
                    for (int k = 0; k < iters; ++k) {
                        acc += A0[k] * B0[k];
                    }
                    *C0 = acc;
                }
            }
        "
        .to_string();
        (struct_body, function_body)
    }

    fn schedule_op(
        &self,
        _: &mut FxHashMap<String, CustomState>,
        _: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        assert_eq!(self.untiled_range.len(), 2);
        let mut m_pos_stride = vec![0.into(); self.range.len()];
        m_pos_stride[self.range.len() - 2] = 'z'.into();
        let mut n_pos_stride = vec![0.into(); self.range.len()];
        n_pos_stride[self.range.len() - 1] = 'z'.into();
        CStructBuilder::new()
            .ints(
                &self
                    .untiled_range
                    .iter()
                    .map(|e| expressions[e])
                    .collect_vec(),
            )
            .int(expressions[&flatten_strides(&self.range, &self.a_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.b_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.out_stride)])
            .int(expressions[&self.iters])
            .int(expressions[&self.a_m_stride])
            .int(expressions[&self.b_n_stride])
            .int(expressions[&self.out_m_stride])
            .int(expressions[&flatten_strides(&self.range, &m_pos_stride)])
            .int(expressions[&flatten_strides(&self.range, &n_pos_stride)])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        let mut m_pos_stride = vec![0.into(); self.range.len()];
        m_pos_stride[self.range.len() - 2] = 'z'.into();
        let mut n_pos_stride = vec![0.into(); self.range.len()];
        n_pos_stride[self.range.len() - 1] = 'z'.into();
        vec![
            self.untiled_range[0],
            self.untiled_range[1],
            flatten_strides(&self.range, &self.a_stride),
            flatten_strides(&self.range, &self.b_stride),
            flatten_strides(&self.range, &self.out_stride),
            self.iters,
            self.a_m_stride,
            self.b_n_stride,
            self.out_m_stride,
            flatten_strides(&self.range, &m_pos_stride),
            flatten_strides(&self.range, &n_pos_stride),
        ]
    }
}

#[derive(Debug, Default)]
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

impl EgglogOp for GQAAttention {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "GQAAttention".to_string(),
            vec![
                EList, Expr, Expr, Expr, Input, EList, Input, EList, Input, EList, EList, Expr, Int,
            ],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a NodeId>,
        list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a NodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                head_dim: extract_expr(egraph, children[1], expr_cache).unwrap(),
                cur_seq: extract_expr(egraph, children[2], expr_cache).unwrap(),
                kv_row_stride: extract_expr(egraph, children[3], expr_cache).unwrap(),
                q_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                k_stride: extract_expr_list(egraph, children[7], list_cache, expr_cache).unwrap(),
                v_stride: extract_expr_list(egraph, children[9], list_cache, expr_cache).unwrap(),
                o_stride: extract_expr_list(egraph, children[10], list_cache, expr_cache).unwrap(),
                prev_seq: extract_expr(egraph, children[11], expr_cache).unwrap(),
                current_layer: egraph.enodes[children[12]].0.parse::<usize>().unwrap(),
            })),
            vec![children[4], children[6], children[8]],
        )
    }
}

impl BlockOp for GQAAttention {
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
        q_pos_stride[self.range.len() - 1] = 'z'.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 'z'.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 'z'.into();
        CStructBuilder::new()
            .int(expressions[&self.head_dim])
            .int(expressions[&self.cur_seq])
            .int(expressions[&self.kv_row_stride])
            .int(expressions[&flatten_strides(&self.range, &self.q_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.k_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.v_stride)])
            .int(expressions[&flatten_strides(&self.range, &self.o_stride)])
            .ptr_mut_f32(k_cache.device_ptr(stream).0 as *mut f32)
            .ptr_mut_f32(v_cache.device_ptr(stream).0 as *mut f32)
            .int(expressions[&self.prev_seq])
            .int(expressions[&flatten_strides(&self.range, &q_pos_stride)])
            .int(expressions[&flatten_strides(&self.range, &group_pos_stride)])
            .int(expressions[&flatten_strides(&self.range, &head_pos_stride)])
            .finish_struct()
    }

    fn expressions(&self) -> Vec<Expression> {
        let mut q_pos_stride = vec![0.into(); self.range.len()];
        q_pos_stride[self.range.len() - 1] = 'z'.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 'z'.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 'z'.into();
        vec![
            flatten_strides(&self.range, &self.q_stride),
            flatten_strides(&self.range, &self.k_stride),
            flatten_strides(&self.range, &self.v_stride),
            flatten_strides(&self.range, &self.o_stride),
            self.head_dim,
            self.cur_seq,
            self.kv_row_stride,
            self.prev_seq,
            flatten_strides(&self.range, &q_pos_stride),
            flatten_strides(&self.range, &group_pos_stride),
            flatten_strides(&self.range, &head_pos_stride),
        ]
    }
}
