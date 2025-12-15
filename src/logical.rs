use std::fmt::Debug;

use crate::utils::{
    EgglogOp,
    OpParam::{self, *},
};

pub type Ops = (
    GMEM,
    Constant,
    Exp2,
    Log2,
    Sin,
    Recip,
    Sqrt,
    Add,
    Mul,
    Mod,
    LessThan,
    Sum,
    Max,
    Exp,
    Sigmoid,
    CubeMul,
    TileSum,
    Softmax,
);

#[allow(unused)]
#[derive(Default, Debug, Clone)]
pub struct GMEM {
    pub node: usize,
    pub label: String,
}

impl EgglogOp for GMEM {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("GMEM".to_string(), vec![Int, Str])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

#[derive(Default, Debug, Clone)]
pub struct Constant;

impl EgglogOp for Constant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Constant".to_string(), vec![Expr])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Recip;
impl EgglogOp for Recip {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Recip".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Sqrt;
impl EgglogOp for Sqrt {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sqrt".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Exp2;
impl EgglogOp for Exp2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Exp2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Log2;
impl EgglogOp for Log2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Log2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Sin;
impl EgglogOp for Sin {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sin".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Add;
impl EgglogOp for Add {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Add".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Mul;
impl EgglogOp for Mul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Mul".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Mod;
impl EgglogOp for Mod {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Mod".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct LessThan;
impl EgglogOp for LessThan {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "LessThan".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Sum;
impl EgglogOp for Sum {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Sum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct Max;
impl EgglogOp for Max {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Max".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
}

#[derive(Default, Debug, Clone)]
pub struct CubeMul;
impl EgglogOp for CubeMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "CubeMul".to_string(),
            vec![
                EList, EList, Input, EList, Expr, Expr, Expr, Input, EList, Expr, Expr, Expr,
                EList, Expr, Expr, Expr,
            ],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                ; get  mul
                (= ?sa (Mul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                (= ?shape_last (nth_from_end ?shape 0))
                (= ?shape_second_to_last (nth_from_end ?shape 1))
                (= ?shape_third_to_last (nth_from_end ?shape 2))
                (!= ?shape_last (MNum 0))
                (!= ?shape_second_to_last (MNum 0))
                (!= ?shape_third_to_last (MNum 0))
                ; get m, n, and k strides for A, B, and outputs
                (= ?a_n_width (nth_from_end ?a_stride 1))
                (= ?b_n_width (nth_from_end ?b_stride 1))
                (= ?out_n_width (nth_from_end ?out_stride 1))
                (= ?a_m_width (nth_from_end ?a_stride 2))
                (= ?b_m_width (nth_from_end ?b_stride 2))
                (= ?out_m_width (nth_from_end ?out_stride 2))
                (= ?a_k_width (nth_from_end ?a_stride 0))
                (= ?b_k_width (nth_from_end ?b_stride 0))
                (= ?out_k_width (nth_from_end ?out_stride 0))
            )
            (
                ; divide the last 3 dimensions by 32
                (let ?new_shape
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd
                                ?shape
                            (MCeilDiv ?shape_last (MNum 32)) 0)
                        (MCeilDiv ?shape_second_to_last (MNum 32)) 1)
                    (MCeilDiv ?shape_third_to_last (MNum 32)) 2)
                )
                ; multiply last 3 strides by 32
                (let ?new_a_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd
                                ?a_stride
                            (MMul (MIter) (MNum 32)) 0)
                        (MMul ?a_n_width (MNum 32)) 1)
                    (MMul ?a_m_width (MNum 32)) 2)
                )
                (let ?new_b_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd
                                ?b_stride
                            (MMul (MIter) (MNum 32)) 0)
                        (MMul ?b_n_width (MNum 32)) 1)
                    (MMul ?b_m_width (MNum 32)) 2)
                )
                (let ?new_out_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd
                                ?out_stride
                            (MMul (MIter) (MNum 32)) 0)
                        (MMul ?out_n_width (MNum 32)) 1)
                    (MMul ?out_m_width (MNum 32)) 2)
                )
                (union ?sa (CubeMul ?new_shape ?shape ?a ?new_a_stride ?a_m_width ?a_n_width ?a_k_width ?b ?new_b_stride ?b_m_width ?b_n_width ?b_k_width ?new_out_stride ?out_m_width ?out_n_width ?out_k_width))
            )
        )".to_string()]
    }
}

#[derive(Default, Debug, Clone)]
pub struct TileSum;
impl EgglogOp for TileSum {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "TileSum".to_string(),
            vec![
                EList, EList, Expr, Input, EList, Expr, Expr, Expr, EList, Expr, Expr,
            ],
        )
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                ; get  sum
                (= ?sa (Sum ?shape ?iters ?a ?a_stride ?a_k_stride ?out_stride))
                (= ?shape_n (nth_from_end ?shape 0)) (!= ?shape_n (MNum 0))
                (= ?shape_m (nth_from_end ?shape 1)) (!= ?shape_m (MNum 0))
                ; get m and n strides for A
                (= ?a_m_stride (nth_from_end ?a_stride 1))
                (= ?a_n_stride (nth_from_end ?a_stride 0))
                ; get m and n strides for out
                (= ?out_m_stride (nth_from_end ?out_stride 1))
                (= ?out_n_stride (nth_from_end ?out_stride 0))
            )
            (
                ; divide second to last and last dimensions by 32
                (let ?new_shape
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            ?shape
                        (MCeilDiv ?shape_n (MNum 32)) 0)
                    (MCeilDiv ?shape_m (MNum 32)) 1)
                )
                ; multiply second to last and last strides by 32
                (let ?new_a_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            ?a_stride
                        (MMul ?a_n_stride (MNum 32)) 0)
                    (MMul ?a_m_stride (MNum 32)) 1)
                )
                (let ?new_out_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd
                            ?out_stride
                        (MMul ?out_n_stride (MNum 32)) 0)
                    (MMul ?out_m_stride (MNum 32)) 1)
                )
                (union
                    ?sa
                    (TileSum ?new_shape ?shape ?iters ?a ?new_a_stride ?a_m_stride ?a_n_stride ?a_k_stride ?new_out_stride ?out_m_stride ?out_n_stride)
                )
            )
        )".to_string()]
    }
}

#[derive(Debug, Default)]
pub struct Exp;
impl EgglogOp for Exp {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Exp".to_string(), vec![EList, Input, EList, EList])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
            (
                (= ?exp_const (Constant (MFloat 1.442695)))
                (= ?mul (Mul ?shape ?x ?x_stride ?exp_const ?const_stride ?intermediate_stride))
                (= ?exp2 (Exp2 ?shape ?mul ?intermediate_stride ?out_stride))
            )
            (
                (union ?exp2 (Exp ?shape ?x ?x_stride ?out_stride))
            )
        )"
            .to_string(),
        ]
    }
}

#[derive(Default, Debug, Clone)]
pub struct Sigmoid;
impl EgglogOp for Sigmoid {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sigmoid".to_string(), vec![EList, Input, EList, EList])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                (= ?neg_input (Mul ?input_range ?input ?input_stride (Constant (MNum -1)) ?const_stride ?intermediate_stride))
                (= ?exp (Exp ?input_range ?neg_input ?intermediate_stride ?exp_stride))
                (= ?plus_one (Add ?input_range ?exp ?exp_stride (Constant (MFloat 1.0)) ?const_stride ?plus_one_stride))
                (= ?sig_out (Recip ?input_range ?plus_one ?plus_one_stride ?out_stride))
            )
            (
                (union
                    ?sig_out
                    (Sigmoid ?input_range ?input ?input_stride ?out_stride)
                )
            )
        )".to_string()]
    }
}

#[derive(Default, Debug, Clone)]
pub struct Softmax;

impl EgglogOp for Softmax {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Softmax".to_string(),
            vec![EList, Expr, Input, EList, EList],
        )
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
          (
            ; rowwise max
            (= ?qk_max (Max ?batches ?row_width ?qk ?max_stride (MIter) ?max_stride_out))
            ; broadcast -max
            (= ?neg1 (MNum -1))
            (= ?qk_max_neg_2d (Mul ?full_range ?qk_max ?broadcast_stride (Constant ?neg1) ?zero_stride ?full_stride))
            (= (MNum 0) (nth_from_end ?broadcast_stride 0)) ; assert broadcasting
            (= ?qk_sub (Add ?full_range ?qk ?full_stride ?qk_max_neg_2d ?full_stride ?full_stride))
            ; exp
            (= ?qk_sub_exp (Exp ?full_range ?qk_sub ?full_stride ?full_stride))
            ; rowwise sum
            (= ?sum (Sum ?batches ?row_width ?qk_sub_exp ?max_stride (MIter) ?max_stride_out))
            ; 2D denom by broadcasting sum to [B,B], then recip producing [B,B]
            (= ?denom_2d (Recip ?full_range ?sum ?broadcast_stride ?full_stride))
            ; scores = exp / sum
            (= ?scores (Mul ?full_range ?qk_sub_exp ?full_stride ?denom_2d ?full_stride ?full_stride))
          )
          ((union ?scores (Softmax ?batches ?row_width ?qk ?max_stride ?max_stride)))
        )".to_string()]
    }
}
