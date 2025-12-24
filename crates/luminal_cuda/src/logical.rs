use std::fmt::Debug;

use luminal::utils::{
    EgglogOp,
    OpParam::{self, *},
};

pub type Ops = (Exp, Sigmoid, CubeMul, TileSum);

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
                ; get mul
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
                (= ?dt (dtype ?a))
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
                (let ?cm (CubeMul ?new_shape ?shape ?a ?new_a_stride ?a_m_width ?a_n_width ?a_k_width ?b ?new_b_stride ?b_m_width ?b_n_width ?b_k_width ?new_out_stride ?out_m_width ?out_n_width ?out_k_width))
                (union ?sa ?cm)
                (set (dtype ?cm) (F32))
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
                (= (F32) (dtype ?a))
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
                (let ?ts (TileSum ?new_shape ?shape ?iters ?a ?new_a_stride ?a_m_stride ?a_n_stride ?a_k_stride ?new_out_stride ?out_m_stride ?out_n_stride))
                (union ?sa ?ts)
                (set (dtype ?ts) (F32))
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
        vec!["(rule
            (
                (= ?exp_const (Constant 1.442695))
                (= ?mul (Mul ?shape ?x ?x_stride ?exp_const ?const_stride ?intermediate_stride))
                (= ?exp2 (Exp2 ?shape ?mul ?intermediate_stride ?out_stride))
                (= ?dt (dtype ?x))
            )
            (
                (let ?exp (Exp ?shape ?x ?x_stride ?out_stride))
                (union ?exp2 ?exp)
                (set (dtype ?exp) ?dt)
            )
        )"
        .to_string()]
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
                (= ?neg_input (Mul ?input_range ?input ?input_stride (Constant -1.0) ?const_stride ?intermediate_stride))
                (= ?exp (Exp ?input_range ?neg_input ?intermediate_stride ?exp_stride))
                (= ?plus_one (Add ?input_range ?exp ?exp_stride (Constant 1.0) ?const_stride ?plus_one_stride))
                (= ?sig_out (Recip ?input_range ?plus_one ?plus_one_stride ?out_stride))
                (= ?dt (dtype ?input))
            )
            (
                (let ?sig (Sigmoid ?input_range ?input ?input_stride ?out_stride))
                (union ?sig_out ?sig)
                (set (dtype ?sig) ?dt)
            )
            :name \"sigmoid\"
        )".to_string()]
    }
}
