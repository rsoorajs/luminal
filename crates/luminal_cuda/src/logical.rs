use std::fmt::Debug;

use luminal::op::{
    EgglogOp,
    OpParam::{self, *},
};

pub type Ops = (Exp, Sigmoid);

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
