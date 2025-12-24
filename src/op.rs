use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::{
    prelude::*,
    utils::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    serialized_egraph::SerializedEGraph,
};

use rustc_hash::FxHashMap;

pub type Ops = (
    GMEM,
    Constant,
    Iota,
    Exp2,
    Log2,
    Sin,
    Recip,
    Sqrt,
    Add,
    Mul,
    Mod,
    LessThan,
    Gather,
    SumReduce,
    MaxReduce,
);

/// The main HLIROp trait.
///
/// Defines an HLIROp that implements a logical operation.
pub trait HLIROp: Debug + as_any::AsAny {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String;
}

impl<T: HLIROp> HLIROp for Box<T> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        <T as HLIROp>::to_egglog(self, inputs)
    }
}
impl<T: HLIROp> HLIROp for Arc<Mutex<T>> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        <T as HLIROp>::to_egglog(&self.lock().unwrap(), inputs)
    }
}

#[allow(unused)]
#[derive(Default, Debug, Clone)]
pub struct GMEM {
    pub node: usize,
    pub label: String,
    pub dtype: DType,
}

impl EgglogOp for GMEM {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("GMEM".to_string(), vec![Int, Str, Dty])
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (GMEM ?node ?label ?dty)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (crate::utils::LLIROp, Vec<&'a ENodeId>) {
        let node = egraph.enodes[children[0]]
            .0
            .replace("\"", "")
            .parse::<usize>()
            .unwrap();
        let label = egraph.enodes[children[1]].0.replace("\"", "");
        (
            LLIROp::new::<GMEM>(Box::new(Self {
                node,
                label,
                dtype: extract_dtype(egraph, children[2]),
            })),
            vec![],
        )
    }
}

impl HLIROp for GMEM {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(GMEM {} \"{}\" ({:?}))", self.node, self.label, self.dtype)
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq, Default)]
pub struct Constant(pub f32);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant(",)?;
        self.0.fmt(f)?;
        write!(f, ")")
    }
}

impl HLIROp for Constant {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Constant {:.6})", self.0)
    }
}

impl EgglogOp for Constant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Constant".to_string(), vec![Float])
    }
    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Constant ?f)))
           ((set (dtype ?e) (F32)))
        )"
            .to_string(),
        ]
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Iota(pub Expression, pub Expression);
impl HLIROp for Iota {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Iota {} {})", self.0.to_egglog(), self.1.to_egglog())
    }
}
impl EgglogOp for Iota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Iota".to_string(), vec![Expr, Expr])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Iota ?expr ?range)))
           ((set (dtype ?e) (Int)))
        )"
            .to_string(),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    Bf16,
    Int,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Cast(pub DType);
impl HLIROp for Cast {
    fn to_egglog(&self, inp: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Cast {} {:?})", inp[0].1, self.0)
    }
}
impl EgglogOp for Cast {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Cast".to_string(), vec![Input, Expr])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Cast ?inp ?dty)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

/// Graph break for chunking search graphs
#[derive(Clone, PartialEq)]
pub struct GraphBreak;
impl Debug for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}

impl HLIROp for GraphBreak {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        panic!("Cannot turn GraphBreak into egglog op!");
    }
}

// Unary Op (A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl HLIROp for Log2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Log2 {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Log2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Log2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Log2 ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2;
impl HLIROp for Exp2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Exp2 {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Exp2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Exp2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Exp2 ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin;
impl HLIROp for Sin {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Sin {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Sin {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sin".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sin ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip;
impl HLIROp for Recip {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Recip {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Recip {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Recip".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Recip ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt;
impl HLIROp for Sqrt {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Sqrt {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Sqrt {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sqrt".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sqrt ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add;
impl HLIROp for Add {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Add {} {} {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            strides_to_egglog(&inputs[1].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

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
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Add ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul;
impl HLIROp for Mul {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Mul {} {} {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            strides_to_egglog(&inputs[1].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

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
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Mul ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod;
impl HLIROp for Mod {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Mod {} {} {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            strides_to_egglog(&inputs[1].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

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
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Mod ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan;
impl HLIROp for LessThan {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(LessThan {} {} {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            strides_to_egglog(&inputs[1].2.strides),
            strides_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

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
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (LessThan ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gather;
impl HLIROp for Gather {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Gather {} {} {} {} {})",
            shape_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            strides_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            strides_to_egglog(&inputs[1].2.strides),
        )
    }
}

impl EgglogOp for Gather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Gather".to_string(),
            vec![EList, Input, EList, Input, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Gather ?shape ?indexes ?a ?data ?b)) (= ?dty (dtype ?data)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SumReduce(pub usize);
impl HLIROp for SumReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        let mut non_reduced_shape = inputs[0].2;
        non_reduced_shape.remove_dim(self.0);
        let reduced_dim = inputs[0].2.dims[self.0];
        let reduced_stride = inputs[0].2.strides[self.0];
        let mut non_reduced_strides = inputs[0].2.strides;
        non_reduced_strides.remove(self.0);
        format!(
            "(Sum {} {} {} {} {} {})",
            shape_to_egglog(&non_reduced_shape.dims),
            reduced_dim.to_egglog(),
            inputs[0].1,
            strides_to_egglog(&non_reduced_strides),
            reduced_stride.to_egglog(),
            strides_to_egglog(&non_reduced_shape.contiguous().strides)
        )
    }
}

impl EgglogOp for SumReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Sum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sum ?shape ?iters ?inp ?a ?stride ?o)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaxReduce(pub usize);
impl HLIROp for MaxReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        let mut non_reduced_shape = inputs[0].2;
        non_reduced_shape.remove_dim(self.0);
        let reduced_dim = inputs[0].2.dims[self.0];
        let reduced_stride = inputs[0].2.strides[self.0];
        let mut non_reduced_strides = inputs[0].2.strides;
        non_reduced_strides.remove(self.0);
        format!(
            "(Max {} {} {} {} {} {})",
            shape_to_egglog(&non_reduced_shape.dims),
            reduced_dim.to_egglog(),
            inputs[0].1,
            strides_to_egglog(&non_reduced_strides),
            reduced_stride.to_egglog(),
            strides_to_egglog(&non_reduced_shape.contiguous().strides)
        )
    }
}

impl EgglogOp for MaxReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Max".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Max ?shape ?iters ?inp ?a ?stride ?o)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
}
