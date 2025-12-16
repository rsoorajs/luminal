use std::{
    any::Any,
    borrow::BorrowMut,
    fmt::Debug,
    ptr::null,
    sync::{Arc, Mutex},
};

use crate::{
    prelude::*,
    utils::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
};

use dyn_clone::{DynClone, clone_trait_object};
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

/// A tensor with data. The data can be anything that implements the Data trait
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Box<dyn Data>,
}

impl Tensor {
    pub fn new<T: Data>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }
    pub fn downcast_ref<T: Data>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref()
    }
    pub fn downcast_mut<T: Data>(&mut self) -> Option<&mut T> {
        self.data.as_any_mut().downcast_mut()
    }
    pub fn is<T: Data>(&self) -> bool {
        self.data.as_any().is::<T>()
    }
}

/// Some sort of data, for instance a Vec<f32> on CPU, CudaSlice<f32> on Nvidia GPUs, or metal::Buffer for Apple GPUs
pub trait Data: Any + Debug + DynClone {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

clone_trait_object!(Data);

impl Data for Vec<f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Either an owned or borrowed tensor that gets consumed by ops
pub enum InputTensor<'a> {
    /// An owned tensor
    Owned(Tensor),
    /// A borrowed tensor
    Borrowed(&'a Tensor),
}

impl<'a> InputTensor<'a> {
    /// Borrow the tensor
    pub fn borrowed(&'a self) -> &'a Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t,
        }
    }

    /// Unwrap or clone the tensor, depending on if it's owned or not
    pub fn cloned(self) -> Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t.clone(),
        }
    }
}

/// The main operator trait.
///
/// Defines an operator that takes in a vector of input tensors and shapes and produces a vector of output tensors
pub trait Operator: Debug + as_any::AsAny {
    /// Process the input tensors and produce output tensors
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>;
    /// Implement custom functionality
    #[allow(unused)]
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        None
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String;
}

impl<T: Operator> Operator for Box<T> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self, key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self, inp)
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
        <T as Operator>::to_egglog(&self, inputs)
    }
}
impl<T: Operator> Operator for Arc<Mutex<T>> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self.lock().unwrap().borrow_mut(), key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self.lock().unwrap().borrow_mut(), inp)
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
        <T as Operator>::to_egglog(&self.lock().unwrap(), inputs)
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
        children: &Vec<&'a egraph_serialize::NodeId>,
        _: &mut FxHashMap<&'a egraph_serialize::NodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a egraph_serialize::NodeId, Expression>,
    ) -> (crate::utils::LLIROp, Vec<&'a egraph_serialize::NodeId>) {
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

impl Operator for GMEM {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }

    fn to_egglog(&self, _: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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

impl Operator for Constant {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor::new(vec![self.0])]
    }

    fn to_egglog(&self, _: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Iota {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }

    fn to_egglog(&self, _: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    Int,
}
impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Cast(pub DType);
impl Operator for Cast {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }

    fn to_egglog(&self, inp: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|(t, _)| t.cloned()).collect() // inefficient, but we don't care as this won't execute on the kernel
    }

    fn to_egglog(&self, _: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
        panic!("Cannot turn GraphBreak into egglog op!");
    }
}

// Unary Op (A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl Operator for Log2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).log2();
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Exp2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).exp2();
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Sin {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sin();
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Recip {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).recip();
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Sqrt {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sqrt();
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Add {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) + get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Mul {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) * get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Mod {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) % get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for LessThan {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = (get_index(lhs, &lexpr, &mut stack, i) < get_index(rhs, &rexpr, &mut stack, i))
                as i32 as f32;
        }
        vec![Tensor::new(out_data)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for Gather {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for SumReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![0.0; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    result[i * back_size + j] += get_index(input, &expr, &mut stack, orig_index);
                }
            }
        }
        vec![Tensor::new(result)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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
impl Operator for MaxReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![-f32::INFINITY; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    result[new_index] =
                        result[new_index].max(get_index(input, &expr, &mut stack, orig_index));
                }
            }
        }
        vec![Tensor::new(result)]
    }

    fn to_egglog(&self, inputs: &Vec<(NodeIndex, String, ShapeTracker)>) -> String {
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

fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap()
}

fn get_index(
    data: &[f32],
    (ind, val): &(Expression, Expression),
    stack: &mut Vec<i64>,
    index: usize,
) -> f32 {
    if val.exec_single_var_stack(index, stack) != 0 {
        let i = ind.exec_single_var_stack(index, stack);
        data[i]
    } else {
        0.0
    }
}
