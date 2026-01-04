use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::prelude::*;
use as_any::{AsAny, Downcast};
use rustc_hash::FxHashMap;

/// Supported dtypes
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DType {
    /// 32-bit float (8e23m)
    #[default]
    F32,
    /// 16-bit float (5e10m)
    F16,
    /// 16-bit float (8e7m)
    Bf16,
    /// 32-bit integer
    Int,
}

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

pub trait EgglogOp: Debug {
    fn term(&self) -> (String, Vec<OpParam>);
    fn rewrites(&self) -> Vec<String> {
        vec![]
    }
    fn cleanup(&self) -> bool;
    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        panic!("Extraction not implemented for {self:?}!");
    }
}

crate::impl_into_ops!(EgglogOp);

#[derive(Debug, Clone)]
pub struct LLIROp(Arc<Box<dyn DialectOpTrait>>);

impl LLIROp {
    /// Store an op in a generic LLIR op. **Make sure to erase type into your dialect trait!** i.e. `as Box<dyn BlockOp>`
    pub fn new<T: ?Sized>(op: Box<T>) -> Self
    where
        Box<T>: Debug + 'static,
    {
        assert!(
            op.type_name().contains("dyn")
                || op.type_name().contains("Input")
                || op.type_name().contains("Output"),
            "op types must be erased into dialect traits for dialect casting to work!"
        );
        Self(Arc::new(Box::new(DialectOp::new(op))))
    }

    pub fn to_dialect<T: ?Sized + 'static>(&self) -> Option<&Arc<Box<T>>> {
        (**self.0).downcast_ref::<DialectOp<Box<T>>>().map(|i| &i.0)
    }

    pub fn to_op<T: 'static>(&self) -> Option<&T> {
        (**self.0)
            .downcast_ref::<DialectOp<Box<T>>>()
            .map(|d| &**d.0)
    }
}

impl std::fmt::Display for LLIROp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct DialectOp<T>(pub Arc<T>);

impl<T> DialectOp<T> {
    pub fn new(op: T) -> Self {
        Self(Arc::new(op))
    }
}

impl<T: Debug + 'static> DialectOpTrait for DialectOp<T> {}

pub trait DialectOpTrait: AsAny + Debug {}

pub enum OpParam {
    EList,
    Expr,
    Input,
    Int,
    Float,
    Str,
    Dty,
}

impl Debug for OpParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpParam::EList => write!(f, "EList"),
            OpParam::Expr => write!(f, "Expression"),
            OpParam::Input => write!(f, "IR"),
            OpParam::Int => write!(f, "i64"),
            OpParam::Str => write!(f, "String"),
            OpParam::Dty => write!(f, "DType",),
            OpParam::Float => write!(f, "f64"),
        }
    }
}

pub fn flatten_z_strides(range: &[Expression], strides: &[Expression]) -> Expression {
    assert_eq!(range.len(), strides.len());
    let mut current_elem_size = Expression::from(1);
    let mut flat_stride = Expression::from(0);
    for (dim, (range, stride)) in range.iter().zip(strides).enumerate().rev() {
        let div = Expression::from('z') / current_elem_size;
        let m = if dim > 0 { div % range } else { div };
        flat_stride += stride.substitute('z', m);
        current_elem_size *= range;
    }
    flat_stride.simplify()
}

pub fn flatten_mul_strides(range: &[Expression], strides: &[Expression]) -> Expression {
    assert_eq!(range.len(), strides.len());
    let mut current_elem_size = Expression::from(1);
    let mut flat_stride = Expression::from(0);
    for (dim, (range, stride)) in range.iter().zip(strides).enumerate().rev() {
        let div = Expression::from('z') / current_elem_size;
        let m = if dim > 0 { div % range } else { div };
        flat_stride += m * stride;
        current_elem_size *= range;
    }
    flat_stride.simplify()
}

pub fn flatten_z_strides_mask(range: &[Expression], strides: &[Expression]) -> Expression {
    assert_eq!(range.len(), strides.len());
    let mut current_elem_size = Expression::from(1);
    let mut flat_stride = Expression::from(1);
    for (dim, (range, stride)) in range.iter().zip(strides).enumerate().rev() {
        let div = Expression::from('z') / current_elem_size;
        let m = if dim > 0 { div % range } else { div };
        flat_stride *= stride.substitute('z', m);
        current_elem_size *= range;
    }
    flat_stride.simplify()
}
