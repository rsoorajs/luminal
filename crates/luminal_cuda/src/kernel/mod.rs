#![allow(unused)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::*;

pub mod ops;
pub use ops::Ops;

pub trait KernelOp: luminal::op::EgglogOp {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    );

    fn output_size(&self) -> Expression;
}

luminal::impl_into_ops!(KernelOp);
