#![allow(unused)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
use luminal::{shape::Expression, utils::EgglogOp};
use rustc_hash::FxHashMap;

pub mod ops;
pub use ops::Ops;

pub trait KernelOp: EgglogOp {
    fn compile(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    );

    fn output_size(&self) -> Expression;
}

luminal::impl_into_ops!(KernelOp);
