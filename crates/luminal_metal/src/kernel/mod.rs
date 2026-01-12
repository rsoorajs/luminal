//! Metal kernel operations

mod ops;
pub use ops::*;

use luminal::prelude::*;
use luminal::utils::EgglogOp;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device};

/// Trait for Metal kernel operations
///
/// Similar to CUDA's `KernelOp` but for Metal backend.
/// Each operation provides a Metal shader and encoding logic.
pub trait MetalKernelOp: EgglogOp {
    /// Compile the Metal shader and return a pipeline state
    fn compile(&self, device: &Device) -> ComputePipelineState;

    /// Get the output buffer size
    fn output_size(&self) -> Expression;

    /// Encode the kernel into a command encoder
    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    );
}

luminal::impl_into_ops!(MetalKernelOp);
