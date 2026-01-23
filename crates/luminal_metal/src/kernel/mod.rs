mod ops;
pub use ops::*;

use luminal::op::EgglogOp;
use luminal::prelude::*;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device};

/// Trait for Metal kernel operations with performance metrics
pub trait MetalKernelOp: EgglogOp {
    fn compile(&self, device: &Device) -> ComputePipelineState;

    fn output_size(&self) -> Expression;

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    );

    // ========================================================================
    // Performance Metrics for MBU/MFU Calculation
    // ========================================================================

    /// Number of bytes loaded from memory by this kernel
    ///
    /// Default implementation returns 0. Override for accurate MBU calculation.
    fn bytes_loaded(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    /// Number of bytes stored to memory by this kernel
    ///
    /// Default implementation returns 0. Override for accurate MBU calculation.
    fn bytes_stored(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    /// Number of floating-point operations performed by this kernel
    ///
    /// Default implementation returns 0. Override for accurate MFU calculation.
    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }
}

luminal::impl_into_ops!(MetalKernelOp);
