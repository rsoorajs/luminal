use luminal::prelude::*;
#[cfg(feature = "cuda")]
use luminal_cuda_lite::runtime::CudaRuntime;
use rustc_hash::FxHashMap;

/// Enum wrapper for runtime backends allowing runtime selection.
pub enum RuntimeBackend {
    Native(NativeRuntime),
    #[cfg(feature = "cuda")]
    Cuda(Box<CudaRuntime>),
}

impl RuntimeBackend {
    /// Set input data for a tensor node.
    pub fn set_data(&mut self, node: NodeIndex, data: Vec<f32>) {
        match self {
            RuntimeBackend::Native(rt) => rt.set_data(node, data),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.set_data(node, data),
        }
    }

    /// Execute the compiled graph.
    pub fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        match self {
            RuntimeBackend::Native(rt) => rt.execute(dyn_map),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.execute(dyn_map),
        }
    }

    /// Get output data from a tensor node.
    pub fn get_f32(&self, node: NodeIndex) -> Vec<f32> {
        match self {
            RuntimeBackend::Native(rt) => rt.get_f32(node).to_vec(),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.get_f32(node),
        }
    }

    /// Get the name of the active backend.
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeBackend::Native(_) => "native",
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(_) => "cuda",
        }
    }
}
