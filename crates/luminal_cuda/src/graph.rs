//! CUDA Graph API wrappers for efficient kernel execution.
//!
//! This module provides low-level wrappers around CUDA Graph APIs for:
//! - Creating CUDA graphs with explicit node dependencies
//! - Adding kernel nodes to graphs
//! - Instantiating and launching graphs
//! - Surgically updating kernel node parameters when dynamic dimensions change

use std::mem::MaybeUninit;

use cudarc::driver::{
    sys::{self, CUfunction, CUgraph, CUgraphExec, CUgraphNode},
    CudaFunction, CudaStream, DriverError,
};

/// A CUDA graph that can be modified and instantiated.
///
/// Unlike cudarc's CudaGraph which uses stream capture, this struct provides
/// direct access to graph handles for surgical updates.
pub struct CudaGraphHandle {
    pub(crate) cu_graph: CUgraph,
}

impl CudaGraphHandle {
    /// Creates a new empty CUDA graph.
    pub fn new() -> Result<Self, DriverError> {
        let mut graph = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphCreate(graph.as_mut_ptr(), 0).result()?;
            Ok(Self {
                cu_graph: graph.assume_init(),
            })
        }
    }

    /// Adds a kernel node to the graph.
    ///
    /// # Arguments
    /// * `dependencies` - Graph nodes that must complete before this kernel runs
    /// * `func` - The CUDA function to execute
    /// * `grid_dim` - Grid dimensions (blocks)
    /// * `block_dim` - Block dimensions (threads per block)
    /// * `shared_mem_bytes` - Dynamic shared memory size
    /// * `kernel_params` - Pointers to kernel parameters
    ///
    /// # Safety
    /// The kernel_params must remain valid for the lifetime of the graph.
    pub unsafe fn add_kernel_node(
        &mut self,
        dependencies: &[CUgraphNode],
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut std::ffi::c_void,
    ) -> Result<CUgraphNode, DriverError> {
        // CUDA_KERNEL_NODE_PARAMS (v2) requires additional kern and ctx fields
        // which can be null/default for basic usage
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(), // Not using CUkernel-based launch
            ctx: std::ptr::null_mut(),  // Use default context
        };

        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddKernelNode_v2(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                &params,
            )
        }
        .result()?;
        Ok(unsafe { node.assume_init() })
    }

    /// Instantiates the graph, creating an executable graph.
    pub fn instantiate(&self) -> Result<CudaGraphExecHandle, DriverError> {
        let mut graph_exec = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), self.cu_graph, 0).result()?;
            Ok(CudaGraphExecHandle {
                cu_graph_exec: graph_exec.assume_init(),
            })
        }
    }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        if !self.cu_graph.is_null() {
            unsafe {
                let _ = sys::cuGraphDestroy(self.cu_graph);
            }
        }
    }
}

/// An instantiated CUDA graph that can be launched and updated.
pub struct CudaGraphExecHandle {
    pub(crate) cu_graph_exec: CUgraphExec,
}

impl CudaGraphExecHandle {
    /// Launches the graph on the given stream.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.context().bind_to_thread()?;
        unsafe { sys::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    /// Updates a kernel node's parameters in the instantiated graph.
    ///
    /// This allows "surgical" updates to kernel launch configurations without
    /// rebuilding the entire graph. Use this when dynamic dimensions change
    /// but buffer pointers remain the same.
    ///
    /// # Safety
    /// The kernel_params must remain valid for the lifetime of the graph.
    pub unsafe fn update_kernel_node(
        &mut self,
        node: CUgraphNode,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut std::ffi::c_void,
    ) -> Result<(), DriverError> {
        let params = sys::CUDA_KERNEL_NODE_PARAMS {
            func,
            gridDimX: grid_dim.0,
            gridDimY: grid_dim.1,
            gridDimZ: grid_dim.2,
            blockDimX: block_dim.0,
            blockDimY: block_dim.1,
            blockDimZ: block_dim.2,
            sharedMemBytes: shared_mem_bytes,
            kernelParams: kernel_params,
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };

        unsafe { sys::cuGraphExecKernelNodeSetParams_v2(self.cu_graph_exec, node, &params) }.result()
    }
}

impl Drop for CudaGraphExecHandle {
    fn drop(&mut self) {
        if !self.cu_graph_exec.is_null() {
            unsafe {
                let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
        }
    }
}

/// Extension trait to get the raw CUfunction handle from CudaFunction.
///
/// This is needed because cudarc makes the cu_function field pub(crate).
pub trait CudaFunctionExt {
    /// Gets the raw CUDA function handle.
    ///
    /// # Safety
    /// This exposes the raw handle which must be used carefully.
    unsafe fn raw_function(&self) -> CUfunction;
}

impl CudaFunctionExt for CudaFunction {
    unsafe fn raw_function(&self) -> CUfunction {
        // CudaFunction is repr(Rust) but its first field is cu_function: CUfunction
        // We use pointer casting to access it since it's pub(crate) in cudarc
        let ptr = self as *const CudaFunction as *const CUfunction;
        unsafe { *ptr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        // Skip if no CUDA device
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }

        let graph = CudaGraphHandle::new();
        assert!(graph.is_ok());
    }
}
