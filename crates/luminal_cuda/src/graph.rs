//! CUDA Graph API wrappers for efficient kernel execution.
//!
//! This module provides low-level wrappers around CUDA Graph APIs for:
//! - Creating CUDA graphs with explicit node dependencies
//! - Adding kernel nodes to graphs
//! - Instantiating and launching graphs
//! - Surgically updating kernel node parameters when dynamic dimensions change

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::{
    sys::{self, CUevent, CUfunction, CUgraph, CUgraphExec, CUgraphNode},
    CudaContext, CudaFunction, CudaStream, DriverError,
};

/// A CUDA graph that can be modified and instantiated.
///
/// Unlike cudarc's CudaGraph which uses stream capture, this struct provides
/// direct access to graph handles for surgical updates.
pub struct CudaGraphHandle {
    pub(crate) cu_graph: CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphHandle {
    /// Creates a new empty CUDA graph.
    pub fn new(ctx: Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut graph = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphCreate(graph.as_mut_ptr(), 0).result()?;
            Ok(Self {
                cu_graph: graph.assume_init(),
                ctx,
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
    /// * `kernel_params` - Pointer to array of kernel parameter pointers
    ///
    /// # Safety
    /// The kernel_params must remain valid for the lifetime of the graph.
    /// The caller must ensure the params struct is properly initialized.
    pub unsafe fn add_kernel_node(
        &mut self,
        dependencies: &[CUgraphNode],
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<CUgraphNode, DriverError> {
        self.ctx.bind_to_thread()?;

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
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Adds an event record node to the graph for timing.
    ///
    /// # Arguments
    /// * `dependencies` - Graph nodes that must complete before this event is recorded
    /// * `event` - The CUDA event to record
    ///
    /// Returns the graph node handle.
    pub fn add_event_record_node(
        &mut self,
        dependencies: &[CUgraphNode],
        event: CUevent,
    ) -> Result<CUgraphNode, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut node = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphAddEventRecordNode(
                node.as_mut_ptr(),
                self.cu_graph,
                dependencies.as_ptr(),
                dependencies.len(),
                event,
            )
            .result()?;
            Ok(node.assume_init())
        }
    }

    /// Instantiates the graph, creating an executable graph.
    pub fn instantiate(&self) -> Result<CudaGraphExecHandle, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut graph_exec = MaybeUninit::uninit();
        unsafe {
            sys::cuGraphInstantiateWithFlags(graph_exec.as_mut_ptr(), self.cu_graph, 0).result()?;
            Ok(CudaGraphExecHandle {
                cu_graph_exec: graph_exec.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
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
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaGraphExecHandle {
    /// Launches the graph on the given stream.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
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
        kernel_params: *mut *mut c_void,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;

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
        let _ = self.ctx.bind_to_thread();
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
        // CudaFunction in cudarc 0.17.8 is declared as:
        //   pub(crate) cu_function: sys::CUfunction,
        //   pub(crate) module: Arc<CudaModule>,
        //
        // However, Rust reorders struct fields for optimization (no #[repr(C)]).
        // Empirically tested: the cu_function is at offset 8, not offset 0.
        // The Arc<CudaModule> (pointer-sized) comes first.

        // Verify our size assumption
        let expected_size = std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>();
        let actual_size = std::mem::size_of::<CudaFunction>();
        debug_assert_eq!(
            actual_size, expected_size,
            "CudaFunction layout changed - expected {} bytes, got {}",
            expected_size, actual_size
        );

        // Read cu_function from offset 8 (after the Arc<CudaModule>)
        unsafe {
            let ptr = (self as *const CudaFunction as *const u8).add(8) as *const CUfunction;
            std::ptr::read(ptr)
        }
    }
}

/// Stored kernel parameters that persist for the lifetime of a CUDA graph.
///
/// CUDA graphs store pointers to kernel parameters, so we need to keep
/// the parameter values alive in stable memory locations.
pub struct KernelParams {
    /// The actual parameter values (device pointers as u64)
    values: Box<[u64]>,
    /// Pointers to each value (what CUDA needs)
    ptrs: Box<[*mut c_void]>,
}

impl KernelParams {
    /// Creates new kernel params from output pointer and input pointers.
    pub fn new(output_ptr: u64, input_ptrs: &[u64]) -> Self {
        // Allocate values: output first, then inputs
        let mut values: Vec<u64> = Vec::with_capacity(1 + input_ptrs.len());
        values.push(output_ptr);
        values.extend_from_slice(input_ptrs);
        let values = values.into_boxed_slice();

        // Create pointers to each value
        let ptrs: Vec<*mut c_void> = values
            .iter()
            .map(|v| v as *const u64 as *mut c_void)
            .collect();
        let ptrs = ptrs.into_boxed_slice();

        Self { values, ptrs }
    }

    /// Returns the pointer array that CUDA expects.
    pub fn as_cuda_params(&mut self) -> *mut *mut c_void {
        self.ptrs.as_mut_ptr()
    }

    /// Updates the output pointer (index 0).
    pub fn update_output(&mut self, ptr: u64) {
        self.values[0] = ptr;
    }

    /// Updates an input pointer (1-indexed since output is 0).
    pub fn update_input(&mut self, index: usize, ptr: u64) {
        self.values[1 + index] = ptr;
    }
}

/// Timing data for a single kernel in a CUDA graph.
#[derive(Clone, Debug)]
pub struct CudaGraphKernelTiming {
    /// Name of the kernel
    pub kernel_name: &'static str,
    /// Start time in nanoseconds (relative to graph start)
    pub start_ns: u64,
    /// End time in nanoseconds (relative to graph start)
    pub end_ns: u64,
}

/// Timing data for a CUDA graph execution.
#[derive(Clone, Debug)]
pub struct CudaGraphTiming {
    /// Per-kernel timing data
    pub kernel_timings: Vec<CudaGraphKernelTiming>,
    /// Host timestamp when the graph started (used to correlate with perfetto)
    pub host_start_ns: u64,
}

/// Helper to create a CUDA event for timing.
pub fn create_cuda_event(ctx: &Arc<CudaContext>) -> Result<CUevent, DriverError> {
    ctx.bind_to_thread()?;
    let mut event = MaybeUninit::uninit();
    unsafe {
        sys::cuEventCreate(
            event.as_mut_ptr(),
            sys::CUevent_flags::CU_EVENT_DEFAULT as u32,
        )
        .result()?;
        Ok(event.assume_init())
    }
}

/// Destroy a CUDA event.
pub fn destroy_cuda_event(ctx: &Arc<CudaContext>, event: CUevent) {
    if !event.is_null() {
        let _ = ctx.bind_to_thread();
        unsafe {
            let _ = sys::cuEventDestroy_v2(event);
        }
    }
}

/// Get elapsed time between two events in milliseconds.
pub fn event_elapsed_ms(ctx: &Arc<CudaContext>, start: CUevent, end: CUevent) -> Result<f32, DriverError> {
    ctx.bind_to_thread()?;
    let mut ms: f32 = 0.0;
    unsafe {
        sys::cuEventElapsedTime(&mut ms, start, end).result()?;
    }
    Ok(ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        // Skip if no CUDA device
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        let graph = CudaGraphHandle::new(ctx);
        assert!(graph.is_ok());
    }

    #[test]
    fn test_kernel_params() {
        let output = 0x1000u64;
        let inputs = vec![0x2000u64, 0x3000u64];
        let mut params = KernelParams::new(output, &inputs);

        // Verify the pointer array is valid
        let cuda_params = params.as_cuda_params();
        assert!(!cuda_params.is_null());

        // Update and verify
        params.update_output(0x4000);
        params.update_input(0, 0x5000);
    }

    #[test]
    fn test_cuda_function_size() {
        // Verify our size assumption for CudaFunction
        // This test will fail if cudarc changes CudaFunction layout
        let expected_size = std::mem::size_of::<CUfunction>() + std::mem::size_of::<usize>();
        let actual_size = std::mem::size_of::<CudaFunction>();
        assert_eq!(
            actual_size, expected_size,
            "CudaFunction size changed! Expected {}, got {}. Update CudaFunctionExt.",
            expected_size, actual_size
        );
    }

    #[test]
    fn test_raw_function_extraction() {
        // Skip if no CUDA device
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Simple kernel that does nothing
        let kernel_src = r#"
            extern "C" __global__ void test_kernel(float* out) {
                out[0] = 1.0f;
            }
        "#;

        // Compile and load the kernel
        let ptx = match cudarc::nvrtc::compile_ptx(kernel_src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to compile PTX: {:?}", e);
                return;
            }
        };

        let module = ctx.load_module(ptx).expect("Failed to load module");
        let func = module.load_function("test_kernel").expect("Failed to load function");

        // Test that we can extract the raw function handle
        let cu_func = unsafe { func.raw_function() };
        assert!(!cu_func.is_null(), "Raw function handle should not be null");

        // Verify the handle is valid by querying an attribute
        let mut max_threads: i32 = 0;
        let result = unsafe {
            sys::cuFuncGetAttribute(
                &mut max_threads,
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                cu_func
            )
        };
        assert!(
            result == sys::cudaError_enum::CUDA_SUCCESS,
            "CUfunction handle should be valid, got {:?}",
            result
        );
        assert!(max_threads > 0, "max_threads should be positive");
    }

    #[test]
    fn test_graph_with_kernel() {
        use cudarc::driver::{CudaSlice, DevicePtr};

        // Skip if no CUDA device
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };

        // Simple kernel that adds 1.0 to the input
        let kernel_src = r#"
            extern "C" __global__ void test_kernel(float* out, float* in1) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx == 0) {
                    out[0] = in1[0] + 1.0f;
                }
            }
        "#;

        // Compile and load the kernel
        let ptx = match cudarc::nvrtc::compile_ptx(kernel_src) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to compile PTX: {:?}", e);
                return;
            }
        };

        let module = ctx.load_module(ptx).expect("Failed to load module");
        let func = module.load_function("test_kernel").expect("Failed to load function");

        // Allocate buffers
        let stream = ctx.default_stream();
        let output: CudaSlice<f32> = unsafe { stream.alloc(1) }.expect("Failed to alloc output");
        let mut input: CudaSlice<f32> = unsafe { stream.alloc(1) }.expect("Failed to alloc input");

        // Initialize input
        stream.memcpy_htod(&[5.0f32], &mut input).expect("Failed to copy input");

        // Get raw function handle
        let cu_func = unsafe { func.raw_function() };
        assert!(!cu_func.is_null(), "CUfunction should not be null");

        // Create graph
        let mut graph = CudaGraphHandle::new(ctx.clone()).expect("Failed to create graph");

        // Get device pointers
        let output_ptr = output.device_ptr(&stream).0;
        let input_ptr = input.device_ptr(&stream).0;

        // Create kernel params
        let mut params = KernelParams::new(output_ptr, &[input_ptr]);

        // Add kernel node to graph
        let _node = unsafe {
            graph.add_kernel_node(
                &[],  // No dependencies
                cu_func,
                (1, 1, 1),  // 1 block
                (1, 1, 1),  // 1 thread
                0,          // No shared memory
                params.as_cuda_params(),
            )
        }.expect("Failed to add kernel node");

        // Instantiate graph
        let exec = graph.instantiate().expect("Failed to instantiate graph");

        // Launch graph
        exec.launch(&stream).expect("Failed to launch graph");

        // Sync and read output
        stream.synchronize().expect("Failed to sync");

        let mut result = [0.0f32];
        stream.memcpy_dtoh(&output, &mut result).expect("Failed to read result");

        assert_eq!(result[0], 6.0f32, "Expected 5.0 + 1.0 = 6.0, got {}", result[0]);
    }
}
