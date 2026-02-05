use crate::{
    block::{BlockOp, N_TIMING_SLOTS, SMEvent, record_block_op_timings},
    host::HostOp,
    kernel::{
        CudaFunctionExt, CudaGraphExecHandle, CudaGraphHandle, CudaGraphKernelTiming,
        CudaGraphTiming, KernelOp, create_cuda_event, destroy_cuda_event, event_elapsed_ms,
        record_cuda_graph_timings,
    },
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, PinnedHostSlice, sys::CUgraphNode,
};

use fixedbitset::FixedBitSet;
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        Directed, Direction,
        algo::{Cycle, toposort},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
    },
    *,
};

use luminal_tracing::PerfettoGuard;
use memmap2::MmapOptions;
use prost::Message;
use safetensors::SafeTensors;
use std::{collections::VecDeque, fmt::Debug, fs::File, mem::size_of, sync::Arc, time::Duration};
use tracing::{Level, enabled, field, span, trace};
use uuid::Uuid;

pub enum CudaInput {
    Buffer(CudaSlice<u8>),
    Ptr(u64),
}

/// Unified kernel node in a CUDA graph.
/// Works for all kernel types - regular kernels and megakernels alike.
#[allow(dead_code)]
struct KernelNode {
    llir_node: NodeIndex,
    kernel: CudaFunction,
    launch_grid: (Expression, Expression, Expression),
    launch_threadblock: (Expression, Expression, Expression),
    shared_mem: Expression,
    inputs: Vec<NodeIndex>,
    output: NodeIndex,
    kernel_name: &'static str,
    kernel_str: String,
    bytes_loaded: Expression,
    bytes_stored: Expression,
    flops: Expression,
    /// Allocated internal buffers (from KernelOp::allocate_internal_buffers)
    internal_bufs: Vec<CudaSlice<u8>>,
    /// Device constants (from compile())
    constants: FxHashMap<char, CudaSlice<u8>>,
    /// Reference to KernelOp for calling trait methods
    kernel_op: Arc<Box<dyn KernelOp>>,
    /// Graph node handle
    graph_node: Option<CUgraphNode>,
}

/// Unified kernel params that can hold any number of u64 values.
struct UnifiedKernelParams {
    values: Vec<u64>,
    ptrs: Vec<*mut std::ffi::c_void>,
}

impl UnifiedKernelParams {
    fn new(values: Vec<u64>) -> Self {
        let ptrs = values
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();
        Self { values, ptrs }
    }

    fn as_cuda_params(&mut self) -> *mut *mut std::ffi::c_void {
        // Rebuild pointers (in case struct was moved)
        for (i, v) in self.values.iter().enumerate() {
            self.ptrs[i] = v as *const u64 as *mut std::ffi::c_void;
        }
        self.ptrs.as_mut_ptr()
    }

    #[allow(dead_code)]
    fn update_value(&mut self, idx: usize, value: u64) {
        self.values[idx] = value;
    }
}

enum ExecutableKernel {
    CudaGraphExec {
        cuda_graph: Option<CudaGraphHandle>,
        cuda_graph_exec: Option<CudaGraphExecHandle>,
        node_to_graph_node: FxHashMap<NodeIndex, CUgraphNode>,
        /// Unified kernel params for each kernel
        kernel_params: Vec<UnifiedKernelParams>,
        /// Kernel nodes (unified - works for all kernel types)
        kernels: Vec<KernelNode>,
        /// Shared device buffer for dynamic dimensions (all kernels read from this)
        dyn_dims_buffer: Option<CudaSlice<i32>>,
        /// Order of dynamic dimensions in the buffer (sorted alphabetically)
        dyn_dims_order: Vec<char>,
        last_dyn_values: FxHashMap<char, usize>,
        last_buffer_ptrs: FxHashMap<NodeIndex, u64>,
        timing_events: Vec<cudarc::driver::sys::CUevent>,
        // Profiling
        total_bytes_loaded: Expression,
        total_bytes_stored: Expression,
        total_flops: Expression,
    },
    HostOp {
        stream: Arc<CudaStream>,
        inputs: Vec<NodeIndex>,
        output: NodeIndex,
        internal: Arc<Box<dyn HostOp>>,
    },
}

/// Statistics for a single kernel execution
#[derive(Debug, Clone)]
pub struct KernelStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
}

/// Statistics for a single block op execution (aggregated across all SMs)
#[derive(Debug, Clone)]
pub struct BlockOpStats {
    pub name: &'static str,
    pub execution_time_us: f64,
    pub bytes_loaded: usize,
    pub bytes_stored: usize,
    pub flops: usize,
    pub bandwidth_gbps: f64,
    pub tflops: f64,
    pub count: usize,
}
impl Drop for ExecutableKernel {
    fn drop(&mut self) {
        match self {
            ExecutableKernel::CudaGraphExec {
                dyn_dims_buffer,
                kernels,
                cuda_graph_exec,
                timing_events,
                ..
            } => {
                // Prevent Drop of dyn_dims_buffer (it's managed by the runtime's allocator)
                if let Some(buf) = dyn_dims_buffer.take() {
                    std::mem::forget(buf);
                }
                // Handle kernel device buffers
                for kernel in kernels.iter_mut() {
                    // Prevent Drop of constants
                    let m = std::mem::take(&mut kernel.constants);
                    for (_k, v) in m {
                        std::mem::forget(v);
                    }
                    // Prevent Drop of internal buffers (they're managed by the runtime's allocator)
                    for buf in kernel.internal_bufs.drain(..) {
                        std::mem::forget(buf);
                    }
                }
                // Destroy timing events using the context from CudaGraphExecHandle
                if let Some(exec) = cuda_graph_exec.as_ref() {
                    for event in timing_events.drain(..) {
                        destroy_cuda_event(&exec.ctx, event);
                    }
                }
                // CudaGraphHandle and CudaGraphExecHandle have proper Drop impls
            }
            ExecutableKernel::HostOp { .. } => {}
        }
    }
}

impl Debug for ExecutableKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CudaGraphExec { kernels, .. } => {
                write!(f, "CudaGraph (kernels={})", kernels.len())
            }
            Self::HostOp { internal, .. } => write!(f, "HostOp: ({internal:?})"),
        }
    }
}

/// Pending timing data using pinned memory for async copies
struct PendingTimingData {
    timing_buffer_idx: usize,
    start_time_idx: usize,
    span_id: Uuid,
}

pub struct CudaRuntime {
    pub hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    pub buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub llir_graph: luminal::graph::LLIRGraph,
    cuda_stream: Arc<CudaStream>,
    timing_stream: Arc<CudaStream>,
    exec_graph: StableGraph<ExecutableKernel, (), Directed>,
    node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) timings: Vec<Vec<(Vec<SMEvent>, u64, Uuid)>>,
    pub(crate) cuda_graph_timings: Vec<(CudaGraphTiming, Uuid)>,
    /// Pending timing data collected asynchronously using pinned memory
    pending_timing_data: Vec<PendingTimingData>,
    /// Pool of pre-allocated pinned timing buffers (one per slot)
    pinned_timing_pool: Vec<PinnedHostSlice<SMEvent>>,
    /// Pool of pre-allocated pinned start time buffers (one per slot)
    pinned_start_pool: Vec<PinnedHostSlice<u64>>,
    /// Index of next available slot in the pinned buffer pools
    next_pinned_slot: usize,
    last_dyn_map: FxHashMap<char, usize>,
    intermediate_buffer_dims: FxHashSet<char>,
    llir_to_hlir: FxHashMap<NodeIndex, NodeIndex>,
    hlir_to_llir: FxHashMap<NodeIndex, NodeIndex>,
    changed_hlir: FxHashSet<NodeIndex>,
    /// Cached buffer pointers to avoid repeated device_ptr() calls (keyed by llir_node)
    cached_buffer_ptrs: FxHashMap<NodeIndex, u64>,
    pub last_kernel_stats: Vec<KernelStats>,
    pub last_total_time_us: f64,
    kernel_cache: FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    num_sms: usize,
}

impl CudaRuntime {
    /// Creates a new CudaRuntime with default configuration:
    /// - Device 0
    /// - Blocking sync scheduling
    /// - Default stream
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = cudarc::driver::CudaContext::new(0)?;
        ctx.bind_to_thread()?;
        ctx.set_flags(cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)?;
        let stream = ctx.default_stream();

        Ok(Self::initialize(stream))
    }

    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>()
                && let Ok(tensor) = st.tensor(label)
            {
                match tensor.dtype() {
                    safetensors::Dtype::F32 => {
                        let bytes = tensor.data();
                        let f32s: &[f32] = bytemuck::cast_slice(bytes);
                        let dev = f32s.to_cuda_input(&self.cuda_stream);
                        self.hlir_buffers.insert(node, dev);
                        self.changed_hlir.insert(node);
                    }
                    dtype => unimplemented!("{dtype} loading not supported yet"),
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaInput) {
        self.hlir_buffers
            .insert(id.to_id(), data.to_cuda_input(&self.cuda_stream));
        self.changed_hlir.insert(id.to_id());
    }

    #[tracing::instrument(skip_all)]
    fn get_output_data(&self, id: impl ToId) -> Vec<u8> {
        let id = id.to_id();
        let output_id = self
            .llir_graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = self.llir_graph[*n].to_op::<Output>() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .expect("Cannot find output tensor!");
        let data_id = self
            .llir_graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap();
        let _span = span!(Level::TRACE, "dtoh").entered();
        self.cuda_stream
            .clone_dtoh(
                self.buffers
                    .get(&data_id)
                    .expect("Cannot find tensor in runtime!"),
            )
            .unwrap()
    }

    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
        let bytes = self.get_output_data(id);
        let bytes = bytes.leak();
        let n_bytes = bytes.len();
        let bytes_ptr = bytes.as_mut_ptr();
        let float_ptr = bytes_ptr as *mut f32;
        unsafe { Vec::from_raw_parts(float_ptr, n_bytes / 4, n_bytes / 4) }
    }

    pub fn get_i32(&self, id: impl ToId) -> Vec<i32> {
        self.get_output_data(id)
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect_vec()
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(&mut self, dyn_dims: &FxHashMap<char, usize>) {
        self.intermediate_buffer_dims.clear();
        for node in self.llir_graph.node_indices().collect_vec() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn BlockOp>() {
                let out_size = op.output_size();
                let exec_size = out_size.exec(dyn_dims).unwrap();
                // Skip allocation for ops with zero output size
                if exec_size == 0 {
                    continue;
                }
                self.intermediate_buffer_dims.extend(out_size.dyn_vars());
                let alloc_bytes = exec_size * size_of::<f32>();
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(alloc_bytes)
                        .unwrap_or_else(|e| {
                            panic!(
                                "Failed to alloc {} bytes for BlockOp node {:?}: {:?}",
                                alloc_bytes, node, e
                            )
                        }),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.cached_buffer_ptrs.insert(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn KernelOp>() {
                let out_size = op.output_size();
                let exec_size = out_size.exec(dyn_dims).unwrap();
                // Skip allocation for kernels with zero output size (e.g., megakernels)
                if exec_size == 0 {
                    continue;
                }
                self.intermediate_buffer_dims.extend(out_size.dyn_vars());
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(exec_size * size_of::<f32>())
                        .unwrap(),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.cached_buffer_ptrs.insert(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn HostOp>() {
                let exec_size = op.output_size().exec(dyn_dims).unwrap();
                // Skip allocation for ops with zero output size
                if exec_size == 0 {
                    continue;
                }
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(exec_size * size_of::<f32>())
                        .unwrap(),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.cached_buffer_ptrs.insert(node, ptr);
            }
        }
    }

    /// Prebuild all CUDA graphs with the given dynamic dimension values.
    #[tracing::instrument(skip_all)]
    pub fn prebuild_graphs(&mut self, dyn_map: &FxHashMap<char, usize>) {
        // 1. Allocate intermediate buffers (needed for buffer pointers)
        if self.buffers.is_empty() {
            self.last_dyn_map = dyn_map.clone();
            self.allocate_intermediate_buffers(dyn_map);
        }

        // 2. Process changed HLIR inputs to get their buffer pointers
        if !self.changed_hlir.is_empty() {
            let to_process: Vec<(NodeIndex, NodeIndex, u64)> = self
                .changed_hlir
                .iter()
                .filter_map(|hlir_node| {
                    self.hlir_buffers.get(hlir_node).map(|input| {
                        let llir_node = self.hlir_to_llir[hlir_node];
                        let ptr = match input {
                            CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                            CudaInput::Ptr(p) => *p,
                        };
                        (*hlir_node, llir_node, ptr)
                    })
                })
                .collect();

            for (hlir_node, llir_node, ptr) in to_process {
                self.cached_buffer_ptrs.insert(llir_node, ptr);
                self.changed_hlir.remove(&hlir_node);
            }
        }

        // 3. Build all CUDA graphs
        let tracing_enabled = enabled!(Level::TRACE);
        for exec_node in toposort(&self.exec_graph, None).unwrap() {
            if let ExecutableKernel::CudaGraphExec {
                cuda_graph,
                cuda_graph_exec,
                node_to_graph_node,
                kernel_params,
                kernels,
                dyn_dims_buffer,
                dyn_dims_order,
                last_dyn_values,
                last_buffer_ptrs,
                timing_events,
                ..
            } = &mut self.exec_graph[exec_node]
            {
                // Skip if already built
                if cuda_graph.is_some() {
                    continue;
                }

                // Collect buffer pointers - skip if any missing
                let mut current_buffer_ptrs = FxHashMap::default();
                let mut all_ptrs_available = true;
                'outer: for kernel in kernels.iter() {
                    if let Some(&output_ptr) = self.cached_buffer_ptrs.get(&kernel.output) {
                        current_buffer_ptrs.insert(kernel.output, output_ptr);
                    } else if kernel
                        .kernel_op
                        .output_size()
                        .to_usize()
                        .map(|s| s > 0)
                        .unwrap_or(true)
                    {
                        // Only require output pointer if kernel has non-zero output size
                        all_ptrs_available = false;
                        break;
                    }
                    for inp in &kernel.inputs {
                        if current_buffer_ptrs.contains_key(inp) {
                            continue;
                        }
                        if let Some(&ptr) = self.cached_buffer_ptrs.get(inp) {
                            current_buffer_ptrs.insert(*inp, ptr);
                        } else {
                            all_ptrs_available = false;
                            break 'outer;
                        }
                    }
                }

                if !all_ptrs_available {
                    continue;
                }

                // Allocate internal buffers for each kernel if needed
                for kernel in kernels.iter_mut() {
                    if kernel.internal_bufs.is_empty() {
                        kernel.internal_bufs = kernel
                            .kernel_op
                            .allocate_internal_buffers(&self.cuda_stream, dyn_map);
                    }
                }

                // Allocate dyn_dims buffer if needed
                if !dyn_dims_order.is_empty() && dyn_dims_buffer.is_none() {
                    *dyn_dims_buffer = Some(
                        self.cuda_stream
                            .alloc_zeros::<i32>(dyn_dims_order.len())
                            .expect("Failed to allocate dyn_dims buffer"),
                    );
                    let values: Vec<i32> = dyn_dims_order
                        .iter()
                        .map(|d| dyn_map.get(d).copied().unwrap_or(0) as i32)
                        .collect();
                    if let Some(buf) = dyn_dims_buffer.as_mut() {
                        self.cuda_stream.memcpy_htod(&values, buf).unwrap();
                    }
                }

                // Build the CUDA graph
                let ctx = self.cuda_stream.context().clone();
                let mut new_graph =
                    CudaGraphHandle::new(ctx.clone()).expect("Failed to create CUDA graph");

                kernel_params.clear();
                kernel_params.reserve(kernels.len());

                // Create timing events if tracing
                if tracing_enabled {
                    let needed_events = kernels.len() + 1;
                    while timing_events.len() < needed_events {
                        timing_events
                            .push(create_cuda_event(&ctx).expect("Failed to create timing event"));
                    }
                }

                let dyn_dims_ptr = dyn_dims_buffer
                    .as_ref()
                    .map(|buf| buf.device_ptr(&self.cuda_stream).0)
                    .unwrap_or(0);

                new_graph.ctx.bind_to_thread().unwrap();

                let mut prev_graph_node: Option<CUgraphNode> = None;

                for (idx, kernel) in kernels.iter_mut().enumerate() {
                    let grid_dim = (
                        kernel.launch_grid.0.exec(dyn_map).unwrap() as u32,
                        kernel.launch_grid.1.exec(dyn_map).unwrap() as u32,
                        kernel.launch_grid.2.exec(dyn_map).unwrap() as u32,
                    );
                    let block_dim = (
                        kernel.launch_threadblock.0.exec(dyn_map).unwrap() as u32,
                        kernel.launch_threadblock.1.exec(dyn_map).unwrap() as u32,
                        kernel.launch_threadblock.2.exec(dyn_map).unwrap() as u32,
                    );
                    let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;

                    let output_ptr = current_buffer_ptrs
                        .get(&kernel.output)
                        .copied()
                        .unwrap_or(0);
                    let input_ptrs: Vec<u64> = kernel
                        .inputs
                        .iter()
                        .map(|inp| current_buffer_ptrs.get(inp).copied().unwrap_or(0))
                        .collect();

                    // Build params using trait method
                    let param_values = kernel.kernel_op.build_params(
                        &self.cuda_stream,
                        output_ptr,
                        &input_ptrs,
                        &kernel.internal_bufs,
                        dyn_dims_ptr,
                    );
                    let mut params = UnifiedKernelParams::new(param_values);

                    let cu_func = unsafe { kernel.kernel.raw_function() };

                    let deps: &[CUgraphNode] = match (&prev_graph_node, tracing_enabled) {
                        (Some(prev), true) => {
                            let event_node = new_graph
                                .add_event_record_node(&[*prev], timing_events[idx])
                                .expect("Failed to add event record node");
                            prev_graph_node = Some(event_node);
                            std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                        }
                        (None, true) => {
                            let event_node = new_graph
                                .add_event_record_node(&[], timing_events[idx])
                                .expect("Failed to add event record node");
                            prev_graph_node = Some(event_node);
                            std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                        }
                        (Some(prev), false) => std::slice::from_ref(prev),
                        (None, false) => &[],
                    };

                    let graph_node = unsafe {
                        new_graph.add_kernel_node(
                            deps,
                            cu_func,
                            grid_dim,
                            block_dim,
                            shared_mem,
                            params.as_cuda_params(),
                        )
                    }
                    .expect("Failed to add kernel node");

                    node_to_graph_node.insert(kernel.llir_node, graph_node);
                    kernel.graph_node = Some(graph_node);
                    kernel_params.push(params);
                    prev_graph_node = Some(graph_node);
                }

                if tracing_enabled && let Some(prev) = prev_graph_node {
                    new_graph
                        .add_event_record_node(&[prev], timing_events[kernels.len()])
                        .expect("Failed to add final event node");
                }

                let new_exec = new_graph
                    .instantiate()
                    .expect("Failed to instantiate CUDA graph");

                *cuda_graph = Some(new_graph);
                *cuda_graph_exec = Some(new_exec);
                *last_dyn_values = dyn_map.clone();
                *last_buffer_ptrs = current_buffer_ptrs;
            }
        }
    }
}

pub trait ToCudaInput {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput;
}

impl ToCudaInput for &[f32] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<i32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for Vec<f32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .clone_htod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl Runtime for CudaRuntime {
    type Ops = (
        crate::logical::Ops,
        crate::kernel::Ops,
        crate::block::Ops,
        crate::host::Ops,
    );
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(stream: Self::CompileArg) -> Self {
        let ctx = stream.context();
        let num_sms = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap() as usize;
        let timing_stream = ctx.new_stream().expect("Failed to create timing stream");

        // Pre-allocate pinned buffer pools for async timing collection
        // We allocate enough slots to handle many megakernels between flushes
        const PINNED_POOL_SIZE: usize = 64;
        let timing_len = num_sms * N_TIMING_SLOTS;
        let pinned_timing_pool;
        let pinned_start_pool;
        if enabled!(Level::TRACE) {
            pinned_timing_pool = (0..PINNED_POOL_SIZE)
                .map(|_| unsafe {
                    ctx.alloc_pinned::<SMEvent>(timing_len)
                        .expect("Failed to allocate pinned timing buffer")
                })
                .collect::<Vec<_>>();
            pinned_start_pool = (0..PINNED_POOL_SIZE)
                .map(|_| unsafe {
                    ctx.alloc_pinned::<u64>(num_sms)
                        .expect("Failed to allocate pinned start time buffer")
                })
                .collect::<Vec<_>>();
        } else {
            pinned_timing_pool = vec![];
            pinned_start_pool = vec![]
        }

        Self {
            num_sms,
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            cuda_stream: stream,
            timing_stream,
            llir_graph: StableGraph::default(),
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            hlir_to_llir: FxHashMap::default(),
            llir_to_hlir: FxHashMap::default(),
            changed_hlir: FxHashSet::default(),
            cached_buffer_ptrs: FxHashMap::default(),
            timings: vec![],
            cuda_graph_timings: vec![],
            pending_timing_data: vec![],
            pinned_timing_pool,
            pinned_start_pool,
            next_pinned_slot: 0,
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            last_kernel_stats: vec![],
            last_total_time_us: 0.0,
            kernel_cache: FxHashMap::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Sync before clearing old data to ensure all operations complete
        self.cuda_stream
            .synchronize()
            .expect("Failed to sync at start of load_llir");

        // Explicitly drop CUDA graph handles before clearing exec_graph
        // to ensure proper destruction order
        let ctx = self.cuda_stream.context().clone();
        for node in self.exec_graph.node_indices().collect::<Vec<_>>() {
            if let ExecutableKernel::CudaGraphExec {
                kernels,
                timing_events,
                dyn_dims_buffer,
                kernel_params,
                cuda_graph_exec,
                cuda_graph,
                ..
            } = &mut self.exec_graph[node]
            {
                // Clear kernels first (drops internal_bufs)
                kernels.clear();
                // Properly destroy timing events before clearing
                for event in timing_events.drain(..) {
                    destroy_cuda_event(&ctx, event);
                }
                // Clear dyn_dims buffer
                *dyn_dims_buffer = None;
                // Clear kernel params
                kernel_params.clear();
                // Finally, drop the CUDA graph handles
                *cuda_graph_exec = None;
                *cuda_graph = None;
            }
        }
        // Sync after dropping CUDA resources
        self.cuda_stream
            .synchronize()
            .expect("Failed to sync after dropping CUDA resources");

        // Clear intermediate buffers when loading new graph - they need to be
        // reallocated and re-registered with the new work_queue
        self.buffers.clear();
        self.cached_buffer_ptrs.clear();
        // Mark all HLIR inputs as changed so their pointers get re-cached in execute
        self.changed_hlir.extend(self.hlir_buffers.keys().copied());
        self.exec_graph.clear();

        // Sync after clearing all buffers to ensure CUDA resources are freed
        self.cuda_stream
            .synchronize()
            .expect("Failed to sync after clearing buffers");

        // Rebind CUDA context to thread after cleanup to ensure valid state
        self.cuda_stream
            .context()
            .bind_to_thread()
            .expect("Failed to bind CUDA context after cleanup");

        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();

        // Clone llir_graph so we can modify it to add megakernel nodes
        let mut llir_graph = llir_graph.clone();

        // Compile BlockOp subgraphs into MegakernelOps and add them to the llir_graph
        let megakernel_to_blocks = crate::block::compile_megakernels(
            &mut llir_graph,
            &self.cuda_stream,
            &mut self.kernel_cache,
        );

        // Add kernels (including megakernels which now implement KernelOp)
        {
            let _span = span!(Level::TRACE, "compile_kernels").entered();
            let kernel_ops_in_graph = llir_graph
                .node_indices()
                .filter(|n| llir_graph[*n].to_dialect::<dyn KernelOp>().is_some())
                .collect::<FxHashSet<_>>();
            let kernel_subgraphs =
                partition_marked_convex(&llir_graph, &kernel_ops_in_graph).unwrap();
            let mut compile_cache = FxHashMap::default();
            for subgraph in kernel_subgraphs {
                let mut kernels = Vec::with_capacity(subgraph.len());
                let mut all_dyn_dims = FxHashSet::default();
                let mut total_bytes_loaded = Expression::from(0);
                let mut total_bytes_stored = Expression::from(0);
                let mut total_flops = Expression::from(0);

                // Get topological order for kernels in this subgraph
                let topo_order: Vec<_> = toposort(&llir_graph, None)
                    .unwrap()
                    .into_iter()
                    .filter(|n| subgraph.contains(n))
                    .collect();

                for kernel_node_idx in &topo_order {
                    let kernel_op_ref = llir_graph[*kernel_node_idx]
                        .to_dialect::<dyn KernelOp>()
                        .unwrap();
                    let span = span!(Level::TRACE, "kernel_compile").entered();
                    let (kernel_function, _, kernel_str, grid, tb, shared_mem, constants) =
                        kernel_op_ref.compile(&self.cuda_stream, &mut compile_cache);
                    drop(span);

                    // For megakernels, inputs should include all block op nodes in the subgraph
                    // (for buffer pointer collection) plus external input edges (for exec_graph deps)
                    let mut inputs: Vec<NodeIndex> = llir_graph
                        .edges_directed(*kernel_node_idx, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect_vec();

                    // If this is a megakernel, include all its block op nodes for buffer access
                    if let Some(block_nodes) = megakernel_to_blocks.get(kernel_node_idx) {
                        inputs.extend(block_nodes.iter().copied());
                    }

                    total_bytes_loaded += kernel_op_ref.bytes_loaded();
                    total_bytes_stored += kernel_op_ref.bytes_stored();
                    total_flops += kernel_op_ref.flops();

                    // Collect all dyn dims used by this kernel's expressions
                    all_dyn_dims.extend(grid.0.dyn_vars());
                    all_dyn_dims.extend(grid.1.dyn_vars());
                    all_dyn_dims.extend(grid.2.dyn_vars());
                    all_dyn_dims.extend(tb.0.dyn_vars());
                    all_dyn_dims.extend(tb.1.dyn_vars());
                    all_dyn_dims.extend(tb.2.dyn_vars());
                    all_dyn_dims.extend(shared_mem.dyn_vars());
                    all_dyn_dims.extend(kernel_op_ref.output_size().dyn_vars());
                    all_dyn_dims.extend(kernel_op_ref.bytes_loaded().dyn_vars());
                    all_dyn_dims.extend(kernel_op_ref.bytes_stored().dyn_vars());

                    // Clone the kernel_op Arc for storage
                    let kernel_op: Arc<Box<dyn KernelOp>> = Arc::clone(kernel_op_ref);

                    kernels.push(KernelNode {
                        llir_node: *kernel_node_idx,
                        kernel: kernel_function,
                        launch_grid: grid,
                        launch_threadblock: tb,
                        shared_mem,
                        inputs,
                        kernel_str,
                        output: *kernel_node_idx,
                        kernel_name: kernel_op.kernel_name(),
                        bytes_loaded: kernel_op.bytes_loaded(),
                        bytes_stored: kernel_op.bytes_stored(),
                        flops: kernel_op.flops(),
                        internal_bufs: Vec::new(),
                        constants,
                        kernel_op,
                        graph_node: None,
                    });
                }

                // Sort dyn dims alphabetically for consistent buffer layout
                let mut dyn_dims_order: Vec<char> = all_dyn_dims.into_iter().collect();
                dyn_dims_order.sort();

                let exec_node = exec_graph.add_node(ExecutableKernel::CudaGraphExec {
                    cuda_graph: None,
                    cuda_graph_exec: None,
                    node_to_graph_node: FxHashMap::default(),
                    kernel_params: Vec::new(),
                    kernels,
                    dyn_dims_buffer: None,
                    dyn_dims_order,
                    last_dyn_values: FxHashMap::default(),
                    last_buffer_ptrs: FxHashMap::default(),
                    timing_events: Vec::new(),
                    total_bytes_loaded,
                    total_bytes_stored,
                    total_flops,
                });

                for node in subgraph {
                    node_to_exec.insert(node, exec_node);
                    // If this is a megakernel node, also map all its block op nodes to this exec_node
                    if let Some(block_nodes) = megakernel_to_blocks.get(&node) {
                        for &block_node in block_nodes {
                            node_to_exec.insert(block_node, exec_node);
                        }
                    }
                }
            }
        }

        // Add host ops
        {
            let _span = span!(Level::TRACE, "compile_host_ops").entered();
            for host_op_node_index in llir_graph.node_indices() {
                if let Some(host_op) = llir_graph[host_op_node_index].to_dialect::<dyn HostOp>() {
                    let inputs = llir_graph
                        .edges_directed(host_op_node_index, Direction::Incoming)
                        .sorted_by_key(|e| e.id())
                        .map(|e| e.source())
                        .collect_vec();
                    node_to_exec.insert(
                        host_op_node_index,
                        exec_graph.add_node(ExecutableKernel::HostOp {
                            stream: Arc::clone(&self.cuda_stream),
                            inputs,
                            output: host_op_node_index,
                            internal: Arc::clone(host_op),
                        }),
                    );
                }
            }
        }

        // Add edges
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            if !node_to_exec.contains_key(&start) || !node_to_exec.contains_key(&end) {
                continue;
            }
            let (exec_start, exec_end) = (node_to_exec[&start], node_to_exec[&end]);
            if exec_start != exec_end
                && exec_graph
                    .edges_connecting(exec_start, exec_end)
                    .next()
                    .is_none()
            {
                exec_graph.add_edge(exec_start, exec_end, ());
            }
        }
        self.exec_graph = exec_graph;
        self.llir_graph = llir_graph.clone();
        self.node_to_exec = node_to_exec;
        self.hlir_to_llir.clear();
        self.llir_to_hlir.clear();
        self.changed_hlir.clear();
        for (hlir_node, llir_node) in self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_op::<Input>().map(|op| (op.node, n)))
            .collect_vec()
        {
            self.llir_to_hlir
                .insert(llir_node, NodeIndex::new(hlir_node));
            self.hlir_to_llir
                .insert(NodeIndex::new(hlir_node), llir_node);
            self.changed_hlir.insert(NodeIndex::new(hlir_node));
        }

        // Prebuild CUDA graphs if we have a previous dyn_map (e.g., from search/profile)
        // This avoids rebuild overhead on first execute after load_llir
        if !self.last_dyn_map.is_empty() {
            let dyn_map = self.last_dyn_map.clone();
            self.prebuild_graphs(&dyn_map);
        }
    }

    #[tracing::instrument(skip_all)]
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
    ) -> (Self::ProfileMetric, String) {
        self.buffers.clear();
        self.load_llir(llir_graph);
        let start = std::time::Instant::now();
        self.execute(dyn_map);
        let duration = start.elapsed();

        // Flush pending timing data so it's available for stats
        self.flush_pending_timings();

        // Compute aggregates for profile display
        let block_op_stats = Self::compute_block_op_stats(
            &self.llir_graph,
            self.timings.last().unwrap_or(&vec![]),
            &self.last_dyn_map,
            self.num_sms,
        );

        let total_bytes: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded + s.bytes_stored)
            .sum::<usize>()
            + block_op_stats
                .iter()
                .map(|s| s.bytes_loaded + s.bytes_stored)
                .sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.flops).sum::<usize>();
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());
        let mbu = peak_bw.map(|p| aggregate_bw / p as f64);
        let mfu = peak_tf.map(|p| aggregate_tf / p as f64);

        let duration_str = pretty_duration::pretty_duration(&duration, None);
        let mbu_str = mbu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let mfu_str = mfu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let display = format!("{duration_str} | MBU: {mbu_str} | MFU: {mfu_str}");

        (duration, display)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let buffers_empty = self.buffers.is_empty();
        let dyn_map_len_changed = dyn_map.len() != self.last_dyn_map.len();
        let dyn_dims_changed = dyn_map
            .iter()
            .filter(|(d, _)| self.intermediate_buffer_dims.contains(*d))
            .any(|(d, v)| self.last_dyn_map.get(d).map(|n| *n != *v).unwrap_or(true));
        let needs_realloc = buffers_empty || dyn_map_len_changed || dyn_dims_changed;
        if needs_realloc {
            self.last_dyn_map = dyn_map.clone();
            self.allocate_intermediate_buffers(dyn_map);
        }

        // Always clear intermediate buffers to ensure correctness for operations using atomicAdd
        // TODO: this is very expensive. Need to eliminate ops that require zeroed outputs
        for buffer in self.buffers.values_mut() {
            self.cuda_stream.memset_zeros(buffer).unwrap();
        }
        self.cuda_stream.synchronize().unwrap();

        // Cache HLIR input pointers
        let had_changed_hlir = !self.changed_hlir.is_empty();
        if had_changed_hlir {
            for hlir_node in self.changed_hlir.clone() {
                let llir_node = self.hlir_to_llir[&hlir_node];
                let ptr = match &self.hlir_buffers[&hlir_node] {
                    CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                    CudaInput::Ptr(p) => *p,
                };
                self.cached_buffer_ptrs.insert(llir_node, ptr);
            }
            self.changed_hlir.clear();
        }
        // Ensure all CUDA graphs are built (handles first execute and any missing graphs)
        self.prebuild_graphs(dyn_map);

        let mut kernel_stats = Vec::new();
        let total_start = std::time::Instant::now();
        for exec_node in toposort(&self.exec_graph, None).unwrap() {
            trace!("Executing: {:?}", self.exec_graph[exec_node]);
            match &mut self.exec_graph[exec_node] {
                ExecutableKernel::HostOp {
                    stream,
                    inputs,
                    output,
                    internal,
                } => {
                    let mut host_op_buffers: Vec<&CudaSlice<u8>> = vec![&self.buffers[output]];
                    host_op_buffers.extend(inputs.iter().map(|inp| {
                        if let Some(buf) = self.buffers.get(inp) {
                            buf
                        } else {
                            match &self.hlir_buffers[&self.llir_to_hlir[inp]] {
                                CudaInput::Buffer(buf) => buf,
                                CudaInput::Ptr(_) => unimplemented!(),
                            }
                        }
                    }));
                    let _span =
                        span!(Level::TRACE, "host_op_execute", n_inputs = inputs.len()).entered();
                    internal.execute(stream, &host_op_buffers, dyn_map).unwrap();
                    self.cuda_stream.synchronize().unwrap();
                }
                ExecutableKernel::CudaGraphExec {
                    cuda_graph,
                    cuda_graph_exec,
                    node_to_graph_node,
                    kernel_params,
                    kernels,
                    dyn_dims_buffer,
                    dyn_dims_order,
                    last_dyn_values,
                    last_buffer_ptrs,
                    timing_events,
                    total_bytes_loaded,
                    total_bytes_stored,
                    total_flops,
                } => {
                    // Generate a unique ID for this graph execution (for perfetto correlation)
                    let graph_span_id = Uuid::new_v4();
                    let span = span!(
                        Level::TRACE,
                        "cuda_graph",
                        kernels = kernels.len(),
                        id = field::Empty
                    );
                    span.record("id", graph_span_id.to_string().as_str());
                    let _entered = span.enter();
                    let span_entry_instant = std::time::Instant::now();

                    let setup_span = span!(Level::TRACE, "cg_setup").entered();

                    // Fast path: check if dyn_map changed before doing any work
                    let dyn_map_changed = dyn_map.len() != last_dyn_values.len()
                        || dyn_map
                            .iter()
                            .any(|(k, v)| last_dyn_values.get(k) != Some(v));

                    // Allocate dyn_dims_buffer if needed (first run or size changed)
                    if !dyn_dims_order.is_empty() && dyn_dims_buffer.is_none() {
                        *dyn_dims_buffer = Some(
                            self.cuda_stream
                                .alloc_zeros::<i32>(dyn_dims_order.len())
                                .expect("Failed to allocate dyn_dims buffer"),
                        );
                    }

                    // Update shared dyn_dims buffer with a single memcpy if dyn_map changed
                    if dyn_map_changed && !dyn_dims_order.is_empty() {
                        let values: Vec<i32> = dyn_dims_order
                            .iter()
                            .map(|d| dyn_map.get(d).copied().unwrap_or(0) as i32)
                            .collect();
                        if let Some(buf) = dyn_dims_buffer.as_mut() {
                            self.cuda_stream.memcpy_htod(&values, buf).unwrap();
                        }
                    }

                    // Check if any kernel's internal buffer dimensions changed
                    // If so, we need to reallocate internal buffers and rebuild the graph
                    let mut needs_internal_realloc = false;
                    for kernel in kernels.iter() {
                        let internal_dims = kernel.kernel_op.internal_buffer_dyn_dims();
                        if internal_dims
                            .iter()
                            .any(|d| dyn_map.get(d) != last_dyn_values.get(d))
                        {
                            needs_internal_realloc = true;
                            break;
                        }
                    }

                    // Reallocate internal buffers if needed
                    if needs_internal_realloc {
                        for kernel in kernels.iter_mut() {
                            kernel.internal_bufs = kernel
                                .kernel_op
                                .allocate_internal_buffers(&self.cuda_stream, dyn_map);
                        }
                        // Internal buffer pointers changed, need to rebuild CUDA graph
                        *cuda_graph = None;
                        *cuda_graph_exec = None;
                        node_to_graph_node.clear();
                        kernel_params.clear();
                    }

                    // Rebuild CUDA graph if needed (after internal buffer reallocation)
                    if cuda_graph.is_none() {
                        // Build the CUDA graph inline (same logic as prebuild_graphs)
                        let ctx = self.cuda_stream.context().clone();
                        let mut new_graph =
                            CudaGraphHandle::new(ctx.clone()).expect("Failed to create CUDA graph");

                        kernel_params.reserve(kernels.len());

                        let tracing_enabled = enabled!(Level::TRACE);
                        if tracing_enabled {
                            let needed_events = kernels.len() + 1;
                            while timing_events.len() < needed_events {
                                timing_events.push(
                                    create_cuda_event(&ctx).expect("Failed to create timing event"),
                                );
                            }
                        }

                        let dyn_dims_ptr = dyn_dims_buffer
                            .as_ref()
                            .map(|buf| buf.device_ptr(&self.cuda_stream).0)
                            .unwrap_or(0);

                        new_graph.ctx.bind_to_thread().unwrap();

                        let mut prev_graph_node: Option<CUgraphNode> = None;

                        for (idx, kernel) in kernels.iter_mut().enumerate() {
                            let grid_dim = (
                                kernel.launch_grid.0.exec(dyn_map).unwrap() as u32,
                                kernel.launch_grid.1.exec(dyn_map).unwrap() as u32,
                                kernel.launch_grid.2.exec(dyn_map).unwrap() as u32,
                            );
                            let block_dim = (
                                kernel.launch_threadblock.0.exec(dyn_map).unwrap() as u32,
                                kernel.launch_threadblock.1.exec(dyn_map).unwrap() as u32,
                                kernel.launch_threadblock.2.exec(dyn_map).unwrap() as u32,
                            );
                            let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;

                            let output_ptr = self
                                .cached_buffer_ptrs
                                .get(&kernel.output)
                                .copied()
                                .unwrap_or(0);
                            let input_ptrs: Vec<u64> = kernel
                                .inputs
                                .iter()
                                .map(|inp| self.cached_buffer_ptrs.get(inp).copied().unwrap_or(0))
                                .collect();

                            let param_values = kernel.kernel_op.build_params(
                                &self.cuda_stream,
                                output_ptr,
                                &input_ptrs,
                                &kernel.internal_bufs,
                                dyn_dims_ptr,
                            );
                            let mut params = UnifiedKernelParams::new(param_values);

                            let cu_func = unsafe { kernel.kernel.raw_function() };

                            let deps: &[CUgraphNode] = match (&prev_graph_node, tracing_enabled) {
                                (Some(prev), true) => {
                                    let event_node = new_graph
                                        .add_event_record_node(&[*prev], timing_events[idx])
                                        .expect("Failed to add event record node");
                                    prev_graph_node = Some(event_node);
                                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                                }
                                (None, true) => {
                                    let event_node = new_graph
                                        .add_event_record_node(&[], timing_events[idx])
                                        .expect("Failed to add event record node");
                                    prev_graph_node = Some(event_node);
                                    std::slice::from_ref(prev_graph_node.as_ref().unwrap())
                                }
                                (Some(prev), false) => std::slice::from_ref(prev),
                                (None, false) => &[],
                            };

                            let graph_node = unsafe {
                                new_graph.add_kernel_node(
                                    deps,
                                    cu_func,
                                    grid_dim,
                                    block_dim,
                                    shared_mem,
                                    params.as_cuda_params(),
                                )
                            }
                            .expect("Failed to add kernel node");

                            node_to_graph_node.insert(kernel.llir_node, graph_node);
                            kernel.graph_node = Some(graph_node);
                            kernel_params.push(params);
                            prev_graph_node = Some(graph_node);
                        }

                        if tracing_enabled && let Some(prev) = prev_graph_node {
                            new_graph
                                .add_event_record_node(&[prev], timing_events[kernels.len()])
                                .expect("Failed to add final event node");
                        }

                        let new_exec = new_graph
                            .instantiate()
                            .expect("Failed to instantiate CUDA graph");

                        *cuda_graph = Some(new_graph);
                        *cuda_graph_exec = Some(new_exec);
                    }

                    debug_assert!(cuda_graph.is_some(), "CUDA graph should be built");

                    // Call pre_execute for each kernel (handles buffer array updates, dyn dims, etc.)
                    for kernel in kernels.iter_mut() {
                        kernel.kernel_op.pre_execute(
                            &self.cuda_stream,
                            &mut kernel.internal_bufs,
                            &mut kernel.constants,
                            &self.cached_buffer_ptrs,
                            dyn_map,
                        );
                    }

                    // Collect buffer pointers if reallocation happened
                    let needs_buffer_update = needs_realloc;
                    let current_buffer_ptrs: FxHashMap<NodeIndex, u64> = if needs_realloc {
                        let mut ptrs = FxHashMap::default();
                        for kernel in kernels.iter() {
                            if let Some(&output_ptr) = self.cached_buffer_ptrs.get(&kernel.output) {
                                ptrs.insert(kernel.output, output_ptr);
                            }
                            for inp in &kernel.inputs {
                                if !ptrs.contains_key(inp) {
                                    if let Some(&ptr) = self.cached_buffer_ptrs.get(inp) {
                                        ptrs.insert(*inp, ptr);
                                    }
                                }
                            }
                        }
                        ptrs
                    } else {
                        std::mem::take(last_buffer_ptrs)
                    };

                    // Check if we need surgical update (dyn dims or buffer pointers changed)
                    let needs_surgical_update = dyn_map_changed || needs_buffer_update;

                    if needs_surgical_update {
                        let exec = cuda_graph_exec.as_mut().unwrap();
                        exec.ctx.bind_to_thread().unwrap();

                        let dyn_dims_ptr = dyn_dims_buffer
                            .as_ref()
                            .map(|buf| buf.device_ptr(&self.cuda_stream).0)
                            .unwrap_or(0);

                        for (idx, kernel) in kernels.iter_mut().enumerate() {
                            // Rebuild params using trait method
                            let output_ptr = current_buffer_ptrs
                                .get(&kernel.output)
                                .copied()
                                .unwrap_or(0);
                            let input_ptrs: Vec<u64> = kernel
                                .inputs
                                .iter()
                                .map(|inp| current_buffer_ptrs.get(inp).copied().unwrap_or(0))
                                .collect();

                            let param_values = kernel.kernel_op.build_params(
                                &self.cuda_stream,
                                output_ptr,
                                &input_ptrs,
                                &kernel.internal_bufs,
                                dyn_dims_ptr,
                            );
                            kernel_params[idx] = UnifiedKernelParams::new(param_values);

                            let grid_dim = (
                                kernel.launch_grid.0.exec(dyn_map).unwrap() as u32,
                                kernel.launch_grid.1.exec(dyn_map).unwrap() as u32,
                                kernel.launch_grid.2.exec(dyn_map).unwrap() as u32,
                            );
                            let block_dim = (
                                kernel.launch_threadblock.0.exec(dyn_map).unwrap() as u32,
                                kernel.launch_threadblock.1.exec(dyn_map).unwrap() as u32,
                                kernel.launch_threadblock.2.exec(dyn_map).unwrap() as u32,
                            );
                            let shared_mem = kernel.shared_mem.exec(dyn_map).unwrap() as u32;
                            let cu_func = unsafe { kernel.kernel.raw_function() };

                            unsafe {
                                exec.update_kernel_node(
                                    node_to_graph_node[&kernel.llir_node],
                                    cu_func,
                                    grid_dim,
                                    block_dim,
                                    shared_mem,
                                    kernel_params[idx].as_cuda_params(),
                                )
                            }
                            .expect("Failed to update kernel node");
                        }
                        *last_dyn_values = dyn_map.clone();
                        *last_buffer_ptrs = current_buffer_ptrs;
                    } else {
                        *last_buffer_ptrs = current_buffer_ptrs;
                    }
                    self.cuda_stream.synchronize().unwrap();
                    drop(setup_span);

                    let ctx = self.cuda_stream.context();

                    // Measure elapsed time from span entry to launch for accurate timeline positioning
                    let setup_duration_ns = span_entry_instant.elapsed().as_nanos() as u64;

                    cuda_graph_exec
                        .as_ref()
                        .unwrap()
                        .launch(&self.cuda_stream)
                        .expect("Failed to launch CUDA graph");
                    self.cuda_stream.synchronize().unwrap();

                    let _teardown_span = span!(Level::TRACE, "cg_teardown").entered();

                    // Only collect timing data when tracing is enabled (timing events exist in graph)
                    let graph_time_us = if enabled!(Level::TRACE) && !timing_events.is_empty() {
                        // With pre-sync, launch latency should be minimal
                        let launch_latency_ns: u64 = 0;

                        let mut kernel_timings = Vec::with_capacity(kernels.len());
                        let mut cumulative_ns: u64 = 0;
                        for (idx, kernel) in kernels.iter().enumerate() {
                            let elapsed_ns =
                                (event_elapsed_ms(ctx, timing_events[idx], timing_events[idx + 1])
                                    .unwrap_or(0.0)
                                    * 1_000_000.0) as u64;
                            kernel_timings.push(CudaGraphKernelTiming {
                                kernel_name: kernel.kernel_name,
                                start_ns: cumulative_ns,
                                end_ns: cumulative_ns + elapsed_ns,
                            });
                            cumulative_ns += elapsed_ns;
                        }

                        self.cuda_graph_timings.push((
                            CudaGraphTiming {
                                kernel_timings,
                                launch_latency_ns,
                                setup_duration_ns,
                            },
                            graph_span_id,
                        ));
                        cumulative_ns as f64 / 1000.0
                    } else {
                        0.0
                    };
                    let loaded = total_bytes_loaded.exec(dyn_map).unwrap_or(0);
                    let stored = total_bytes_stored.exec(dyn_map).unwrap_or(0);
                    let flop_count = total_flops.exec(dyn_map).unwrap_or(0);

                    let total_bytes = loaded + stored;
                    let bandwidth_gbps = if graph_time_us > 0.0 {
                        (total_bytes as f64) / (graph_time_us * 1e-6) / 1e9
                    } else {
                        0.0
                    };
                    let tflops = if graph_time_us > 0.0 {
                        (flop_count as f64) / (graph_time_us * 1e-6) / 1e12
                    } else {
                        0.0
                    };

                    kernel_stats.push(KernelStats {
                        name: "CudaGraph",
                        execution_time_us: graph_time_us,
                        bytes_loaded: loaded,
                        bytes_stored: stored,
                        flops: flop_count,
                        bandwidth_gbps,
                        tflops,
                    });
                }
            }
        }
        // Save raw stats for lazy interpretation in print_execution_stats
        self.last_kernel_stats = kernel_stats;
        self.last_total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;
        // Final sync to ensure all operations completed successfully
        self.cuda_stream
            .synchronize()
            .expect("Final sync failed in execute");
    }
}

impl CudaRuntime {
    fn compute_block_op_stats(
        llir_graph: &LLIRGraph,
        timings: &[(Vec<SMEvent>, u64, Uuid)],
        dyn_map: &FxHashMap<char, usize>,
        sm_count: usize,
    ) -> Vec<BlockOpStats> {
        // Get unique op names (same order as in interpreter)
        let op_names: Vec<&'static str> = llir_graph
            .node_indices()
            .filter_map(|n| llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<FxHashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(n, _)| n)
            .collect();

        if op_names.is_empty() {
            return vec![];
        }

        // Build map from op_name to index for event decoding
        let op_name_to_idx: FxHashMap<&'static str, usize> =
            op_names.iter().enumerate().map(|(i, n)| (*n, i)).collect();

        // Sum up bytes_loaded, bytes_stored, flops across ALL instances of each op type
        let mut op_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut op_bytes_stored: Vec<usize> = vec![0; op_names.len()];
        let mut op_flops: Vec<usize> = vec![0; op_names.len()];

        // Sum up prologue metrics across ALL instances of each op type
        let mut prologue_a_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_a_flops: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_b_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_b_flops: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_c_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut prologue_c_flops: Vec<usize> = vec![0; op_names.len()];

        for node in llir_graph.node_indices() {
            if let Some(op) = llir_graph[node].to_dialect::<dyn BlockOp>()
                && let Some(&idx) = op_name_to_idx.get(op.op_name())
            {
                let flops_expr = op.flops();
                let flops_val = flops_expr.exec(dyn_map).unwrap();
                op_bytes_loaded[idx] += op.bytes_loaded().exec(dyn_map).unwrap();
                op_bytes_stored[idx] += op.bytes_stored().exec(dyn_map).unwrap();
                op_flops[idx] += flops_val;

                // Aggregate prologue metrics
                prologue_a_bytes_loaded[idx] +=
                    op.prologue_a_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_a_flops[idx] += op.prologue_a_flops().exec(dyn_map).unwrap_or(0);
                prologue_b_bytes_loaded[idx] +=
                    op.prologue_b_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_b_flops[idx] += op.prologue_b_flops().exec(dyn_map).unwrap_or(0);
                prologue_c_bytes_loaded[idx] +=
                    op.prologue_c_bytes_loaded().exec(dyn_map).unwrap_or(0);
                prologue_c_flops[idx] += op.prologue_c_flops().exec(dyn_map).unwrap_or(0);
            }
        }

        // Aggregate timing per op type across all SMs
        // Event encoding:
        // 0: Issue
        // 1: Wait
        // 2 to 2 + n_ops - 1: Main ops
        // 2 + n_ops + op_idx * 3 + 0: Prologue A
        // 2 + n_ops + op_idx * 3 + 1: Prologue B
        // 2 + n_ops + op_idx * 3 + 2: Prologue C
        let n_ops = op_names.len();
        let mut op_times_ns: Vec<u64> = vec![0; n_ops];
        let mut op_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_a_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_a_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_b_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_b_counts: Vec<usize> = vec![0; n_ops];
        let mut prologue_c_times_ns: Vec<u64> = vec![0; n_ops];
        let mut prologue_c_counts: Vec<usize> = vec![0; n_ops];
        let mut issue_time_ns: u64 = 0;
        let mut issue_count: usize = 0;
        let mut wait_time_ns: u64 = 0;
        let mut wait_count: usize = 0;

        for (sm_timings, _start_time, _) in timings {
            for sm_chunk in sm_timings.chunks(N_TIMING_SLOTS) {
                for event in sm_chunk.iter() {
                    if event.start == 0 {
                        break; // No more events recorded for this SM
                    }
                    let stop = if event.stop == 0 {
                        event.start
                    } else {
                        event.stop
                    };
                    let duration = stop.saturating_sub(event.start);
                    let event_code = event.event as usize;
                    if event_code == 0 {
                        issue_time_ns += duration;
                        issue_count += 1;
                    } else if event_code == 1 {
                        wait_time_ns += duration;
                        wait_count += 1;
                    } else if event_code >= 2 && event_code < 2 + n_ops {
                        // Main op
                        let op_idx = event_code - 2;
                        op_times_ns[op_idx] += duration;
                        op_counts[op_idx] += 1;
                    } else if event_code >= 2 + n_ops {
                        // Prologue event
                        let prologue_event = event_code - 2 - n_ops;
                        let op_idx = prologue_event / 3;
                        let prologue_type = prologue_event % 3;
                        if op_idx < n_ops {
                            match prologue_type {
                                0 => {
                                    prologue_a_times_ns[op_idx] += duration;
                                    prologue_a_counts[op_idx] += 1;
                                }
                                1 => {
                                    prologue_b_times_ns[op_idx] += duration;
                                    prologue_b_counts[op_idx] += 1;
                                }
                                2 => {
                                    prologue_c_times_ns[op_idx] += duration;
                                    prologue_c_counts[op_idx] += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Compute bandwidth and TFLOPS from aggregated metrics
        // Divide total SM-time by sm_count to get wall-clock time
        let mut stats: Vec<BlockOpStats> = op_names
            .iter()
            .enumerate()
            .filter(|(i, _)| op_times_ns[*i] > 0)
            .map(|(i, &name)| {
                // Total SM-time divided by number of SMs gives wall-clock time
                let time_us = (op_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = op_bytes_loaded[i];
                let bytes_stored = op_bytes_stored[i];
                let flop_count = op_flops[i];

                let total_bytes = bytes_loaded + bytes_stored;
                let bandwidth_gbps = if time_us > 0.0 {
                    (total_bytes as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };

                BlockOpStats {
                    name,
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: op_counts[i],
                }
            })
            .collect();

        // Add prologue timing stats (only for prologues that were recorded)
        for (i, &name) in op_names.iter().enumerate() {
            if prologue_a_times_ns[i] > 0 {
                let time_us = (prologue_a_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_a_bytes_loaded[i];
                let flop_count = prologue_a_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue A)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_a_counts[i],
                });
            }
            if prologue_b_times_ns[i] > 0 {
                let time_us = (prologue_b_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_b_bytes_loaded[i];
                let flop_count = prologue_b_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue B)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_b_counts[i],
                });
            }
            if prologue_c_times_ns[i] > 0 {
                let time_us = (prologue_c_times_ns[i] as f64 / sm_count as f64) / 1000.0;
                let bytes_loaded = prologue_c_bytes_loaded[i];
                let flop_count = prologue_c_flops[i];
                let bandwidth_gbps = if time_us > 0.0 && bytes_loaded > 0 {
                    (bytes_loaded as f64) / (time_us * 1e-6) / 1e9
                } else {
                    0.0
                };
                let tflops = if time_us > 0.0 && flop_count > 0 {
                    (flop_count as f64) / (time_us * 1e-6) / 1e12
                } else {
                    0.0
                };
                stats.push(BlockOpStats {
                    name: Box::leak(format!("{} (prologue C)", name).into_boxed_str()),
                    execution_time_us: time_us,
                    bytes_loaded,
                    bytes_stored: 0,
                    flops: flop_count,
                    bandwidth_gbps,
                    tflops,
                    count: prologue_c_counts[i],
                });
            }
        }

        // Add Issue and Wait timing stats
        if issue_time_ns > 0 {
            let time_us = (issue_time_ns as f64 / sm_count as f64) / 1000.0;
            stats.push(BlockOpStats {
                name: "Issue",
                execution_time_us: time_us,
                bytes_loaded: 0,
                bytes_stored: 0,
                flops: 0,
                bandwidth_gbps: 0.0,
                tflops: 0.0,
                count: issue_count,
            });
        }
        if wait_time_ns > 0 {
            let time_us = (wait_time_ns as f64 / sm_count as f64) / 1000.0;
            stats.push(BlockOpStats {
                name: "Wait",
                execution_time_us: time_us,
                bytes_loaded: 0,
                bytes_stored: 0,
                flops: 0,
                bandwidth_gbps: 0.0,
                tflops: 0.0,
                count: wait_count,
            });
        }

        stats
    }

    /// Print execution statistics for the last execution (computes block op stats lazily).
    pub fn print_execution_stats(&self) {
        if self.last_kernel_stats.is_empty() && self.timings.is_empty() {
            println!("No execution stats available.");
            return;
        }

        // Compute block op stats lazily
        let sm_count = self
            .cuda_stream
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap() as usize;
        let block_op_stats = Self::compute_block_op_stats(
            &self.llir_graph,
            self.timings.last().map(|v| v.as_slice()).unwrap_or(&[]),
            &self.last_dyn_map,
            sm_count,
        );

        // Compute aggregates
        let total_bytes_loaded: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_loaded)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.bytes_loaded).sum::<usize>();
        let total_bytes_stored: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.bytes_stored)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.bytes_stored).sum::<usize>();
        let total_flops: usize = self
            .last_kernel_stats
            .iter()
            .map(|s| s.flops)
            .sum::<usize>()
            + block_op_stats.iter().map(|s| s.flops).sum::<usize>();
        let total_bytes = total_bytes_loaded + total_bytes_stored;
        let aggregate_bw = if self.last_total_time_us > 0.0 {
            (total_bytes as f64) / (self.last_total_time_us * 1e-6) / 1e9
        } else {
            0.0
        };
        let aggregate_tf = if self.last_total_time_us > 0.0 {
            (total_flops as f64) / (self.last_total_time_us * 1e-6) / 1e12
        } else {
            0.0
        };

        let peak_bw = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tf = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        // Print kernel stats
        if !self.last_kernel_stats.is_empty() {
            println!("\n=== Kernel Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "Kernel",
                "Time (us)",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(116));
            for s in &self.last_kernel_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    None,
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(116));
        }

        // Print block op stats
        if !block_op_stats.is_empty() {
            println!("\n=== Block Op Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "BlockOp",
                "Time (us)",
                "Count",
                "Loaded",
                "Stored",
                "Agg FLOPS",
                "BW (GB/s)",
                "TFLOPS",
                "MBU",
                "MFU"
            );
            println!("{}", "-".repeat(124));
            for s in &block_op_stats {
                self.print_stat_row(
                    s.name,
                    s.execution_time_us,
                    Some(s.count),
                    s.bytes_loaded,
                    s.bytes_stored,
                    s.flops,
                    s.bandwidth_gbps,
                    s.tflops,
                    peak_bw,
                    peak_tf,
                );
            }
            println!("{}", "-".repeat(124));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS", "MBU", "MFU"
        );
        println!("{}", "-".repeat(116));
        let (mbu, mfu) = match (peak_bw, peak_tf) {
            (Some(pb), Some(pt)) => (
                format!("{:.1}%", aggregate_bw / pb as f64 * 100.0),
                format!("{:.1}%", aggregate_tf / pt as f64 * 100.0),
            ),
            _ => ("-".into(), "-".into()),
        };
        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Total",
            self.last_total_time_us,
            format_size(total_bytes_loaded),
            format_size(total_bytes_stored),
            format_flops(total_flops),
            format!("{:.2}", aggregate_bw),
            format!("{:.4}", aggregate_tf),
            mbu,
            mfu
        );

        if let (Some(pb), Some(pt)) = (peak_bw, peak_tf) {
            println!("\nDevice peak: {} GB/s bandwidth, {} TFLOPS (F32)", pb, pt);
        }
        println!();
    }

    #[allow(clippy::too_many_arguments)]
    fn print_stat_row(
        &self,
        name: &str,
        time_us: f64,
        count: Option<usize>,
        loaded: usize,
        stored: usize,
        flops: usize,
        bw: f64,
        tf: f64,
        peak_bw: Option<usize>,
        peak_tf: Option<usize>,
    ) {
        let total = loaded + stored;
        let ld = if loaded > 0 {
            format_size(loaded)
        } else {
            "-".into()
        };
        let st = if stored > 0 {
            format_size(stored)
        } else {
            "-".into()
        };
        let fl = if flops > 0 {
            format_flops(flops)
        } else {
            "-".into()
        };
        let bw_s = if total > 0 {
            format!("{bw:.2}")
        } else {
            "-".into()
        };
        let tf_s = if flops > 0 {
            format!("{tf:.4}")
        } else {
            "-".into()
        };
        let mbu = peak_bw
            .filter(|_| total > 0)
            .map(|p| format!("{:.1}%", bw / p as f64 * 100.0))
            .unwrap_or("-".into());
        let mfu = peak_tf
            .filter(|_| flops > 0)
            .map(|p| format!("{:.1}%", tf / p as f64 * 100.0))
            .unwrap_or("-".into());

        match count {
            Some(c) => println!(
                "{name:<20} {time_us:>12.2} {c:>8} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
            None => println!(
                "{name:<20} {time_us:>12.2} {ld:>12} {st:>12} {fl:>12} {bw_s:>12} {tf_s:>12} {mbu:>8} {mfu:>8}"
            ),
        }
    }

    /// Flush pending timing data to the timings collection.
    fn flush_pending_timings(&mut self) {
        if self.pending_timing_data.is_empty() {
            return;
        }

        // Sync timing stream first to ensure all async copies are complete
        self.timing_stream
            .synchronize()
            .expect("Failed to sync timing stream");

        // Extract data from pinned memory pool
        let mut timing_entries = Vec::with_capacity(self.pending_timing_data.len());
        for pending in self.pending_timing_data.drain(..) {
            let timing_data = self.pinned_timing_pool[pending.timing_buffer_idx]
                .as_slice()
                .expect("Failed to read timing buffer")
                .to_vec();
            let min_start = self.pinned_start_pool[pending.start_time_idx]
                .as_slice()
                .expect("Failed to read start time buffer")
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            timing_entries.push((timing_data, min_start, pending.span_id));
        }

        // Reset pool index so buffers can be reused
        self.next_pinned_slot = 0;

        if !timing_entries.is_empty() {
            self.timings.push(timing_entries);
        }
    }

    /// Record GPU timings to an existing perfetto trace file.
    pub fn record_cuda_perfetto_trace(&mut self, perfetto_guard: PerfettoGuard) {
        // Flush any pending timing copies first
        self.flush_pending_timings();

        perfetto_guard.stop();
        let ops: Vec<Arc<Box<dyn BlockOp>>> = self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<std::collections::HashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(_, o)| o)
            .collect();
        let data = std::fs::read(&perfetto_guard.path).unwrap();
        let mut trace = tracing_perfetto_sdk_schema::Trace::decode(data.as_slice()).unwrap();
        let mut extra_packets = record_block_op_timings(&trace, &ops, &self.timings);
        extra_packets.extend(record_cuda_graph_timings(&trace, &self.cuda_graph_timings));
        trace.packet.extend(extra_packets);
        // Sort ALL packets by timestamp for proper Perfetto visualization
        trace.packet.sort_by_key(|p| p.timestamp.unwrap_or(0));
        let mut buf = Vec::with_capacity(trace.encoded_len());
        trace.encode(&mut buf).unwrap();
        std::fs::write(perfetto_guard.path, buf).unwrap();
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}

fn format_flops(flops: usize) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.2} T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.2} G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.2} M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.2} K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
    }
}

pub(crate) fn partition_marked_convex<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Result<Vec<FxHashSet<NodeIndex>>, Cycle<NodeIndex>> {
    if marked.is_empty() {
        return Ok(vec![]);
    }

    // --- Global topo order (also validates DAG) ---
    let topo = toposort(g, None)?;
    let topo_len = topo.len();

    // Map NodeIndex <-> topo position
    let mut idx_to_pos: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut pos_to_idx: Vec<NodeIndex> = Vec::with_capacity(topo_len);
    for (pos, &ni) in topo.iter().enumerate() {
        idx_to_pos.insert(ni, pos);
        pos_to_idx.push(ni);
    }

    // --- Full-graph reachability: reach[upos] contains all vpos reachable from u ---
    // (Bitset DP over topo order)
    let mut reach: Vec<FixedBitSet> = (0..topo_len)
        .map(|_| {
            let mut b = FixedBitSet::with_capacity(topo_len);
            b.grow(topo_len);
            b
        })
        .collect();

    for &u in topo.iter().rev() {
        let upos = idx_to_pos[&u];
        for v in g.neighbors_directed(u, Direction::Outgoing) {
            if let Some(&vpos) = idx_to_pos.get(&v) {
                reach[upos].insert(vpos);
                let rv = reach[vpos].clone();
                reach[upos].union_with(&rv);
            }
        }
    }

    // --- 1) Weakly-connected components in the marked-induced subgraph ---
    let components = marked_weak_components(g, marked);

    let mut results: Vec<FxHashSet<NodeIndex>> = Vec::new();

    for comp in components {
        // Component nodes in topo positions (sorted)
        let mut comp_pos: Vec<usize> = comp
            .iter()
            .filter_map(|ni| idx_to_pos.get(ni).copied())
            .collect();
        comp_pos.sort_unstable();

        // Membership: in_comp_pos bitset over topo positions
        let mut in_comp_pos = FixedBitSet::with_capacity(topo_len);
        in_comp_pos.grow(topo_len);
        for &p in &comp_pos {
            in_comp_pos.insert(p);
        }

        // Membership: in_comp_idx vec over NodeIndex::index() for component-relative DP
        let mut in_comp_idx = vec![false; g.node_bound()];
        for &n in &comp {
            in_comp_idx[n.index()] = true;
        }

        // --- Component-relative "between" witnesses (path-wise, correct) ---
        // has_comp_anc[x] == true if x has a component node as an ancestor (or is in comp)
        let mut has_comp_anc = vec![false; g.node_bound()];
        for &u in &topo {
            let mut v = in_comp_idx[u.index()];
            for p in g.neighbors_directed(u, Direction::Incoming) {
                v |= has_comp_anc[p.index()];
                if v {
                    break;
                }
            }
            has_comp_anc[u.index()] = v;
        }

        // has_comp_des[x] == true if x has a component node as a descendant (or is in comp)
        let mut has_comp_des = vec![false; g.node_bound()];
        for &u in topo.iter().rev() {
            let mut v = in_comp_idx[u.index()];
            for s in g.neighbors_directed(u, Direction::Outgoing) {
                v |= has_comp_des[s.index()];
                if v {
                    break;
                }
            }
            has_comp_des[u.index()] = v;
        }

        // --- Build witness constraints Px/Sx only for true witnesses of THIS component ---
        // Witness x is UNMARKED and lies on some path comp_node ->* x ->* comp_node.
        // For each witness x:
        //   Px(x) = {u in comp | u ->* x}
        //   Sx(x) = {v in comp | x ->* v}
        // A valid block cannot contain nodes from both Px(x) and Sx(x).
        let mut px_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut sx_map: FxHashMap<NodeIndex, FixedBitSet> = FxHashMap::default();
        let mut px_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // upos -> witnesses where upos ∈ Px
        let mut sx_witnesses: FxHashMap<usize, Vec<NodeIndex>> = FxHashMap::default(); // vpos -> witnesses where vpos ∈ Sx

        for x in g.node_indices() {
            if marked.contains(&x) {
                continue; // must be outside the block (unmarked) to be a witness
            }
            if !(has_comp_anc[x.index()] && has_comp_des[x.index()]) {
                continue; // not between this component's marked nodes
            }

            let Some(&xpos) = idx_to_pos.get(&x) else {
                continue;
            };
            // Sx = reachable-from-x ∩ component
            let mut sx = reach[xpos].clone();
            sx.intersect_with(&in_comp_pos);
            if sx.is_empty() {
                continue;
            }

            // Px = {u in comp | u can reach x}
            let mut px = FixedBitSet::with_capacity(topo_len);
            px.grow(topo_len);
            for &upos in &comp_pos {
                if reach[upos].contains(xpos) {
                    px.insert(upos);
                }
            }
            if px.is_empty() {
                continue;
            }

            px_map.insert(x, px.clone());
            sx_map.insert(x, sx.clone());

            for upos in px.ones() {
                px_witnesses.entry(upos).or_default().push(x);
            }
            for vpos in sx.ones() {
                sx_witnesses.entry(vpos).or_default().push(x);
            }
        }

        // --- 3) Deterministic topo sweep partition within this component ---
        let mut current: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut block_bits = FixedBitSet::with_capacity(topo_len);
        block_bits.grow(topo_len);

        for &p in &comp_pos {
            let violates = would_violate(
                p,
                &block_bits,
                &px_witnesses,
                &sx_witnesses,
                &px_map,
                &sx_map,
            );

            if violates && !current.is_empty() {
                results.push(std::mem::take(&mut current));
                block_bits.clear(); // keeps length
            }

            let ni = pos_to_idx[p];
            current.insert(ni);
            block_bits.insert(p);
        }

        if !current.is_empty() {
            results.push(current);
        }
    }

    Ok(results)
}

/// Deterministic “contiguous marked” components: weakly-connected in the marked-induced subgraph.
fn marked_weak_components<T, E>(
    g: &StableGraph<T, E, Directed>,
    marked: &FxHashSet<NodeIndex>,
) -> Vec<Vec<NodeIndex>> {
    let mut seen: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut comps: Vec<Vec<NodeIndex>> = Vec::new();

    for start in g.node_indices() {
        if !marked.contains(&start) || seen.contains(&start) {
            continue;
        }

        let mut q = VecDeque::new();
        q.push_back(start);
        seen.insert(start);

        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for v in g.neighbors_undirected(u) {
                if marked.contains(&v) && seen.insert(v) {
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }

    comps
}

fn would_violate(
    p: usize,
    block_bits: &FixedBitSet,
    px_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    sx_witnesses: &FxHashMap<usize, Vec<NodeIndex>>,
    px_map: &FxHashMap<NodeIndex, FixedBitSet>,
    sx_map: &FxHashMap<NodeIndex, FixedBitSet>,
) -> bool {
    // If p ∈ Px(x), block cannot contain any node in Sx(x)
    if let Some(ws) = px_witnesses.get(&p) {
        for &x in ws {
            if let Some(sx) = sx_map.get(&x)
                && intersects(block_bits, sx)
            {
                return true;
            }
        }
    }

    // If p ∈ Sx(x), block cannot contain any node in Px(x)
    if let Some(ws) = sx_witnesses.get(&p) {
        for &x in ws {
            if let Some(px) = px_map.get(&x)
                && intersects(block_bits, px)
            {
                return true;
            }
        }
    }

    false
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let mut tmp = a.clone();
    tmp.intersect_with(b);
    // Note: is_empty() checks if length is 0, not if there are no bits set
    // Use count_ones() to check if there are any set bits after intersection
    tmp.count_ones(..) > 0
}
