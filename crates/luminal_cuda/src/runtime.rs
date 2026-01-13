use crate::{block::*, kernel::KernelOp};
use cudarc::driver::{
    sys::CUevent_flags, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use luminal::hlir::*;
use luminal::prelude::{
    petgraph::{
        algo::{toposort, Cycle},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
        Directed, Direction,
    },
    *,
};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{collections::VecDeque, fmt::Debug, fs::File, sync::Arc, time::Duration};
use tracing::{field, span, Level};
use uuid::Uuid;

pub enum CudaInput {
    Buffer(CudaSlice<u8>),
    Ptr(u64),
}

#[derive(Clone)]
enum ExecutableKernel {
    Megakernel {
        interpreter: CudaFunction,
        interpreter_constants: FxHashMap<char, CudaSlice<u8>>,
        n_barriers: Expression,
        work_queue: TaskQueue,
        node_to_task_index: FxHashMap<NodeIndex, usize>,
    },
    Kernel {
        kernel: CudaFunction,
        launch_grid: (Expression, Expression, Expression),
        launch_threadblock: (Expression, Expression, Expression),
        shared_mem: Expression,
        inputs: Vec<NodeIndex>,
        output: NodeIndex,
        constants: FxHashMap<char, CudaSlice<u8>>,
        // Profiling metrics
        kernel_name: &'static str,
        bytes_loaded: Expression,
        bytes_stored: Expression,
        flops: Expression,
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
}

/// Aggregated execution statistics
#[derive(Debug, Default, Clone)]
pub struct ExecutionStats {
    pub kernel_stats: Vec<KernelStats>,
    pub block_op_stats: Vec<BlockOpStats>,
    pub total_time_us: f64,
    /// Total bytes loaded across all ops
    pub total_bytes_loaded: usize,
    /// Total bytes stored across all ops
    pub total_bytes_stored: usize,
    /// Total floating point operations across all ops
    pub total_flops: usize,
    /// Aggregate bandwidth in GB/s (total bytes / total time)
    pub aggregate_bandwidth_gbps: f64,
    /// Aggregate compute in TFLOPS (total flops / total time)
    pub aggregate_tflops: f64,
}

impl Drop for ExecutableKernel {
    fn drop(&mut self) {
        match self {
            ExecutableKernel::Megakernel {
                interpreter_constants,
                ..
            } => {
                // Prevent Drop of CudaSlice<u8> (likely calls cuMemFree).
                let m = std::mem::take(interpreter_constants);
                for (_k, v) in m {
                    std::mem::forget(v);
                }
            }
            ExecutableKernel::Kernel { constants, .. } => {
                let m = std::mem::take(constants);
                for (_k, v) in m {
                    std::mem::forget(v);
                }
            }
        }
    }
}

impl Debug for ExecutableKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Megakernel { work_queue, .. } => format!("Megakernel ({})", work_queue.len()),
                Self::Kernel { .. } => "Kernel".to_string(),
            }
        )
    }
}

pub struct CudaRuntime {
    pub hlir_buffers: FxHashMap<NodeIndex, CudaInput>,
    pub buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub llir_graph: luminal::graph::LLIRGraph,
    cuda_stream: Arc<CudaStream>,
    exec_graph: StableGraph<ExecutableKernel, (), Directed>,
    node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    pub(crate) timings: Vec<(Vec<SMEvent>, u64, Uuid)>,
    last_dyn_map: FxHashMap<char, usize>,
    intermediate_buffer_dims: FxHashSet<char>,
    /// Statistics from the last execution
    pub last_execution_stats: ExecutionStats,
}

impl CudaRuntime {
    #[tracing::instrument(skip_all)]
    pub fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        let f = File::open(file_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for node in cx.graph.node_indices() {
            if let Some(Input { label, .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                if let Ok(tensor) = st.tensor(label) {
                    match tensor.dtype() {
                        safetensors::Dtype::F32 => {
                            let bytes = tensor.data();
                            let f32s: &[f32] = bytemuck::cast_slice(bytes);
                            let dev = f32s.to_cuda_input(&self.cuda_stream);
                            self.hlir_buffers.insert(node, dev);
                        }
                        dtype => unimplemented!("{dtype} loading not supported yet"),
                    }
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaInput) {
        self.hlir_buffers
            .insert(id.to_id(), data.to_cuda_input(&self.cuda_stream));
    }

    #[tracing::instrument(skip_all)]
    pub fn get_f32(&self, id: impl ToId) -> Vec<f32> {
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
        self.cuda_stream
            .memcpy_dtov(
                self.buffers
                    .get(&data_id)
                    .expect("Cannot find tensor in runtime!"),
            )
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect_vec()
    }

    fn register_buffer(&mut self, llir_node: NodeIndex, ptr: u64) {
        // Remap pointers in work queue
        if let Some(ExecutableKernel::Megakernel {
            work_queue,
            node_to_task_index,
            ..
        }) = self
            .node_to_exec
            .get(&llir_node)
            .and_then(|n| self.exec_graph.node_weight_mut(*n))
        {
            if self.llir_graph[llir_node].to_op::<Input>().is_none() {
                work_queue.set_out_ptr(node_to_task_index[&llir_node], ptr as *mut f32);
            }
        }
        for edge in self
            .llir_graph
            .edges_directed(llir_node, Direction::Outgoing)
        {
            let dest = edge.target();
            let n_input = self
                .llir_graph
                .edges_directed(dest, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .position(|e| e.id() == edge.id())
                .unwrap();
            if let Some(ExecutableKernel::Megakernel {
                work_queue,
                node_to_task_index,
                ..
            }) = self
                .node_to_exec
                .get(&dest)
                .and_then(|n| self.exec_graph.node_weight_mut(*n))
            {
                work_queue.set_source_ptr(node_to_task_index[&dest], n_input, ptr as *const f32);
            }
        }
    }

    #[tracing::instrument(skip_all)]
    fn allocate_intermediate_buffers(&mut self, dyn_dims: &FxHashMap<char, usize>) {
        self.intermediate_buffer_dims.clear();
        for node in self.llir_graph.node_indices().collect_vec() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn BlockOp>() {
                self.intermediate_buffer_dims
                    .extend(op.output_size().dyn_vars());
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(op.output_size().exec(dyn_dims).unwrap() * size_of::<f32>())
                        .unwrap(),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.register_buffer(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn KernelOp>() {
                self.intermediate_buffer_dims
                    .extend(op.output_size().dyn_vars());
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(op.output_size().exec(dyn_dims).unwrap() * size_of::<f32>())
                        .unwrap(),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.register_buffer(node, ptr);
            }
        }
    }
}

pub trait ToCudaInput {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput;
}

impl ToCudaInput for Vec<f32> {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .memcpy_stod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl ToCudaInput for &[f32] {
    fn to_cuda_input(self, stream: &Arc<CudaStream>) -> CudaInput {
        CudaInput::Buffer(
            stream
                .memcpy_stod(unsafe {
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
                .memcpy_stod(unsafe {
                    std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
                })
                .unwrap(),
        )
    }
}

impl Runtime for CudaRuntime {
    type Ops = (crate::logical::Ops, crate::kernel::Ops, crate::block::Ops);
    type CompileArg = Arc<CudaStream>;
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize(stream: Self::CompileArg) -> Self {
        Self {
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            cuda_stream: stream,
            llir_graph: StableGraph::default(),
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            timings: vec![],
            last_dyn_map: FxHashMap::default(),
            intermediate_buffer_dims: FxHashSet::default(),
            last_execution_stats: ExecutionStats::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.exec_graph.clear();
        let block_ops_in_graph = llir_graph
            .node_indices()
            .filter(|n| llir_graph[*n].to_dialect::<dyn BlockOp>().is_some())
            .collect::<FxHashSet<_>>();
        let block_subgraphs = partition_marked_convex(llir_graph, &block_ops_in_graph).unwrap();

        // Add megakernels
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();
        for subgraph in block_subgraphs {
            let (interpreter, constants, n_barriers, tasks, node_to_task_index) =
                make_megakernel_from_llir_graph(llir_graph, &subgraph, &self.cuda_stream);
            let exec_node = exec_graph.add_node(ExecutableKernel::Megakernel {
                interpreter,
                interpreter_constants: constants,
                n_barriers,
                work_queue: tasks,
                node_to_task_index,
            });
            for node in subgraph {
                node_to_exec.insert(node, exec_node);
            }
        }
        // Add kernels
        for kernel in llir_graph.node_indices() {
            if let Some(kernel_op) = llir_graph[kernel].to_dialect::<dyn KernelOp>() {
                let (kernel_function, _, _, grid, tb, shared_mem, constants) =
                    kernel_op.compile(&self.cuda_stream);
                let inputs = llir_graph
                    .edges_directed(kernel, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect_vec();
                node_to_exec.insert(
                    kernel,
                    exec_graph.add_node(ExecutableKernel::Kernel {
                        kernel: kernel_function,
                        launch_grid: grid,
                        launch_threadblock: tb,
                        inputs,
                        output: kernel,
                        shared_mem,
                        constants,
                        kernel_name: kernel_op.kernel_name(),
                        bytes_loaded: kernel_op.bytes_loaded(),
                        bytes_stored: kernel_op.bytes_stored(),
                        flops: kernel_op.flops(),
                    }),
                );
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
        self.timings.clear();

        let duration = start.elapsed();
        let stats = &self.last_execution_stats;

        // Compute MBU and MFU from execution stats
        let peak_bandwidth_gbps = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tflops = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        let mbu =
            peak_bandwidth_gbps.map(|peak_bw| stats.aggregate_bandwidth_gbps / peak_bw as f64);
        let mfu = peak_tflops.map(|peak_tf| stats.aggregate_tflops / peak_tf as f64);

        let duration_str = pretty_duration::pretty_duration(&duration, None);
        let mbu_str = mbu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let mfu_str = mfu.map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0));
        let display = format!("{duration_str} | MBU: {mbu_str} | MFU: {mfu_str}");

        (duration, display)
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        if self.buffers.is_empty()
            || dyn_map.len() != self.last_dyn_map.len()
            || dyn_map
                .iter()
                .filter(|(d, _)| self.intermediate_buffer_dims.contains(*d))
                .any(|(d, v)| self.last_dyn_map.get(d).map(|n| *n != *v).unwrap_or(true))
        {
            self.last_dyn_map = dyn_map.clone();
            self.allocate_intermediate_buffers(dyn_map);
        }
        let mut llir_to_hlir: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
        for (hlir_node, llir_node) in self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_op::<Input>().map(|op| (op.node, n)))
            .collect_vec()
        {
            llir_to_hlir.insert(llir_node, NodeIndex::new(hlir_node));
            let ptr = match &self.hlir_buffers[&NodeIndex::new(hlir_node)] {
                CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                CudaInput::Ptr(p) => *p,
            };
            self.register_buffer(llir_node, ptr);
        }
        let mut timings = vec![];
        let mut kernel_stats = Vec::new();
        let total_start = std::time::Instant::now();
        for exec_node in toposort(&self.exec_graph, None).unwrap() {
            match &mut self.exec_graph[exec_node] {
                ExecutableKernel::Kernel {
                    kernel,
                    launch_grid,
                    launch_threadblock,
                    inputs,
                    output,
                    shared_mem,
                    constants,
                    kernel_name,
                    bytes_loaded,
                    bytes_stored,
                    flops,
                } => {
                    for (dyn_dim, val) in dyn_map {
                        if let Some(global) = constants.get_mut(dyn_dim) {
                            let mut view = global.as_view_mut();
                            let mut symbol = unsafe { view.transmute_mut::<i32>(1).unwrap() };
                            self.cuda_stream
                                .memcpy_htod(&[*val as i32], &mut symbol)
                                .unwrap();
                        }
                    }
                    let cfg = LaunchConfig {
                        grid_dim: (
                            launch_grid.0.exec(dyn_map).unwrap() as u32,
                            launch_grid.1.exec(dyn_map).unwrap() as u32,
                            launch_grid.2.exec(dyn_map).unwrap() as u32,
                        ),
                        block_dim: (
                            launch_threadblock.0.exec(dyn_map).unwrap() as u32,
                            launch_threadblock.1.exec(dyn_map).unwrap() as u32,
                            launch_threadblock.2.exec(dyn_map).unwrap() as u32,
                        ),
                        shared_mem_bytes: shared_mem.exec(dyn_map).unwrap() as u32,
                    };
                    let mut ptrs = vec![];
                    for inp in inputs {
                        if let Some(buf) = self.buffers.get(inp) {
                            ptrs.push(buf.device_ptr(&self.cuda_stream).0);
                        } else {
                            ptrs.push(match &self.hlir_buffers[&llir_to_hlir[inp]] {
                                CudaInput::Buffer(buf) => buf.device_ptr(&self.cuda_stream).0,
                                CudaInput::Ptr(p) => *p,
                            });
                        }
                    }
                    let mut lb = self.cuda_stream.launch_builder(kernel);
                    lb.arg(&self.buffers[output]);
                    for ptr in &ptrs {
                        lb.arg(ptr);
                    }
                    let span = span!(Level::INFO, "kernel", kernel = field::Empty);
                    span.record("kernel", &kernel_name);
                    let _entered = span.enter();

                    // Use CUDA events for accurate GPU-side timing
                    let start_event = self
                        .cuda_stream
                        .context()
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    let end_event = self
                        .cuda_stream
                        .context()
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();

                    start_event.record(&self.cuda_stream).unwrap();
                    unsafe { lb.launch(cfg) }.unwrap();
                    end_event.record(&self.cuda_stream).unwrap();

                    // elapsed_ms synchronizes internally
                    let kernel_time_ms = start_event.elapsed_ms(&end_event).unwrap();
                    let kernel_time_us = kernel_time_ms as f64 * 1000.0;

                    // Calculate metrics
                    let loaded = bytes_loaded.exec(dyn_map).unwrap_or(0);
                    let stored = bytes_stored.exec(dyn_map).unwrap_or(0);
                    let flop_count = flops.exec(dyn_map).unwrap_or(0);

                    // Calculate bandwidth (GB/s) and compute (TFLOPS)
                    // Total memory traffic = bytes loaded + bytes stored
                    let total_bytes = loaded + stored;
                    let bandwidth_gbps = (total_bytes as f64) / (kernel_time_us * 1e-6) / 1e9;
                    let tflops = (flop_count as f64) / (kernel_time_us * 1e-6) / 1e12;

                    kernel_stats.push(KernelStats {
                        name: *kernel_name,
                        execution_time_us: kernel_time_us,
                        bytes_loaded: loaded,
                        bytes_stored: stored,
                        flops: flop_count,
                        bandwidth_gbps,
                        tflops,
                    });

                    drop(_entered);
                    drop(span);
                }
                ExecutableKernel::Megakernel {
                    interpreter,
                    interpreter_constants,
                    n_barriers,
                    work_queue,
                    ..
                } => {
                    let sm_count = self
                        .cuda_stream
                        .context()
                        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                        .unwrap();
                    let span = span!(Level::INFO, "megakernel_setup");
                    let _entered = span.enter();
                    // Upload queue, barriers and program counter
                    let d_barriers = self
                        .cuda_stream
                        .alloc_zeros::<i32>(n_barriers.exec(dyn_map).unwrap())
                        .unwrap();
                    let d_tasks = self.cuda_stream.memcpy_stod(work_queue.as_slice()).unwrap();
                    let d_head = self.cuda_stream.memcpy_stod(&[0i32]).unwrap();
                    let queue_lock = self.cuda_stream.memcpy_stod(&[0i32]).unwrap();
                    // Set up timing buffer storing (start_u64, stop_u64, event_i32) tuples
                    // for up to 1000 events per SM
                    let timing_buffer = self
                        .cuda_stream
                        .alloc_zeros::<SMEvent>(sm_count as usize * 1000)
                        .unwrap();
                    let start_time = self
                        .cuda_stream
                        .alloc_zeros::<u64>(sm_count as usize)
                        .unwrap();

                    // Set up dyn dims
                    for (dyn_dim, val) in dyn_map {
                        if let Some(global) = interpreter_constants.get_mut(dyn_dim) {
                            let mut view = global.as_view_mut();
                            let mut symbol = unsafe { view.transmute_mut::<i32>(1).unwrap() };
                            self.cuda_stream
                                .memcpy_htod(&[*val as i32], &mut symbol)
                                .unwrap();
                        }
                    }

                    let shared_mem_max = self
                        .cuda_stream
                        .context()
                        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
                        .unwrap();

                    interpreter.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_max / 2, // Half shared mem, half L2
                    ).unwrap();

                    // Launch kernel
                    let cfg = LaunchConfig {
                        grid_dim: (sm_count as u32, 1, 1), // One block per SM
                        block_dim: (256, 1, 1),            // 1024 threads (32 warps) per block
                        shared_mem_bytes: (shared_mem_max / 2) as u32,
                    };
                    let mut lb = self.cuda_stream.launch_builder(interpreter);
                    let n_tasks = work_queue.len() as i32;
                    lb.arg(&d_tasks);
                    lb.arg(&n_tasks);
                    lb.arg(&d_head);
                    lb.arg(&d_barriers);
                    lb.arg(&queue_lock);
                    lb.arg(&timing_buffer);
                    lb.arg(&start_time);
                    drop(_entered);
                    drop(span);
                    let mk_span_id = Uuid::new_v4();
                    let span = span!(Level::INFO, "megakernel", id = field::Empty);
                    // Record fields after span creation to work around tracing-perfetto-sdk-layer sync span bug
                    span.record("id", format!("{}", mk_span_id).as_str());
                    let _entered = span.enter();
                    unsafe { lb.launch(cfg) }.unwrap();
                    self.cuda_stream.synchronize().unwrap();
                    drop(_entered);
                    drop(span);
                    timings.push((
                        self.cuda_stream.memcpy_dtov(&timing_buffer).unwrap(),
                        self.cuda_stream
                            .memcpy_dtov(&start_time)
                            .unwrap()
                            .into_iter()
                            .min()
                            .unwrap(),
                        mk_span_id,
                    ));
                }
            }
        }
        self.timings.extend(timings.clone());

        {
            let span = span!(Level::TRACE, "timings");
            let _entered = span.enter();
            // Compute block op stats from SMEvent timings
            let sm_count = self
            .cuda_stream
            .context()
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap() as usize;
            let block_op_stats =
                Self::compute_block_op_stats(&self.llir_graph, &timings, dyn_map, sm_count);

            // Compute aggregate stats from kernel and block op stats
            let total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;

            let total_bytes_loaded: usize =
                kernel_stats.iter().map(|s| s.bytes_loaded).sum::<usize>()
                    + block_op_stats.iter().map(|s| s.bytes_loaded).sum::<usize>();
            let total_bytes_stored: usize =
                kernel_stats.iter().map(|s| s.bytes_stored).sum::<usize>()
                    + block_op_stats.iter().map(|s| s.bytes_stored).sum::<usize>();
            let total_flops: usize = kernel_stats.iter().map(|s| s.flops).sum::<usize>()
                + block_op_stats.iter().map(|s| s.flops).sum::<usize>();

            let total_bytes = total_bytes_loaded + total_bytes_stored;
            let aggregate_bandwidth_gbps = if total_time_us > 0.0 {
                (total_bytes as f64) / (total_time_us * 1e-6) / 1e9
            } else {
                0.0
            };
            let aggregate_tflops = if total_time_us > 0.0 {
                (total_flops as f64) / (total_time_us * 1e-6) / 1e12
            } else {
                0.0
            };

            // Store execution stats
            self.last_execution_stats = ExecutionStats {
                kernel_stats,
                block_op_stats,
                total_time_us,
                total_bytes_loaded,
                total_bytes_stored,
                total_flops,
                aggregate_bandwidth_gbps,
                aggregate_tflops,
            };
        }
    }
}

impl CudaRuntime {
    fn compute_block_op_stats(
        llir_graph: &LLIRGraph,
        timings: &[(Vec<SMEvent>, u64, Uuid)],
        dyn_map: &FxHashMap<char, usize>,
        sm_count: usize,
    ) -> Vec<BlockOpStats> {
        use std::collections::HashMap;

        // Get unique op names (same order as in interpreter)
        let op_names: Vec<&'static str> = llir_graph
            .node_indices()
            .filter_map(|n| llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<HashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(n, _)| n)
            .collect();

        if op_names.is_empty() {
            return vec![];
        }

        // Build map from op_name to index for event decoding
        let op_name_to_idx: HashMap<&'static str, usize> =
            op_names.iter().enumerate().map(|(i, n)| (*n, i)).collect();

        // Sum up bytes_loaded, bytes_stored, flops across ALL instances of each op type
        let mut op_bytes_loaded: Vec<usize> = vec![0; op_names.len()];
        let mut op_bytes_stored: Vec<usize> = vec![0; op_names.len()];
        let mut op_flops: Vec<usize> = vec![0; op_names.len()];

        for node in llir_graph.node_indices() {
            if let Some(op) = llir_graph[node].to_dialect::<dyn BlockOp>() {
                if let Some(&idx) = op_name_to_idx.get(op.op_name()) {
                    let flops_expr = op.flops();
                    let flops_val = flops_expr.exec(dyn_map).unwrap();
                    op_bytes_loaded[idx] += op.bytes_loaded().exec(dyn_map).unwrap();
                    op_bytes_stored[idx] += op.bytes_stored().exec(dyn_map).unwrap();
                    op_flops[idx] += flops_val;
                }
            }
        }

        // Aggregate timing per op type across all SMs
        // event codes: 0=Issue, 1=Wait, 2+=BlockOps (index in ops list)
        let mut op_times_ns: Vec<u64> = vec![0; op_names.len()];

        for (sm_timings, _start_time, _) in timings {
            for sm_chunk in sm_timings.chunks(1000) {
                for event in sm_chunk.iter() {
                    if event.start == 0 {
                        break; // No more events recorded for this SM
                    }
                    // event >= 2 means it's a block op (0=Issue, 1=Wait)
                    if event.event >= 2 {
                        let op_idx = (event.event - 2) as usize;
                        if op_idx < op_names.len() {
                            let stop = if event.stop == 0 {
                                event.start
                            } else {
                                event.stop
                            };
                            let duration = stop.saturating_sub(event.start);
                            op_times_ns[op_idx] += duration;
                        }
                    }
                }
            }
        }

        // Compute bandwidth and TFLOPS from aggregated metrics
        // Divide total SM-time by sm_count to get wall-clock time
        op_names
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
                }
            })
            .collect()
    }

    /// Print execution statistics for the last execution.
    /// Shows bandwidth and compute utilization for each kernel and block op.
    pub fn print_execution_stats(&self) {
        let stats = &self.last_execution_stats;
        if stats.kernel_stats.is_empty() && stats.block_op_stats.is_empty() {
            println!("No execution stats available.");
            return;
        }

        // Get device peak performance
        let peak_bandwidth_gbps = crate::cuda_bandwidth_gbps(self.cuda_stream.context());
        let peak_tflops = crate::cuda_compute_f32_tflops(self.cuda_stream.context());

        // Print kernel stats if any
        if !stats.kernel_stats.is_empty() {
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

            for stat in &stats.kernel_stats {
                self.print_stat_row(
                    stat.name,
                    stat.execution_time_us,
                    stat.bytes_loaded,
                    stat.bytes_stored,
                    stat.flops,
                    stat.bandwidth_gbps,
                    stat.tflops,
                    peak_bandwidth_gbps,
                    peak_tflops,
                );
            }

            println!("{}", "-".repeat(116));
        }

        // Print block op stats if any
        if !stats.block_op_stats.is_empty() {
            println!("\n=== Block Op Execution Statistics ===\n");
            println!(
                "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
                "BlockOp",
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

            for stat in &stats.block_op_stats {
                self.print_stat_row(
                    stat.name,
                    stat.execution_time_us,
                    stat.bytes_loaded,
                    stat.bytes_stored,
                    stat.flops,
                    stat.bandwidth_gbps,
                    stat.tflops,
                    peak_bandwidth_gbps,
                    peak_tflops,
                );
            }

            println!("{}", "-".repeat(116));
        }

        // Print aggregate stats
        println!("\n=== Aggregate Statistics ===\n");
        println!(
            "{:<20} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "", "Time (us)", "Loaded", "Stored", "Agg FLOPS", "BW (GB/s)", "TFLOPS", "MBU", "MFU"
        );
        println!("{}", "-".repeat(116));

        let (mbu_str, mfu_str) =
            if let (Some(peak_bw), Some(peak_tf)) = (peak_bandwidth_gbps, peak_tflops) {
                let mbu = (stats.aggregate_bandwidth_gbps / peak_bw as f64) * 100.0;
                let mfu = (stats.aggregate_tflops / peak_tf as f64) * 100.0;
                (format!("{:.1}%", mbu), format!("{:.1}%", mfu))
            } else {
                ("-".to_string(), "-".to_string())
            };

        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Total",
            stats.total_time_us,
            format_size(stats.total_bytes_loaded),
            format_size(stats.total_bytes_stored),
            format_flops(stats.total_flops),
            format!("{:.2}", stats.aggregate_bandwidth_gbps),
            format!("{:.4}", stats.aggregate_tflops),
            mbu_str,
            mfu_str
        );

        // Print device info
        if let (Some(peak_bw), Some(peak_tf)) = (peak_bandwidth_gbps, peak_tflops) {
            println!(
                "\nDevice peak: {} GB/s bandwidth, {} TFLOPS (F32)",
                peak_bw, peak_tf
            );
        }
        println!();
    }

    fn print_stat_row(
        &self,
        name: &str,
        execution_time_us: f64,
        bytes_loaded: usize,
        bytes_stored: usize,
        flops: usize,
        bandwidth_gbps: f64,
        tflops: f64,
        peak_bandwidth_gbps: Option<usize>,
        peak_tflops: Option<usize>,
    ) {
        let total_bytes = bytes_loaded + bytes_stored;

        // Show "-" if bytes are 0 (not specified)
        let loaded_str = if bytes_loaded > 0 {
            format_size(bytes_loaded)
        } else {
            "-".to_string()
        };
        let stored_str = if bytes_stored > 0 {
            format_size(bytes_stored)
        } else {
            "-".to_string()
        };
        let flops_str = if flops > 0 {
            format_flops(flops)
        } else {
            "-".to_string()
        };
        let bw_str = if total_bytes > 0 {
            format!("{:.2}", bandwidth_gbps)
        } else {
            "-".to_string()
        };
        let tflops_str = if flops > 0 {
            format!("{:.4}", tflops)
        } else {
            "-".to_string()
        };

        // Calculate MBU (Memory Bandwidth Utilization) and MFU (Model FLOPS Utilization)
        let mbu_str = if let Some(peak_bw) = peak_bandwidth_gbps {
            if total_bytes > 0 {
                let mbu = (bandwidth_gbps / peak_bw as f64) * 100.0;
                format!("{:.1}%", mbu)
            } else {
                "-".to_string()
            }
        } else {
            "-".to_string()
        };

        let mfu_str = if let Some(peak_tf) = peak_tflops {
            if flops > 0 {
                let mfu = (tflops / peak_tf as f64) * 100.0;
                format!("{:.1}%", mfu)
            } else {
                "-".to_string()
            }
        } else {
            "-".to_string()
        };

        println!(
            "{:<20} {:>12.2} {:>12} {:>12} {:>12} {:>12} {:>12} {:>8} {:>8}",
            name,
            execution_time_us,
            loaded_str,
            stored_str,
            flops_str,
            bw_str,
            tflops_str,
            mbu_str,
            mfu_str
        );
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

fn partition_marked_convex<T, E>(
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
            if let Some(sx) = sx_map.get(&x) {
                if intersects(block_bits, sx) {
                    return true;
                }
            }
        }
    }

    // If p ∈ Sx(x), block cannot contain any node in Px(x)
    if let Some(ws) = sx_witnesses.get(&p) {
        for &x in ws {
            if let Some(px) = px_map.get(&x) {
                if intersects(block_bits, px) {
                    return true;
                }
            }
        }
    }

    false
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    let mut tmp = a.clone();
    tmp.intersect_with(b);
    !tmp.is_empty()
}
