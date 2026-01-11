use crate::{block::BlockOp, kernel::KernelOp};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
        LaunchConfig, PushKernelArg, ValidAsZeroBits,
    },
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use luminal::prelude::{
    petgraph::{
        algo::{toposort, Cycle},
        prelude::StableGraph,
        visit::{EdgeRef, NodeIndexable},
        Directed, Direction,
    },
    *,
};
use luminal::{hlir::*, shape::flatten_z_strides};
use memmap2::MmapOptions;
use prost::Message as _;
use safetensors::SafeTensors;
use std::{
    collections::{HashMap, VecDeque},
    ffi::c_void,
    fmt::Debug,
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    io::Read,
    iter::once,
    ptr::{null, null_mut},
    sync::Arc,
    time::Duration,
};
use tracing::{span, Level};
use tracing_perfetto_sdk_schema::{
    self as schema, trace_packet, track_descriptor, track_event, TrackEvent,
};

#[allow(dead_code)]
pub enum CustomState {
    DebugBuffers(FxHashMap<(usize, String), &'static mut [f32]>),
    KVCache(
        Vec<(
            cudarc::driver::CudaSlice<f32>,
            cudarc::driver::CudaSlice<f32>,
        )>,
    ),
}

#[derive(Clone)]
pub enum ExecutableKernel {
    Megakernel {
        interpreter: CudaFunction,
        interpreter_constants: FxHashMap<char, CudaSlice<u8>>,
        n_barriers: Expression,
        work_queue: Vec<Task>,
        node_to_task_index: FxHashMap<NodeIndex, usize>,
        module: Arc<CudaModule>,
    },
    Kernel {
        kernel: CudaFunction,
        code: String,
        launch_grid: (Expression, Expression, Expression),
        launch_threadblock: (Expression, Expression, Expression),
        shared_mem: Expression,
        inputs: Vec<NodeIndex>,
        output: NodeIndex,
        constants: FxHashMap<char, CudaSlice<u8>>,
        module: Arc<CudaModule>,
    },
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
    pub hlir_buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub buffers: FxHashMap<NodeIndex, CudaSlice<u8>>,
    pub llir_graph: luminal::graph::LLIRGraph,
    cuda_stream: Arc<CudaStream>,
    cuda_context: Arc<CudaContext>,
    pub custom_state: FxHashMap<String, CustomState>,
    exec_graph: StableGraph<ExecutableKernel, (), Directed>,
    node_to_exec: FxHashMap<NodeIndex, NodeIndex>,
    timings: Vec<(Vec<SMEvent>, u64)>,
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
                            let dev = f32s.to_cuda_buffer(&self.cuda_context, &self.cuda_stream);
                            self.hlir_buffers.insert(node, dev);
                        }
                        dtype => unimplemented!("{dtype} loading not supported yet"),
                    }
                }
            }
        }
    }

    pub fn set_data(&mut self, id: impl ToId, data: impl ToCudaBuffer) {
        self.hlir_buffers.insert(
            id.to_id(),
            data.to_cuda_buffer(&self.cuda_context, &self.cuda_stream),
        );
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
                work_queue[node_to_task_index[&llir_node]].out_ptr = ptr as *mut f32;
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
                work_queue[node_to_task_index[&dest]].source_ptrs[n_input] = ptr as *const f32;
            }
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn allocate_intermediate_buffers(&mut self, dyn_dims: &FxHashMap<char, usize>) {
        for node in self.llir_graph.node_indices().collect_vec() {
            if self.llir_graph[node].to_op::<Input>().is_some() {
                continue;
            }
            if let Some(op) = self.llir_graph[node].to_dialect::<dyn BlockOp>() {
                self.buffers.insert(
                    node,
                    self.cuda_stream
                        .alloc_zeros(op.output_size().exec(dyn_dims).unwrap() * size_of::<f32>())
                        .unwrap(),
                );
                let ptr = self.buffers[&node].device_ptr(&self.cuda_stream).0;
                self.register_buffer(node, ptr);
            } else if let Some(op) = self.llir_graph[node].to_dialect::<dyn KernelOp>() {
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

    pub fn record_cuda_perfetto_trace(&self, file_path: impl AsRef<std::path::Path>) {
        let ops = self
            .llir_graph
            .node_indices()
            .filter_map(|n| self.llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<HashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(_, o)| o)
            .collect_vec();
        let data = std::fs::read(&file_path).unwrap();
        let mut trace = tracing_perfetto_sdk_schema::Trace::decode(data.as_slice()).unwrap();

        let host_start_times: Vec<(u64, u32)> = trace
            .packet
            .iter()
            .filter_map(|p| match &p.data {
                Some(tracing_perfetto_sdk_schema::trace_packet::Data::TrackEvent(TrackEvent {
                    name_field: Some(tracing_perfetto_sdk_schema::track_event::NameField::Name(s)),
                    r#type: ty,
                    ..
                })) if s == "megakernel"
                    && *ty
                        == Some(
                            tracing_perfetto_sdk_schema::track_event::Type::SliceBegin as i32,
                        ) =>
                {
                    Some((p.timestamp?, p.timestamp_clock_id?))
                }
                _ => None,
            })
            .sorted_by_key(|i| *i)
            .collect_vec();
        let mut extra_packets = Vec::new();
        for (run, (device_timings, device_start_time)) in self.timings.iter().enumerate() {
            let (host_time, host_clock_id) = host_start_times[run];
            for (sm, sm_timings) in device_timings.chunks(1000).enumerate() {
                let mut builder = ManualTrackBuilder::new(sm as u32, host_time, host_clock_id);
                for n_op in 0..sm_timings.len() - 1 {
                    let op = sm_timings[n_op].event as usize;
                    let op_label = if op == 0 {
                        "Issue".to_string()
                    } else if op == 1 {
                        "Wait".to_string()
                    } else {
                        ops[op - 2].op_name().to_string()
                    };
                    if sm_timings[n_op + 1].start == 0 {
                        break;
                    }
                    builder.push_slice(
                        &op_label,
                        sm_timings[n_op].start - *device_start_time,
                        sm_timings[n_op + 1].start - *device_start_time,
                        host_time,
                        host_clock_id,
                    );
                }
                extra_packets.extend(builder.into_packets());
            }
        }
        trace.packet.extend(extra_packets);
        let mut buf = Vec::with_capacity(trace.encoded_len());
        trace.encode(&mut buf).unwrap();
        std::fs::write(file_path, buf).unwrap();
    }
}

pub trait ToCudaBuffer {
    fn to_cuda_buffer(&self, ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> CudaSlice<u8>;
}

impl ToCudaBuffer for Vec<f32> {
    fn to_cuda_buffer(&self, _: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> CudaSlice<u8> {
        stream
            .memcpy_stod(unsafe {
                std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
            })
            .unwrap()
    }
}

impl ToCudaBuffer for &[f32] {
    fn to_cuda_buffer(&self, _: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> CudaSlice<u8> {
        stream
            .memcpy_stod(unsafe {
                std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
            })
            .unwrap()
    }
}

impl ToCudaBuffer for Vec<i32> {
    fn to_cuda_buffer(&self, _: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> CudaSlice<u8> {
        stream
            .memcpy_stod(unsafe {
                std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4)
            })
            .unwrap()
    }
}

impl Runtime for CudaRuntime {
    type Ops = (crate::logical::Ops, crate::kernel::Ops, crate::block::Ops);
    type CompileArg = (
        Arc<CudaContext>,
        Arc<CudaStream>,
        FxHashMap<String, CustomState>,
    );
    type ExecReturn = ();
    type ProfileMetric = Duration;

    fn initialize((ctx, stream, custom_state): Self::CompileArg) -> Self {
        Self {
            hlir_buffers: FxHashMap::default(),
            buffers: FxHashMap::default(),
            cuda_stream: stream,
            cuda_context: ctx,
            llir_graph: StableGraph::default(),
            custom_state,
            exec_graph: StableGraph::default(),
            node_to_exec: FxHashMap::default(),
            timings: vec![],
        }
    }

    #[tracing::instrument(skip_all)]
    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        self.exec_graph.clear();
        // clear kv cache
        for s in self.custom_state.values_mut() {
            if let CustomState::KVCache(layers) = s {
                for (k, v) in layers {
                    self.cuda_stream.memset_zeros(k).unwrap();
                    self.cuda_stream.memset_zeros(v).unwrap();
                }
            }
        }
        let block_ops = llir_graph
            .node_indices()
            .filter_map(|n| llir_graph[n].to_dialect::<dyn BlockOp>())
            .map(|bo| (bo.op_name(), bo.clone()))
            .collect::<HashMap<_, _>>()
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(_, o)| o)
            .collect_vec();
        let block_ops_in_graph = llir_graph
            .node_indices()
            .filter(|n| llir_graph[*n].to_dialect::<dyn BlockOp>().is_some())
            .collect::<FxHashSet<_>>();
        let block_subgraphs = partition_marked_convex(llir_graph, &block_ops_in_graph).unwrap();

        // Add megakernels
        let mut exec_graph = StableGraph::default();
        let mut node_to_exec = FxHashMap::default();
        for subgraph in block_subgraphs {
            // Render expressions
            let (
                producer_barrier_strides,
                consumer_barrier_strides,
                mut producer_barrier_bases,
                n_barriers,
            ) = get_barrier_strides(llir_graph, &subgraph);
            for node in llir_graph
                .node_indices()
                .filter(|n| llir_graph[*n].to_op::<Input>().is_some())
            {
                producer_barrier_bases.insert(node, 0.into());
            }
            #[allow(clippy::mutable_key_type)]
            let expressions = llir_graph
                .node_weights()
                .filter_map(|op| op.to_dialect::<dyn BlockOp>())
                .flat_map(|op| {
                    op.expressions()
                        .into_iter()
                        .chain(once(op.launch_range().iter().copied().product()))
                })
                .chain(producer_barrier_strides.iter().map(|(n, e)| {
                    flatten_z_strides(
                        &llir_graph[*n]
                            .to_dialect::<dyn BlockOp>()
                            .unwrap()
                            .launch_range(),
                        e,
                    )
                }))
                .chain(consumer_barrier_strides.iter().map(|((n, _), e)| {
                    flatten_z_strides(
                        &llir_graph[*n]
                            .to_dialect::<dyn BlockOp>()
                            .unwrap()
                            .launch_range(),
                        e,
                    )
                }))
                .chain(producer_barrier_bases.values().copied())
                .chain(once(0.into()))
                .chain(once(1.into()))
                .collect::<FxHashSet<_>>();
            let (interpreter, module, expressions, interpreter_constants) = compile_interpreter(
                &self.cuda_context,
                &self.cuda_stream,
                &block_ops,
                &expressions,
            );
            // Build task queue
            let mut tasks: Vec<Task> = vec![];
            let mut node_to_task_index = FxHashMap::default();
            for node in toposort(&llir_graph, None).unwrap() {
                if !subgraph.contains(&node) {
                    continue;
                }
                let sources = llir_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect_vec();
                let op = llir_graph[node].to_dialect::<dyn BlockOp>().unwrap();
                let op_code = block_ops
                    .iter()
                    .position(|o| o.op_name() == op.op_name())
                    .unwrap();
                let mut payload =
                    op.schedule_op(&mut self.custom_state, &self.cuda_stream, &expressions);
                payload.extend(vec![0; size_of::<Payload>() - payload.len()]);
                let range = op.launch_range();
                let in_dep_a_stride = consumer_barrier_strides
                    .get(&(node, 0))
                    .map(|s| flatten_z_strides(&range, s))
                    .unwrap_or(0.into());
                let in_dep_b_stride = consumer_barrier_strides
                    .get(&(node, 1))
                    .map(|s| flatten_z_strides(&range, s))
                    .unwrap_or(0.into());
                let in_dep_c_stride = consumer_barrier_strides
                    .get(&(node, 2))
                    .map(|s| flatten_z_strides(&range, s))
                    .unwrap_or(0.into());
                let out_dep_stride = producer_barrier_strides
                    .get(&node)
                    .map(|s| flatten_z_strides(&range, s))
                    .unwrap_or(0.into());
                node_to_task_index.insert(node, tasks.len());
                tasks.push(Task {
                    op: op_code as i32,
                    range: expressions[&range.iter().copied().product()],
                    remaining: -1,
                    in_dep_a_stride: expressions[&in_dep_a_stride],
                    in_dep_a_base: expressions[&producer_barrier_bases
                        .get(&sources[0])
                        .copied()
                        .unwrap_or(0.into())],
                    in_dep_b_stride: expressions[&in_dep_b_stride],
                    in_dep_b_base: expressions[&sources
                        .get(1)
                        .and_then(|n| producer_barrier_bases.get(n).copied())
                        .unwrap_or(0.into())],
                    in_dep_c_stride: expressions[&in_dep_c_stride],
                    in_dep_c_base: expressions[&sources
                        .get(2)
                        .and_then(|n| producer_barrier_bases.get(n).copied())
                        .unwrap_or(0.into())],
                    out_dep_stride: expressions[&out_dep_stride],
                    out_dep_base: expressions[&producer_barrier_bases[&node]],
                    source_ptrs: [null(); 3],
                    out_ptr: null_mut(),
                    payload: PayloadBytes {
                        bytes: payload.try_into().unwrap(),
                    },
                });
            }
            let exec_node = exec_graph.add_node(ExecutableKernel::Megakernel {
                interpreter,
                module,
                interpreter_constants,
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
                let (kernel_function, module, code, grid, tb, shared_mem, constants) =
                    kernel_op.compile(&self.cuda_context, &self.cuda_stream);
                self.cuda_stream.synchronize().unwrap();
                let inputs = llir_graph
                    .edges_directed(kernel, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source())
                    .collect_vec();
                node_to_exec.insert(
                    kernel,
                    exec_graph.add_node(ExecutableKernel::Kernel {
                        kernel: kernel_function,
                        module,
                        code,
                        launch_grid: grid,
                        launch_threadblock: tb,
                        inputs,
                        output: kernel,
                        shared_mem,
                        constants,
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
        self.allocate_intermediate_buffers(dyn_map);
        let start = std::time::Instant::now();
        self.execute(dyn_map);
        self.timings.clear();
        (
            start.elapsed(),
            pretty_duration::pretty_duration(&start.elapsed(), None),
        )
    }

    #[tracing::instrument(skip_all)]
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        let mut llir_to_hlir: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
        for (hlir_node, llir_node) in self
            .llir_graph
            .node_indices()
            .filter_map(|n| {
                if let Some(Input { node, .. }) = self.llir_graph[n].to_op::<Input>() {
                    Some((*node, n))
                } else {
                    None
                }
            })
            .collect_vec()
        {
            llir_to_hlir.insert(llir_node, NodeIndex::new(hlir_node));
            let ptr = self.hlir_buffers[&NodeIndex::new(hlir_node)]
                .device_ptr(&self.cuda_stream)
                .0;
            self.register_buffer(llir_node, ptr);
        }
        let mut timings = vec![];
        for exec_node in toposort(&self.exec_graph, None).unwrap() {
            match &mut self.exec_graph[exec_node] {
                ExecutableKernel::Kernel {
                    kernel,
                    module: _,
                    code: _,
                    launch_grid,
                    launch_threadblock,
                    inputs,
                    output,
                    shared_mem,
                    constants,
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
                    let mut lb = self.cuda_stream.launch_builder(kernel);
                    lb.arg(&self.buffers[output]);
                    for inp in inputs {
                        if let Some(buf) = self.buffers.get(inp) {
                            lb.arg(buf);
                        } else {
                            lb.arg(&self.hlir_buffers[&llir_to_hlir[inp]]);
                        }
                    }
                    let span = span!(Level::INFO, "kernel");
                    let _entered = span.enter();
                    unsafe { lb.launch(cfg) }.unwrap();
                    self.cuda_stream.synchronize().unwrap();
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
                        .cuda_context
                        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                        .unwrap();
                    let span = span!(Level::INFO, "megakernel_setup");
                    let _entered = span.enter();
                    // Upload queue, barriers and program counter
                    let d_barriers = self
                        .cuda_stream
                        .alloc_zeros::<i32>(n_barriers.exec(dyn_map).unwrap())
                        .unwrap();
                    let d_tasks = self.cuda_stream.memcpy_stod(work_queue).unwrap();
                    let d_head = self.cuda_stream.memcpy_stod(&[0i32]).unwrap();
                    let queue_lock = self.cuda_stream.memcpy_stod(&[0i32]).unwrap();
                    // Set up timing buffer (start_time_u64,[[event_start_u64,event_type_i32 for sm_event in sm[:1000] for sm in sms[:sm_count]])
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
                        .cuda_context
                        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
                        .unwrap();

                    interpreter.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_max / 2, // Half shared mem, half L2
                    ).unwrap();

                    // Launch kernel
                    let cfg = LaunchConfig {
                        grid_dim: (sm_count as u32, 1, 1), // One block per SM
                        block_dim: (1024, 1, 1),           // 1024 threads (32 warps) per block
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
                    let span = span!(Level::INFO, "megakernel");
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
                    ));
                }
            }
        }
        self.timings.extend(timings);
    }
}

// TODO: get rid of this, go through all ops to get largest payload, and build the Task struct with CStructBuilder
// this is only here because it is currently the largest payload so it lets us have the Task struct with correct alignment / sizing
#[repr(C)]
#[derive(Copy, Clone)]
struct AttentionOp {
    row_width: i32,
    n_rows: i32,
    kv_row_stride: i32,
    q: i32,
    k: i32,
    v: i32,
    out: i32,
    key_cache: *mut f32,
    val_cache: *mut f32,
    prev_seq: i32,
    q_pos_stride: i32,
    group_pos_stride: i32,
    head_pos_stride: i32,
}
unsafe impl DeviceRepr for AttentionOp {}

#[repr(C)]
#[derive(Copy, Clone)]
pub union Payload {
    attention: AttentionOp,
}
unsafe impl DeviceRepr for Payload {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SMEvent {
    start: u64,
    event: i32,
}
unsafe impl DeviceRepr for SMEvent {}
unsafe impl ValidAsZeroBits for SMEvent {}

const PAYLOAD_SIZE: usize = std::mem::size_of::<Payload>();

#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct PayloadBytes {
    pub bytes: [u8; PAYLOAD_SIZE],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Task {
    pub op: i32,
    pub range: i32,
    pub remaining: i32,
    pub in_dep_a_stride: i32,
    pub in_dep_a_base: i32,
    pub in_dep_b_stride: i32,
    pub in_dep_b_base: i32,
    pub in_dep_c_stride: i32,
    pub in_dep_c_base: i32,
    pub out_dep_stride: i32,
    pub out_dep_base: i32,
    pub source_ptrs: [*const f32; 3],
    pub out_ptr: *mut f32,
    pub payload: PayloadBytes,
}
unsafe impl DeviceRepr for Task {}

#[tracing::instrument(skip_all)]
fn compute_barrier_strides(
    mut prod_range: Vec<Expression>,
    mut cons_range: Vec<Vec<Expression>>,
    mut cons_shared: Vec<Vec<bool>>,
) -> (Vec<Expression>, Vec<Vec<Expression>>) {
    // returns (producer strides, consumer strides)
    fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if v.is_empty() {
            return vec![];
        }
        let len = v[0].len();
        let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
        (0..len)
            .map(|_| {
                iters
                    .iter_mut()
                    .map(|n| n.next().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect()
    }
    let max_range_len = prod_range
        .len()
        .max(cons_range.iter().map(|i| i.len()).max().unwrap_or_default());
    let prod_range_len = prod_range.len();
    let cons_range_lens = cons_range.iter().map(|c| c.len()).collect_vec();
    prod_range.append(&mut vec![1.into(); max_range_len - prod_range.len()]);
    for v in &mut cons_range {
        v.append(&mut vec![1.into(); max_range_len - v.len()]);
    }
    for v in &mut cons_shared {
        v.append(&mut vec![false; max_range_len - v.len()]);
    }
    let cons_range_t = transpose(cons_range);
    let cons_shared_t = transpose(cons_shared);
    assert_eq!(cons_shared_t.len(), prod_range.len());
    let r = prod_range
        .iter()
        .zip(&cons_range_t)
        .zip(cons_shared_t)
        .rev()
        .scan(Expression::from(1), |acc, ((pr, cr), cs)| {
            let prev = *acc;
            if cs.iter().all(|i| *i) {
                if cr.iter().all(|cr| *pr == *cr) {
                    *acc *= *pr;
                    Some((prev, vec![prev; cr.len()]))
                } else if let Some(Some(factor)) = cr.iter().try_fold(None, |acc, cr| {
                    // Multiple producers per consumer
                    if !(*pr % *cr).to_usize().map(|i| i == 0).unwrap_or_default() {
                        return None;
                    }
                    if let Some(prev) = acc {
                        if prev != (*pr / *cr) {
                            return None;
                        }
                    }
                    Some(Some(*pr / *cr))
                }) {
                    *acc *= *pr / factor;
                    assert!(factor.to_usize().map(|i| i > 0).unwrap_or(true));
                    Some((
                        Expression::from('z') / factor * prev,
                        vec![prev * 'z'; cr.len()],
                    ))
                } else if let Some(Some(factor)) = cr.iter().try_fold(None, |acc, cr| {
                    // Multiple consumers per producer
                    if !(*cr % *pr).to_usize().map(|i| i == 0).unwrap_or_default() {
                        return None;
                    }
                    if let Some(prev) = acc {
                        if prev != (*cr / *pr) {
                            return None;
                        }
                    }
                    Some(Some(*cr / *pr))
                }) {
                    assert!(factor.to_usize().map(|i| i > 0).unwrap_or(true));
                    *acc *= cr[0] / factor;
                    Some((
                        prev * 'z',
                        vec![Expression::from('z') / factor * prev; cr.len()],
                    ))
                } else {
                    Some((0.into(), vec![0.into(); cr.len()]))
                }
            } else {
                Some((0.into(), vec![0.into(); cr.len()]))
            }
        })
        .collect_vec();
    let (mut p, c): (Vec<Expression>, Vec<Vec<Expression>>) = r.into_iter().rev().unzip();
    let mut c = transpose(c);
    // Re-trim down to original range lengths
    p = p[..prod_range_len].to_vec();
    for (c, r) in c.iter_mut().zip(cons_range_lens) {
        *c = c[..r].to_vec();
    }
    (p, c)
}

#[tracing::instrument(skip_all)]
pub fn allocate_input_buffers(
    stream: &Arc<CudaStream>,
    inputs: &FxHashMap<usize, Vec<f32>>,
    graph: &LLIRGraph,
) -> FxHashMap<NodeIndex, CudaSlice<f32>> {
    let mut buffers = FxHashMap::default();
    for node in graph.node_indices() {
        if let Some(gmem) = graph[node].to_op::<Input>() {
            if let Some(buf) = inputs.get(&gmem.node) {
                buffers.insert(node, stream.memcpy_stod(buf).unwrap());
            }
        }
    }
    buffers
}

struct ManualTrackBuilder {
    packets: Vec<schema::TracePacket>,
    track_uuid: u64,
    sequence_id: u32,
    state_cleared: bool,
    core_index: u32,
}

impl ManualTrackBuilder {
    fn new(core_index: u32, ts0: u64, clock_id: u32) -> Self {
        let track_uuid = manual_track_uuid(core_index);
        let sequence_id = manual_sequence_id(core_index);
        let track_name = format!("SM {core_index}");
        let synthetic_tid = 10_000 + core_index;
        let descriptor = schema::TracePacket {
            timestamp: Some(ts0.saturating_sub(1)),
            timestamp_clock_id: Some(clock_id),
            data: Some(trace_packet::Data::TrackDescriptor(
                schema::TrackDescriptor {
                    parent_uuid: None,
                    uuid: Some(track_uuid),
                    static_or_dynamic_name: Some(track_descriptor::StaticOrDynamicName::Name(
                        track_name.clone(),
                    )),
                    thread: Some(schema::ThreadDescriptor {
                        pid: Some(std::process::id() as i32),
                        tid: Some(synthetic_tid as i32),
                        thread_name: Some(track_name),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let mut builder = Self {
            packets: Vec::new(),
            track_uuid,
            sequence_id,
            state_cleared: false,
            core_index,
        };
        builder.push_packet(descriptor);
        builder
    }

    fn push_slice(&mut self, label: &str, start: u64, end: u64, ts0: u64, clock_id: u32) {
        self.push_packet(self.slice_packet(label, ts0 + start, clock_id, true));
        self.push_packet(self.slice_packet(label, ts0 + end, clock_id, false));
    }

    fn slice_packet(
        &self,
        label: &str,
        timestamp_ns: u64,
        clock_id: u32,
        is_begin: bool,
    ) -> schema::TracePacket {
        let mut debug_annotations = Vec::new();
        debug_annotations.push(schema::DebugAnnotation {
            name_field: Some(schema::debug_annotation::NameField::Name("sm".into())),
            value: Some(schema::debug_annotation::Value::IntValue(
                self.core_index as i64,
            )),
            ..Default::default()
        });
        debug_annotations.push(schema::DebugAnnotation {
            name_field: Some(schema::debug_annotation::NameField::Name(
                "span.label".into(),
            )),
            value: Some(schema::debug_annotation::Value::StringValue(label.into())),
            ..Default::default()
        });

        schema::TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(clock_id),
            data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                track_uuid: Some(self.track_uuid),
                r#type: Some(if is_begin {
                    track_event::Type::SliceBegin as i32
                } else {
                    track_event::Type::SliceEnd as i32
                }),
                name_field: Some(track_event::NameField::Name(label.to_owned())),
                debug_annotations,
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    fn push_packet(&mut self, mut packet: schema::TracePacket) {
        packet.optional_trusted_packet_sequence_id = Some(
            trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                self.sequence_id,
            ),
        );
        if !self.state_cleared {
            packet.sequence_flags =
                Some(trace_packet::SequenceFlags::SeqIncrementalStateCleared as i32 as u32);
            self.state_cleared = true;
        }
        self.packets.push(packet);
    }

    fn into_packets(self) -> Vec<schema::TracePacket> {
        self.packets
    }
}

fn manual_track_uuid(core_index: u32) -> u64 {
    hash64((1u32, 42u32, core_index))
}

fn manual_sequence_id(core_index: u32) -> u32 {
    hash32((2u32, 42u32, core_index))
}

fn hash64<T: Hash>(val: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    hasher.finish()
}

fn hash32<T: Hash>(val: T) -> u32 {
    (hash64(val) & 0xffff_ffff) as u32
}

type BarrierStrides = (
    FxHashMap<NodeIndex, Vec<Expression>>,
    FxHashMap<(NodeIndex, usize), Vec<Expression>>,
    FxHashMap<NodeIndex, Expression>,
    Expression,
);

type InterpreterCompileResult = (
    CudaFunction,
    Arc<CudaModule>,
    FxHashMap<Expression, i32>,
    FxHashMap<char, CudaSlice<u8>>,
);

#[tracing::instrument(skip_all)]
pub fn get_barrier_strides(graph: &LLIRGraph, block_ops: &FxHashSet<NodeIndex>) -> BarrierStrides {
    // Resolve dependencies
    let mut producer_barrier_strides = FxHashMap::default();
    let mut consumer_barrier_strides = FxHashMap::default();
    for node in block_ops {
        if !graph
            .neighbors_directed(*node, Direction::Outgoing)
            .any(|n| block_ops.contains(&n))
        {
            producer_barrier_strides.insert(
                *node,
                vec![
                    0.into();
                    graph[*node]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .launch_range()
                        .len()
                ],
            ); // TODO: is this right?
            continue;
        }
        let consumers = graph
            .edges_directed(*node, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| {
                let n_input = graph
                    .edges_directed(e.target(), Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .position(|ie| ie.id() == e.id())
                    .unwrap();
                (e.target(), n_input)
            })
            .filter(|(n, _)| block_ops.contains(n))
            .collect_vec();
        let prod_range = graph[*node]
            .to_dialect::<dyn BlockOp>()
            .unwrap()
            .launch_range();
        let cons_range: Vec<Vec<Expression>> = consumers
            .iter()
            .map(|(n, _)| {
                graph[*n]
                    .to_dialect::<dyn BlockOp>()
                    .unwrap()
                    .launch_range()
            })
            .collect();
        let (producer_strides, consumer_strides) = compute_barrier_strides(
            prod_range.clone(),
            cons_range.clone(),
            consumers
                .iter()
                .map(|(n, i)| {
                    graph[*n]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .consumer_barriers_seperate()
                        .remove(*i)
                })
                .collect(),
        );

        producer_barrier_strides.insert(*node, producer_strides);
        assert_eq!(consumers.len(), consumer_strides.len());
        for ((cons, inp), strides) in consumers.into_iter().zip(consumer_strides) {
            consumer_barrier_strides.insert((cons, inp), strides);
        }
    }
    let mut n_barriers = Expression::from(1); // Starts at 1 to account for GMEM producers
    let mut producer_barrier_bases = FxHashMap::default();
    for op in block_ops {
        producer_barrier_bases.insert(*op, n_barriers);
        n_barriers = (n_barriers
            + producer_barrier_strides[op]
                .iter()
                .zip(
                    graph[*op]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .launch_range(),
                )
                .map(|(stride, range)| stride.substitute('z', range))
                .sum::<Expression>()
            + 1)
        .simplify();
    }
    (
        producer_barrier_strides,
        consumer_barrier_strides,
        producer_barrier_bases,
        n_barriers,
    )
}

#[allow(clippy::mutable_key_type)]
#[tracing::instrument(skip_all)]
pub fn compile_interpreter(
    cuda_ctx: &Arc<CudaContext>,
    cuda_stream: &Arc<CudaStream>,
    ops: &Vec<Arc<Box<dyn BlockOp>>>,
    expressions: &FxHashSet<Expression>,
) -> InterpreterCompileResult {
    // Compile the interpreter
    let mut kernel = include_str!("block/interpreter.cu").to_string();
    kernel = kernel.replace(
        "const int N_OPS = 0;",
        &format!(
            "const int N_OPS = {};",
            ops.iter().filter(|op| !op.cuda_op().0.is_empty()).count()
        ),
    );
    kernel = kernel.replace(
        "//%extra_op_codes%",
        &ops.iter()
            .enumerate()
            .map(|(i, op)| format!("{}Op = {i}", op.op_name()))
            .join(", "),
    );
    kernel = kernel.replace(
        "//%extra_op_structs%",
        &ops.iter()
            .map(|op| format!("struct {}Payload {{{}}};", op.op_name(), op.cuda_op().0))
            .join("\n"),
    );
    kernel = kernel.replace(
        "//%extra_op_functions%",
        &ops
            .iter()
            .map(|op| {
                let op_name = op.op_name();
                let (_, op_body) = op.cuda_op();
                format!(
                    "__device__ void {op_name}_function({op_name}Payload payload, const float* const source_ptrs[3], float* out_ptr, const int current, int t) {{
{op_body}
}}"
                )
            })
            .join("\n"),
    );
    kernel = kernel.replace(
        "//%extra_op_payloads%",
        &ops.iter()
            .map(|op| {
                let op_name = op.op_name();
                format!("{op_name}Payload {op_name};")
            })
            .join(" "),
    );
    kernel = kernel.replace("//%extra_op_calls%", &ops.iter().map(|op| {
            let op_name = op.op_name();
            format!("case OpCode::{op_name}Op: {op_name}_function(t->payload.{op_name}, t->source_ptrs, t->out_ptr, nt.current, threadIdx.x); break;")
        }).join("\n"));
    let constants = expressions
        .iter()
        .flat_map(|e| e.dyn_vars())
        .collect::<FxHashSet<_>>();
    let constant_string = constants
        .iter()
        .map(|v| format!("__constant__ int const_{v}[1];"))
        .join("\n");
    let expression_map = expressions
        .iter()
        .enumerate()
        .map(|(i, e)| (*e, i as i32))
        .collect::<FxHashMap<_, _>>();
    let lambdas = expression_map
        .iter()
        .sorted_by_key(|(_, i)| **i)
        .map(|(e, i)| format!("case {i}: return {};", e.to_kernel()))
        .join("\n");
    kernel = kernel.replace("//%expr_fns%", &lambdas);
    kernel = kernel.replace("//%constants%", &constant_string);

    let ptx = compile_ptx_with_opts(
        &kernel,
        CompileOptions {
            arch: Some("sm_75"),
            ..Default::default()
        },
    )
    .unwrap();
    cuda_stream.synchronize().unwrap();
    let module = cuda_ctx.load_module(ptx).unwrap();
    let func = module.load_function("worker_kernel").unwrap();
    let constants = constants
        .into_iter()
        .map(|d| {
            (
                d,
                module
                    .get_global(&format!("const_{d}"), cuda_stream)
                    .unwrap(),
            )
        })
        .collect();
    cuda_stream.synchronize().unwrap();
    (func, module, expression_map, constants)
}

#[allow(unused)]
pub fn free_debug_buffers(debug_buffers: FxHashMap<(usize, String), &mut [f32]>) {
    for (_, buffer) in debug_buffers {
        unsafe {
            assert_eq!(
                cudarc::driver::sys::cuMemFreeHost(buffer.as_mut_ptr() as *mut c_void),
                cudarc::driver::sys::CUresult::CUDA_SUCCESS
            );
        }
    }
}

#[allow(unused)]
pub fn print_debug_buffers(debug_buffers: FxHashMap<(usize, String), &'static mut [f32]>) {
    'debug: for ((_, label), buf) in debug_buffers.iter().sorted_by_key(|((i, _), _)| *i) {
        if label.contains("diff:") {
            let mut file = std::fs::File::open(label.replace("diff:", "")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            let file_floats: Vec<f32> = file_buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            for (i, (a, b)) in buf.iter().zip(file_floats).enumerate() {
                if (*a - b).abs() > 1e-4 {
                    println!(
                        "{} mismatch at index {i}: {a} != {b}",
                        label.replace("diff:", "")
                    );
                    continue 'debug;
                }
            }
            println!("{} matches", label.replace("diff:", ""));
        } else {
            println!(
                "{label} ({}): {:?}...{:?}",
                buf.len(),
                buf.iter().take(3).map(|f| format!("{f:.4}")).collect_vec(),
                buf.iter()
                    .skip(buf.len() - 3)
                    .map(|f| format!("{f:.4}"))
                    .collect_vec(),
            );
        }
    }
    free_debug_buffers(debug_buffers);
}

#[allow(unused)]
#[tracing::instrument(skip_all)]
pub fn dtoh_outputs(
    stream: &Arc<CudaStream>,
    buffers: &FxHashMap<NodeIndex, CudaSlice<f32>>,
    graph: &StableGraph<Arc<Box<dyn BlockOp>>, (), Directed>,
) -> Vec<Vec<f32>> {
    graph
        .externals(Direction::Outgoing)
        .sorted()
        .map(|n| stream.memcpy_dtov(&buffers[&n]).unwrap())
        .collect()
}

pub fn partition_marked_convex<T, E>(
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
