// =========================================================================
// Region codegen for FusionStart / FusionEnd-bracketed fused regions.
//
// PR1 left FusedX / FusionStart / FusionEnd nodes in the post-extraction
// LLIR, each compiling to its own standalone CUDA kernel. PR2 collapses
// every FusionEnd-rooted region into ONE fused CUDA kernel at codegen
// time — without rewriting the LLIR.
//
// Pipeline:
//   `kernel_to_host` builds a Vec<CompileUnit> from the topo order:
//     - CompileUnit::Single(node)  — un-fused KernelX, compiled as before.
//     - CompileUnit::Region(rgn)   — one FE + its interior FusedX DAG +
//                                    its FS leaves. Compiled here as a
//                                    single CUDA kernel that reads from
//                                    the region's external inputs once,
//                                    chains all FusedX bodies through
//                                    register-resident locals, and writes
//                                    the FE's output.
//
// The CompiledKernel for a Region is keyed on the FE node and stores
// `inputs = external producer NodeIndices` (one per interior FusionStart),
// so the existing buffer-pointer wiring in to_host.rs picks up the right
// device pointers at execute time. Interior FusedX / FusionStart nodes
// never enter the kernels Vec — they have no buffers, no launches.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    graph::LLIRGraph,
    prelude::{
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
        *,
    },
};

use as_any::Downcast;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
    kernel::other_ops::{FusionEnd, FusionStart},
};

// =========================================================================
// Compile units — what `kernel_to_host` iterates over instead of nodes.
// =========================================================================

#[derive(Debug, Clone)]
pub(crate) struct RegionUnit {
    /// The FusionEnd node that anchors this region.
    pub fe_node: NodeIndex,
    /// Interior FusedX nodes, in topological order (predecessors before
    /// consumers). Used to emit register-binding statements in dependency
    /// order in the fused CUDA kernel body.
    pub fusedx_topo: Vec<NodeIndex>,
    /// FusionStart nodes that bound the region's leaves. One per external
    /// read site — duplicates (different FS LLIR nodes wrapping the same
    /// upstream tensor) are kept separate so each read uses its own
    /// strides; the host launch passes the same device pointer twice.
    pub fs_nodes: Vec<NodeIndex>,
    /// External producer NodeIndices, one per `fs_nodes` entry in the same
    /// order. Becomes the `inputs` field of the FE's `CompiledKernel`, and
    /// the kernel function's `in0`, `in1`, ... parameters in that order.
    pub external_inputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
pub(crate) enum CompileUnit {
    Single(NodeIndex),
    Region(RegionUnit),
}

// =========================================================================
// Region detection.
// =========================================================================

/// Group a sub-DAG's topo order into compile units. Each FusionEnd node
/// becomes the root of a `CompileUnit::Region`; the region's interior
/// FusedX and FusionStart nodes are absorbed into that region and removed
/// from the per-node iteration. Anything else is wrapped in
/// `CompileUnit::Single`.
pub(crate) fn build_compile_units(
    topo_order: &[NodeIndex],
    llir_graph: &LLIRGraph,
) -> Vec<CompileUnit> {
    let name_of = |idx: NodeIndex| -> Option<&'static str> {
        llir_graph
            .node_weight(idx)
            .and_then(|op| op.to_dialect::<dyn KernelOp>().map(|k| k.kernel_name()))
    };

    // First pass: every FusionEnd in the subgraph anchors a region; gather
    // the region's interior + FS leaves by walking incoming edges
    // backward, stopping at FusionStart (a leaf — its predecessor is the
    // external producer, outside the region).
    let mut absorbed: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut regions: FxHashMap<NodeIndex, RegionUnit> = FxHashMap::default();

    for &node in topo_order {
        if name_of(node) != Some("FusionEnd") {
            continue;
        }

        let mut interior: Vec<NodeIndex> = Vec::new();
        let mut fs_nodes: Vec<NodeIndex> = Vec::new();
        let mut visited: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut stack: Vec<NodeIndex> = Vec::new();
        stack.push(node);
        visited.insert(node);

        while let Some(cur) = stack.pop() {
            for pred in llir_graph.neighbors_directed(cur, Direction::Incoming) {
                if !visited.insert(pred) {
                    continue;
                }
                match name_of(pred) {
                    Some("FusionStart") => {
                        fs_nodes.push(pred);
                        // Don't recurse past FS — its predecessor is
                        // external (outside the region).
                    }
                    Some("FusionEnd") => {
                        // A nested FE inside a region. Under the current
                        // rule design these are cascade artifacts — treat
                        // them as transparent (walk through) rather than
                        // as a separate region. The outer region absorbs
                        // them. They do not become CompileUnit::Region
                        // anchors because their eclass is already the
                        // outer region's.
                        absorbed.insert(pred);
                        stack.push(pred);
                    }
                    Some(other) if other.starts_with("Fused") => {
                        interior.push(pred);
                        stack.push(pred);
                    }
                    _ => {
                        // Non-marker, non-FusedX predecessor inside what
                        // we thought was a region. Shouldn't happen with
                        // the current rules; treat conservatively: do
                        // not absorb — let the kernel_to_host single
                        // path handle it. This means the region is
                        // malformed and we likely should not have a
                        // region at all. Caller will see incomplete
                        // interior; the safer thing is to fall back.
                    }
                }
            }
        }

        // Topological order on the interior + FS nodes (so the kernel
        // emits `let v = ...;` lines after their inputs are bound). We
        // use the parent graph's toposort filtered to in-region nodes.
        let mut region_set: FxHashSet<NodeIndex> = FxHashSet::default();
        region_set.extend(interior.iter().copied());
        region_set.extend(fs_nodes.iter().copied());
        let topo = toposort(llir_graph, None).expect("LLIR cycle in region detection");
        let interior_topo: Vec<NodeIndex> = topo
            .iter()
            .copied()
            .filter(|n| region_set.contains(n) && interior.contains(n))
            .collect();
        let fs_topo: Vec<NodeIndex> = topo
            .iter()
            .copied()
            .filter(|n| region_set.contains(n) && fs_nodes.contains(n))
            .collect();

        // External producer for each FS leaf, in the same order.
        let external_inputs: Vec<NodeIndex> = fs_topo
            .iter()
            .map(|&fs| {
                llir_graph
                    .neighbors_directed(fs, Direction::Incoming)
                    .next()
                    .expect("FusionStart with no predecessor")
            })
            .collect();

        absorbed.extend(interior_topo.iter().copied());
        absorbed.extend(fs_topo.iter().copied());

        regions.insert(
            node,
            RegionUnit {
                fe_node: node,
                fusedx_topo: interior_topo,
                fs_nodes: fs_topo,
                external_inputs,
            },
        );
    }

    // Second pass: emit compile units in original topo order, replacing
    // FE nodes with their RegionUnit and skipping anything absorbed.
    let mut units: Vec<CompileUnit> = Vec::new();
    for &node in topo_order {
        if let Some(region) = regions.remove(&node) {
            units.push(CompileUnit::Region(region));
        } else if absorbed.contains(&node) {
            // Interior FusedX / FS / nested FE — folded into a region.
            continue;
        } else {
            units.push(CompileUnit::Single(node));
        }
    }
    units
}

// =========================================================================
// Per-FusedX body templates.
//
// Each entry takes the names of the local variables holding the op's
// inputs and returns a CUDA expression evaluating to the op's output
// (a register-resident value, no buffer involved).
// =========================================================================

fn fused_body(name: &str, locals: &[&str]) -> String {
    match name {
        "FusedSin" => format!("sinf({})", locals[0]),
        "FusedSqrt" => format!("sqrtf({})", locals[0]),
        "FusedExp" => format!("expf({})", locals[0]),
        "FusedExp2" => format!("exp2f({})", locals[0]),
        "FusedLog2" => format!("log2f({})", locals[0]),
        "FusedRecip" => format!("1.0f / {}", locals[0]),
        "FusedAdd" => format!("{} + {}", locals[0], locals[1]),
        "FusedMul" => format!("{} * {}", locals[0], locals[1]),
        other => panic!("region_codegen: unknown FusedX op {other}"),
    }
}

// =========================================================================
// Region compilation — emit one CUDA kernel for the whole region.
// =========================================================================

#[allow(clippy::type_complexity)]
pub(crate) struct CompiledRegion {
    pub function: CudaFunction,
    pub module: Arc<CudaModule>,
    pub kernel_str: String,
    pub grid: (Expression, Expression, Expression),
    pub block: (Expression, Expression, Expression),
    pub shared_mem: Expression,
    pub constants: FxHashMap<char, CudaSlice<u8>>,
}

#[allow(clippy::type_complexity)]
pub(crate) fn compile_region(
    region: &RegionUnit,
    llir_graph: &LLIRGraph,
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> CompiledRegion {
    // Resolve FE: shape, strides (for the write), dtype.
    let fe_op = llir_graph[region.fe_node]
        .to_dialect::<dyn KernelOp>()
        .expect("FE node must be a KernelOp");
    let fe_struct: &FusionEnd = (***fe_op)
        .downcast_ref::<FusionEnd>()
        .expect("region root must be FusionEnd");
    let out_shape: &[Expression] = &fe_struct.shape;
    let out_strides: &[Expression] = &fe_struct.strides;
    let dtype: DType = fe_struct.dtype;

    // Aggregate all dynamic vars used anywhere in the region (FS strides,
    // FE strides, FusedX shape — all FusedX share `out_shape`, but their
    // own strides are likewise relevant for any future stride-affine ops).
    let mut all_vars: FxHashSet<char> = FxHashSet::default();
    all_vars.extend(out_shape.iter().flat_map(|e| e.dyn_vars()));
    all_vars.extend(out_strides.iter().flat_map(|e| e.dyn_vars()));
    for &fs_idx in &region.fs_nodes {
        let fs_op = llir_graph[fs_idx].to_dialect::<dyn KernelOp>().unwrap();
        let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
        all_vars.extend(fs_struct.strides.iter().flat_map(|e| e.dyn_vars()));
    }

    let cuda_ty = cuda_dtype(dtype);
    let includes = dtype_includes(&[dtype]);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
    let dyn_dims_param = if all_vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };

    let n_elements = out_shape
        .iter()
        .copied()
        .product::<Expression>()
        .to_kernel();

    // Build kernel signature: out, then one input per FS leaf in
    // `region.fs_nodes` order. The `external_inputs` list (parallel to
    // `fs_nodes`) is what the host wires into the launch params.
    let mut signature_params: Vec<String> = vec![format!("{cuda_ty} *out")];
    for i in 0..region.fs_nodes.len() {
        signature_params.push(format!("const {cuda_ty} *in{i}"));
    }
    let signature = signature_params.join(", ");

    // Body: read FS leaves, then walk FusedX in topo order emitting a
    // local per op, then write FE output. Every node gets a local
    // `v_<idx>` keyed by its NodeIndex, so an op's inputs are resolved
    // by looking up its predecessors' locals.
    let local_name = |n: NodeIndex| format!("v_{}", n.index());

    let mut body = String::new();
    body.push_str(&format!(
        "        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20       if (const_z >= {n_elements}) return;\n"
    ));

    // FS leaves: each reads from its corresponding `in_i` parameter using
    // its own strides.
    for (i, &fs_idx) in region.fs_nodes.iter().enumerate() {
        let fs_op = llir_graph[fs_idx].to_dialect::<dyn KernelOp>().unwrap();
        let fs_struct: &FusionStart = (***fs_op).downcast_ref::<FusionStart>().unwrap();
        let read_idx = flatten_strides(out_shape, &fs_struct.strides).to_kernel();
        body.push_str(&format!(
            "        {cuda_ty} {name} = in{i}[{read_idx}];\n",
            name = local_name(fs_idx),
        ));
    }

    // FusedX ops in topo order. Each looks up its predecessor locals
    // (in incoming-edge id order to match the original op's input
    // arity / position).
    for &op_idx in &region.fusedx_topo {
        let op_ref = llir_graph[op_idx].to_dialect::<dyn KernelOp>().unwrap();
        let op_name = op_ref.kernel_name();

        let mut input_locals: Vec<String> = llir_graph
            .edges_directed(op_idx, Direction::Incoming)
            .map(|e| (e.id(), e.source()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(_, src)| local_name(src))
            .collect();
        // Sort by edge id like the rest of the codegen does for stable
        // input ordering.
        let mut edges: Vec<(_, NodeIndex)> = llir_graph
            .edges_directed(op_idx, Direction::Incoming)
            .map(|e| (e.id(), e.source()))
            .collect();
        edges.sort_by_key(|(eid, _)| *eid);
        input_locals = edges.into_iter().map(|(_, src)| local_name(src)).collect();
        let inputs_ref: Vec<&str> = input_locals.iter().map(|s| s.as_str()).collect();

        let expr = fused_body(op_name, &inputs_ref);
        body.push_str(&format!(
            "        {cuda_ty} {name} = {expr};\n",
            name = local_name(op_idx),
        ));
    }

    // FE write: pick the FusedX feeding FE (its single incoming edge in
    // the region — a FusedX or, in degenerate single-FS regions which
    // shouldn't arise, an FS).
    let fe_input: NodeIndex = llir_graph
        .neighbors_directed(region.fe_node, Direction::Incoming)
        .next()
        .expect("FusionEnd with no predecessor");
    let fe_input_local = local_name(fe_input);
    let write_idx = flatten_strides(out_shape, out_strides).to_kernel();
    body.push_str(&format!(
        "        out[{write_idx}] = {fe_input_local};\n"
    ));

    let kernel = format!(
        "{includes}\n\
         {dyn_defines}\n\
         extern \"C\" {{\n\
         \x20   __global__ void fused_region_k({signature}{dyn_dims_param}) {{\n\
         {body}\
         \x20   }}\n\
         }}"
    );

    let (module, function) = if let Some((m, f)) = compile_cache.get(&kernel) {
        (m.clone(), f.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel)
            .expect("region kernel PTX compile failed");
        let module = stream.context().load_module(ptx).expect("module load failed");
        let function = module
            .load_function("fused_region_k")
            .expect("region kernel function not found");
        compile_cache.insert(kernel.clone(), (module.clone(), function.clone()));
        (module, function)
    };

    let out_size = out_shape.iter().copied().product::<Expression>();

    CompiledRegion {
        function,
        module,
        kernel_str: kernel,
        grid: (out_size.ceil_div(256), 1.into(), 1.into()),
        block: (out_size.min(256), 1.into(), 1.into()),
        shared_mem: 0.into(),
        constants: FxHashMap::default(),
    }
}
