use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hlir_subgraph_to_egglog, hlir_to_egglog,
    random_initial_choice, run_egglog, stitch_llir_graphs,
};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpHLIR, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{Direction, algo::toposort, stable_graph::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    any::TypeId,
    fmt::Debug,
    io::Write,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use tracing::{self, trace};

pub type LLIRGraph = StableGraph<LLIROp, ()>;
pub type HLIRGraph = StableGraph<Box<dyn HLIROp>, ShapeTracker>;

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
#[derive(Debug, Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: HLIRGraph,
    /// E-Graph search spaces (one per subgraph when graph breaks are used)
    egraphs: Vec<SerializedEGraph>,
    /// Subgraph descriptors (one per chunk when graph breaks are used)
    subgraph_descriptors: Vec<SubgraphDescriptor>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("Tensor", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        let id = self.graph.add_node(Box::new(crate::hlir::Input {
            node: 0,
            label: name.to_string(),
            dtype: DType::default(),
        }));
        self.get_op_mut::<crate::hlir::Input>(id).node = id.index();
        GraphTensor {
            id,
            graph_ref: self,
            shape: ShapeTracker::new(shape),
            dtype: DType::default(),
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), *e.weight()))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn HLIROp>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.target(), &self.graph[e.target()]))
            .collect()
    }

    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx
    ///     .add_op(luminal::hlir::Mul::default())
    ///     .input(a.id, a.shape)
    ///     .finish();
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph(), a.dtype);
    /// ```
    pub fn add_op<O: HLIROp + 'static>(&mut self, op: O) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Add op on the graph, and get back a NewOp. Just like add_op, except a boxed op is expected.
    pub fn add_boxed_op(&mut self, op: Box<dyn HLIROp + 'static>) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(op),
            graph_ref: self,
            num_srcs: 0,
        }
    }

    pub fn try_get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> Option<&T> {
        self.node_weight(node).unwrap().as_any().downcast_ref::<T>()
    }
    pub fn get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> &T {
        self.try_get_op(node).unwrap()
    }
    pub fn try_get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.node_weight_mut(node)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<T>()
    }
    pub fn get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> &mut T {
        self.try_get_op_mut(node).unwrap()
    }

    pub fn custom_op(
        &mut self,
        op: impl CustomOp + 'static,
        inputs: impl ToIds,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        self.custom_ops.push(Box::new(op));
        let mut add = self.add_op(CustomOpHLIR {
            id: self.custom_ops.len() - 1,
            dtype,
        });
        for input in inputs.to_ids() {
            add = add.input(input, ShapeTracker::new(()));
        }
        GraphTensor::from_id(add.finish(), ShapeTracker::new(shape), self, dtype)
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = split_at_graph_breaks(self);

        if subgraphs.len() <= 1 {
            // No graph breaks — original single-egraph path
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        } else {
            println!(
                "   {:>6}  {} chunks from graph breaks",
                "Split".cyan().bold(),
                subgraphs.len()
            );
            self.egraphs = subgraphs
                .iter()
                .map(|sg| {
                    let (program, root) = hlir_subgraph_to_egglog(self, sg);
                    run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()
                })
                .collect();
        }
        self.subgraph_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.term().0)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.term().0));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = split_at_graph_breaks(self);
        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        } else {
            self.egraphs = subgraphs
                .iter()
                .map(|sg| {
                    let (program, root) = hlir_subgraph_to_egglog(self, sg);
                    run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()
                })
                .collect();
        }
        self.subgraph_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraphs.first()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    const DEFAULT_GENERATION_SIZE: usize = 30;
    const MUTATIONS_PER_OFFSPRING: usize = 30;
    const TRIALS_PER_PROFILE: usize = 10;

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
        if self.egraphs.len() <= 1 {
            return self.search_single(runtime, limit);
        }

        let n_chunks = self.egraphs.len();
        let ops = self.ops.as_ref().unwrap();
        let start = std::time::Instant::now();

        // Allocate dummy buffers for all boundary inputs so any chunk can be profiled
        for desc in &self.subgraph_descriptors {
            for bi in &desc.boundary_inputs {
                let n_elements = bi
                    .shape
                    .n_elements()
                    .exec(&self.dyn_map)
                    .expect("Failed to resolve boundary input shape");
                runtime.allocate_dummy_input(bi.break_node.index(), n_elements);
            }
        }

        // Search each chunk independently
        let mut rng = rand::rng();
        let mut chunk_best_llirs: Vec<LLIRGraph> = Vec::with_capacity(n_chunks);
        let mut bars_drawn = false;

        fn make_bar(searched: usize, total: usize) -> String {
            let bar_width = 24;
            let head = ((searched as f32 / total as f32) * bar_width as f32)
                .clamp(0.0, bar_width as f32)
                .floor() as usize;
            if head == 0 {
                format!("[>{}]", " ".repeat(bar_width - 1))
            } else if head >= bar_width {
                format!("[{}>]", "=".repeat(bar_width))
            } else {
                format!(
                    "[{}>{}]",
                    "=".repeat(head),
                    " ".repeat(bar_width - head - 1)
                )
            }
        }

        for chunk_idx in 0..n_chunks {
            let egraph = &self.egraphs[chunk_idx];

            let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();

            // Clear intermediate buffers from previous chunk's profiling
            runtime.clear_intermediate_buffers();

            let mut best_genome = random_initial_choice(egraph, &mut rng);
            prev_selected.insert(hash_choice_set(&best_genome));

            let mut best_graph = egglog_to_llir(
                egraph,
                best_genome.clone(),
                ops,
                &self.custom_ops,
                &mut list_cache,
                &mut expr_cache,
            );
            let (mut best_metric, display) =
                runtime.profile(&best_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE);

            let mut n_graphs = 1;

            // Print initial result above progress bars
            {
                let msg = format!(
                    "   {:>8} {display}",
                    format!("Chunk {chunk_idx}").cyan().bold(),
                );
                if bars_drawn {
                    print!("\x1b[1A\r\x1b[2K");
                }
                println!("{msg}");
                print!(
                    "\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {chunk_idx}/{n_chunks}",
                    "Chunk".cyan().bold(),
                    make_bar(n_graphs, limit),
                    "Total".cyan().bold(),
                    make_bar(chunk_idx, n_chunks)
                );
                std::io::stdout().flush().unwrap();
                bars_drawn = true;
            }

            while n_graphs < limit {
                let offspring = extract_generation(
                    egraph,
                    &best_genome,
                    (limit - n_graphs).min(Self::DEFAULT_GENERATION_SIZE),
                    Self::MUTATIONS_PER_OFFSPRING,
                    &mut prev_selected,
                    &mut rng,
                );
                if offspring.is_empty() {
                    break;
                }

                for genome in offspring {
                    n_graphs += 1;
                    list_cache.clear();
                    expr_cache.clear();

                    let llir_graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                    );

                    // Use catch_unwind to handle CUDA errors from invalid LLIR
                    runtime.clear_intermediate_buffers();
                    let profile_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            runtime.profile(&llir_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE)
                        }));

                    let (new_metric, display_metric) = match profile_result {
                        Ok(result) => result,
                        Err(_) => {
                            // Just update progress bars on skip
                            print!(
                                "\x1b[1A\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {chunk_idx}/{n_chunks}",
                                "Chunk".cyan().bold(),
                                make_bar(n_graphs, limit),
                                "Total".cyan().bold(),
                                make_bar(chunk_idx, n_chunks)
                            );
                            std::io::stdout().flush().unwrap();
                            continue;
                        }
                    };

                    let mut new_best = false;
                    if best_metric.gt(&new_metric) {
                        best_metric = new_metric;
                        best_graph = llir_graph;
                        best_genome = genome;
                        new_best = true;
                    }

                    if new_best {
                        // Print result above progress bars
                        let msg = format!("   {:>6} {display_metric}", "Searched".green().bold(),);
                        print!("\x1b[1A\r\x1b[2K");
                        println!("{msg}");
                        print!(
                            "\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {chunk_idx}/{n_chunks}",
                            "Chunk".cyan().bold(),
                            make_bar(n_graphs, limit),
                            "Total".cyan().bold(),
                            make_bar(chunk_idx, n_chunks)
                        );
                        std::io::stdout().flush().unwrap();
                    } else {
                        // Just update progress bars
                        print!(
                            "\x1b[1A\r\x1b[2K  {:>6}  {} {n_graphs}/{limit}\n\x1b[2K  {:>6}  {} {chunk_idx}/{n_chunks}",
                            "Chunk".cyan().bold(),
                            make_bar(n_graphs, limit),
                            "Total".cyan().bold(),
                            make_bar(chunk_idx, n_chunks)
                        );
                        std::io::stdout().flush().unwrap();
                    }
                }
            }

            chunk_best_llirs.push(best_graph);
        }

        // Clear progress bars
        if bars_drawn {
            print!("\x1b[1A\r\x1b[2K\n\x1b[2K\x1b[1A\r");
            std::io::stdout().flush().unwrap();
        }

        // Stitch all chunk LLIRs into a single graph
        let stitched = stitch_llir_graphs(&chunk_best_llirs, &self.subgraph_descriptors);

        println!(
            "   {:>6}  {} chunks in {}",
            "Searched".green().bold(),
            n_chunks,
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );

        // Clear stale buffers from chunk profiling before loading the final graph
        runtime.clear_intermediate_buffers();
        runtime.load_llir(&stitched);
        runtime
    }

    /// Search a single e-graph (original behavior, no graph breaks).
    fn search_single<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
        let mut rng = rand::rng();
        let egraph = &self.egraphs[0];
        let ops = self.ops.as_ref().unwrap();

        let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();

        let mut best_genome = random_initial_choice(egraph, &mut rng);
        prev_selected.insert(hash_choice_set(&best_genome));

        let start = std::time::Instant::now();
        let mut best_graph;
        let mut best_metric;
        let bar_width = 24;
        let mut n_graphs;

        let progress_bar = |searched: usize, limit: usize| {
            let total = limit;
            let head = ((searched as f32 / total as f32) * bar_width as f32)
                .clamp(0.0, bar_width as f32)
                .floor() as usize;
            let bar = if head == 0 {
                format!("[>{}]", " ".repeat(bar_width - 1))
            } else if head >= bar_width {
                format!("[{}>]", "=".repeat(bar_width))
            } else {
                format!(
                    "[{}>{}]",
                    "=".repeat(head),
                    " ".repeat(bar_width - head - 1)
                )
            };
            print!(
                "\r\x1b[2K  {:>6}  {bar} {searched}/{total}",
                "Searching".cyan().bold(),
            );
            std::io::stdout().flush().unwrap();
        };

        {
            let llir_graph = egglog_to_llir(
                egraph,
                best_genome.clone(),
                ops,
                &self.custom_ops,
                &mut list_cache,
                &mut expr_cache,
            );
            let (new_metric, display_metric) =
                runtime.profile(&llir_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE);
            best_metric = Some(new_metric);
            best_graph = llir_graph;
            n_graphs = 1;
            progress_bar(n_graphs, limit);
            print!("\r\x1b[2K");
            std::io::stdout().flush().unwrap();
            println!(
                "   {:>6}  Graph {n_graphs}[0]: {}",
                "Searched".green().bold(),
                display_metric.bold().green()
            );
        }

        let mut generation = 0;
        while n_graphs < limit {
            generation += 1;
            let offspring = extract_generation(
                egraph,
                &best_genome,
                (limit - n_graphs).min(Self::DEFAULT_GENERATION_SIZE),
                Self::MUTATIONS_PER_OFFSPRING,
                &mut prev_selected,
                &mut rng,
            );

            if offspring.is_empty() {
                break;
            }

            for genome in offspring {
                n_graphs += 1;
                progress_bar(n_graphs, limit);
                list_cache.clear();
                expr_cache.clear();

                let llir_graph = egglog_to_llir(
                    egraph,
                    genome.clone(),
                    ops,
                    &self.custom_ops,
                    &mut list_cache,
                    &mut expr_cache,
                );
                let (new_metric, display_metric) =
                    runtime.profile(&llir_graph, &self.dyn_map, Self::TRIALS_PER_PROFILE);

                let mut new_best = false;
                if let Some(old_metric) = &best_metric {
                    if old_metric.gt(&new_metric) {
                        best_metric = Some(new_metric);
                        best_graph = llir_graph;
                        best_genome = genome;
                        new_best = true;
                    }
                } else {
                    best_metric = Some(new_metric);
                    best_graph = llir_graph;
                    best_genome = genome;
                    new_best = true;
                }

                print!("\r\x1b[2K");
                std::io::stdout().flush().unwrap();
                println!(
                    "   {:>6}  Graph {n_graphs}[{generation}]: {}",
                    "Searched".green().bold(),
                    if new_best {
                        display_metric.bold().green().to_string()
                    } else {
                        display_metric
                    }
                );
            }
        }

        trace!(
            target: "luminal::search",
            n_graphs,
            limit,
            limit_reached = n_graphs >= limit,
            duration_ms = start.elapsed().as_millis() as u64,
            "search completed"
        );
        println!(
            "   {:>6}  {n_graphs} graphs in {}",
            "Searched".green().bold(),
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );
        runtime.load_llir(&best_graph);
        runtime
    }
}

impl Deref for Graph {
    type Target = HLIRGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl NewOp<'_> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(mut self, id: NodeIndex, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(id, self.new_op_id, shape);
        self.num_srcs += 1;
        self
    }
}

/// Describes a tensor value crossing a graph break boundary into a chunk.
#[derive(Debug, Clone)]
pub struct BoundaryInput {
    /// The HLIR NodeIndex of the GraphBreak node (unique ID for matching)
    pub break_node: NodeIndex,
    /// Shape of the tensor at the boundary
    pub shape: ShapeTracker,
    /// DType of the tensor at the boundary
    pub dtype: DType,
}

/// Describes a subgraph (chunk) of the HLIR graph between graph breaks.
#[derive(Debug, Clone)]
pub struct SubgraphDescriptor {
    /// HLIR nodes in this chunk (excludes GraphBreak nodes themselves)
    pub nodes: FxHashSet<NodeIndex>,
    /// Boundary inputs entering from prior chunks
    pub boundary_inputs: Vec<BoundaryInput>,
    /// GraphBreak node indices whose predecessor is in this chunk
    pub boundary_outputs: Vec<NodeIndex>,
}

/// Split the HLIR graph at GraphBreak nodes into independent subgraphs.
///
/// Each non-GraphBreak node is assigned to a chunk based on the latest
/// GraphBreak in its transitive dependency chain. Real Input nodes (weights,
/// data) are included in every chunk that uses them.
pub fn split_at_graph_breaks(graph: &Graph) -> Vec<SubgraphDescriptor> {
    use crate::hlir::GraphBreak;

    // Find all GraphBreak nodes
    let break_nodes: FxHashSet<NodeIndex> = graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<GraphBreak>(*n).is_some())
        .collect();

    if break_nodes.is_empty() {
        // No breaks: single chunk with all nodes
        return vec![SubgraphDescriptor {
            nodes: graph.graph.node_indices().collect(),
            boundary_inputs: vec![],
            boundary_outputs: vec![],
        }];
    }

    // Topological sort of the full graph
    let topo = toposort(&graph.graph, None).expect("HLIR graph has a cycle");

    // Assign each GraphBreak a sequential break index (0, 1, 2, ...)
    // so chunk 0 is before break 0, chunk 1 is between break 0 and break 1, etc.
    let mut break_index: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut next_break_idx = 0;
    for &n in &topo {
        if break_nodes.contains(&n) {
            break_index.insert(n, next_break_idx);
            next_break_idx += 1;
        }
    }
    let n_chunks = next_break_idx + 1;

    // Assign each non-break node to a chunk.
    // A node's chunk = max over all predecessors' chunks.
    // A GraphBreak's "output chunk" = break_index + 1 (it pushes successors to the next chunk).
    let mut node_chunk: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    for &n in &topo {
        if break_nodes.contains(&n) {
            continue; // GraphBreak nodes aren't in any chunk
        }
        let mut chunk = 0usize;
        for pred in graph.graph.neighbors_directed(n, Direction::Incoming) {
            if let Some(&bi) = break_index.get(&pred) {
                // Predecessor is a GraphBreak: this node is in chunk bi+1
                chunk = chunk.max(bi + 1);
            } else if let Some(&pred_chunk) = node_chunk.get(&pred) {
                chunk = chunk.max(pred_chunk);
            }
        }
        node_chunk.insert(n, chunk);
    }

    // Build per-chunk node sets.
    // For non-Input nodes: they go to their assigned chunk.
    // For Input nodes (no predecessors): they go to EVERY chunk where they have
    // at least one direct consumer. This avoids polluting chunk 0 with hundreds
    // of weight tensors that are only used by later layers.
    let mut chunk_nodes: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); n_chunks];

    let input_nodes: FxHashSet<NodeIndex> = graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<crate::hlir::Input>(*n).is_some())
        .collect();

    for (&node, &chunk) in &node_chunk {
        if input_nodes.contains(&node) {
            continue; // Handle Input nodes separately below
        }
        chunk_nodes[chunk].insert(node);
    }

    // Place each Input node in every chunk that has a consumer of it
    for &inp in &input_nodes {
        let mut target_chunks = FxHashSet::default();
        for consumer in graph.graph.neighbors_directed(inp, Direction::Outgoing) {
            if break_nodes.contains(&consumer) {
                // Consumer is a GraphBreak — the chunk after it needs this input
                let bi = break_index[&consumer];
                let target_chunk = bi + 1;
                if target_chunk < n_chunks {
                    target_chunks.insert(target_chunk);
                }
            } else if let Some(&consumer_chunk) = node_chunk.get(&consumer) {
                target_chunks.insert(consumer_chunk);
            }
        }
        // If no consumers found (orphan input), put it in chunk 0
        if target_chunks.is_empty() {
            target_chunks.insert(0);
        }
        for chunk in target_chunks {
            chunk_nodes[chunk].insert(inp);
        }
    }

    // Closure: for each chunk, ensure all transitive predecessors of its nodes
    // are included (except GraphBreak boundary nodes). This handles shared
    // computation nodes (e.g. RoPE frequencies, sqrt(d_k)) that are assigned to
    // an earlier chunk but used by later chunks.
    for chunk_idx in 0..n_chunks {
        let mut to_visit: Vec<NodeIndex> = chunk_nodes[chunk_idx].iter().copied().collect();
        while let Some(node) = to_visit.pop() {
            for pred in graph.graph.neighbors_directed(node, Direction::Incoming) {
                if break_nodes.contains(&pred) {
                    continue; // Boundary handled separately
                }
                if chunk_nodes[chunk_idx].insert(pred) {
                    // Newly added — also visit its predecessors
                    to_visit.push(pred);
                }
            }
        }
    }

    // Build SubgraphDescriptors
    let mut descriptors: Vec<SubgraphDescriptor> = Vec::with_capacity(n_chunks);
    for chunk_idx in 0..n_chunks {
        let nodes = chunk_nodes[chunk_idx].clone();

        // Boundary inputs: GraphBreak nodes whose successors are in this chunk
        let mut boundary_inputs = vec![];
        for &brk in &break_nodes {
            let bi = break_index[&brk];
            if bi + 1 == chunk_idx {
                // This break feeds into this chunk
                // Get shape/dtype from the incoming edge of the GraphBreak
                let edge = graph
                    .graph
                    .edges_directed(brk, Direction::Incoming)
                    .next()
                    .expect("GraphBreak must have exactly one input");
                let shape = *edge.weight();
                // Get dtype from the predecessor
                let pred = edge.source();
                let dtype = if let Some(inp) = graph.try_get_op::<crate::hlir::Input>(pred) {
                    inp.dtype
                } else {
                    DType::F32 // Default; most intermediate values are F32
                };
                boundary_inputs.push(BoundaryInput {
                    break_node: brk,
                    shape,
                    dtype,
                });
            }
        }

        // Boundary outputs: GraphBreak nodes whose predecessor is in this chunk
        let mut boundary_outputs = vec![];
        for &brk in &break_nodes {
            let bi = break_index[&brk];
            if bi + 1 == chunk_idx + 1 {
                // This break's predecessor should be in this chunk
                let pred = graph
                    .graph
                    .neighbors_directed(brk, Direction::Incoming)
                    .next()
                    .expect("GraphBreak must have exactly one input");
                if nodes.contains(&pred) || node_chunk.get(&pred) == Some(&chunk_idx) {
                    boundary_outputs.push(brk);
                }
            }
        }

        descriptors.push(SubgraphDescriptor {
            nodes,
            boundary_inputs,
            boundary_outputs,
        });
    }

    descriptors
}
