use crate::egglog_utils::{
    egglog_to_llir, egglog_to_llir_from_root, extract_generation, hash_choice_set,
    hash_egglog_normalized, hlir_subgraph_to_egglog, hlir_to_egglog, random_initial_choice,
    run_egglog, run_egglog_multi_roots, stitch_llir_graphs,
};
use crate::visualization::ToDot;
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpKind, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{Direction, algo::toposort, stable_graph::StableGraph, visit::EdgeRef};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    any::TypeId,
    fmt::Debug,
    fs,
    io::Write,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Arc,
};
use tracing;

pub type LLIRGraph = StableGraph<LLIROp, ()>;
pub type HLIRGraph = StableGraph<Box<dyn HLIROp>, ()>;

#[derive(Debug, Clone)]
pub struct RegionalLLIRRegion {
    /// Representative chunk index this region was optimized from.
    pub representative_region: usize,
    /// All chunk indices covered by this region (including representative).
    pub member_regions: Vec<usize>,
    /// Best LLIR found for the representative.
    pub representative_llir: LLIRGraph,
}

#[derive(Debug, Clone)]
pub struct RegionalLLIR {
    pub region_descriptors: Vec<SubgraphDescriptor>,
    pub regions: Vec<RegionalLLIRRegion>,
}

#[derive(Debug, Clone)]
struct RollingOccurrence {
    nodes: Vec<NodeIndex>,
    boundary_inputs: Vec<NodeIndex>,
    output_nodes: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
struct RollingCandidate {
    signature: String,
    occurrences: Vec<RollingOccurrence>,
    state_param_indices: Vec<usize>,
    savings: usize,
}

#[derive(Debug, Clone)]
struct RollingRun {
    occurrences: Vec<RollingOccurrence>,
    starts: Vec<usize>,
    window: usize,
}

#[derive(Debug, Clone, Default)]
struct RollingSearchDiagnostics {
    windows_probed: usize,
    adjacent_hash_matches: usize,
    repeated_signature_runs: usize,
    rejected_zero_state_params: usize,
    best_rejected: Option<RollingRejectedCandidate>,
    top_runs: Vec<String>,
}

#[derive(Debug, Clone)]
struct RollingRejectedCandidate {
    window: usize,
    repetitions: usize,
    boundary_inputs: usize,
    state_params: usize,
    savings: usize,
}

#[derive(Debug, Clone)]
struct RollingSearchReport {
    candidate: Option<RollingCandidate>,
    diagnostics: RollingSearchDiagnostics,
}

#[derive(Debug, Clone)]
struct AutoRegionPlan {
    descriptors: Vec<SubgraphDescriptor>,
    #[allow(dead_code)] // read only in tests
    loop_region_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct SingleRegionalizedEGraphPlan {
    representative_descriptors: Vec<SubgraphDescriptor>,
    stitched_representative_descriptors: Vec<SubgraphDescriptor>,
    region_groups: Vec<RegionGroup>,
    representative_root_indices: Vec<usize>,
}

/// A compiled bucket: (bucket_indices, representative_dyn_map, stitched_llir).
pub type BucketLLIR = (FxHashMap<char, usize>, FxHashMap<char, usize>, LLIRGraph);

/// A group of structurally identical chunks that share the same e-graph.
#[derive(Debug, Clone)]
struct RegionGroup {
    /// The representative chunk index (used for e-graph building and search)
    representative: usize,
    /// All chunk indices in this group (including the representative)
    members: Vec<usize>,
}

/// A bucket for a dynamic dimension, defining a range of valid values.
/// For an exact value, use `min == max` (zero-length range).
#[derive(Debug, Clone)]
pub struct DimBucket {
    pub min: usize,
    pub max: usize,
    representative_override: Option<usize>,
}

impl DimBucket {
    /// Create a new bucket covering `[min, max]` inclusive.
    /// For an exact value, pass `min == max`.
    pub fn new(min: usize, max: usize) -> Self {
        assert!(min <= max, "DimBucket min ({min}) must be <= max ({max})");
        DimBucket {
            min,
            max,
            representative_override: None,
        }
    }

    /// Override the representative value used during search profiling.
    /// Must be within `[min, max]`.
    pub fn representative(mut self, val: usize) -> Self {
        assert!(
            val >= self.min && val <= self.max,
            "Representative {val} must be in [{}, {}]",
            self.min,
            self.max
        );
        self.representative_override = Some(val);
        self
    }

    /// The representative value used during search profiling.
    /// Defaults to midpoint `(min + max) / 2`.
    pub fn representative_value(&self) -> usize {
        self.representative_override
            .unwrap_or((self.min + self.max) / 2)
    }

    /// Check if `val` falls within this bucket's range.
    pub fn contains(&self, val: usize) -> bool {
        val >= self.min && val <= self.max
    }
}

/// Options for controlling the genetic search algorithm.
///
/// Use the builder pattern to configure search parameters:
/// ```
/// use luminal::prelude::SearchOptions;
/// let opts = SearchOptions::new(5)
///     .generation_size(50)
///     .mutations(40)
///     .trials(15);
/// ```
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of graphs to evaluate
    pub limit: usize,
    /// Number of offspring per generation (default: 30)
    pub generation_size: usize,
    /// Number of mutations applied to each offspring (default: 30)
    pub mutations: usize,
    /// Number of profiling trials per candidate (default: 10)
    pub trials: usize,
    /// Number of best genomes to keep as parents per generation (default: 1)
    pub keep_best: usize,
    /// Optional per-candidate profiling timeout.
    pub profile_timeout: Option<std::time::Duration>,
    /// Optional per-group search timeout.
    pub group_timeout: Option<std::time::Duration>,
    /// Optional profiling dimension overrides.
    pub profile_dims: FxHashMap<char, usize>,
}

impl SearchOptions {
    /// Create new search options with the given limit. Other fields use defaults.
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            generation_size: 30,
            mutations: 30,
            trials: 10,
            keep_best: 1,
            profile_timeout: None,
            group_timeout: None,
            profile_dims: FxHashMap::default(),
        }
    }

    /// Set the number of offspring per generation.
    pub fn generation_size(mut self, generation_size: usize) -> Self {
        self.generation_size = generation_size;
        self
    }

    /// Set the number of mutations per offspring.
    pub fn mutations(mut self, mutations: usize) -> Self {
        self.mutations = mutations;
        self
    }

    /// Set the number of profiling trials per candidate.
    pub fn trials(mut self, trials: usize) -> Self {
        self.trials = trials;
        self
    }

    /// Set the number of best genomes to keep as parents per generation.
    pub fn keep_best(mut self, keep_best: usize) -> Self {
        self.keep_best = keep_best;
        self
    }

    /// Set an optional per-candidate profiling timeout.
    pub fn profile_timeout(mut self, profile_timeout: std::time::Duration) -> Self {
        self.profile_timeout = Some(profile_timeout);
        self
    }

    /// Set an optional per-group search timeout.
    pub fn group_timeout(mut self, group_timeout: std::time::Duration) -> Self {
        self.group_timeout = Some(group_timeout);
        self
    }

    /// Override a dynamic dimension value used during search profiling.
    pub fn profile_dim(mut self, dim: char, value: usize) -> Self {
        self.profile_dims.insert(dim, value);
        self
    }
}

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
#[derive(Debug)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: HLIRGraph,
    /// E-Graph search spaces (one per unique group when graph breaks are used)
    egraphs: Vec<SerializedEGraph>,
    /// Subgraph descriptors (one per chunk when graph breaks are used)
    region_descriptors: Vec<SubgraphDescriptor>,
    /// Groups of structurally identical chunks
    region_groups: Vec<RegionGroup>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
    /// Bucket definitions per dynamic dimension. Dimensions without buckets use a
    /// single implicit bucket (current behavior). When set, search compiles a
    /// separate LLIR per bucket combination and runtime dispatches automatically.
    pub dim_buckets: FxHashMap<char, Vec<DimBucket>>,
    /// Metadata for Input nodes: NodeIndex -> (label, dtype).
    /// Stored as plain data so it survives cross-binary type identity mismatches
    /// when external backend plugins are compiled separately.
    pub input_meta: FxHashMap<NodeIndex, (String, DType)>,
    /// Whether explicit loop regions were produced by the automatic loop-rolling prepass.
    auto_rolled_regions: bool,
    /// Virtual region plan produced by automatic loop rolling.
    auto_region_plan: Option<AutoRegionPlan>,
    /// Most recent regionalized LLIR result from search.
    pub last_regional_llir: Option<RegionalLLIR>,
    /// Single-egraph regionalized search plan used for deduplicated loop-region search.
    single_regional_egraph: Option<SingleRegionalizedEGraphPlan>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            dyn_map: Default::default(),
            graph: Default::default(),
            egraphs: Default::default(),
            region_descriptors: Default::default(),
            region_groups: Default::default(),
            ops: Default::default(),
            custom_ops: Default::default(),
            dim_buckets: Default::default(),
            input_meta: Default::default(),
            auto_rolled_regions: Default::default(),
            auto_region_plan: Default::default(),
            last_regional_llir: Default::default(),
            single_regional_egraph: Default::default(),
        }
    }
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    fn run_auto_loop_rolling_prepass(&mut self) {
        let before = self.graph.node_count();
        let inserted = if std::env::var("LUMINAL_NO_ROLL").is_ok() {
            0
        } else {
            self.auto_roll_loops_prepass()
        };
        if inserted == 0 {
            println!(
                "   {:>6}  no loop regions found (max body={})",
                "Rolled".cyan().bold(),
                before / 2,
            );
        }
    }

    /// Mutate the HLIR graph in place to fold N repeated body occurrences into
    /// a single body plus loop-marker ops. See `auto_roll_loops_prepass`.
    fn insert_loop_region_ops(&mut self, candidate: RollingCandidate) -> usize {
        use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopStart};
        use petgraph::visit::EdgeRef;

        let n_iters = candidate.occurrences.len();
        let loop_id = 0usize;

        let body_nodes: FxHashSet<NodeIndex> =
            candidate.occurrences[0].nodes.iter().copied().collect();
        let mut duplicate_body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
        for occ in &candidate.occurrences[1..] {
            for &n in &occ.nodes {
                duplicate_body_nodes.insert(n);
            }
        }

        let n_boundary = candidate.occurrences[0].boundary_inputs.len();
        let state_set: FxHashSet<usize> =
            candidate.state_param_indices.iter().copied().collect();

        let mut state_out_pos_per_slot: Vec<usize> =
            Vec::with_capacity(candidate.state_param_indices.len());
        let mut state_output_positions: FxHashSet<usize> = FxHashSet::default();
        for &p in &candidate.state_param_indices {
            let next_val = candidate.occurrences[1].boundary_inputs[p];
            let pos = candidate.occurrences[0]
                .output_nodes
                .iter()
                .position(|&n| n == next_val)
                .expect("state param must have a producer in output_nodes");
            state_out_pos_per_slot.push(pos);
            state_output_positions.insert(pos);
        }

        let mut created = 0usize;
        // Track all NodeIndex slots we newly assign for loop-marker ops.
        // StableGraph reuses freed node indices; removals later in this
        // function might target slots that happen to coincide with a new
        // loop-marker's NodeIndex, so we explicitly exclude those.
        let mut added_loop_ops: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut pending_output_removals: FxHashSet<NodeIndex> = FxHashSet::default();

        for (slot_idx, (&p, &out_pos)) in candidate
            .state_param_indices
            .iter()
            .zip(state_out_pos_per_slot.iter())
            .enumerate()
        {
            let initial = candidate.occurrences[0].boundary_inputs[p];
            let body_state_out = candidate.occurrences[0].output_nodes[out_pos];
            let last_state_out = candidate.occurrences[n_iters - 1].output_nodes[out_pos];
            let dtype = self.infer_node_dtype(initial);

            let loop_start = self.graph.add_node(Box::new(LoopStart {
                loop_id,
                slot_idx,
                iters: Expression::from(n_iters as i32),
                dtype,
            }));
            added_loop_ops.insert(loop_start);
            self.graph.add_edge(initial, loop_start, ());

            let edges_out_of_initial: Vec<_> = self
                .graph
                .edges_directed(initial, Direction::Outgoing)
                .filter(|e| body_nodes.contains(&e.target()))
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in edges_out_of_initial {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_start, dst, ());
            }

            let loop_end = self.graph.add_node(Box::new(LoopEnd {
                loop_id,
                slot_idx,
                dtype,
            }));
            added_loop_ops.insert(loop_end);
            self.graph.add_edge(body_state_out, loop_end, ());

            let external_edges: Vec<_> = self
                .graph
                .edges_directed(last_state_out, Direction::Outgoing)
                .filter(|e| {
                    let t = e.target();
                    !body_nodes.contains(&t) && !duplicate_body_nodes.contains(&t)
                })
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in external_edges {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_end, dst, ());
            }

            created += 2;
        }

        // Cache of structural hashes per HLIR node. Used to detect when two
        // boundary inputs are STRUCTURALLY identical (e.g., `Iota(z, (d1,d2))`
        // nodes appear as separate NodeIndex per layer but compute the same
        // value). Those should NOT be wrapped in LoopInput — they should stay
        // as shared constants so downstream egglog rules (e.g., MoE fusion,
        // which requires matching on `(Op (Iota ...) ...)` directly) can fire.
        let mut struct_hash_cache: FxHashMap<NodeIndex, u64> = FxHashMap::default();
        for p in 0..n_boundary {
            if state_set.contains(&p) {
                continue;
            }
            let per_iter_sources: Vec<NodeIndex> = candidate
                .occurrences
                .iter()
                .map(|occ| occ.boundary_inputs[p])
                .collect();
            if per_iter_sources.windows(2).all(|w| w[0] == w[1]) {
                continue;
            }
            // Structural-equality bypass: if all per-iter sources produce the
            // same structural hash, they're semantically one value. Rewire
            // iter-0's source to feed all body consumers directly, and skip
            // creating a LoopInput.
            let hashes: Vec<u64> = per_iter_sources
                .iter()
                .map(|&n| structural_hash_of(&self.graph, n, &mut struct_hash_cache))
                .collect();
            if hashes.windows(2).all(|w| w[0] == w[1]) {
                // All structurally identical — nothing to do. iter-0's source
                // already feeds iter-0 body nodes; for iter>0 clones we want
                // them to reference the same source. Since we're NOT creating
                // a LoopInput, unroll's `resolve_src` will fall through to the
                // `src` branch (not body_nodes, not markers, not input_per_iter)
                // and each iter i clone will read from the same shared source.
                continue;
            }

            let body_input = candidate.occurrences[0].boundary_inputs[p];
            let dtype = self.infer_node_dtype(body_input);
            let loop_input = self.graph.add_node(Box::new(LoopInput {
                loop_id,
                stream_id: p,
                dtype,
            }));
            added_loop_ops.insert(loop_input);
            for &src in &per_iter_sources {
                self.graph.add_edge(src, loop_input, ());
            }

            let body_edges: Vec<_> = self
                .graph
                .edges_directed(body_input, Direction::Outgoing)
                .filter(|e| body_nodes.contains(&e.target()))
                .map(|e| (e.id(), e.target()))
                .collect();
            for (eid, dst) in body_edges {
                self.graph.remove_edge(eid);
                self.graph.add_edge(loop_input, dst, ());
            }

            created += 1;
        }

        let n_outputs = candidate.occurrences[0].output_nodes.len();
        for q in 0..n_outputs {
            if state_output_positions.contains(&q) {
                continue;
            }

            // Two shapes are possible at position q:
            //   (a) the body exposes an intermediate value that downstream
            //       (outside-body) code then wraps in an Output HLIR node.
            //   (b) the body contains the Output HLIR directly (the `.output()`
            //       call put it inside the rolling window).
            // In (b) each occurrence's `output_nodes[q]` IS an Output node;
            // in (a) each needs an Output consumer via an outgoing edge.
            let mut targets: Vec<usize> = Vec::with_capacity(n_iters);
            let mut target_output_nodes: Vec<NodeIndex> = Vec::with_capacity(n_iters);
            let mut body_producer_for_loopoutput: Option<NodeIndex> = None;
            let mut complete = true;
            for occ in &candidate.occurrences {
                let node = occ.output_nodes[q];
                if let Some(op) = self.try_get_op::<crate::hlir::Output>(node) {
                    // Case (b): the node itself is an Output. Its producer is
                    // what LoopOutput will wrap.
                    let pred = self
                        .graph
                        .neighbors_directed(node, Direction::Incoming)
                        .next();
                    match pred {
                        Some(p) => {
                            if body_producer_for_loopoutput.is_none() {
                                body_producer_for_loopoutput = Some(p);
                            }
                            target_output_nodes.push(node);
                            targets.push(op.node);
                        }
                        None => {
                            complete = false;
                            break;
                        }
                    }
                } else {
                    // Case (a): find an Output consumer.
                    let out = self
                        .graph
                        .edges_directed(node, Direction::Outgoing)
                        .filter_map(|e| {
                            let t = e.target();
                            self.try_get_op::<crate::hlir::Output>(t).map(|op| (t, op.node))
                        })
                        .next();
                    match out {
                        Some((out_node, out_id)) => {
                            if body_producer_for_loopoutput.is_none() {
                                body_producer_for_loopoutput = Some(node);
                            }
                            target_output_nodes.push(out_node);
                            targets.push(out_id);
                        }
                        None => {
                            complete = false;
                            break;
                        }
                    }
                }
            }
            if !complete {
                continue;
            }
            let body_output = body_producer_for_loopoutput
                .expect("complete LoopOutput detection must have set body_producer");

            let loop_output = self.graph.add_node(Box::new(LoopOutput {
                loop_id,
                stream_id: q,
                targets,
            }));
            self.graph.add_edge(body_output, loop_output, ());
            added_loop_ops.insert(loop_output);

            // Mark iter 1..N's Output HLIR nodes for removal (defer so we
            // don't free NodeIndex slots that subsequent LoopOutput adds
            // would reuse — that reuse caused newly-created LoopOutputs to
            // be misidentified as duplicates below).
            for &out_node in &target_output_nodes[1..] {
                pending_output_removals.insert(out_node);
            }

            created += 1;
        }

        // Perform deferred Output HLIR removals (iter 1..N's Outputs whose
        // ids are now owned by LoopOutput.targets).
        for node in &pending_output_removals {
            self.graph.remove_node(*node);
        }

        // Delete duplicate body nodes. Skip any node that we just ADDED as
        // a LoopOutput (its NodeIndex might collide with a removed node's
        // slot).
        for &node in &duplicate_body_nodes {
            if added_loop_ops.contains(&node) {
                continue;
            }
            self.graph.remove_node(node);
        }

        if created > 0 {
            println!(
                "   {:>6}  rolled HLIR: {} loop ops inserted, {} duplicate body nodes deleted",
                "Rolled".cyan().bold(),
                created,
                duplicate_body_nodes.len(),
            );
        }
        created
    }

    /// Best-effort dtype lookup for a NodeIndex in the HLIR graph.
    fn infer_node_dtype(&self, node: NodeIndex) -> DType {
        if let Some((_, dt)) = self.input_meta.get(&node) {
            return *dt;
        }
        if let Some(op) = self.try_get_op::<crate::hlir::Input>(node) {
            return op.dtype;
        }
        if let Some(op) = self.try_get_op::<crate::hlir::Cast>(node) {
            return op.1;
        }
        for pred in self
            .graph
            .neighbors_directed(node, Direction::Incoming)
            .collect::<Vec<_>>()
        {
            let dt = self.infer_node_dtype(pred);
            if dt != DType::F32 || self.input_meta.contains_key(&pred) {
                return dt;
            }
        }
        DType::F32
    }

    fn regionalized_hlir_node_count(
        &self,
        descriptors: &[SubgraphDescriptor],
        groups: &[RegionGroup],
    ) -> usize {
        groups
            .iter()
            .flat_map(|group| descriptors[group.representative].nodes.iter().copied())
            .collect::<FxHashSet<_>>()
            .len()
    }

    fn full_hlir_op_count(&self) -> usize {
        self.graph
            .node_indices()
            .filter(|n| self.try_get_op::<crate::hlir::Input>(*n).is_none())
            .filter(|n| self.try_get_op::<crate::hlir::Output>(*n).is_none())
            .count()
    }

    fn regionalized_hlir_op_count(
        &self,
        descriptors: &[SubgraphDescriptor],
        groups: &[RegionGroup],
    ) -> usize {
        groups
            .iter()
            .flat_map(|group| descriptors[group.representative].nodes.iter().copied())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .filter(|n| self.try_get_op::<crate::hlir::Input>(*n).is_none())
            .filter(|n| self.try_get_op::<crate::hlir::Output>(*n).is_none())
            .count()
    }

    fn missing_graph_outputs(&self, llir: &LLIRGraph) -> Vec<usize> {
        let llir_outputs: FxHashSet<usize> = llir
            .node_indices()
            .filter_map(|n| llir[n].to_op::<crate::hlir::Output>().map(|out| out.node))
            .collect();
        self.graph
            .externals(Direction::Outgoing)
            .filter_map(|n| {
                self.try_get_op::<crate::hlir::Output>(n)
                    .map(|out| out.node)
            })
            .filter(|output_id| !llir_outputs.contains(output_id))
            .collect()
    }

    fn debug_regional_output_coverage(&self, missing_outputs: &[usize]) {
        let Some(regional) = self.last_regional_llir.as_ref() else {
            return;
        };
        let missing: FxHashSet<usize> = missing_outputs.iter().copied().collect();
        for (idx, region) in regional.regions.iter().enumerate() {
            let outputs: Vec<usize> = region
                .representative_llir
                .node_indices()
                .filter_map(|n| {
                    region.representative_llir[n]
                        .to_op::<crate::hlir::Output>()
                        .map(|out| out.node)
                })
                .sorted()
                .collect();
            let expected_outputs: Vec<usize> = self.region_descriptors
                [region.representative_region]
                .nodes
                .iter()
                .filter_map(|&n| {
                    self.try_get_op::<crate::hlir::Output>(n)
                        .map(|out| out.node)
                })
                .sorted()
                .collect();
            let covered_missing: Vec<usize> = outputs
                .iter()
                .copied()
                .filter(|id| missing.contains(id))
                .collect();
            let expected_missing: Vec<usize> = expected_outputs
                .iter()
                .copied()
                .filter(|id| missing.contains(id))
                .collect();
            println!(
                "   {:>6}  region {} rep={} members={} outputs={}/{} missing_hits={:?} expected_missing={:?}",
                "Rolled".yellow().bold(),
                idx,
                region.representative_region,
                region.member_regions.len(),
                outputs.len(),
                expected_outputs.len(),
                covered_missing,
                expected_missing,
            );
        }
        for &output_id in missing_outputs.iter().take(16) {
            let desc_indices: Vec<usize> = self
                .region_descriptors
                .iter()
                .enumerate()
                .filter_map(|(idx, desc)| {
                    desc.nodes
                        .contains(&NodeIndex::new(output_id))
                        .then_some(idx)
                })
                .collect();
            let group_indices: Vec<usize> = self
                .region_groups
                .iter()
                .enumerate()
                .filter_map(|(idx, group)| {
                    group
                        .members
                        .iter()
                        .any(|member| desc_indices.contains(member))
                        .then_some(idx)
                })
                .collect();
            println!(
                "   {:>6}  missing output {} lives in descriptors {:?} groups {:?}",
                "Rolled".yellow().bold(),
                output_id,
                desc_indices,
                group_indices,
            );
        }
    }

    pub fn regional_llir(&self) -> Option<&RegionalLLIR> {
        self.last_regional_llir.as_ref()
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Define buckets for a dynamic dimension.
    ///
    /// When buckets are set, `search()` compiles a separate optimized LLIR graph
    /// for each bucket combination. At runtime, `execute()` dispatches to the
    /// appropriate compiled graph based on the current `dyn_map` values.
    ///
    /// Buckets must not overlap and must cover all values that will be used at runtime.
    pub fn set_dim_buckets(&mut self, dimension: char, buckets: &[DimBucket]) {
        // Validate no overlapping ranges
        for (i, a) in buckets.iter().enumerate() {
            for b in buckets.iter().skip(i + 1) {
                assert!(
                    a.max < b.min || b.max < a.min,
                    "Overlapping buckets for dim '{}': [{}, {}] and [{}, {}]",
                    dimension,
                    a.min,
                    a.max,
                    b.min,
                    b.max,
                );
            }
        }
        self.dim_buckets.insert(dimension, buckets.to_vec());
    }

    /// Attempt to discover repeated HLIR regions and build explicit region
    /// descriptors for loop-carried state edges.
    /// Returns the number of detected inter-region boundaries.
    ///
    /// This is a conservative prepass:
    /// - only rolls candidates with at least one loop-carried state parameter
    /// - only inserts when the carried edge shapes can be inferred
    pub fn auto_roll_loops_prepass(&mut self) -> usize {
        self.auto_rolled_regions = false;
        self.auto_region_plan = None;
        let max_region_size = self.graph.node_count() / 2;
        if max_region_size < 1 {
            return 0;
        }
        println!(
            "   {:>6}  scanning {} HLIR nodes for loop regions (max body={})",
            "Rolled".cyan().bold(),
            self.graph.node_count(),
            max_region_size,
        );
        let report = self.best_rolling_candidate(max_region_size);
        let Some(candidate) = report.candidate else {
            self.print_rolling_search_diagnostics(&report.diagnostics);
            return 0;
        };
        println!(
            "   {:>6}  candidate: body={} trips={} boundary_inputs={} state_params={:?}",
            "Rolled".yellow().bold(),
            candidate.occurrences[0].nodes.len(),
            candidate.occurrences.len(),
            candidate.occurrences[0].boundary_inputs.len(),
            candidate.state_param_indices,
        );
        if let Some(rejected) = &report.diagnostics.best_rejected {
            println!(
                "   {:>6}  best rejected: body={} trips={} boundary_inputs={} state_params={} savings={}",
                "Rolled".yellow().bold(),
                rejected.window,
                rejected.repetitions,
                rejected.boundary_inputs,
                rejected.state_params,
                rejected.savings,
            );
        }
        for run in report.diagnostics.top_runs.iter().take(5) {
            println!("   {:>6}  run: {}", "Rolled".yellow().bold(), run);
        }
        if candidate.occurrences.len() < 2 {
            return 0;
        }

        // Mutate the HLIR in place — insert LoopStart/LoopEnd/LoopInput/
        // LoopOutput markers, delete N-1 duplicate bodies. `auto_rolled_regions`
        // and `auto_region_plan` remain false/None so downstream takes the
        // plain single-root egglog path; the loop structure is encoded in the
        // HLIR graph itself.
        self.insert_loop_region_ops(candidate)
    }

    fn print_rolling_search_diagnostics(&self, diagnostics: &RollingSearchDiagnostics) {
        let best_rejected = diagnostics
            .best_rejected
            .as_ref()
            .map(|candidate| {
                format!(
                    "best rejected: body={} trips={} boundary_inputs={} state_params={} savings={}",
                    candidate.window,
                    candidate.repetitions,
                    candidate.boundary_inputs,
                    candidate.state_params,
                    candidate.savings
                )
            })
            .unwrap_or_else(|| "best rejected: none".to_string());
        println!(
            "   {:>6}  diagnostics: windows={} hash_matches={} repeated_runs={} rejected(zero_state={}); {}",
            "Rolled".yellow().bold(),
            diagnostics.windows_probed,
            diagnostics.adjacent_hash_matches,
            diagnostics.repeated_signature_runs,
            diagnostics.rejected_zero_state_params,
            best_rejected,
        );
        for run in diagnostics.top_runs.iter().take(5) {
            println!("   {:>6}  run: {}", "Rolled".yellow().bold(), run);
        }
    }

    fn best_rolling_candidate(&self, max_region_size: usize) -> RollingSearchReport {
        let Some(full_topo) = toposort(&self.graph, None).ok() else {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        };
        let topo: Vec<NodeIndex> = full_topo
            .into_iter()
            .filter(|n| self.try_get_op::<crate::hlir::Input>(*n).is_none())
            .collect();
        if topo.len() < 2 {
            return RollingSearchReport {
                candidate: None,
                diagnostics: RollingSearchDiagnostics::default(),
            };
        }
        let uses = build_uses(&self.graph);
        let topo_index: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let max_window = max_region_size.min(topo.len() / 2);
        let probe_windows = rolling_probe_window_sizes(max_window);
        let node_hashes: Vec<u64> = topo
            .iter()
            .map(|&node| cheap_rolling_node_hash(&self.graph, node))
            .collect();
        let rolling_hash = RollingHash64::new(&node_hashes);
        let mut diagnostics = RollingSearchDiagnostics::default();
        let mut best_overall: Option<RollingCandidate> = None;
        let mut discovered_runs: Vec<RollingRun> = Vec::new();

        // Search all window sizes down to 1, using cheap rolling hashes only as a
        // gate for expensive canonicalization. Candidate selection remains purely
        // based on valid HLIR-op reduction.
        for window in probe_windows {
            let mut start = 0usize;
            while start + window * 2 <= topo.len() {
                diagnostics.windows_probed += 1;
                let first_hash = rolling_hash.window_hash(start, window);
                let second_hash = rolling_hash.window_hash(start + window, window);
                if first_hash != second_hash {
                    start += 1;
                    continue;
                }
                diagnostics.adjacent_hash_matches += 1;

                let mut occs = vec![];
                let mut starts = vec![];
                let first_nodes = topo[start..start + window].to_vec();
                let Some((sig, first_boundary, first_outputs)) =
                    canonicalize_occurrence(&self.graph, &first_nodes, &uses, &topo_index)
                else {
                    start += 1;
                    continue;
                };
                starts.push(start);
                occs.push(RollingOccurrence {
                    nodes: first_nodes,
                    boundary_inputs: first_boundary,
                    output_nodes: first_outputs,
                });

                let mut pos = start + window;
                while pos + window <= topo.len() {
                    if rolling_hash.window_hash(pos, window) != first_hash {
                        break;
                    }
                    let nodes = topo[pos..pos + window].to_vec();
                    let Some((next_sig, boundary_inputs, output_nodes)) =
                        canonicalize_occurrence(&self.graph, &nodes, &uses, &topo_index)
                    else {
                        break;
                    };
                    if next_sig != sig {
                        break;
                    }
                    starts.push(pos);
                    occs.push(RollingOccurrence {
                        nodes,
                        boundary_inputs,
                        output_nodes,
                    });
                    pos += window;
                }
                if occs.len() < 2 {
                    start += 1;
                    continue;
                }
                diagnostics.repeated_signature_runs += 1;
                discovered_runs.push(RollingRun {
                    occurrences: occs.clone(),
                    starts: starts.clone(),
                    window,
                });
                let stride = starts
                    .windows(2)
                    .next()
                    .map(|w| w[1].saturating_sub(w[0]))
                    .unwrap_or(0);
                let summary = format!(
                    "body={} trips={} stride={} boundary_inputs={} state_params={} starts={:?}",
                    window,
                    occs.len(),
                    stride,
                    occs[0].boundary_inputs.len(),
                    collect_state_params(&occs, &uses, &self.graph).len(),
                    starts.iter().copied().take(4).collect::<Vec<_>>()
                );
                if occs.len() >= 20 && diagnostics.top_runs.len() < 16 {
                    diagnostics.top_runs.push(summary);
                }

                let state_params = collect_state_params(&occs, &uses, &self.graph);
                if state_params.is_empty() {
                    let rejected = RollingRejectedCandidate {
                        window,
                        repetitions: occs.len(),
                        boundary_inputs: occs[0].boundary_inputs.len(),
                        state_params: state_params.len(),
                        savings: window * (occs.len() - 1),
                    };
                    diagnostics.rejected_zero_state_params += 1;
                    let replace = diagnostics.best_rejected.as_ref().is_none_or(|best| {
                        (rejected.savings, rejected.repetitions, rejected.window)
                            > (best.savings, best.repetitions, best.window)
                    });
                    if replace {
                        diagnostics.best_rejected = Some(rejected);
                    }
                    start = pos.saturating_sub(window).max(start + 1);
                    continue;
                }

                let savings = window * (occs.len() - 1);
                let candidate = RollingCandidate {
                    signature: sig,
                    occurrences: occs,
                    state_param_indices: state_params,
                    savings,
                };
                let replace = best_overall.as_ref().is_none_or(|b| {
                    (candidate.savings, candidate.occurrences.len())
                        > (b.savings, b.occurrences.len())
                });
                if replace {
                    best_overall = Some(candidate);
                }
                start = pos.saturating_sub(window).max(start + 1);
            }
        }
        if let Some(best) = best_overall.take() {
            best_overall = Some(grow_rolling_candidate(
                &self.graph,
                &uses,
                &topo_index,
                best,
                &discovered_runs,
            ));
        }
        RollingSearchReport {
            candidate: best_overall,
            diagnostics,
        }
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        let name = name.to_string();
        let id = self.graph.add_node(Box::new(crate::hlir::Input {
            node: 0,
            label: name.clone(),
            dtype: DType::default(),
        }));
        self.get_op_mut::<crate::hlir::Input>(id).node = id.index();
        self.input_meta.insert(id, (name.clone(), DType::default()));
        GraphTensor {
            id,
            graph_ref: self,
            shape: ShapeTracker::new(shape),
            dtype: DType::default(),
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<NodeIndex> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
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

    /// Add an op to the graph with the given input edges. Returns the new node's index.
    ///
    /// ```rust
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx.add_op(
    ///     luminal::hlir::Mul { input_shapes: vec![a.shape, a.shape], ..Default::default() },
    ///     &[a.id],
    /// );
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph(), a.dtype);
    /// ```
    pub fn add_op<O: HLIROp + 'static>(&mut self, op: O, inputs: &[NodeIndex]) -> NodeIndex {
        let id = self.graph.add_node(Box::new(op));
        for &src in inputs {
            self.graph.add_edge(src, id, ());
        }
        id
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
        let input_ids = inputs.to_ids();
        let id = self.add_op(
            CustomOpKind {
                id: self.custom_ops.len() - 1,
                dtype,
            },
            &input_ids,
        );
        GraphTensor::from_id(
            id,
            ShapeTracker::new_with_element_bits(shape, dtype.bits()),
            self,
            dtype,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
        self.last_regional_llir = None;
        self.single_regional_egraph = None;
        self.run_auto_loop_rolling_prepass();
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = self
            .auto_region_plan
            .as_ref()
            .map(|p| p.descriptors.clone())
            .unwrap_or_else(|| default_region_descriptors(self));

        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];

            self.region_groups = vec![RegionGroup {
                representative: 0,
                members: vec![0],
            }];
        } else {
            println!(
                "   {:>6}  {} regions from boundaries",
                "Split".cyan().bold(),
                subgraphs.len()
            );
            let groups = self.auto_rolled_region_groups(&subgraphs);
            self.build_single_regionalized_egraph(&subgraphs, groups, &ops, cleanup_hlir);
        }
        self.region_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        self.last_regional_llir = None;
        self.single_regional_egraph = None;
        self.run_auto_loop_rolling_prepass();
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = self
            .auto_region_plan
            .as_ref()
            .map(|p| p.descriptors.clone())
            .unwrap_or_else(|| default_region_descriptors(self));
        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
            self.region_groups = vec![RegionGroup {
                representative: 0,
                members: vec![0],
            }];
        } else {
            let groups = self.auto_rolled_region_groups(&subgraphs);
            self.build_single_regionalized_egraph(&subgraphs, groups, &ops, cleanup_hlir);
        }
        self.region_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    fn auto_rolled_region_groups(&self, subgraphs: &[SubgraphDescriptor]) -> Vec<RegionGroup> {
        let egglog_texts: Vec<(String, String)> = subgraphs
            .iter()
            .map(|sg| hlir_subgraph_to_egglog(self, sg))
            .collect();
        let mut hash_to_chunks: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        for (i, (text, _)) in egglog_texts.iter().enumerate() {
            let h = hash_egglog_normalized(text);
            hash_to_chunks.entry(h).or_default().push(i);
        }
        let mut groups: Vec<RegionGroup> = hash_to_chunks
            .into_values()
            .map(|mut members| {
                members.sort_unstable();
                RegionGroup {
                    representative: members[0],
                    members,
                }
            })
            .collect();
        if groups.is_empty() {
            groups = (0..subgraphs.len())
                .map(|i| RegionGroup {
                    representative: i,
                    members: vec![i],
                })
                .collect();
        }
        groups.sort_by_key(|g| (std::cmp::Reverse(g.members.len()), g.representative));
        groups
    }

    fn build_single_regionalized_egraph(
        &mut self,
        subgraphs: &[SubgraphDescriptor],
        groups: Vec<RegionGroup>,
        ops: &[Arc<Box<dyn EgglogOp>>],
        cleanup_hlir: bool,
    ) {
        let mut ordered_groups = groups;
        ordered_groups.sort_by_key(|g| g.representative);
        let representative_descriptors: Vec<_> = ordered_groups
            .iter()
            .map(|group| subgraphs[group.representative].clone())
            .collect();
        let stitched_representative_descriptors =
            deduped_representative_descriptors(subgraphs, &ordered_groups, &self.graph);
        let (program, roots) = build_regionalized_egglog_program(self, &representative_descriptors);
        println!(
            "   {:>6}  building 1 regionalized e-graph from {} representative regions ({} total chunks)",
            "Graphs".cyan().bold(),
            representative_descriptors.len(),
            subgraphs.len()
        );
        let start = std::time::Instant::now();
        let egraph = run_egglog_multi_roots(&program, &roots, ops, cleanup_hlir).unwrap();
        println!(
            "   {:>6}  regionalized e-graph ready in {}",
            "Graphs".cyan().bold(),
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );
        self.egraphs = vec![egraph];
        self.region_groups = ordered_groups.clone();
        self.single_regional_egraph = Some(SingleRegionalizedEGraphPlan {
            representative_descriptors,
            stitched_representative_descriptors,
            region_groups: ordered_groups,
            representative_root_indices: (0..roots.len()).collect(),
        });
    }

    /// Get a reference to the first e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraphs.first()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime + 'static>(&mut self, runtime: R, limit: usize) -> R {
        let mut rng = rand::rng();
        self.search_options(runtime, SearchOptions::new(limit), &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn search_options<R: Runtime + 'static, G: rand::Rng>(
        &mut self,
        mut runtime: R,
        options: SearchOptions,
        rng: &mut G,
    ) -> R {
        runtime.set_profile_timeout(options.profile_timeout);
        self.dump_regionalized_hlir_before_search();
        if self.dim_buckets.is_empty() {
            // No buckets: existing single-search path
            let stitched =
                self.search_single(&mut runtime, &options, rng, &self.dyn_map.clone(), None);

            runtime.clear_intermediate_buffers();
            runtime.load_llir(&stitched);
            runtime.set_profile_timeout(None);
            runtime
        } else {
            // Bucketed search: compile one LLIR per bucket combination
            let bucket_combos = self.bucket_combinations();
            let n_combos = bucket_combos.len();
            let mut bucket_llirs: Vec<BucketLLIR> = Vec::with_capacity(n_combos);

            for (combo_idx, (bucket_indices, representative_dyn_map)) in
                bucket_combos.into_iter().enumerate()
            {
                let bucket_label = self.format_bucket_label(&bucket_indices);
                println!(
                    "   {:>6}  Group {}/{}: {}",
                    "Search".cyan().bold(),
                    combo_idx + 1,
                    n_combos,
                    bucket_label,
                );

                let stitched = self.search_single(
                    &mut runtime,
                    &options,
                    rng,
                    &representative_dyn_map,
                    Some((combo_idx, n_combos)),
                );
                bucket_llirs.push((bucket_indices, representative_dyn_map, stitched));
            }

            runtime.clear_intermediate_buffers();
            runtime.load_llir_buckets(&self.dim_buckets, &bucket_llirs);
            runtime.set_profile_timeout(None);
            runtime
        }
    }

    /// Compute cartesian product of all bucket combinations.
    /// Returns Vec of (bucket_indices, representative_dyn_map).
    fn bucket_combinations(&self) -> Vec<(FxHashMap<char, usize>, FxHashMap<char, usize>)> {
        let mut dims: Vec<(char, &Vec<DimBucket>)> =
            self.dim_buckets.iter().map(|(c, b)| (*c, b)).collect();
        dims.sort_by_key(|(c, _)| *c);

        let mut combos: Vec<(FxHashMap<char, usize>, FxHashMap<char, usize>)> =
            vec![(FxHashMap::default(), self.dyn_map.clone())];

        for (dim, buckets) in &dims {
            let mut new_combos = Vec::new();
            for (existing_indices, existing_dyn_map) in &combos {
                for (bucket_idx, bucket) in buckets.iter().enumerate() {
                    let mut indices = existing_indices.clone();
                    indices.insert(*dim, bucket_idx);
                    let mut dyn_map = existing_dyn_map.clone();
                    dyn_map.insert(*dim, bucket.representative_value());
                    new_combos.push((indices, dyn_map));
                }
            }
            combos = new_combos;
        }

        combos
    }

    /// Format a human-readable label for a bucket combination.
    fn format_bucket_label(&self, bucket_indices: &FxHashMap<char, usize>) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut dims: Vec<_> = bucket_indices.iter().collect();
        dims.sort_by_key(|(c, _)| **c);
        for (dim, &idx) in dims {
            let bucket = &self.dim_buckets[dim][idx];
            if bucket.min == bucket.max {
                parts.push(format!("{}={}", dim, bucket.min));
            } else {
                parts.push(format!(
                    "{}=[{},{}]@{}",
                    dim,
                    bucket.min,
                    bucket.max,
                    bucket.representative_value()
                ));
            }
        }
        parts.join(", ")
    }

    /// Run the genetic search for all chunk groups using the given dyn_map for profiling.
    /// Returns the stitched LLIR graph.
    /// `bucket_progress`: if `Some((current_bucket_idx, total_buckets))`, shows an extra
    /// "Bucket" bar for bucket-level progress and renames the group-level "Bucket" to "Group".
    fn search_single<R: Runtime + 'static, G: rand::Rng>(
        &mut self,
        runtime: &mut R,
        options: &SearchOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
    ) -> LLIRGraph {
        if self.single_regional_egraph.is_some() {
            let stitched = self.search_single_regionalized_deduped(
                runtime,
                options,
                rng,
                dyn_map,
                bucket_progress,
            );
            let missing_outputs = self.missing_graph_outputs(&stitched);
            if missing_outputs.is_empty() {
                return stitched;
            }
            println!(
                "   {:>6}  regionalized LLIR missing graph outputs {:?}, falling back to full-graph search",
                "Rolled".yellow().bold(),
                missing_outputs.iter().copied().take(8).collect::<Vec<_>>(),
            );
            self.debug_regional_output_coverage(&missing_outputs);
            self.single_regional_egraph = None;
            self.last_regional_llir = None;
            self.auto_rolled_regions = false;
            self.auto_region_plan = None;
            self.region_descriptors = default_region_descriptors(self);
            self.region_groups = vec![RegionGroup {
                representative: 0,
                members: vec![0],
            }];
            let cleanup_hlir = TypeId::of::<R>() != TypeId::of::<NativeRuntime>();
            let ops = self.ops.as_ref().unwrap().clone();
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
            return self.search_single(runtime, options, rng, dyn_map, bucket_progress);
        }
        let mut profile_dyn_map = dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        let limit = options.limit;
        let n_chunks = self.region_descriptors.len();
        let n_groups = self.region_groups.len();
        let multi_chunk = n_chunks > 1;
        let ops = self.ops.as_ref().unwrap();
        let start = std::time::Instant::now();

        // Label for the group-level "total" bar: "Group" when in bucketed mode, "Bucket" otherwise
        let group_total_label = if bucket_progress.is_some() {
            "Group"
        } else {
            "Bucket"
        };
        // How many progress bar lines we render (for ANSI cursor management)
        // multi_chunk without buckets: 2 lines (Graph + Bucket)
        // multi_chunk with buckets: 3 lines (Graph + Group + Bucket)
        // single chunk without buckets: 1 line (Search)
        // single chunk with buckets: 2 lines (Search + Bucket)
        let n_bar_lines =
            if multi_chunk { 2 } else { 1 } + if bucket_progress.is_some() { 1 } else { 0 };

        // Allocate dummy buffers for boundary inputs so groups can be profiled
        for desc in &self.region_descriptors {
            for bi in &desc.boundary_inputs {
                if !runtime.has_hlir_buffer(bi.break_node.index()) {
                    let n_elements = bi
                        .shape
                        .n_elements()
                        .exec(&profile_dyn_map)
                        .expect("Failed to resolve boundary input shape");
                    let n_bytes = n_elements * bi.dtype.bits() / 8;
                    runtime.allocate_dummy_input(bi.break_node.index(), n_bytes);
                }
            }
        }

        // Search each group's representative.
        let mut group_best_llirs: Vec<Option<LLIRGraph>> = (0..n_groups).map(|_| None).collect();
        let mut group_best_genomes: Vec<Option<crate::egglog_utils::EGraphChoiceSet>> =
            (0..n_groups).map(|_| None).collect();
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

        // Render progress bars. Prints the right number of lines depending on mode.
        // Returns the number of lines printed (for ANSI cursor management).
        let render_bars = |n_graphs: usize,
                           limit: usize,
                           group_idx: usize,
                           n_groups: usize,
                           multi_chunk: bool,
                           group_total_label: &str,
                           bucket_progress: Option<(usize, usize)>|
         -> usize {
            let mut lines = 0;
            if multi_chunk {
                print!(
                    "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                    "Graph".cyan().bold(),
                    make_bar(n_graphs, limit),
                );
                lines += 1;
                print!(
                    "\n\x1b[2K  {:>6}  {} {group_idx}/{n_groups}",
                    group_total_label.cyan().bold(),
                    make_bar(group_idx, n_groups),
                );
                lines += 1;
            } else {
                print!(
                    "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
                    "Search".cyan().bold(),
                    make_bar(n_graphs, limit),
                );
                lines += 1;
            }
            if let Some((bucket_idx, n_buckets)) = bucket_progress {
                print!(
                    "\n\x1b[2K  {:>6}  {} {}/{n_buckets}",
                    "Bucket".cyan().bold(),
                    make_bar(bucket_idx, n_buckets),
                    bucket_idx,
                );
                lines += 1;
            }
            lines
        };

        for (group_idx, group) in self.region_groups.iter().enumerate() {
            let group_start = std::time::Instant::now();
            let egraph = &self.egraphs[group_idx];
            let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();

            // Clear intermediate buffers from previous group's profiling
            runtime.clear_intermediate_buffers();

            // Find a viable initial genome (may need multiple attempts if some panic)
            let (mut best_genome, mut best_graph, mut best_metric, display, mut n_graphs);
            let mut init_attempts = 0;
            loop {
                init_attempts += 1;
                if init_attempts > 100 {
                    panic!(
                        "Failed to find a viable initial genome for group {group_idx} after 100 attempts"
                    );
                }
                let genome = random_initial_choice(egraph, rng);
                prev_selected.insert(hash_choice_set(&genome));

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let graph = egglog_to_llir(
                        egraph,
                        genome.clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );
                    runtime.clear_intermediate_buffers();
                    let (rep_metric, rep_display) =
                        runtime.profile(&graph, &profile_dyn_map, options.trials);
                    let mut metrics = vec![rep_metric.clone()];
                    let mut has_nan = runtime.has_nan_outputs(&graph, &profile_dyn_map);

                    if self.auto_rolled_regions && group.members.len() > 1 {
                        for &chunk_idx in &group.members {
                            if chunk_idx == group.representative {
                                continue;
                            }
                            let (node_remap, custom_op_id_remap) = build_region_remaps(
                                &self.region_descriptors[group.representative],
                                &self.region_descriptors[chunk_idx],
                                &self.graph,
                            );
                            let mut local_list_cache = FxHashMap::default();
                            let mut local_expr_cache = FxHashMap::default();
                            let custom_remap = if custom_op_id_remap.is_empty() {
                                None
                            } else {
                                Some(&custom_op_id_remap)
                            };
                            let mut llir = egglog_to_llir(
                                egraph,
                                genome.clone(),
                                ops,
                                &self.custom_ops,
                                &mut local_list_cache,
                                &mut local_expr_cache,
                                custom_remap,
                            );
                            remap_llir_io_nodes(&mut llir, &node_remap, &self.graph);
                            runtime.clear_intermediate_buffers();
                            let (metric, _) =
                                runtime.profile(&llir, &profile_dyn_map, options.trials);
                            metrics.push(metric);
                            has_nan |= runtime.has_nan_outputs(&llir, &profile_dyn_map);
                        }
                    }
                    let combined_metric = R::aggregate_profile_metrics(&metrics);
                    let display = if self.auto_rolled_regions && group.members.len() > 1 {
                        format!("{rep_display} (regional x{})", group.members.len())
                    } else {
                        rep_display
                    };
                    (graph, (combined_metric, display), has_nan)
                }));

                match result {
                    Ok((graph, (metric, disp), has_nan)) if !has_nan => {
                        best_genome = genome;
                        best_graph = graph;
                        best_metric = metric;
                        display = disp;
                        n_graphs = 1;
                        break;
                    }
                    Ok(_) | Err(_) => {
                        if options
                            .group_timeout
                            .is_some_and(|timeout| group_start.elapsed() >= timeout)
                        {
                            panic!(
                                "Failed to find a viable initial genome for group {group_idx} before timeout"
                            );
                        }
                        list_cache.clear();
                        expr_cache.clear();
                        continue;
                    }
                }
            }

            // Print initial result and progress
            {
                let multiplier = if group.members.len() > 1 {
                    format!(" ({}x)", group.members.len())
                } else {
                    String::new()
                };
                let msg = format!(
                    "   {:>8} {}{multiplier}",
                    format!("Graph {group_idx}").cyan().bold(),
                    display,
                );
                if bars_drawn {
                    // Move up past existing bar lines to overwrite them
                    for _ in 1..n_bar_lines {
                        print!("\x1b[1A");
                    }
                    print!("\r\x1b[2K");
                }
                println!("{msg}");
                render_bars(
                    n_graphs,
                    limit,
                    group_idx,
                    n_groups,
                    multi_chunk,
                    group_total_label,
                    bucket_progress,
                );
                std::io::stdout().flush().unwrap();
                bars_drawn = true;
            }

            // Track top-N parents for offspring generation
            let mut parents: Vec<(R::ProfileMetric, crate::egglog_utils::EGraphChoiceSet<'_>)> =
                vec![(best_metric.clone(), best_genome.clone())];

            while n_graphs < limit {
                if options
                    .group_timeout
                    .is_some_and(|timeout| group_start.elapsed() >= timeout)
                {
                    break;
                }

                // Generate offspring from all parents, dividing budget evenly
                let budget = (limit - n_graphs).min(options.generation_size);
                let per_parent = budget.div_ceil(parents.len());
                let mut all_offspring = Vec::new();
                for (_, parent_genome) in &parents {
                    let remaining = budget.saturating_sub(all_offspring.len());
                    if remaining == 0 {
                        break;
                    }
                    all_offspring.extend(extract_generation(
                        egraph,
                        parent_genome,
                        per_parent.min(remaining),
                        options.mutations,
                        &mut prev_selected,
                        rng,
                    ));
                }
                if all_offspring.is_empty() {
                    break;
                }

                for genome in all_offspring {
                    if options
                        .group_timeout
                        .is_some_and(|timeout| group_start.elapsed() >= timeout)
                    {
                        break;
                    }
                    n_graphs += 1;
                    list_cache.clear();
                    expr_cache.clear();

                    let profile_result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let llir_graph = egglog_to_llir(
                                egraph,
                                genome.clone(),
                                ops,
                                &self.custom_ops,
                                &mut list_cache,
                                &mut expr_cache,
                                None,
                            );
                            runtime.clear_intermediate_buffers();
                            let (rep_metric, rep_display) =
                                runtime.profile(&llir_graph, &profile_dyn_map, options.trials);
                            let mut metrics = vec![rep_metric.clone()];
                            let mut has_nan =
                                runtime.has_nan_outputs(&llir_graph, &profile_dyn_map);

                            if self.auto_rolled_regions && group.members.len() > 1 {
                                for &chunk_idx in &group.members {
                                    if chunk_idx == group.representative {
                                        continue;
                                    }
                                    let (node_remap, custom_op_id_remap) = build_region_remaps(
                                        &self.region_descriptors[group.representative],
                                        &self.region_descriptors[chunk_idx],
                                        &self.graph,
                                    );
                                    let mut local_list_cache = FxHashMap::default();
                                    let mut local_expr_cache = FxHashMap::default();
                                    let custom_remap = if custom_op_id_remap.is_empty() {
                                        None
                                    } else {
                                        Some(&custom_op_id_remap)
                                    };
                                    let mut llir = egglog_to_llir(
                                        egraph,
                                        genome.clone(),
                                        ops,
                                        &self.custom_ops,
                                        &mut local_list_cache,
                                        &mut local_expr_cache,
                                        custom_remap,
                                    );
                                    remap_llir_io_nodes(&mut llir, &node_remap, &self.graph);
                                    runtime.clear_intermediate_buffers();
                                    let (metric, _) =
                                        runtime.profile(&llir, &profile_dyn_map, options.trials);
                                    metrics.push(metric);
                                    has_nan |= runtime.has_nan_outputs(&llir, &profile_dyn_map);
                                }
                            }
                            let combined_metric = R::aggregate_profile_metrics(&metrics);
                            let display = if self.auto_rolled_regions && group.members.len() > 1 {
                                format!("{rep_display} (regional x{})", group.members.len())
                            } else {
                                rep_display
                            };
                            (((combined_metric, display), llir_graph), has_nan)
                        }));

                    let ((new_metric, display_metric), llir_graph) = match profile_result {
                        Ok((((metric, display), graph), false)) => ((metric, display), graph),
                        Ok((_, true)) | Err(_) => {
                            // NaN or panic — redraw bars and skip
                            for _ in 1..n_bar_lines {
                                print!("\x1b[1A");
                            }
                            print!("\r\x1b[2K");
                            render_bars(
                                n_graphs,
                                limit,
                                group_idx,
                                n_groups,
                                multi_chunk,
                                group_total_label,
                                bucket_progress,
                            );
                            std::io::stdout().flush().unwrap();
                            continue;
                        }
                    };

                    // Update parents list (keep top-N for next generation)
                    let dominated_by_all = parents.len() >= options.keep_best
                        && !parents.last().unwrap().0.gt(&new_metric);
                    if !dominated_by_all {
                        let pos = parents
                            .iter()
                            .position(|(m, _)| {
                                new_metric
                                    .partial_cmp(m)
                                    .is_some_and(|o| o == std::cmp::Ordering::Less)
                            })
                            .unwrap_or(parents.len());
                        parents.insert(pos, (new_metric.clone(), genome.clone()));
                        if parents.len() > options.keep_best {
                            parents.truncate(options.keep_best);
                        }
                    }

                    let new_best = best_metric.gt(&new_metric);
                    if new_best {
                        best_metric = new_metric;
                        best_graph = llir_graph;
                        best_genome = genome.clone();
                    }

                    if new_best {
                        let msg = format!("   {:>6} {display_metric}", "Searched".green().bold());
                        // Move cursor up to overwrite bars, print new best, re-render bars
                        for _ in 1..n_bar_lines {
                            print!("\x1b[1A");
                        }
                        print!("\r\x1b[2K");
                        println!("{msg}");
                    } else {
                        // Just move cursor up to overwrite bars
                        for _ in 1..n_bar_lines {
                            print!("\x1b[1A");
                        }
                        print!("\r\x1b[2K");
                    }
                    render_bars(
                        n_graphs,
                        limit,
                        group_idx,
                        n_groups,
                        multi_chunk,
                        group_total_label,
                        bucket_progress,
                    );
                    std::io::stdout().flush().unwrap();
                }
            }

            group_best_llirs[group_idx] = Some(best_graph);
            group_best_genomes[group_idx] = Some(best_genome);
        }

        // Clear progress bars
        if bars_drawn {
            // Move up to first bar line and clear all bar lines
            for _ in 1..n_bar_lines {
                print!("\x1b[1A");
            }
            print!("\r");
            for _ in 0..n_bar_lines {
                println!("\x1b[2K");
            }
            // Move back up to where we started
            for _ in 0..n_bar_lines {
                print!("\x1b[1A");
            }
            print!("\r");
            std::io::stdout().flush().unwrap();
        }

        // Build explicit regionalized LLIR, then unroll into full LLIR.
        let mut regions = Vec::with_capacity(self.region_groups.len());
        for (group_idx, group) in self.region_groups.iter().enumerate() {
            let representative_llir = group_best_llirs[group_idx]
                .take()
                .unwrap_or_else(|| panic!("Missing representative LLIR for group {group_idx}"));
            regions.push(RegionalLLIRRegion {
                representative_region: group.representative,
                member_regions: group.members.clone(),
                representative_llir,
            });
        }
        let regional = RegionalLLIR {
            region_descriptors: self.region_descriptors.clone(),
            regions,
        };
        let stitched = regional.unroll(&self.graph);
        self.last_regional_llir = Some(regional);

        println!(
            "   {:>6}  {} groups ({} regions) in {}",
            "Searched".green().bold(),
            n_groups,
            n_chunks,
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );

        stitched
    }

    fn search_single_regionalized_deduped<R: Runtime, G: rand::Rng>(
        &mut self,
        runtime: &mut R,
        options: &SearchOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
    ) -> LLIRGraph {
        let plan = self
            .single_regional_egraph
            .clone()
            .expect("single regionalized e-graph plan should be present");
        let mut profile_dyn_map = dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        let limit = options.limit.max(1);
        let egraph = &self.egraphs[0];
        let ops = self.ops.as_ref().unwrap();
        let start = std::time::Instant::now();

        if bucket_progress.is_some() {
            println!(
                "   {:>6}  single regionalized e-graph search for bucketed execution is experimental",
                "Search".cyan().bold()
            );
        }

        for desc in &plan.stitched_representative_descriptors {
            for bi in &desc.boundary_inputs {
                if !runtime.has_hlir_buffer(bi.break_node.index()) {
                    let n_elements = bi
                        .shape
                        .n_elements()
                        .exec(&profile_dyn_map)
                        .expect("Failed to resolve boundary input shape");
                    let n_bytes = n_elements * bi.dtype.bits() / 8;
                    runtime.allocate_dummy_input(bi.break_node.index(), n_bytes);
                }
            }
        }

        let mut eval_genome = |genome: &crate::egglog_utils::EGraphChoiceSet<'_>| {
            let mut representative_llirs =
                Vec::with_capacity(plan.representative_root_indices.len());
            let mut stitched_representative_llirs =
                Vec::with_capacity(plan.representative_root_indices.len());
            for (rep_idx, &root_idx) in plan.representative_root_indices.iter().enumerate() {
                let mut list_cache = FxHashMap::default();
                let mut expr_cache = FxHashMap::default();
                let llir = egglog_to_llir_from_root(
                    egraph,
                    genome.clone(),
                    ops,
                    &self.custom_ops,
                    &mut list_cache,
                    &mut expr_cache,
                    None,
                    &egraph.roots[root_idx],
                );
                let mut stitched_llir = llir.clone();
                let (node_remap, _custom_op_id_remap) = build_region_remaps(
                    &plan.representative_descriptors[rep_idx],
                    &plan.stitched_representative_descriptors[rep_idx],
                    &self.graph,
                );
                remap_llir_io_nodes(&mut stitched_llir, &node_remap, &self.graph);
                representative_llirs.push(llir);
                stitched_representative_llirs.push(stitched_llir);
            }
            let deduped = stitch_llir_graphs(
                &stitched_representative_llirs,
                &plan.stitched_representative_descriptors,
            );
            runtime.clear_intermediate_buffers();
            let (metric, display) = runtime.profile(&deduped, &profile_dyn_map, options.trials);
            let has_nan = runtime.has_nan_outputs(&deduped, &profile_dyn_map);
            (representative_llirs, deduped, metric, display, has_nan)
        };

        let (mut best_genome, mut best_rep_llirs, mut best_metric, mut n_graphs);
        let mut attempts = 0usize;
        loop {
            attempts += 1;
            if attempts > 100 {
                panic!("Failed to find a viable initial regionalized genome after 100 attempts");
            }
            let genome = random_initial_choice(egraph, rng);
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_genome(&genome)));
            match result {
                Ok((rep_llirs, _deduped, metric, display, false)) => {
                    println!(
                        "   {:>6}  regionalized candidate: {}",
                        "Graph".cyan().bold(),
                        display
                    );
                    best_genome = genome;
                    best_rep_llirs = rep_llirs;
                    best_metric = metric;
                    n_graphs = 1usize;
                    break;
                }
                Ok(_) | Err(_) => continue,
            }
        }

        let mut prev_selected = FxHashSet::default();
        prev_selected.insert(hash_choice_set(&best_genome));
        while n_graphs < limit {
            let remaining = limit - n_graphs;
            let offspring = extract_generation(
                egraph,
                &best_genome,
                remaining.min(options.generation_size.max(1)),
                options.mutations,
                &mut prev_selected,
                rng,
            );
            if offspring.is_empty() {
                break;
            }
            for genome in offspring {
                n_graphs += 1;
                let eval_result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_genome(&genome)));
                let (rep_llirs, _deduped, metric, display, has_nan) = match eval_result {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if has_nan {
                    continue;
                }
                if best_metric.gt(&metric) {
                    println!(
                        "   {:>6}  regionalized best: {}",
                        "Search".green().bold(),
                        display
                    );
                    best_metric = metric;
                    best_genome = genome;
                    best_rep_llirs = rep_llirs;
                }
            }
        }

        let mut regions = Vec::with_capacity(plan.region_groups.len());
        for (group, representative_llir) in plan
            .region_groups
            .iter()
            .cloned()
            .zip(best_rep_llirs.into_iter())
        {
            regions.push(RegionalLLIRRegion {
                representative_region: group.representative,
                member_regions: group.members,
                representative_llir,
            });
        }
        let regional = RegionalLLIR {
            region_descriptors: self.region_descriptors.clone(),
            regions,
        };
        let stitched = regional.unroll(&self.graph);
        self.last_regional_llir = Some(regional);
        println!(
            "   {:>6}  single regionalized e-graph searched in {}",
            "Searched".green().bold(),
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );
        stitched
    }

    /// Run joint search over all regional groups as a single candidate.
    /// Each candidate contains one genome per region-group and is scored by
    /// profiling the fully unrolled stitched LLIR.
    #[allow(dead_code)]
    fn search_single_regionalized<R: Runtime, G: rand::Rng>(
        &mut self,
        runtime: &mut R,
        options: &SearchOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        _bucket_progress: Option<(usize, usize)>,
    ) -> LLIRGraph {
        let mut profile_dyn_map = dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        let limit = options.limit.max(1);
        let n_groups = self.region_groups.len();
        let n_chunks = self.region_descriptors.len();
        let ops = self.ops.as_ref().unwrap();
        let start = std::time::Instant::now();

        // Allocate dummy buffers for virtual boundary inputs.
        for desc in &self.region_descriptors {
            for bi in &desc.boundary_inputs {
                if !runtime.has_hlir_buffer(bi.break_node.index()) {
                    let n_elements = bi
                        .shape
                        .n_elements()
                        .exec(&profile_dyn_map)
                        .expect("Failed to resolve boundary input shape");
                    let n_bytes = n_elements * bi.dtype.bits() / 8;
                    runtime.allocate_dummy_input(bi.break_node.index(), n_bytes);
                }
            }
        }

        type GenomeSet<'a> = Vec<crate::egglog_utils::EGraphChoiceSet<'a>>;

        let combo_hash = |combo: &GenomeSet<'_>| -> u64 {
            let mut hash: u64 = 1469598103934665603;
            for genome in combo {
                let g = hash_choice_set(genome);
                hash ^= g
                    .wrapping_add(0x9e3779b97f4a7c15)
                    .wrapping_add(hash << 6)
                    .wrapping_add(hash >> 2);
            }
            hash
        };

        let mut eval_combo =
            |combo: &GenomeSet<'_>| -> (RegionalLLIR, LLIRGraph, R::ProfileMetric, String, bool) {
                let mut regions = Vec::with_capacity(n_groups);
                for (group_idx, group) in self.region_groups.iter().enumerate() {
                    let egraph = &self.egraphs[group_idx];
                    let mut list_cache = FxHashMap::default();
                    let mut expr_cache = FxHashMap::default();
                    let llir = egglog_to_llir(
                        egraph,
                        combo[group_idx].clone(),
                        ops,
                        &self.custom_ops,
                        &mut list_cache,
                        &mut expr_cache,
                        None,
                    );
                    regions.push(RegionalLLIRRegion {
                        representative_region: group.representative,
                        member_regions: group.members.clone(),
                        representative_llir: llir,
                    });
                }
                let regional = RegionalLLIR {
                    region_descriptors: self.region_descriptors.clone(),
                    regions,
                };
                let stitched = regional.unroll(&self.graph);
                runtime.clear_intermediate_buffers();
                let (metric, display) =
                    runtime.profile(&stitched, &profile_dyn_map, options.trials);
                let has_nan = runtime.has_nan_outputs(&stitched, &profile_dyn_map);
                (regional, stitched, metric, display, has_nan)
            };

        // Find a viable initial regionalized candidate.
        let (mut best_combo, mut best_regional, mut best_stitched, mut best_metric, mut n_graphs);
        let mut attempts = 0usize;
        loop {
            attempts += 1;
            if attempts > 100 {
                panic!("Failed to find a viable initial regionalized genome after 100 attempts");
            }
            let combo: GenomeSet<'_> = self
                .egraphs
                .iter()
                .map(|egraph| random_initial_choice(egraph, rng))
                .collect();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_combo(&combo)));
            match result {
                Ok((regional, stitched, metric, display, false)) => {
                    best_combo = combo;
                    best_regional = regional;
                    best_stitched = stitched;
                    best_metric = metric;
                    n_graphs = 1usize;
                    println!(
                        "   {:>6}  full regionalized candidate: {}",
                        "Graph".cyan().bold(),
                        display
                    );
                    break;
                }
                Ok(_) | Err(_) => continue,
            }
        }

        // Mutation/refinement loop over full regionalized candidates.
        let mut seen = FxHashSet::default();
        seen.insert(combo_hash(&best_combo));
        while n_graphs < limit {
            let mut candidate = best_combo.clone();
            let n_mutations = options.mutations.max(1);
            for _ in 0..n_mutations {
                let group_idx = rng.random_range(0..n_groups);
                let egraph = &self.egraphs[group_idx];
                let mut prev_selected = FxHashSet::default();
                prev_selected.insert(hash_choice_set(&candidate[group_idx]));
                let mut offspring = extract_generation(
                    egraph,
                    &candidate[group_idx],
                    1,
                    1,
                    &mut prev_selected,
                    rng,
                );
                let mutated = offspring
                    .pop()
                    .unwrap_or_else(|| random_initial_choice(egraph, rng));
                candidate[group_idx] = mutated;
            }
            let h = combo_hash(&candidate);
            if !seen.insert(h) {
                continue;
            }
            n_graphs += 1;

            let eval_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_combo(&candidate)));
            let (regional, stitched, metric, display, has_nan) = match eval_result {
                Ok(v) => v,
                Err(_) => continue,
            };
            if has_nan {
                continue;
            }

            if best_metric.gt(&metric) {
                best_combo = candidate;
                best_regional = regional;
                best_stitched = stitched;
                best_metric = metric;
                println!(
                    "   {:>6}  full regionalized best: {} ({}/{})",
                    "Searched".green().bold(),
                    display,
                    n_graphs,
                    limit
                );
            }
        }

        self.last_regional_llir = Some(best_regional);
        println!(
            "   {:>6}  full regionalized search: {} groups ({} regions) in {}",
            "Searched".green().bold(),
            n_groups,
            n_chunks,
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );
        best_stitched
    }

    fn regionalized_hlir_debug_graph(&self) -> Option<StableGraph<String, ()>> {
        if !self.auto_rolled_regions
            || self.region_descriptors.is_empty()
            || self.region_groups.is_empty()
        {
            return None;
        }

        let representative_regions: FxHashSet<usize> = self
            .region_groups
            .iter()
            .map(|group| group.representative)
            .collect();
        let included_nodes: FxHashSet<NodeIndex> = representative_regions
            .iter()
            .flat_map(|&region_idx| self.region_descriptors[region_idx].nodes.iter().copied())
            .collect();
        if included_nodes.is_empty() {
            return None;
        }

        let mut debug_graph = StableGraph::<String, ()>::default();
        let mut node_remap = FxHashMap::default();
        for node in included_nodes.iter().copied().sorted_by_key(|n| n.index()) {
            let label = format!("{}: {}", node.index(), self.graph[node]);
            let mapped = debug_graph.add_node(label);
            node_remap.insert(node, mapped);
        }

        for edge in self.graph.edge_indices() {
            let (src, dst) = self.graph.edge_endpoints(edge).unwrap();
            if let (Some(&mapped_src), Some(&mapped_dst)) =
                (node_remap.get(&src), node_remap.get(&dst))
            {
                debug_graph.add_edge(mapped_src, mapped_dst, ());
            }
        }

        Some(debug_graph)
    }

    fn dump_regionalized_hlir_before_search(&self) {
        let Some(debug_graph) = self.regionalized_hlir_debug_graph() else {
            return;
        };

        let out_dir = PathBuf::from("target/luminal_debug");
        let out_path = out_dir.join("regionalized_hlir_before_search.dot");
        if let Err(err) = fs::create_dir_all(&out_dir) {
            println!(
                "   {:>6}  regionalized HLIR: {} nodes (failed to create {}: {})",
                "Debug".cyan().bold(),
                debug_graph.node_count(),
                out_dir.display(),
                err,
            );
            return;
        }

        match debug_graph
            .to_dot()
            .and_then(|dot| fs::write(&out_path, dot).map_err(anyhow::Error::from))
        {
            Ok(()) => println!(
                "   {:>6}  regionalized HLIR: {} nodes, {} edges ({} representative regions / {} total regions) -> {}",
                "Debug".cyan().bold(),
                debug_graph.node_count(),
                debug_graph.edge_count(),
                self.region_groups.len(),
                self.region_descriptors.len(),
                out_path.display(),
            ),
            Err(err) => println!(
                "   {:>6}  regionalized HLIR: {} nodes, {} edges (failed to write {}: {})",
                "Debug".cyan().bold(),
                debug_graph.node_count(),
                debug_graph.edge_count(),
                out_path.display(),
                err,
            ),
        }
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

fn build_uses(graph: &HLIRGraph) -> FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>> {
    let mut uses: FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>> = FxHashMap::default();
    for n in graph.node_indices() {
        uses.entry(n).or_default();
    }
    for dst in graph.node_indices() {
        let sources: Vec<_> = graph
            .edges_directed(dst, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        for (port, src) in sources.into_iter().enumerate() {
            if let Some(v) = uses.get_mut(&src) {
                v.push((dst, port));
            }
        }
    }
    uses
}

struct RollingHash64 {
    prefix: Vec<u64>,
    powers: Vec<u64>,
}

impl RollingHash64 {
    const BASE: u64 = 1_000_000_007;

    fn new(tokens: &[u64]) -> Self {
        let mut prefix = Vec::with_capacity(tokens.len() + 1);
        let mut powers = Vec::with_capacity(tokens.len() + 1);
        prefix.push(0u64);
        powers.push(1u64);
        for &token in tokens {
            let next_prefix = prefix
                .last()
                .copied()
                .unwrap()
                .wrapping_mul(Self::BASE)
                .wrapping_add(token.wrapping_add(1));
            prefix.push(next_prefix);
            let next_power = powers.last().copied().unwrap().wrapping_mul(Self::BASE);
            powers.push(next_power);
        }
        Self { prefix, powers }
    }

    fn window_hash(&self, start: usize, len: usize) -> u64 {
        self.prefix[start + len].wrapping_sub(self.prefix[start].wrapping_mul(self.powers[len]))
    }
}

fn cheap_rolling_node_hash(graph: &HLIRGraph, node: NodeIndex) -> u64 {
    let op = graph[node].to_string();
    let mut hash: u64 = 1469598103934665603;
    for byte in op.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    let in_degree = graph.neighbors_directed(node, Direction::Incoming).count() as u64;
    let out_degree = graph.neighbors_directed(node, Direction::Outgoing).count() as u64;
    hash ^= in_degree.wrapping_mul(0x9e3779b185ebca87);
    hash = hash.rotate_left(13);
    hash ^= out_degree.wrapping_mul(0xc2b2ae3d27d4eb4f);
    hash
}

fn rolling_probe_window_sizes(max_window: usize) -> Vec<usize> {
    if max_window == 0 {
        return vec![];
    }
    (1..=max_window).rev().collect()
}

fn build_regionalized_egglog_program(
    graph: &Graph,
    representative_descriptors: &[SubgraphDescriptor],
) -> (String, Vec<String>) {
    let token_re = Regex::new(r"\bt\d+\b").expect("valid egglog temp-var regex");
    let mut merged = String::new();
    let mut roots = Vec::with_capacity(representative_descriptors.len());
    for (idx, desc) in representative_descriptors.iter().enumerate() {
        let (program, root) = hlir_subgraph_to_egglog(graph, desc);
        let prefix = format!("r{idx}_");
        let renamed = token_re.replace_all(&program, |caps: &regex::Captures<'_>| {
            format!("{prefix}{}", &caps[0])
        });
        let renamed_root = token_re
            .replace_all(&root, |caps: &regex::Captures<'_>| {
                format!("{prefix}{}", &caps[0])
            })
            .to_string();
        merged.push_str(&renamed);
        if !merged.ends_with('\n') {
            merged.push('\n');
        }
        roots.push(renamed_root);
    }
    (merged.replace("(MVar \"z\")", "(MIter)"), roots)
}

fn deduped_representative_descriptors(
    subgraphs: &[SubgraphDescriptor],
    groups: &[RegionGroup],
    hlir_graph: &HLIRGraph,
) -> Vec<SubgraphDescriptor> {
    let mut chunk_to_group_rep = FxHashMap::default();
    for group in groups {
        for &member in &group.members {
            chunk_to_group_rep.insert(member, group.representative);
        }
    }

    let mut node_to_chunk = FxHashMap::default();
    for (chunk_idx, desc) in subgraphs.iter().enumerate() {
        for &node in &desc.nodes {
            node_to_chunk.insert(node, chunk_idx);
        }
    }

    groups
        .iter()
        .map(|group| {
            let mut desc = subgraphs[group.representative].clone();
            for bi in &mut desc.boundary_inputs {
                let Some(&src_chunk) = node_to_chunk.get(&bi.break_node) else {
                    continue;
                };
                let Some(&src_rep) = chunk_to_group_rep.get(&src_chunk) else {
                    continue;
                };
                if src_rep == src_chunk {
                    continue;
                }
                let (rep_to_target, _) =
                    build_region_remaps(&subgraphs[src_rep], &subgraphs[src_chunk], hlir_graph);
                let target_to_rep: FxHashMap<usize, usize> = rep_to_target
                    .into_iter()
                    .map(|(rep, target)| (target, rep))
                    .collect();
                if let Some(&mapped) = target_to_rep.get(&bi.break_node.index()) {
                    bi.break_node = NodeIndex::new(mapped);
                }
            }
            desc
        })
        .collect()
}

fn canonicalize_occurrence(
    graph: &HLIRGraph,
    ordered_nodes: &[NodeIndex],
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    topo_index: &FxHashMap<NodeIndex, usize>,
) -> Option<(String, Vec<NodeIndex>, Vec<NodeIndex>)> {
    let region: FxHashSet<NodeIndex> = ordered_nodes.iter().copied().collect();
    if region.is_empty() {
        return None;
    }
    let internal_index: FxHashMap<NodeIndex, usize> = ordered_nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let mut param_index: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let mut boundary_inputs = vec![];
    let mut node_parts = vec![];

    for &node in ordered_nodes {
        let op = graph[node].to_string();
        let inputs: Vec<NodeIndex> = graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        let mut inp_parts = vec![];
        for src in inputs {
            if let Some(&idx) = internal_index.get(&src) {
                inp_parts.push(format!("n{idx}"));
            } else {
                let p = *param_index.entry(src).or_insert_with(|| {
                    boundary_inputs.push(src);
                    boundary_inputs.len() - 1
                });
                inp_parts.push(format!("p{p}"));
            }
        }
        node_parts.push(format!("{op}({})", inp_parts.join(",")));
    }

    let mut output_nodes: Vec<NodeIndex> = ordered_nodes
        .iter()
        .copied()
        .filter(|n| {
            uses.get(n)
                .is_some_and(|out_uses| out_uses.iter().any(|(user, _)| !region.contains(user)))
                || graph.externals(Direction::Outgoing).any(|root| root == *n)
        })
        .collect();
    output_nodes.sort_by_key(|n| topo_index[n]);
    let outputs: Vec<String> = output_nodes
        .iter()
        .filter_map(|n| internal_index.get(n).copied())
        .map(|idx| format!("o{idx}"))
        .collect();

    let sig = format!("{}|{}", node_parts.join(";"), outputs.join(","));
    Some((sig, boundary_inputs, output_nodes))
}

fn collect_state_params(
    occurrences: &[RollingOccurrence],
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    graph: &HLIRGraph,
) -> Vec<usize> {
    if occurrences.len() < 2 {
        return vec![];
    }
    let param_count = occurrences[0].boundary_inputs.len();
    let mut state_params = vec![];

    for p in 0..param_count {
        let mut is_state = true;
        for i in 1..occurrences.len() {
            let earlier = &occurrences[i - 1];
            let later = &occurrences[i];
            let val = later.boundary_inputs.get(p).copied();
            let Some(val) = val else {
                is_state = false;
                break;
            };
            if !earlier.output_nodes.contains(&val) {
                is_state = false;
                break;
            }
            let external_uses: Vec<_> = uses
                .get(&val)
                .map(|u| {
                    u.iter()
                        .copied()
                        .filter(|(user, _)| !earlier.nodes.contains(user))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if external_uses.is_empty() {
                is_state = false;
                break;
            }
            if graph.externals(Direction::Outgoing).any(|root| root == val) {
                is_state = false;
                break;
            }
            if external_uses
                .iter()
                .any(|(user, _)| !later.nodes.contains(user))
            {
                is_state = false;
                break;
            }
        }
        if is_state {
            state_params.push(p);
        }
    }
    state_params
}

fn grow_rolling_candidate(
    graph: &HLIRGraph,
    uses: &FxHashMap<NodeIndex, Vec<(NodeIndex, usize)>>,
    topo_index: &FxHashMap<NodeIndex, usize>,
    mut candidate: RollingCandidate,
    discovered_runs: &[RollingRun],
) -> RollingCandidate {
    loop {
        let candidate_starts: Vec<usize> = candidate
            .occurrences
            .iter()
            .map(|occ| {
                occ.nodes
                    .first()
                    .map(|n| topo_index[n])
                    .unwrap_or(usize::MAX)
            })
            .collect();
        let candidate_ends: Vec<usize> = candidate
            .occurrences
            .iter()
            .map(|occ| occ.nodes.last().map(|n| topo_index[n] + 1).unwrap_or(0))
            .collect();

        let mut best_growth: Option<RollingCandidate> = None;
        for run in discovered_runs {
            for shift in 0..=1usize {
                if run.occurrences.len() < candidate.occurrences.len() + shift {
                    continue;
                }
                let aligned = (0..candidate.occurrences.len()).all(|i| {
                    run.starts[i + shift] == candidate_ends[i]
                        || run.starts[i + shift] + run.window == candidate_starts[i]
                });
                if !aligned {
                    continue;
                }

                let mut merged_occs = Vec::with_capacity(candidate.occurrences.len());
                for i in 0..candidate.occurrences.len() {
                    let run_occ = &run.occurrences[i + shift];
                    let mut nodes = if run.starts[i + shift] + run.window == candidate_starts[i] {
                        let mut n = run_occ.nodes.clone();
                        n.extend(candidate.occurrences[i].nodes.iter().copied());
                        n
                    } else {
                        let mut n = candidate.occurrences[i].nodes.clone();
                        n.extend(run_occ.nodes.iter().copied());
                        n
                    };
                    nodes.sort_by_key(|n| topo_index[n]);
                    let Some((sig, boundary_inputs, output_nodes)) =
                        canonicalize_occurrence(graph, &nodes, uses, topo_index)
                    else {
                        merged_occs.clear();
                        break;
                    };
                    if i == 0 && sig.is_empty() {
                        merged_occs.clear();
                        break;
                    }
                    merged_occs.push(RollingOccurrence {
                        nodes,
                        boundary_inputs,
                        output_nodes,
                    });
                }
                if merged_occs.len() != candidate.occurrences.len() {
                    continue;
                }
                let first_sig =
                    canonicalize_occurrence(graph, &merged_occs[0].nodes, uses, topo_index)
                        .map(|(sig, _, _)| sig);
                let Some(first_sig) = first_sig else { continue };
                if merged_occs.iter().skip(1).any(|occ| {
                    canonicalize_occurrence(graph, &occ.nodes, uses, topo_index)
                        .map(|(sig, _, _)| sig != first_sig)
                        .unwrap_or(true)
                }) {
                    continue;
                }
                let state_param_indices = collect_state_params(&merged_occs, uses, graph);
                if state_param_indices.is_empty() {
                    continue;
                }
                let savings = merged_occs[0].nodes.len() * (merged_occs.len() - 1);
                let grown = RollingCandidate {
                    signature: first_sig,
                    occurrences: merged_occs,
                    state_param_indices,
                    savings,
                };
                let replace = best_growth.as_ref().is_none_or(|best| {
                    (
                        grown.savings,
                        grown.occurrences[0].nodes.len(),
                        grown.occurrences.len(),
                    ) > (
                        best.savings,
                        best.occurrences[0].nodes.len(),
                        best.occurrences.len(),
                    )
                });
                if replace {
                    best_growth = Some(grown);
                }
            }
        }

        match best_growth {
            Some(grown) if grown.savings > candidate.savings => candidate = grown,
            _ => return candidate,
        }
    }
}

fn build_virtual_loop_region_subgraphs(
    graph: &Graph,
    candidate: &RollingCandidate,
    state_param_indices: &[usize],
) -> Result<(Vec<SubgraphDescriptor>, Vec<usize>), String> {
    if candidate.occurrences.len() < 2 {
        return Err("candidate has fewer than 2 occurrences".to_string());
    }
    let topo = toposort(&graph.graph, None)
        .map_err(|_| "HLIR graph has cycles during region construction".to_string())?;
    let topo_index: FxHashMap<NodeIndex, usize> =
        topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut node_to_chunk: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let first_start = candidate
        .occurrences
        .first()
        .ok_or_else(|| "missing first occurrence".to_string())?
        .nodes
        .iter()
        .map(|n| topo_index[n])
        .min()
        .ok_or_else(|| "first occurrence has no nodes".to_string())?;
    let last_end = candidate
        .occurrences
        .last()
        .ok_or_else(|| "missing last occurrence".to_string())?
        .nodes
        .iter()
        .map(|n| topo_index[n])
        .max()
        .ok_or_else(|| "last occurrence has no nodes".to_string())?;
    let n_loop_chunks = candidate.occurrences.len();
    let pre_chunk = 0usize;
    let post_chunk = n_loop_chunks + 1;
    let total_chunks = n_loop_chunks + 2;

    for (&n, _) in &topo_index {
        if graph.try_get_op::<crate::hlir::Input>(n).is_some()
            || graph.try_get_op::<crate::hlir::Output>(n).is_some()
        {
            continue;
        }
        for (i, occ) in candidate.occurrences.iter().enumerate() {
            if occ.nodes.contains(&n) {
                node_to_chunk.insert(n, i + 1);
                break;
            }
        }
    }

    // Grow anchored loop chunks through uniquely connected surrounding ops.
    loop {
        let mut changed = false;
        for &n in &topo {
            if node_to_chunk.contains_key(&n)
                || graph.try_get_op::<crate::hlir::Input>(n).is_some()
                || graph.try_get_op::<crate::hlir::Output>(n).is_some()
            {
                continue;
            }
            let mut neighbor_chunks: FxHashSet<usize> = FxHashSet::default();
            for pred in graph.graph.neighbors_directed(n, Direction::Incoming) {
                if let Some(&chunk) = node_to_chunk.get(&pred)
                    && (1..=n_loop_chunks).contains(&chunk)
                {
                    neighbor_chunks.insert(chunk);
                }
            }
            for succ in graph.graph.neighbors_directed(n, Direction::Outgoing) {
                if let Some(&chunk) = node_to_chunk.get(&succ)
                    && (1..=n_loop_chunks).contains(&chunk)
                {
                    neighbor_chunks.insert(chunk);
                }
            }
            if neighbor_chunks.len() == 1 {
                node_to_chunk.insert(n, *neighbor_chunks.iter().next().unwrap());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (&n, &ti) in &topo_index {
        if graph.try_get_op::<crate::hlir::Input>(n).is_some()
            || graph.try_get_op::<crate::hlir::Output>(n).is_some()
            || node_to_chunk.contains_key(&n)
        {
            continue;
        }
        if ti < first_start {
            node_to_chunk.insert(n, pre_chunk);
        } else if ti > last_end {
            node_to_chunk.insert(n, post_chunk);
        } else {
            node_to_chunk.insert(n, pre_chunk);
        }
    }
    // Place Output nodes in the same chunk as their producer.
    for n in graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<crate::hlir::Output>(*n).is_some())
    {
        let pred = graph
            .graph
            .neighbors_directed(n, Direction::Incoming)
            .next()
            .ok_or_else(|| format!("output node {} has no producer", n.index()))?;
        let chunk = if graph.try_get_op::<crate::hlir::Input>(pred).is_some() {
            // Support persisted / passthrough roots like Input -> Output generically.
            // These represent external state or weights being preserved, not values
            // produced inside a rolled region, so they live in the pre-loop chunk.
            pre_chunk
        } else {
            *node_to_chunk.get(&pred).ok_or_else(|| {
                format!(
                    "producer node {} for output {} was not assigned to any chunk",
                    pred.index(),
                    n.index()
                )
            })?
        };
        node_to_chunk.insert(n, chunk);
    }

    // Force explicit loop-carried edges between successive loop chunks.
    let mut forced_boundaries: FxHashSet<NodeIndex> = FxHashSet::default();
    for pair in candidate.occurrences.windows(2) {
        let [earlier, later] = pair else { continue };
        for &state_param_index in state_param_indices {
            let Some(&src) = later.boundary_inputs.get(state_param_index) else {
                continue;
            };
            if earlier.output_nodes.contains(&src) {
                forced_boundaries.insert(src);
            }
        }
    }

    let mut chunk_nodes: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); total_chunks];
    let mut boundary_inputs: Vec<FxHashMap<NodeIndex, BoundaryInput>> =
        vec![FxHashMap::default(); total_chunks];
    let mut boundary_outputs: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); total_chunks];
    let mut dtype_cache: FxHashMap<NodeIndex, DType> = FxHashMap::default();

    for (&node, &chunk) in &node_to_chunk {
        chunk_nodes[chunk].insert(node);
    }

    // Include real Inputs in every chunk that uses them.
    for input in graph
        .graph
        .node_indices()
        .filter(|n| graph.try_get_op::<crate::hlir::Input>(*n).is_some())
    {
        let users: Vec<NodeIndex> = graph
            .graph
            .neighbors_directed(input, Direction::Outgoing)
            .collect();
        for user in users {
            if let Some(&dst_chunk) = node_to_chunk.get(&user) {
                chunk_nodes[dst_chunk].insert(input);
            }
        }
    }

    for dst in graph.graph.node_indices() {
        let Some(&dst_chunk) = node_to_chunk.get(&dst) else {
            continue;
        };
        for src in graph
            .graph
            .edges_directed(dst, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
        {
            if graph.try_get_op::<crate::hlir::Input>(src).is_some() {
                chunk_nodes[dst_chunk].insert(src);
                continue;
            }
            let Some(&src_chunk) = node_to_chunk.get(&src) else {
                continue;
            };
            let forced = forced_boundaries.contains(&src)
                && src_chunk + 1 == dst_chunk
                && (1..=n_loop_chunks).contains(&dst_chunk);
            if src_chunk != dst_chunk || forced {
                let port = graph
                    .get_sources(dst)
                    .iter()
                    .position(|s| *s == src)
                    .ok_or_else(|| {
                        format!(
                            "could not find input port for edge {} -> {}",
                            src.index(),
                            dst.index()
                        )
                    })?;
                let shape =
                    infer_input_shape_for_port(&graph.graph[dst], port).ok_or_else(|| {
                        format!(
                            "failed to infer input shape for node {} port {}",
                            dst.index(),
                            port
                        )
                    })?;
                let dtype =
                    infer_node_output_dtype(graph, src, &mut dtype_cache).unwrap_or(DType::F32);
                boundary_inputs[dst_chunk]
                    .entry(src)
                    .or_insert(BoundaryInput {
                        break_node: src,
                        shape,
                        dtype,
                    });
                boundary_outputs[src_chunk].insert(src);
            } else {
                chunk_nodes[dst_chunk].insert(src);
            }
        }
    }

    let mut descriptors = Vec::with_capacity(total_chunks);
    let mut old_to_new: FxHashMap<usize, usize> = FxHashMap::default();
    for i in 0..total_chunks {
        if chunk_nodes[i].is_empty() {
            continue;
        }
        let mut chunk_boundary_inputs: Vec<BoundaryInput> =
            boundary_inputs[i].values().cloned().collect();
        chunk_boundary_inputs.sort_by_key(|bi| bi.break_node.index());
        let mut chunk_boundary_outputs: Vec<NodeIndex> =
            boundary_outputs[i].iter().copied().collect();
        chunk_boundary_outputs.sort_by_key(|n| n.index());
        old_to_new.insert(i, descriptors.len());
        descriptors.push(SubgraphDescriptor {
            nodes: chunk_nodes[i].clone(),
            boundary_inputs: chunk_boundary_inputs,
            boundary_outputs: chunk_boundary_outputs,
        });
    }
    let mut loop_region_indices = Vec::new();
    let start_loop_group = if n_loop_chunks >= 2 { 2 } else { 1 };
    for old_idx in start_loop_group..=n_loop_chunks {
        if let Some(&new_idx) = old_to_new.get(&old_idx) {
            loop_region_indices.push(new_idx);
        }
    }
    if loop_region_indices.is_empty() {
        return Err("no loop region indices survived descriptor compaction".to_string());
    }
    Ok((descriptors, loop_region_indices))
}

/// Recursively compute a structural hash of an HLIR node. Two nodes with the
/// same hash produce the same value (e.g., `Iota(z, (d,d))` or `Constant(1.0)`
/// appearing separately per layer all hash identically). Used by the rolling
/// prepass to detect "per-iter boundary inputs" that are actually one shared
/// value, so they can be left unwrapped instead of hidden behind a LoopInput
/// (which breaks downstream egglog rules that pattern-match on op kind).
fn structural_hash_of(
    graph: &HLIRGraph,
    node: NodeIndex,
    cache: &mut FxHashMap<NodeIndex, u64>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    if let Some(&h) = cache.get(&node) {
        return h;
    }
    // Cycle-break: insert a placeholder before recursing. In a DAG this
    // shouldn't fire, but is a safety net.
    cache.insert(node, 0);
    let sources: Vec<NodeIndex> = graph
        .edges_directed(node, Direction::Incoming)
        .sorted_by_key(|e| e.id())
        .map(|e| e.source())
        .collect();
    let source_names: Vec<(NodeIndex, String)> = sources
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, format!("src{i}")))
        .collect();
    let op_repr = graph[node].to_egglog(&source_names);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    op_repr.hash(&mut hasher);
    for &src in &sources {
        let child = structural_hash_of(graph, src, cache);
        child.hash(&mut hasher);
    }
    let h = hasher.finish();
    cache.insert(node, h);
    h
}

fn infer_input_shape_for_port(op: &Box<dyn HLIROp>, port: usize) -> Option<ShapeTracker> {
    use crate::hlir::*;
    if let Some(o) = op.as_any().downcast_ref::<Log2>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Exp2>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Sin>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Recip>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Sqrt>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Softmax>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<SumReduce>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<MaxReduce>() {
        return Some(o.input_shape);
    }
    if let Some(o) = op.as_any().downcast_ref::<Add>() {
        return o.input_shapes.get(port).copied();
    }
    if let Some(o) = op.as_any().downcast_ref::<Mul>() {
        return o.input_shapes.get(port).copied();
    }
    if let Some(o) = op.as_any().downcast_ref::<Mod>() {
        return o.input_shapes.get(port).copied();
    }
    if let Some(o) = op.as_any().downcast_ref::<LessThan>() {
        return o.input_shapes.get(port).copied();
    }
    if let Some(o) = op.as_any().downcast_ref::<Gather>() {
        return match port {
            0 => Some(ShapeTracker::new(o.index_shape.clone())),
            1 => Some(ShapeTracker::new(o.data_shape.clone())),
            _ => None,
        };
    }
    if let Some(o) = op.as_any().downcast_ref::<Scatter>() {
        return match port {
            0 => Some(ShapeTracker::new(o.dest_shape.clone())),
            1 | 2 => Some(ShapeTracker::new(o.index_shape.clone())),
            _ => None,
        };
    }
    if let Some(o) = op.as_any().downcast_ref::<Cast>() {
        return Some(ShapeTracker::new(o.0));
    }
    None
}

fn infer_node_output_dtype(
    graph: &Graph,
    node: NodeIndex,
    cache: &mut FxHashMap<NodeIndex, DType>,
) -> Option<DType> {
    if let Some(&dtype) = cache.get(&node) {
        return Some(dtype);
    }

    let sources = graph.get_sources(node);
    let source_dtype = |idx: usize, cache: &mut FxHashMap<NodeIndex, DType>| -> Option<DType> {
        sources
            .get(idx)
            .and_then(|&src| infer_node_output_dtype(graph, src, cache))
    };

    use crate::hlir::*;
    let dtype = if let Some(op) = graph.try_get_op::<Input>(node) {
        Some(op.dtype)
    } else if graph.try_get_op::<Output>(node).is_some() {
        source_dtype(0, cache)
    } else if let Some(op) = graph.try_get_op::<CustomOpKind>(node) {
        Some(op.dtype)
    } else if graph.try_get_op::<Constant>(node).is_some() {
        Some(DType::F32)
    } else if graph.try_get_op::<Iota>(node).is_some() {
        Some(DType::Int)
    } else if let Some(op) = graph.try_get_op::<Cast>(node) {
        Some(op.1)
    } else if graph.try_get_op::<LessThan>(node).is_some() {
        Some(DType::Bool)
    } else if graph.try_get_op::<Gather>(node).is_some() {
        source_dtype(1, cache)
    } else if graph.try_get_op::<Scatter>(node).is_some() {
        source_dtype(2, cache)
    } else if graph.try_get_op::<Log2>(node).is_some()
        || graph.try_get_op::<Exp2>(node).is_some()
        || graph.try_get_op::<Sin>(node).is_some()
        || graph.try_get_op::<Recip>(node).is_some()
        || graph.try_get_op::<Sqrt>(node).is_some()
        || graph.try_get_op::<Softmax>(node).is_some()
        || graph.try_get_op::<SumReduce>(node).is_some()
        || graph.try_get_op::<MaxReduce>(node).is_some()
        || graph.try_get_op::<Add>(node).is_some()
        || graph.try_get_op::<Mul>(node).is_some()
        || graph.try_get_op::<Mod>(node).is_some()
    {
        source_dtype(0, cache)
    } else {
        source_dtype(0, cache)
    };

    if let Some(dtype) = dtype {
        cache.insert(node, dtype);
    }
    dtype
}

/// Describes a tensor value crossing a region boundary.
#[derive(Debug, Clone)]
pub struct BoundaryInput {
    /// The HLIR NodeIndex used as the boundary identifier for matching.
    pub break_node: NodeIndex,
    /// Shape of the tensor at the boundary
    pub shape: ShapeTracker,
    /// DType of the tensor at the boundary
    pub dtype: DType,
}

/// Describes a region of the HLIR graph.
#[derive(Debug, Clone)]
pub struct SubgraphDescriptor {
    /// HLIR nodes in this region
    pub nodes: FxHashSet<NodeIndex>,
    /// Boundary inputs entering from prior regions
    pub boundary_inputs: Vec<BoundaryInput>,
    /// Boundary output node indices produced by this region
    pub boundary_outputs: Vec<NodeIndex>,
}

/// Default regionization fallback when no explicit auto-rolling plan is present.
/// Produces a single region spanning the full HLIR graph.
pub fn default_region_descriptors(graph: &Graph) -> Vec<SubgraphDescriptor> {
    vec![SubgraphDescriptor {
        nodes: graph.graph.node_indices().collect(),
        boundary_inputs: vec![],
        boundary_outputs: vec![],
    }]
}

/// Clone a representative chunk's LLIR graph and remap Input/Output node indices
/// to match a different (structurally identical) target chunk.
///
/// The remapping handles three categories of nodes:
/// 1. **Boundary inputs**: Matched positionally via SubgraphDescriptor.boundary_inputs
/// 2. **Boundary outputs**: Matched positionally via SubgraphDescriptor.boundary_outputs
/// 3. **Weight/data inputs**: Chunk-specific Input nodes (in one subgraph but not the other) are sorted by index and matched positionally.
///
/// Shared Input nodes (same NodeIndex in both subgraphs) need no remapping.
/// Build remapping tables for cloning a representative chunk's LLIR to a target chunk.
///
/// Returns:
/// - `node_remap`: maps rep HLIR node indices → target HLIR node indices (for Input/Output nodes)
/// - `custom_op_id_remap`: maps rep CustomOpKind IDs → target CustomOpKind IDs
fn build_region_remaps(
    rep_desc: &SubgraphDescriptor,
    target_desc: &SubgraphDescriptor,
    hlir_graph: &HLIRGraph,
) -> (FxHashMap<usize, usize>, FxHashMap<usize, usize>) {
    let mut node_remap: FxHashMap<usize, usize> = FxHashMap::default();

    // 1. Boundary inputs: positional match
    for (r, t) in rep_desc
        .boundary_inputs
        .iter()
        .zip(&target_desc.boundary_inputs)
    {
        node_remap.insert(r.break_node.index(), t.break_node.index());
    }

    // 2. Boundary outputs: positional match
    for (r, t) in rep_desc
        .boundary_outputs
        .iter()
        .zip(&target_desc.boundary_outputs)
    {
        node_remap.insert(r.index(), t.index());
    }

    // 3. Weight/data inputs: chunk-specific Input HLIR nodes
    let rep_input_nodes: FxHashSet<NodeIndex> = rep_desc
        .nodes
        .iter()
        .filter(|n| {
            hlir_graph
                .node_weight(**n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Input>())
                .is_some()
        })
        .copied()
        .collect();
    let target_input_nodes: FxHashSet<NodeIndex> = target_desc
        .nodes
        .iter()
        .filter(|n| {
            hlir_graph
                .node_weight(**n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Input>())
                .is_some()
        })
        .copied()
        .collect();

    // Chunk-specific = in one subgraph but not the other
    let rep_specific: Vec<usize> = rep_input_nodes
        .difference(&target_input_nodes)
        .map(|n| n.index())
        .sorted()
        .collect();
    let target_specific: Vec<usize> = target_input_nodes
        .difference(&rep_input_nodes)
        .map(|n| n.index())
        .sorted()
        .collect();

    for (r, t) in rep_specific.iter().zip(&target_specific) {
        node_remap.insert(*r, *t);
    }

    // 4. Internal Output nodes: Output nodes may reference internal computation nodes
    //    (e.g., scatter results marked with .output()). These need remapping too.
    //    Match positionally by sorted `node` value, skipping those already remapped.
    let rep_output_refs: Vec<usize> = rep_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Output>())
                .map(|o| o.node)
        })
        .filter(|n| !node_remap.contains_key(n))
        .sorted()
        .collect();
    let target_output_refs: Vec<usize> = target_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Output>())
                .map(|o| o.node)
        })
        .filter(|n| !node_remap.values().any(|v| v == n))
        .sorted()
        .collect();
    for (r, t) in rep_output_refs.iter().zip(&target_output_refs) {
        if r != t {
            node_remap.insert(*r, *t);
        }
    }

    // 5. CustomOpKind ID remapping: match positionally by sorted HLIR node index
    let mut custom_op_id_remap: FxHashMap<usize, usize> = FxHashMap::default();
    let rep_custom_ops: Vec<usize> = rep_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::CustomOpKind>())
                .map(|op| op.id)
        })
        .sorted()
        .collect();
    let target_custom_ops: Vec<usize> = target_desc
        .nodes
        .iter()
        .filter_map(|n| {
            hlir_graph
                .node_weight(*n)
                .and_then(|w| w.as_any().downcast_ref::<crate::hlir::CustomOpKind>())
                .map(|op| op.id)
        })
        .sorted()
        .collect();
    for (r, t) in rep_custom_ops.iter().zip(&target_custom_ops) {
        if r != t {
            custom_op_id_remap.insert(*r, *t);
        }
    }

    (node_remap, custom_op_id_remap)
}

/// Apply Input/Output node index remapping to an LLIR graph (in-place modification).
fn remap_llir_io_nodes(
    llir: &mut LLIRGraph,
    node_remap: &FxHashMap<usize, usize>,
    hlir_graph: &HLIRGraph,
) {
    // We need to replace nodes in-place. Collect node indices first.
    let node_indices: Vec<NodeIndex> = llir.node_indices().collect();
    for node_idx in node_indices {
        let op = &llir[node_idx];
        let new_op = if let Some(input_op) = op.to_op::<crate::hlir::Input>() {
            if let Some(&new_node) = node_remap.get(&input_op.node) {
                // Look up the target HLIR Input's label so chunk copies get correct names
                let new_label = hlir_graph
                    .node_weight(NodeIndex::new(new_node))
                    .and_then(|w| w.as_any().downcast_ref::<crate::hlir::Input>())
                    .map(|inp| inp.label.clone())
                    .unwrap_or_else(|| input_op.label.clone());
                Some(LLIROp::new::<crate::hlir::Input>(Box::new(
                    crate::hlir::Input {
                        node: new_node,
                        label: new_label,
                        dtype: input_op.dtype,
                    },
                )))
            } else {
                None
            }
        } else if let Some(output_op) = op.to_op::<crate::hlir::Output>() {
            if let Some(&new_node) = node_remap.get(&output_op.node) {
                Some(LLIROp::new::<crate::hlir::Output>(Box::new(
                    crate::hlir::Output { node: new_node },
                )))
            } else {
                None
            }
        } else {
            None
        };
        if let Some(new_op) = new_op {
            llir[node_idx] = new_op;
        }
    }
}

impl RegionalLLIR {
    /// Expand a regionalized LLIR artifact into a full stitched LLIR graph.
    pub fn unroll(&self, hlir_graph: &HLIRGraph) -> LLIRGraph {
        let n_chunks = self.region_descriptors.len();
        let mut chunk_best_llirs: Vec<Option<LLIRGraph>> = (0..n_chunks).map(|_| None).collect();

        for region in &self.regions {
            for &chunk_idx in &region.member_regions {
                if chunk_idx == region.representative_region {
                    chunk_best_llirs[chunk_idx] = Some(region.representative_llir.clone());
                    continue;
                }
                let (node_remap, _custom_op_id_remap) = build_region_remaps(
                    &self.region_descriptors[region.representative_region],
                    &self.region_descriptors[chunk_idx],
                    hlir_graph,
                );
                let mut llir = region.representative_llir.clone();
                remap_llir_io_nodes(&mut llir, &node_remap, hlir_graph);
                chunk_best_llirs[chunk_idx] = Some(llir);
            }
        }

        let chunk_best_llirs: Vec<LLIRGraph> = chunk_best_llirs
            .into_iter()
            .enumerate()
            .map(|(i, opt)| opt.unwrap_or_else(|| panic!("Missing LLIR for chunk {i}")))
            .collect();

        stitch_llir_graphs(&chunk_best_llirs, &self.region_descriptors)
    }
}

/// Expand all loop-region markers in an LLIR graph into fully unrolled bodies.
///
/// Reads `LoopStart` / `LoopEnd` / `LoopInput` / `LoopOutput` metadata placed
/// by the auto-roll prepass, clones the loop body `iters-1` additional times,
/// threads loop-carried state between clones, routes per-iteration inputs and
/// per-iteration outputs, and removes the four marker op types.
///
/// Incoming-edge ORDER is preserved for every affected node — ops read their
/// inputs by edge-id order, so edges are rebuilt in position.
pub fn unroll_loops_in_llir(llir: &mut LLIRGraph) {
    use crate::hlir::{LoopEnd, LoopInput, LoopOutput, LoopStart, Output};
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut starts: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut ends: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut inputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut outputs: BTreeMap<usize, (NodeIndex, Vec<usize>)> = BTreeMap::new();

    let mut iters = 0usize;
    for n in llir.node_indices() {
        let op = &llir[n];
        if let Some(ls) = op.to_op::<LoopStart>() {
            iters = iters.max(ls.iters.to_usize().unwrap_or(1));
            starts.insert(ls.slot_idx, n);
        } else if let Some(le) = op.to_op::<LoopEnd>() {
            ends.insert(le.slot_idx, n);
        } else if let Some(li) = op.to_op::<LoopInput>() {
            inputs.insert(li.stream_id, n);
        } else if let Some(lo) = op.to_op::<LoopOutput>() {
            outputs.insert(lo.stream_id, (n, lo.targets.clone()));
        }
    }
    if iters <= 1 || starts.is_empty() {
        return;
    }

    let loop_markers: FxHashSet<NodeIndex> = starts
        .values()
        .copied()
        .chain(ends.values().copied())
        .chain(inputs.values().copied())
        .chain(outputs.values().map(|(n, _)| *n))
        .collect();

    let mut body_nodes: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut worklist: Vec<NodeIndex> = starts
        .values()
        .flat_map(|n| llir.neighbors_directed(*n, Direction::Outgoing).collect::<Vec<_>>())
        .chain(
            inputs
                .values()
                .flat_map(|n| llir.neighbors_directed(*n, Direction::Outgoing).collect::<Vec<_>>()),
        )
        .collect();
    while let Some(n) = worklist.pop() {
        if body_nodes.contains(&n) || loop_markers.contains(&n) {
            continue;
        }
        if llir[n].to_op::<Output>().is_some() {
            continue;
        }
        body_nodes.insert(n);
        for succ in llir.neighbors_directed(n, Direction::Outgoing).collect::<Vec<_>>() {
            worklist.push(succ);
        }
    }

    let mut start_meta: FxHashMap<NodeIndex, (NodeIndex, NodeIndex)> = FxHashMap::default();
    for (slot_idx, &start_node) in &starts {
        let end_node = *ends
            .get(slot_idx)
            .unwrap_or_else(|| panic!("missing LoopEnd for slot {slot_idx}"));
        let initial = llir
            .neighbors_directed(start_node, Direction::Incoming)
            .next()
            .expect("LoopStart must have an initial-value producer");
        let body_producer = llir
            .neighbors_directed(end_node, Direction::Incoming)
            .next()
            .expect("LoopEnd must have a body producer");
        start_meta.insert(start_node, (initial, body_producer));
    }

    let mut input_per_iter: FxHashMap<NodeIndex, Vec<NodeIndex>> = FxHashMap::default();
    for input_node in inputs.values() {
        let srcs: Vec<NodeIndex> = llir
            .edges_directed(*input_node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        assert_eq!(srcs.len(), iters, "LoopInput stream must have `iters` sources");
        input_per_iter.insert(*input_node, srcs);
    }

    let mut clone_map: Vec<FxHashMap<NodeIndex, NodeIndex>> = vec![FxHashMap::default(); iters];
    for &b in &body_nodes {
        clone_map[0].insert(b, b);
    }
    for i in 1..iters {
        for &b in &body_nodes {
            let cloned = llir.add_node(llir[b].clone());
            clone_map[i].insert(b, cloned);
        }
    }

    let resolve_src = |src: NodeIndex, i: usize, clone_map: &[FxHashMap<NodeIndex, NodeIndex>]| {
        if let Some(&(initial, body_producer)) = start_meta.get(&src) {
            if i == 0 {
                initial
            } else {
                clone_map[i - 1][&body_producer]
            }
        } else if let Some(sources) = input_per_iter.get(&src) {
            sources[i]
        } else if body_nodes.contains(&src) {
            clone_map[i][&src]
        } else {
            src
        }
    };

    let body_incoming: FxHashMap<NodeIndex, Vec<NodeIndex>> = body_nodes
        .iter()
        .map(|&b| {
            let srcs: Vec<NodeIndex> = llir
                .edges_directed(b, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| e.source())
                .collect();
            (b, srcs)
        })
        .collect();

    // For iter 0, we rebuild each body node's incoming edges in place: we
    // remove each old edge and immediately re-add a new edge with the
    // resolved source. petgraph::stable_graph reuses freed edge indices
    // LIFO, so interleaving remove+add for each edge causes the new edge
    // to reuse exactly the freed slot, preserving edge-id ordering (which
    // the runtime relies on for input positions).
    for &b in &body_nodes {
        let pairs: Vec<(NodeIndex, petgraph::graph::EdgeIndex)> = llir
            .edges_directed(b, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), e.id()))
            .collect();
        for (src, eid) in pairs {
            let new_src = resolve_src(src, 0, &clone_map);
            llir.remove_edge(eid);
            llir.add_edge(new_src, b, ());
        }
    }
    // For iter > 0 clones, there are no existing edges — add fresh ones in
    // body_incoming order so edge-id ordering matches.
    for i in 1..iters {
        for &b in &body_nodes {
            let target = clone_map[i][&b];
            let srcs = &body_incoming[&b];
            for &src in srcs {
                let new_src = resolve_src(src, i, &clone_map);
                llir.add_edge(new_src, target, ());
            }
        }
    }

    let post_loop_consumers: FxHashSet<NodeIndex> = loop_markers
        .iter()
        .flat_map(|n| llir.neighbors_directed(*n, Direction::Outgoing).collect::<Vec<_>>())
        .filter(|n| !loop_markers.contains(n) && !body_nodes.contains(n))
        .collect();

    let mut marker_post_sub: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &end_node in ends.values() {
        let body_producer = llir
            .neighbors_directed(end_node, Direction::Incoming)
            .next()
            .expect("LoopEnd missing body producer during rewire");
        marker_post_sub.insert(end_node, clone_map[iters - 1][&body_producer]);
    }
    for (output_node, _) in outputs.values() {
        let body_producer = llir
            .neighbors_directed(*output_node, Direction::Incoming)
            .next()
            .expect("LoopOutput missing body producer during rewire");
        marker_post_sub.insert(*output_node, clone_map[iters - 1][&body_producer]);
    }

    for &consumer in &post_loop_consumers {
        let srcs: Vec<NodeIndex> = llir
            .edges_directed(consumer, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        // Per-edge replace to preserve edge-id ordering via LIFO reuse.
        let pairs: Vec<(NodeIndex, petgraph::graph::EdgeIndex)> = llir
            .edges_directed(consumer, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), e.id()))
            .collect();
        let _ = srcs;
        for (src, eid) in pairs {
            let new_src = marker_post_sub.get(&src).copied().unwrap_or(src);
            llir.remove_edge(eid);
            llir.add_edge(new_src, consumer, ());
        }
    }

    for (output_node, targets) in outputs.values() {
        let body_producer = llir
            .neighbors_directed(*output_node, Direction::Incoming)
            .next()
            .expect("LoopOutput missing body producer during output lift");
        for i in 1..iters {
            let Some(&target_id) = targets.get(i) else {
                break;
            };
            let new_output = llir.add_node(LLIROp::new::<Output>(Box::new(Output {
                node: target_id,
            })));
            let src = clone_map[i][&body_producer];
            llir.add_edge(src, new_output, ());
        }
    }

    for &n in &loop_markers {
        llir.remove_node(n);
    }

    debug_assert_eq!(
        llir.node_indices()
            .filter(|n| {
                let op = &llir[*n];
                op.to_op::<LoopStart>().is_some()
                    || op.to_op::<LoopEnd>().is_some()
                    || op.to_op::<LoopInput>().is_some()
                    || op.to_op::<LoopOutput>().is_some()
            })
            .count(),
        0,
        "unroll left stray loop marker ops in LLIR"
    );

    // Compact the graph into a freshly-allocated StableGraph so all edge
    // IDs are re-assigned sequentially in our chosen insertion order.
    // Without this, later kernel_to_host add_edge calls can reuse edge
    // indices that our unroll freed via remove_node on loop markers,
    // producing sort-by-edge-id orderings where a later-added scheduling
    // edge ends up at a low index — which the runtime interprets as a
    // primary input position and crashes looking up a buffer.
    let compacted = compact_llir_preserving_input_order(llir);
    *llir = compacted;
}

/// Rebuild an LLIR graph into a fresh StableGraph, copying nodes and edges
/// such that edge IDs are sequential in the insertion order we choose
/// (per-node incoming edges in their original edge-id order). This erases
/// any free-list reuse artifacts from prior `remove_edge` / `remove_node`
/// calls.
fn compact_llir_preserving_input_order(old: &LLIRGraph) -> LLIRGraph {
    use petgraph::visit::EdgeRef;
    let mut new_graph = LLIRGraph::default();
    let mut old_to_new: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    // Topo sort to add nodes in a deterministic order. If the graph has
    // cycles (shouldn't for LLIR), fall back to node_indices order.
    let topo = match petgraph::algo::toposort(old, None) {
        Ok(v) => v,
        Err(_) => old.node_indices().collect(),
    };
    for n in &topo {
        let new_n = new_graph.add_node(old[*n].clone());
        old_to_new.insert(*n, new_n);
    }
    // Add edges in topo order, per-node incoming sorted by old edge id.
    // This reassigns new edge indices sequentially so sort-by-id matches
    // the intended input position.
    for n in &topo {
        let incoming: Vec<NodeIndex> = old
            .edges_directed(*n, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        for src in incoming {
            if let (Some(&new_src), Some(&new_dst)) =
                (old_to_new.get(&src), old_to_new.get(n))
            {
                new_graph.add_edge(new_src, new_dst, ());
            }
        }
    }
    new_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egglog_utils::hash_egglog_normalized;
    use crate::tests::{assert_close, random_vec};

    #[test]
    fn test_hash_egglog_normalized_same_structure() {
        // Two egglog texts differing only in Input node indices and labels
        let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Input 100 "layers.0.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 42))
"#;
        let text_b = r#"(let t0 (Input 84 "boundary" (F32)))
(let t1 (Input 200 "layers.1.wq.weight" (F32)))
(let t2 (Add (ECons 128 (ECons 4096 (ENil))) t1 (ECons 1 (ECons 128 (ENil))) t0 (ECons 1 (ECons 1 (ENil))) (ECons 1 (ECons 128 (ENil)))))
(let t3 (Output t2 84))
"#;
        assert_eq!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Structurally identical chunks should hash the same"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_different_structure() {
        let text_a = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Add (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
        let text_b = r#"(let t0 (Input 42 "boundary" (F32)))
(let t1 (Mul (ECons 128 (ENil)) t0 (ECons 1 (ENil)) t0 (ECons 1 (ENil)) (ECons 1 (ENil))))
"#;
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Different op types should produce different hashes"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_different_dtypes() {
        let text_a = "(let t0 (Input 42 \"boundary\" (F32)))\n";
        let text_b = "(let t0 (Input 42 \"boundary\" (F16)))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "Different dtypes should produce different hashes"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_output_join_not_normalized() {
        // OutputJoin lines should be hashed verbatim, not treated as Output
        let text_a = "(let t0 (OutputJoin t1 t2))\n";
        let text_b = "(let t0 (OutputJoin t3 t4))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "OutputJoin lines should be hashed verbatim"
        );
    }

    #[test]
    fn test_build_region_remaps_and_remap_io() {
        use petgraph::stable_graph::NodeIndex as NI;

        // Build a simple LLIR: Input(node=10) -> Output(node=20), Input(node=100) -> Output(node=20)
        let mut llir = LLIRGraph::default();
        let input_node = llir.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 10,
                label: "boundary".to_string(),
                dtype: DType::F32,
            },
        )));
        let weight_node = llir.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 100,
                label: "weight_a".to_string(),
                dtype: DType::F32,
            },
        )));
        let output_node = llir.add_node(LLIROp::new::<crate::hlir::Output>(Box::new(
            crate::hlir::Output { node: 20 },
        )));
        llir.add_edge(input_node, output_node, ());
        llir.add_edge(weight_node, output_node, ());

        // Build minimal HLIR graph with Input nodes for both rep and target
        let mut hlir_graph = HLIRGraph::default();
        while hlir_graph.node_count() < 10 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 10,
            label: "boundary".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 20 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 20,
            label: "brk".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 50 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 50,
            label: "boundary".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 60 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 60,
            label: "brk".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 100 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 100,
            label: "weight_a".to_string(),
            dtype: DType::F32,
        }));
        while hlir_graph.node_count() < 200 {
            hlir_graph.add_node(Box::new(crate::hlir::Input {
                node: 0,
                label: "pad".to_string(),
                dtype: DType::F32,
            }));
        }
        hlir_graph.add_node(Box::new(crate::hlir::Input {
            node: 200,
            label: "weight_b".to_string(),
            dtype: DType::F32,
        }));

        let rep_desc = SubgraphDescriptor {
            nodes: [NI::new(10), NI::new(100)].into_iter().collect(),
            boundary_inputs: vec![BoundaryInput {
                break_node: NI::new(10),
                shape: ShapeTracker::new(()),
                dtype: DType::F32,
            }],
            boundary_outputs: vec![NI::new(20)],
        };
        let target_desc = SubgraphDescriptor {
            nodes: [NI::new(50), NI::new(200)].into_iter().collect(),
            boundary_inputs: vec![BoundaryInput {
                break_node: NI::new(50),
                shape: ShapeTracker::new(()),
                dtype: DType::F32,
            }],
            boundary_outputs: vec![NI::new(60)],
        };

        let (node_remap, custom_op_remap) =
            build_region_remaps(&rep_desc, &target_desc, &hlir_graph);

        // No custom ops in this test
        assert!(custom_op_remap.is_empty());

        // Apply IO remap
        remap_llir_io_nodes(&mut llir, &node_remap, &hlir_graph);

        // Verify remapped nodes
        let mut input_nodes: Vec<(usize, String)> = vec![];
        let mut output_nodes: Vec<usize> = vec![];
        for node in llir.node_indices() {
            let op = &llir[node];
            if let Some(inp) = op.to_op::<crate::hlir::Input>() {
                input_nodes.push((inp.node, inp.label.clone()));
            }
            if let Some(out) = op.to_op::<crate::hlir::Output>() {
                output_nodes.push(out.node);
            }
        }
        input_nodes.sort_by_key(|(n, _)| *n);
        assert_eq!(input_nodes.len(), 2);
        assert_eq!(
            input_nodes[0].0, 50,
            "Boundary input should be remapped to 50"
        );
        assert_eq!(
            input_nodes[1].0, 200,
            "Weight input should be remapped to 200"
        );
        assert_eq!(output_nodes, vec![60], "Output should be remapped to 60");
        assert_eq!(llir.edge_count(), 2, "Should have 2 edges");
    }

    #[test]
    fn test_stitch_keeps_real_output_when_boundary_duplicates_id() {
        use petgraph::stable_graph::NodeIndex as NI;

        let mut chunk0 = LLIRGraph::default();
        let input = chunk0.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 2,
                label: "kv_cache.0.k".to_string(),
                dtype: DType::F32,
            },
        )));
        let real_output = chunk0.add_node(LLIROp::new::<crate::hlir::Output>(Box::new(
            crate::hlir::Output { node: 3 },
        )));
        let boundary_wrapper = chunk0.add_node(LLIROp::new::<crate::hlir::Output>(Box::new(
            crate::hlir::Output { node: 3 },
        )));
        chunk0.add_edge(input, real_output, ());
        chunk0.add_edge(real_output, boundary_wrapper, ());

        let mut chunk1 = LLIRGraph::default();
        let boundary_input = chunk1.add_node(LLIROp::new::<crate::hlir::Input>(Box::new(
            crate::hlir::Input {
                node: 3,
                label: "boundary".to_string(),
                dtype: DType::F32,
            },
        )));
        let downstream_output = chunk1.add_node(LLIROp::new::<crate::hlir::Output>(Box::new(
            crate::hlir::Output { node: 100 },
        )));
        chunk1.add_edge(boundary_input, downstream_output, ());

        let descriptors = vec![
            SubgraphDescriptor {
                nodes: FxHashSet::default(),
                boundary_inputs: vec![],
                boundary_outputs: vec![NI::new(3)],
            },
            SubgraphDescriptor {
                nodes: FxHashSet::default(),
                boundary_inputs: vec![BoundaryInput {
                    break_node: NI::new(3),
                    shape: ShapeTracker::new(()),
                    dtype: DType::F32,
                }],
                boundary_outputs: vec![],
            },
        ];

        let stitched = stitch_llir_graphs(&[chunk0, chunk1], &descriptors);
        let mut output_ids: Vec<usize> = stitched
            .node_indices()
            .filter_map(|n| {
                stitched[n]
                    .to_op::<crate::hlir::Output>()
                    .map(|out| out.node)
            })
            .collect();
        output_ids.sort_unstable();
        assert_eq!(
            output_ids.iter().filter(|&&id| id == 3).count(),
            1,
            "real passthrough output should survive stitching exactly once"
        );
        assert!(
            output_ids.contains(&100),
            "downstream outputs should remain after stitching"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_id() {
        // CustomOpKind lines differ only in the integer ID (layer index)
        let text_a = r#"(let t0 (Input 441 "boundary" (F32)))
(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 585))
"#;
        let text_b = r#"(let t0 (Input 585 "boundary" (F32)))
(let t1 (Op (CustomOpKind 2 (F32)) (ICons t74 (ICons t120 (ICons t28 (INil))))))
(let t2 (Output t1 729))
"#;
        assert_eq!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpKind with different IDs should hash the same"
        );
    }

    #[test]
    fn test_hash_egglog_normalized_custom_op_different_structure() {
        // CustomOpKind lines with different input lists should hash differently
        let text_a = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t120 (INil)))))\n";
        let text_b = "(let t1 (Op (CustomOpKind 1 (F32)) (ICons t74 (ICons t99 (INil)))))\n";
        assert_ne!(
            hash_egglog_normalized(text_a),
            hash_egglog_normalized(text_b),
            "CustomOpKind with different input lists should hash differently"
        );
    }

    #[test]
    fn test_auto_roll_loops_prepass_creates_regions_for_chain_recurrence() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let out = x.exp2().sin().exp2().sin().exp2().sin().output();

        let inserted = cx.auto_roll_loops_prepass();
        assert!(
            inserted >= 2,
            "expected at least two loop boundaries for 3 repeated bodies, got {inserted}"
        );

        assert!(cx.auto_rolled_regions);
        assert!(
            cx.auto_region_plan
                .as_ref()
                .is_some_and(|p| p.descriptors.len() >= 3)
        );
        assert!(
            cx.auto_region_plan
                .as_ref()
                .is_some_and(|p| p.loop_region_indices.len() >= 2)
        );
        let plan = cx
            .auto_region_plan
            .as_ref()
            .expect("expected auto region plan after rolling");
        for desc in &plan.descriptors {
            assert!(
                desc.boundary_inputs
                    .windows(2)
                    .all(|w| w[0].break_node.index() <= w[1].break_node.index()),
                "boundary inputs must be sorted for positional remapping"
            );
            assert!(
                desc.boundary_outputs
                    .windows(2)
                    .all(|w| w[0].index() <= w[1].index()),
                "boundary outputs must be sorted for positional remapping"
            );
        }

        let vals = random_vec(8);
        let mut rt = NativeRuntime::default();
        cx.build_search_space::<NativeRuntime>();
        rt = cx.search(rt, 1);
        rt.set_data(x.id, vals.clone());
        rt.execute(&cx.dyn_map);

        let expected = vals
            .into_iter()
            .map(|v| v.exp2().sin().exp2().sin().exp2().sin())
            .collect::<Vec<f32>>();
        assert_close(rt.get_f32(out.id), &expected);
    }

    #[test]
    fn test_auto_roll_loops_prepass_skips_non_recurrent_branches() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let y = cx.tensor(8);
        let _out = (x.exp().sin() + y.exp().sin()).output();

        let inserted = cx.auto_roll_loops_prepass();
        assert_eq!(inserted, 0, "branch-only reuse should not roll into loops");
    }

    #[test]
    fn test_regionalized_hlir_debug_graph_collapses_repeated_regions() {
        let mut cx = Graph::new();
        let x = cx.tensor(8);
        let _out = x
            .exp2()
            .sin()
            .exp2()
            .sin()
            .exp2()
            .sin()
            .exp2()
            .sin()
            .exp2()
            .sin()
            .output();

        cx.build_search_space::<NativeRuntime>();
        let debug_graph = cx
            .regionalized_hlir_debug_graph()
            .expect("expected auto-rolled regionalized debug graph");

        assert!(
            debug_graph.node_count() < cx.graph.node_count(),
            "regionalized debug graph should collapse repeated regions: {} !< {}",
            debug_graph.node_count(),
            cx.graph.node_count()
        );
    }
}
