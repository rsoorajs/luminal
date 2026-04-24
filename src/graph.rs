use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hlir_to_egglog, random_initial_choice,
    run_egglog,
};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpKind, op::*, prelude::*};
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
        let inserted = self.auto_roll_loops_prepass();
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
        // Rolling has rough edges on graphs with fewer than 3 repetitions —
        // proptest-generated test cases hit body×2 patterns that round-trip
        // incorrectly through egglog + unroll. Real models roll 20–50
        // repetitions of a transformer block, so this threshold doesn't
        // affect any production path.
        if candidate.occurrences.len() < 3 {
            return 0;
        }

        // Mutate the HLIR in place — insert LoopStart/LoopEnd/LoopInput/
        // LoopOutput markers, delete N-1 duplicate bodies. The loop structure
        // is encoded in the HLIR graph itself and the downstream single-root
        // egglog path picks it up unchanged.
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
                let _ = sig;
                let candidate = RollingCandidate {
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
        self.run_auto_loop_rolling_prepass();
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let (program, root) = hlir_to_egglog(self);
        self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        self.region_descriptors = default_region_descriptors(self);
        self.region_groups = vec![RegionGroup {
            representative: 0,
            members: vec![0],
        }];
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        self.run_auto_loop_rolling_prepass();
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let (program, root) = hlir_to_egglog(self);
        self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
        self.region_descriptors = default_region_descriptors(self);
        self.region_groups = vec![RegionGroup {
            representative: 0,
            members: vec![0],
        }];
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
                    let metrics = vec![rep_metric.clone()];
                    let has_nan = runtime.has_nan_outputs(&graph, &profile_dyn_map);

                    let combined_metric = R::aggregate_profile_metrics(&metrics);
                    let display = rep_display;
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
                            let metrics = vec![rep_metric.clone()];
                            let has_nan =
                                runtime.has_nan_outputs(&llir_graph, &profile_dyn_map);

                            let combined_metric = R::aggregate_profile_metrics(&metrics);
                            let display = rep_display;
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

        println!(
            "   {:>6}  {} groups ({} regions) in {}",
            "Searched".green().bold(),
            n_groups,
            n_chunks,
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );

        stitched
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
                let _ = first_sig;
                let grown = RollingCandidate {
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

impl RegionalLLIR {
    /// Expand a regionalized LLIR artifact into a full stitched LLIR graph.
    /// With the loop-rolling refactor the search pipeline always produces a
    /// single region whose representative LLIR is the full graph, so unroll
    /// is just a clone.
    pub fn unroll(&self, _hlir_graph: &HLIRGraph) -> LLIRGraph {
        assert_eq!(
            self.regions.len(),
            1,
            "RegionalLLIR now always holds exactly one region"
        );
        self.regions[0].representative_llir.clone()
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
    use crate::hlir::{LoopEnd, LoopInput, LoopInputStatic, LoopOutput, LoopStart, Output};
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut starts: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut ends: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut inputs: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    // Iteration-independent inputs. Keyed on LLIR NodeIndex (stream_id is not
    // unique when a single LoopInput splits into multiple LoopInputStatics via
    // egglog rewrites of the same stream).
    let mut static_inputs: FxHashSet<NodeIndex> = FxHashSet::default();
    let mut outputs: BTreeMap<usize, (NodeIndex, Vec<usize>)> = BTreeMap::new();

    let mut iters = 0usize;
    for n in llir.node_indices() {
        let op = &llir[n];
        if let Some(ls) = op.to_op::<LoopStart>() {
            iters = iters.max(ls.iters.to_usize().unwrap_or(1));
            starts.insert(ls.slot_idx, n);
        } else if let Some(le) = op.to_op::<LoopEnd>() {
            ends.insert(le.slot_idx, n);
        } else if op.to_op::<LoopInputStatic>().is_some() {
            // Must be checked before LoopInput because LoopInputStatic is a
            // distinct op with its own sort name, but we want it recognized as
            // a separate category here.
            static_inputs.insert(n);
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
        .chain(static_inputs.iter().copied())
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
        .chain(static_inputs.iter().flat_map(|n| {
            llir.neighbors_directed(*n, Direction::Outgoing).collect::<Vec<_>>()
        }))
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
    // LoopInputStatic: single shared source reused across all iterations.
    let mut static_source: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    for &static_node in &static_inputs {
        let srcs: Vec<NodeIndex> = llir
            .edges_directed(static_node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect();
        assert_eq!(
            srcs.len(),
            1,
            "LoopInputStatic must have exactly 1 source (got {})",
            srcs.len()
        );
        static_source.insert(static_node, srcs[0]);
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
        } else if let Some(&shared) = static_source.get(&src) {
            // LoopInputStatic: same source for every iteration's clone.
            shared
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

}
