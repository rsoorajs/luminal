use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hash_egglog_normalized,
    hlir_subgraph_to_egglog, hlir_to_egglog, random_initial_choice, run_egglog, stitch_llir_graphs,
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
    signature: String,
    occurrences: Vec<RollingOccurrence>,
    state_param_index: usize,
    savings: usize,
}

#[derive(Debug, Clone)]
struct AutoRegionPlan {
    descriptors: Vec<SubgraphDescriptor>,
    loop_region_indices: Vec<usize>,
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
#[derive(Debug, Default)]
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
    /// Enable automatic loop-rolling prepass before e-graph search-space build.
    pub enable_auto_loop_rolling: bool,
    /// Whether explicit loop regions were produced by the automatic loop-rolling prepass.
    auto_rolled_regions: bool,
    /// Virtual region plan produced by automatic loop rolling.
    auto_region_plan: Option<AutoRegionPlan>,
    /// Most recent regionalized LLIR result from search.
    pub last_regional_llir: Option<RegionalLLIR>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn set_auto_loop_rolling(&mut self, enabled: bool) {
        self.enable_auto_loop_rolling = enabled;
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
    /// - only rolls candidates with a single loop-carried state parameter
    /// - only inserts when the carried edge shape can be inferred
    pub fn auto_roll_loops_prepass(&mut self, max_region_size: usize) -> usize {
        self.auto_rolled_regions = false;
        self.auto_region_plan = None;
        if max_region_size < 1 {
            return 0;
        }
        let Some(candidate) = self.best_rolling_candidate(max_region_size) else {
            return 0;
        };
        if candidate.occurrences.len() < 2 {
            return 0;
        }

        let Some((descriptors, loop_region_indices)) =
            build_virtual_loop_region_subgraphs(self, &candidate, candidate.state_param_index)
        else {
            return 0;
        };
        let inserted = descriptors
            .iter()
            .map(|d| d.boundary_inputs.len())
            .sum::<usize>();
        if inserted > 0 {
            self.auto_rolled_regions = true;
            self.auto_region_plan = Some(AutoRegionPlan {
                descriptors,
                loop_region_indices,
            });
            println!(
                "   {:>6}  created {} loop boundaries ({}, trips={}, body={})",
                "Rolled".cyan().bold(),
                inserted,
                candidate.signature,
                candidate.occurrences.len(),
                candidate.occurrences[0].nodes.len(),
            );
        }
        inserted
    }

    fn best_rolling_candidate(&self, max_region_size: usize) -> Option<RollingCandidate> {
        let topo = toposort(&self.graph, None).ok()?;
        let uses = build_uses(&self.graph);
        let topo_index: FxHashMap<NodeIndex, usize> =
            topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        let mut best: Option<RollingCandidate> = None;

        for window in 1..=max_region_size {
            if topo.len() < window * 2 {
                continue;
            }
            for start in 0..=topo.len() - window * 2 {
                let mut occs = vec![];
                let first_nodes = topo[start..start + window].to_vec();
                let Some((sig, first_boundary, first_outputs)) =
                    canonicalize_occurrence(&self.graph, &first_nodes, &uses, &topo_index)
                else {
                    continue;
                };
                occs.push(RollingOccurrence {
                    nodes: first_nodes,
                    boundary_inputs: first_boundary,
                    output_nodes: first_outputs,
                });

                let mut pos = start + window;
                while pos + window <= topo.len() {
                    let nodes = topo[pos..pos + window].to_vec();
                    let Some((next_sig, boundary_inputs, output_nodes)) =
                        canonicalize_occurrence(&self.graph, &nodes, &uses, &topo_index)
                    else {
                        break;
                    };
                    if next_sig != sig {
                        break;
                    }
                    occs.push(RollingOccurrence {
                        nodes,
                        boundary_inputs,
                        output_nodes,
                    });
                    pos += window;
                }
                if occs.len() < 2 {
                    continue;
                }

                let state_params = collect_state_params(&occs, &uses, &self.graph);
                if state_params.len() != 1 {
                    continue;
                }

                let savings = window * (occs.len() - 1);
                let candidate = RollingCandidate {
                    signature: sig,
                    occurrences: occs,
                    state_param_index: state_params[0],
                    savings,
                };
                let replace = best.as_ref().is_none_or(|b| {
                    (candidate.savings, candidate.occurrences.len())
                        > (b.savings, b.occurrences.len())
                });
                if replace {
                    best = Some(candidate);
                }
            }
        }
        best
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
        if self.enable_auto_loop_rolling {
            self.auto_roll_loops_prepass(16);
        } else {
            self.auto_rolled_regions = false;
            self.auto_region_plan = None;
        }
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = if self.auto_rolled_regions {
            self.auto_region_plan
                .as_ref()
                .map(|p| p.descriptors.clone())
                .unwrap_or_else(|| default_region_descriptors(self))
        } else {
            default_region_descriptors(self)
        };

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
            if self.auto_rolled_regions {
                let mut groups: Vec<RegionGroup> = Vec::new();
                if let Some(plan) = &self.auto_region_plan {
                    let loop_members: Vec<usize> = plan
                        .loop_region_indices
                        .iter()
                        .copied()
                        .filter(|&i| {
                            !subgraphs[i]
                                .nodes
                                .iter()
                                .any(|n| self.try_get_op::<crate::hlir::Output>(*n).is_some())
                        })
                        .collect();
                    if loop_members.len() >= 2 {
                        let rep = loop_members[0];
                        groups.push(RegionGroup {
                            representative: rep,
                            members: loop_members.clone(),
                        });
                        let loop_set: FxHashSet<usize> = loop_members.iter().copied().collect();
                        for i in 0..subgraphs.len() {
                            if !loop_set.contains(&i) {
                                groups.push(RegionGroup {
                                    representative: i,
                                    members: vec![i],
                                });
                            }
                        }
                        groups.sort_by_key(|g| g.representative);
                    }
                }
                if groups.is_empty() {
                    groups = (0..subgraphs.len())
                        .map(|i| RegionGroup {
                            representative: i,
                            members: vec![i],
                        })
                        .collect();
                }
                self.egraphs = groups
                    .iter()
                    .map(|sg| {
                        let (program, root) =
                            hlir_subgraph_to_egglog(self, &subgraphs[sg.representative]);
                        run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()
                    })
                    .collect();
                self.region_groups = groups;
            } else {
                self.build_grouped_egraphs(&subgraphs, &ops, cleanup_hlir);
            }
        }
        self.region_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space_exclude_ops<Rt: Runtime + 'static, Ex: IntoEgglogOp>(&mut self) {
        self.last_regional_llir = None;
        if self.enable_auto_loop_rolling {
            self.auto_roll_loops_prepass(16);
        } else {
            self.auto_rolled_regions = false;
            self.auto_region_plan = None;
        }
        let exclude_ops = Ex::into_vec()
            .into_iter()
            .map(|e| e.sort().name)
            .collect::<FxHashSet<_>>();
        let mut ops = Rt::Ops::into_vec();
        ops.retain(|o| !exclude_ops.contains(&o.sort().name));
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>();

        let subgraphs = if self.auto_rolled_regions {
            self.auto_region_plan
                .as_ref()
                .map(|p| p.descriptors.clone())
                .unwrap_or_else(|| default_region_descriptors(self))
        } else {
            default_region_descriptors(self)
        };
        if subgraphs.len() <= 1 {
            let (program, root) = hlir_to_egglog(self);
            self.egraphs = vec![run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()];
            self.region_groups = vec![RegionGroup {
                representative: 0,
                members: vec![0],
            }];
        } else {
            if self.auto_rolled_regions {
                let mut groups: Vec<RegionGroup> = Vec::new();
                if let Some(plan) = &self.auto_region_plan {
                    let loop_members: Vec<usize> = plan
                        .loop_region_indices
                        .iter()
                        .copied()
                        .filter(|&i| {
                            !subgraphs[i]
                                .nodes
                                .iter()
                                .any(|n| self.try_get_op::<crate::hlir::Output>(*n).is_some())
                        })
                        .collect();
                    if loop_members.len() >= 2 {
                        let rep = loop_members[0];
                        groups.push(RegionGroup {
                            representative: rep,
                            members: loop_members.clone(),
                        });
                        let loop_set: FxHashSet<usize> = loop_members.iter().copied().collect();
                        for i in 0..subgraphs.len() {
                            if !loop_set.contains(&i) {
                                groups.push(RegionGroup {
                                    representative: i,
                                    members: vec![i],
                                });
                            }
                        }
                        groups.sort_by_key(|g| g.representative);
                    }
                }
                if groups.is_empty() {
                    groups = (0..subgraphs.len())
                        .map(|i| RegionGroup {
                            representative: i,
                            members: vec![i],
                        })
                        .collect();
                }
                self.egraphs = groups
                    .iter()
                    .map(|sg| {
                        let (program, root) =
                            hlir_subgraph_to_egglog(self, &subgraphs[sg.representative]);
                        run_egglog(&program, &root, &ops, cleanup_hlir).unwrap()
                    })
                    .collect();
                self.region_groups = groups;
            } else {
                self.build_grouped_egraphs(&subgraphs, &ops, cleanup_hlir);
            }
        }
        self.region_descriptors = subgraphs;
        self.ops = Some(ops);
    }

    /// Build e-graphs for multi-chunk subgraphs, grouping structurally identical
    /// chunks and only building one e-graph per unique group.
    fn build_grouped_egraphs(
        &mut self,
        subgraphs: &[SubgraphDescriptor],
        ops: &[Arc<Box<dyn EgglogOp>>],
        cleanup_hlir: bool,
    ) {
        // Get egglog text for each subgraph
        let egglog_texts: Vec<(String, String)> = subgraphs
            .iter()
            .map(|sg| hlir_subgraph_to_egglog(self, sg))
            .collect();

        // Group by normalized egglog hash
        let mut hash_to_chunks: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
        for (i, (text, _)) in egglog_texts.iter().enumerate() {
            let h = hash_egglog_normalized(text);
            hash_to_chunks.entry(h).or_default().push(i);
        }
        let mut groups: Vec<RegionGroup> = hash_to_chunks
            .into_values()
            .map(|members| RegionGroup {
                representative: members[0],
                members,
            })
            .collect();
        groups.sort_by_key(|g| g.representative);

        println!(
            "   {:>6}  {} unique groups from {} chunks",
            "Graphs".cyan().bold(),
            groups.len(),
            subgraphs.len()
        );

        self.egraphs = groups
            .iter()
            .map(|g| {
                let (ref program, ref root) = egglog_texts[g.representative];
                run_egglog(program, root, ops, cleanup_hlir).unwrap()
            })
            .collect();

        self.region_groups = groups;
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
    pub fn search<R: Runtime>(&mut self, runtime: R, limit: usize) -> R {
        let mut rng = rand::rng();
        self.search_options(runtime, SearchOptions::new(limit), &mut rng)
    }

    #[tracing::instrument(skip_all)]
    pub fn search_options<R: Runtime, G: rand::Rng>(
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
    fn search_single<R: Runtime, G: rand::Rng>(
        &mut self,
        runtime: &mut R,
        options: &SearchOptions,
        rng: &mut G,
        dyn_map: &FxHashMap<char, usize>,
        bucket_progress: Option<(usize, usize)>,
    ) -> LLIRGraph {
        if self.auto_rolled_regions && self.region_groups.len() > 1 {
            return self.search_single_regionalized(runtime, options, rng, dyn_map, bucket_progress);
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
                            let (metric, _) = runtime.profile(&llir, &profile_dyn_map, options.trials);
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

    /// Run joint search over all regional groups as a single candidate.
    /// Each candidate contains one genome per region-group and is scored by
    /// profiling the fully unrolled stitched LLIR.
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

        let mut eval_combo = |combo: &GenomeSet<'_>| -> (RegionalLLIR, LLIRGraph, R::ProfileMetric, String, bool) {
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
            let (metric, display) = runtime.profile(&stitched, &profile_dyn_map, options.trials);
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
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| eval_combo(&combo)));
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
                let mut offspring =
                    extract_generation(egraph, &candidate[group_idx], 1, 1, &mut prev_selected, rng);
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

fn build_virtual_loop_region_subgraphs(
    graph: &Graph,
    candidate: &RollingCandidate,
    state_param_index: usize,
) -> Option<(Vec<SubgraphDescriptor>, Vec<usize>)> {
    if candidate.occurrences.len() < 2 {
        return None;
    }
    let topo = toposort(&graph.graph, None).ok()?;
    let topo_index: FxHashMap<NodeIndex, usize> =
        topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut node_to_chunk: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let first_start = candidate
        .occurrences
        .first()?
        .nodes
        .iter()
        .map(|n| topo_index[n])
        .min()?;
    let last_end = candidate
        .occurrences
        .last()?
        .nodes
        .iter()
        .map(|n| topo_index[n])
        .max()?;
    let n_loop_chunks = candidate.occurrences.len();
    let pre_chunk = 0usize;
    let post_chunk = n_loop_chunks + 1;
    let total_chunks = n_loop_chunks + 2;

    for (&n, &ti) in &topo_index {
        if graph.try_get_op::<crate::hlir::Input>(n).is_some() {
            continue;
        }
        if graph.try_get_op::<crate::hlir::Output>(n).is_some() {
            continue;
        }
        if ti < first_start {
            node_to_chunk.insert(n, pre_chunk);
        } else if ti > last_end {
            node_to_chunk.insert(n, post_chunk);
        } else {
            // Assign to loop chunk if present in an occurrence
            let mut assigned = None;
            for (i, occ) in candidate.occurrences.iter().enumerate() {
                if occ.nodes.contains(&n) {
                    assigned = Some(i + 1);
                    break;
                }
            }
            node_to_chunk.insert(n, assigned.unwrap_or(pre_chunk));
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
            .next()?;
        let chunk = *node_to_chunk.get(&pred)?;
        node_to_chunk.insert(n, chunk);
    }

    // Force explicit loop-carried edges between successive loop chunks.
    let mut forced_boundaries: FxHashSet<NodeIndex> = FxHashSet::default();
    for pair in candidate.occurrences.windows(2) {
        let [earlier, later] = pair else { continue };
        let src = later.boundary_inputs[state_param_index];
        if earlier.output_nodes.contains(&src) {
            forced_boundaries.insert(src);
        }
    }

    let mut chunk_nodes: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); total_chunks];
    let mut boundary_inputs: Vec<FxHashMap<NodeIndex, BoundaryInput>> =
        vec![FxHashMap::default(); total_chunks];
    let mut boundary_outputs: Vec<FxHashSet<NodeIndex>> = vec![FxHashSet::default(); total_chunks];

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
                let port = graph.get_sources(dst).iter().position(|s| *s == src)?;
                let shape = infer_input_shape_for_port(&graph.graph[dst], port)?;
                let dtype = if let Some(inp) = graph.try_get_op::<crate::hlir::Input>(src) {
                    inp.dtype
                } else {
                    DType::F32
                };
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
        old_to_new.insert(i, descriptors.len());
        descriptors.push(SubgraphDescriptor {
            nodes: chunk_nodes[i].clone(),
            boundary_inputs: boundary_inputs[i].values().cloned().collect(),
            boundary_outputs: boundary_outputs[i].iter().copied().collect(),
        });
    }
    let mut loop_region_indices = Vec::new();
    let start_loop_group = if n_loop_chunks >= 2 { 2 } else { 1 };
    for old_idx in start_loop_group..=n_loop_chunks {
        if let Some(&new_idx) = old_to_new.get(&old_idx) {
            loop_region_indices.push(new_idx);
        }
    }
    Some((descriptors, loop_region_indices))
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
        cx.set_auto_loop_rolling(true);
        let x = cx.tensor(8);
        let out = x.exp2().sin().exp2().sin().exp2().sin().output();

        let inserted = cx.auto_roll_loops_prepass(6);
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
        cx.set_auto_loop_rolling(true);
        let x = cx.tensor(8);
        let y = cx.tensor(8);
        let _out = (x.exp().sin() + y.exp().sin()).output();

        let inserted = cx.auto_roll_loops_prepass(6);
        assert_eq!(inserted, 0, "branch-only reuse should not roll into loops");
    }
}
