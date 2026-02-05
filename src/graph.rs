use crate::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, hlir_to_egglog, random_initial_choice,
    run_egglog,
};
use crate::{
    egglog_utils::SerializedEGraph,
    op::{EgglogOp, IntoEgglogOp, LLIROp},
};
use crate::{hlir::CustomOpHLIR, op::*, prelude::*};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{Direction, stable_graph::StableGraph, visit::EdgeRef};
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
    /// E-Graph search space
    egraph: Option<SerializedEGraph>,
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
        let (program, root) = hlir_to_egglog(self);
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>(); // need to ignore hlir op cleanups if we're on native runtime
        self.egraph = Some(run_egglog(&program, &root, &ops, cleanup_hlir).unwrap());
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
        let (program, root) = hlir_to_egglog(self);
        let cleanup_hlir = TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>(); // need to ignore hlir op cleanups if we're on native runtime
        self.egraph = Some(run_egglog(&program, &root, &ops, cleanup_hlir).unwrap());
        self.ops = Some(ops);
    }

    /// Get a reference to the e-graph search space (if built)
    pub fn egraph(&self) -> Option<&SerializedEGraph> {
        self.egraph.as_ref()
    }

    /// Get a reference to the available ops (if search space is built)
    pub fn egglog_ops(&self) -> Option<&Vec<Arc<Box<dyn EgglogOp>>>> {
        self.ops.as_ref()
    }

    const DEFAULT_GENERATION_SIZE: usize = 50;
    const MUTATIONS_PER_OFFSPRING: usize = 40;
    const TRIALS_PER_PROFILE: usize = 5;

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
        let mut rng = rand::rng();
        let egraph = self.egraph.as_ref().unwrap();
        let ops = self.ops.as_ref().unwrap();

        // Initialize tracking state
        let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();

        // Start with a random initial genome
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

        // Profile initial genome
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

        // Genetic algorithm search loop
        let mut generation = 0;
        while n_graphs < limit {
            generation += 1;
            // Extract offspring from the current best genome
            let offspring = extract_generation(
                egraph,
                &best_genome,
                (limit - n_graphs).min(Self::DEFAULT_GENERATION_SIZE),
                Self::MUTATIONS_PER_OFFSPRING,
                &mut prev_selected,
                &mut rng,
            );

            // If no offspring could be generated, search space is exhausted
            if offspring.is_empty() {
                break;
            }

            // Profile each offspring
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
