use crate::{
    prelude::*,
    utils::{EgglogOp, LLIROp},
};
use std::{
    io::Write,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::Duration,
};

use super::compiler_utils::{ToIds, ToIdsMut};
use colored::Colorize;
use egglog::{prelude::RustSpan, var};
use egglog_ast::span::Span;
use egraph_serialize::{ClassId, NodeId};
use itertools::Itertools;
use petgraph::{Direction, stable_graph::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};

pub type LLIRGraph = StableGraph<LLIROp, (), petgraph::Directed>;

pub type StorageGraph = StableGraph<Box<dyn Operator>, Dependency>;

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
/// All data is stored inside this object as well.
#[derive(Debug, Default)]
pub struct Graph {
    /// The store of tensors in the graph. Indexed by node index and output index.
    pub tensors: FxHashMap<(NodeIndex, u8), Tensor>,
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: StorageGraph,
    /// Tensors marked in this set will not get deleted when the graph is ran
    pub no_delete: FxHashSet<NodeIndex>,
    /// Tensors marked in this set need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub to_retrieve: FxHashMap<NodeIndex, (u8, ShapeTracker)>,
    /// A cached list of nodes to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<(NodeIndex, u8, ShapeTracker)>)>>,
    /// Cached consumers (for execution only)
    consumers_map: Option<FxHashMap<(NodeIndex, u8), usize>>,
    /// E-Graph search space
    egraph: Option<SerializedEGraph>,
}

/// A dependency between two nodes
#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum Dependency {
    /// A data dependency (transferring a tensor from one node to the next)
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },
    /// Explicit dependency for ordering. No tensors are transferred through this dependency
    Schedule,
}

impl Dependency {
    /// Try to extract dependency data
    pub fn as_data(self) -> Option<(u8, u8, ShapeTracker)> {
        if let Self::Data {
            input_order,
            output_order,
            shape,
        } = self
        {
            Some((input_order, output_order, shape))
        } else {
            None
        }
    }

    /// Is this a schedule dependency?
    pub fn is_schedule(&self) -> bool {
        matches!(self, Self::Schedule)
    }
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Try to remove the tensor data from the graph
    pub fn get_tensor(&mut self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.tensors.remove(&(id, ind))
    }

    /// Try to get the tensor data in the graph
    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<&Tensor> {
        self.tensors.get(&(id, ind))
    }

    /// Delete the tensor data from the graph
    pub fn drop_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.tensors.remove(&(id, 0));
        }
    }

    /// Mark tensors to be kept
    pub fn keep_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.no_delete.insert(id);
        }
    }

    /// Set a tensor's data
    pub fn set_tensor(&mut self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.tensors.insert((id, ind), tensor);
    }

    /// Set a dynamic dimension
    pub fn set_dyn_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("Tensor", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: &str, shape: impl ToShape) -> GraphTensor {
        let name = name.to_string();
        GraphTensor {
            id: self.graph.add_node(Box::new(Function(
                format!("{name} Load"),
                Box::new(move |_| panic!("You must set a value for this tensor! ({name})")),
            ))),
            graph_ref: self,
            shape: ShapeTracker::new(shape),
        }
    }

    /// Compile the graph using the given compiler
    pub fn compile<T: ToIdsMut, C: Compiler>(&mut self, compiler: C, remap: T) -> C::Output {
        let output = compiler.compile(self, remap);
        self.toposort();
        self.reset();
        output
    }

    /// Refresh the internally sorted graph
    pub(crate) fn toposort(&mut self) {
        self.linearized_graph = Some(
            petgraph::algo::toposort(&self.graph, None)
                .unwrap()
                .into_iter()
                .map(|node| (node, self.get_sources(node)))
                .collect(),
        );

        // Refresh the internal remaining consumers map
        self.consumers_map = Some(
            self.graph
                .node_indices()
                .flat_map(|i| {
                    self.graph
                        .edges_directed(i, Direction::Outgoing)
                        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                        .group_by(|(_, (_, i, _))| *i)
                        .into_iter()
                        .map(|(ind, g)| ((i, ind), g.count()))
                        .collect::<Vec<_>>()
                })
                .collect(),
        );
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors(&mut self, a: GraphTensor, b: GraphTensor) {
        // Swap tensors
        for i in 0.. {
            let a_t = self.tensors.remove(&(a.id, i));
            let b_t = self.tensors.remove(&(b.id, i));
            if a_t.is_none() && b_t.is_none() {
                break;
            }
            if let Some(a_t) = a_t {
                self.tensors.insert((b.id, i), a_t);
            }
            if let Some(b_t) = b_t {
                self.tensors.insert((a.id, i), b_t);
            }
        }
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        self.tensors.retain(|(n, _), _| self.no_delete.contains(n));
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }
        self.reset();
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let mut srcs = src_ids
                .iter()
                .map(|(id, ind, st)| {
                    (
                        InputTensor::Borrowed(self.tensors.get(&(*id, *ind)).unwrap()),
                        *st,
                    )
                })
                .collect_vec();

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready, execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
        }
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        fn format_duration(duration: &Duration) -> String {
            if duration.as_secs() > 0 {
                format!("{:.2}s", duration.as_secs_f32())
            } else if duration.as_millis() > 0 {
                format!("{}ms", duration.as_millis())
            } else {
                format!("{}µs", duration.as_micros())
            }
        }
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut op_times = FxHashMap::default();
        let width = term_size::dimensions().unwrap().0;

        println!(
            "{:->2$} Executing {:->2$}",
            "",
            "",
            (width.saturating_sub(" Executing ".len())) / 2
        );
        let start = std::time::Instant::now();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let op_name = format!("{:?} | {}", self.node_weight(*node).unwrap(), node.index());
            print!("{}", op_name.bold().bright_green());

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready
            let mut shapes_string = srcs.iter().map(|(_, s)| format!("{:?}", s.dims)).join(", ");
            if !shapes_string.is_empty() {
                shapes_string = format!(" ({shapes_string})");
            }
            print!("{shapes_string}");
            std::io::stdout().flush().unwrap();
            // Execute
            let now = std::time::Instant::now();
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            let elapsed = now.elapsed();
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width
                    .saturating_sub(op_name.len())
                    .saturating_sub(shapes_string.len()),
            );
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
            let timed_op_name = format!("{:?}", self.node_weight(*node).unwrap());
            if let Some(t) = op_times.get_mut(&timed_op_name) {
                *t += elapsed;
            } else {
                op_times.insert(timed_op_name, elapsed);
            }

            // Check if we can delete the source tensors now
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }

        // Print out total times
        println!();
        println!(
            "{:->2$} Total Times {:->2$}",
            "",
            "",
            (width.saturating_sub(" Total Times ".len())) / 2
        );
        for (name, elapsed) in op_times.into_iter().sorted_by(|(_, a), (_, b)| b.cmp(a)) {
            print!("{}", name.bold().bright_green());
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width.saturating_sub(name.len()),
            );
        }
        println!("Total: {}", format_duration(&start.elapsed()).bold());
        self.reset();
    }

    pub fn build_search_space(&mut self, ops: &[Arc<Box<dyn EgglogOp>>]) {
        let (program, root) = hlir_to_egglog(self);
        self.egraph = Some(run_egglog(&program, &root, ops).unwrap());
    }

    pub fn search<R: Runtime>(
        &mut self,
        mut runtime: R,
        ops: &Vec<Arc<Box<dyn EgglogOp>>>,
        limit: usize,
    ) -> R {
        let llir_graphs = egglog_to_llir(self.egraph.as_ref().unwrap(), ops, limit);
        runtime.compile(llir_graphs.last().unwrap());
        runtime
    }
}

pub trait Runtime {
    type CompileArg;
    fn initialize(arg: Self::CompileArg) -> Self;
    fn compile(&mut self, llir_graph: &LLIRGraph);
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>);
}

impl Deref for Graph {
    type Target = StorageGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

/// Get source tensor array for a node
fn get_source_tensors<'a>(
    no_delete: &'a FxHashSet<NodeIndex>,
    tensors: *mut FxHashMap<(NodeIndex, u8), Tensor>,
    src_ids: &'a [(NodeIndex, u8, ShapeTracker)],
    consumers: &'a FxHashMap<(NodeIndex, u8), usize>,
) -> Vec<(InputTensor<'a>, ShapeTracker)> {
    let mut srcs = vec![];
    for (id, ind, sh) in src_ids {
        let id = &(*id, *ind);
        if consumers[id] == 1 && !no_delete.contains(&id.0) {
            srcs.push((
                InputTensor::Owned(unsafe { tensors.as_mut().unwrap() }.remove(id).unwrap()),
                *sh,
            ));
        } else {
            srcs.push((
                InputTensor::Borrowed(unsafe { tensors.as_ref().unwrap() }.get(id).unwrap()),
                *sh,
            ));
        }
    }
    srcs
}

fn hlir_to_egglog(graph: &Graph) -> (String, String) {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    // 1. Topo-order with tie-break: lower NodeIndex first
    let mut indeg: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|n| (n, graph.neighbors_directed(n, Direction::Incoming).count()))
        .collect();

    let mut ready: BinaryHeap<(Reverse<usize>, NodeIndex)> = BinaryHeap::new();
    for (n, &d) in &indeg {
        if d == 0 {
            ready.push((Reverse(n.index()), *n));
        }
    }

    let mut topo_order: Vec<NodeIndex> = Vec::with_capacity(indeg.len());
    while let Some((_, n)) = ready.pop() {
        topo_order.push(n);
        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            let e = indeg.get_mut(&succ).unwrap();
            *e -= 1;
            if *e == 0 {
                ready.push((Reverse(succ.index()), succ));
            }
        }
    }

    // 2. Map <node-id> → <egglog var name>
    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut next_id = 0usize;
    let mut out = String::new();

    for n in topo_order {
        let var = format!("t{next_id}");
        next_id += 1;
        let sources = graph
            .get_sources(n)
            .into_iter()
            .zip(
                graph
                    .edges_directed(n, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| names[&e.source()].clone()),
            )
            .map(|((n, _, sh), name)| (n, name, sh))
            .collect_vec();
        let node_weight = graph.node_weight(n).unwrap();
        let op_name_full = format!("{node_weight:?}");
        let op = op_name_full
            .split('|')
            .next()
            .unwrap_or(&op_name_full)
            .trim();
        let code = if op.contains("Load") {
            format!("(GMEM {} \"{}\")", n.index(), op.replace(" Load", ""))
        } else {
            node_weight.to_egglog(&sources)
        };

        out.push_str(&format!("(let {var} {code})\n"));
        names.insert(n, var);
    }

    let root = graph
        .node_indices()
        .find(|&idx| {
            graph
                .neighbors_directed(idx, Direction::Outgoing)
                .next()
                .is_none()
        })
        .and_then(|idx| names.get(&idx))
        .cloned()
        .unwrap_or_else(|| "t0".into());
    (out.replace("(MVar \"z\")", "(MIter)"), root)
}

pub fn shape_to_egglog(shape: &[Expression]) -> String {
    if shape.is_empty() {
        "(ENil)".to_string()
    } else {
        format!(
            "(ECons {} {})",
            shape[0].to_egglog(),
            shape_to_egglog(&shape[1..])
        )
    }
}

pub fn strides_to_egglog(strides: &[Expression]) -> String {
    if strides.is_empty() {
        "(ENil)".to_string()
    } else {
        format!(
            "(ECons {} {})",
            strides[0].to_egglog(),
            strides_to_egglog(&strides[1..])
        )
    }
}

#[derive(Debug)]
pub struct SerializedEGraph {
    pub enodes: FxHashMap<NodeId, (String, Vec<ClassId>)>,
    pub eclasses: FxHashMap<ClassId, (String, Vec<NodeId>)>,
    pub node_to_class: FxHashMap<NodeId, ClassId>,
    pub roots: Vec<ClassId>,
}

#[tracing::instrument(skip_all)]
fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
) -> Result<SerializedEGraph, egglog::Error> {
    let mut egraph = egglog::EGraph::default();
    let mut code = include_str!("egglog.egg").replace("{program}", &program);
    code = code.replace(
        "{ops}",
        &ops.iter()
            .map(|o| {
                let (name, body) = o.term();
                format!(
                    "({name} {})",
                    body.into_iter().map(|j| format!("{j:?}")).join(" ")
                )
            })
            .join("\n"),
    );
    code = code.replace(
        "{rewrites}",
        &ops.iter().map(|o| o.rewrites().join("\n")).join("\n"),
    );
    code = code.replace(
        "{cleanups}",
        &ops.iter()
            .filter(|op| op.cleanup())
            .filter(|op| op.term().0 != "GMEM")
            .map(|o| {
                let (name, body) = o.term();
                let body_terms = (0..body.len()).map(|i| (b'a' + i as u8) as char).join(" ");
                format!(
                    "(rule
                ((= ?m ({name} {body_terms})))
                ((delete ({name} {body_terms})))
                :ruleset cleanup
            )"
                )
            })
            .join("\n"),
    );

    let commands = egraph.parser.get_program_from_string(None, &code)?;
    let start = std::time::Instant::now();
    let msgs = egraph.run_program(commands)?;
    println!("Egglog Took {}ms", start.elapsed().as_millis());

    let (sort, value) = egraph.eval_expr(&var!(root))?;
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        max_functions: None,
        include_temporary_functions: false,
        max_calls_per_function: None,
    });
    // Convert to SerializedEGraph
    let mut classes = FxHashMap::default();
    for (node_id, node) in &s.egraph.nodes {
        classes
            .entry(node.eclass.clone())
            .or_insert(vec![])
            .push(node_id.clone())
    }
    let mut egraph = SerializedEGraph {
        roots: s.egraph.root_eclasses,
        node_to_class: s
            .egraph
            .nodes
            .iter()
            .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
            .collect(),
        enodes: s
            .egraph
            .nodes
            .iter()
            .map(|(n, enode)| {
                (
                    n.clone(),
                    (
                        enode.op.clone(),
                        enode
                            .children
                            .iter()
                            .map(|n| s.egraph.nodes[n].eclass.clone())
                            .collect(),
                    ),
                )
            })
            .collect(),
        eclasses: s
            .egraph
            .class_data
            .iter()
            .map(|(c, eclass)| (c.clone(), (eclass.typ.clone().unwrap(), classes[c].clone())))
            .collect(),
    };
    // Strip out all [...] enodes
    egraph.enodes.retain(|_, (label, _)| label != "[...]");
    loop {
        let mut to_remove = vec![];
        for (id, (_, children)) in &egraph.enodes {
            if children.iter().any(|c| {
                !egraph.eclasses[c]
                    .1
                    .iter()
                    .any(|n| egraph.enodes.contains_key(n))
            }) {
                to_remove.push(id.clone());
            }
        }
        for n in &to_remove {
            egraph.enodes.remove(n);
        }
        if to_remove.is_empty() {
            break;
        }
    }
    // Correct the eclass mapping
    for (_, (_, enodes)) in &mut egraph.eclasses {
        enodes.retain(|n| egraph.enodes.contains_key(n));
    }
    egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
    egraph
        .node_to_class
        .retain(|n, _| egraph.enodes.contains_key(n));
    assert!(
        egraph.roots.iter().all(|c| egraph.eclasses.contains_key(c)),
        "No valid graphs present in the e-graph!"
    );
    if msgs.iter().any(|m| !m.to_string().is_empty()) {
        println!("Messages:");
        for m in msgs {
            if !m.to_string().is_empty() {
                println!("{m}");
            }
        }
    }

    Ok(egraph)
}

pub fn extract_expr_list<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> Option<Vec<Expression>> {
    if let Some(l) = list_cache.get(node) {
        return Some(l.clone());
    }
    let expr = extract_expr(
        egraph,
        &egraph.eclasses[&egraph.enodes[node].1[0]].1[0],
        expr_cache,
    )?;
    match egraph.enodes[&egraph.eclasses[&egraph.enodes[node].1[1]].1[0]]
        .0
        .as_str()
    {
        "ENil" => Some(vec![expr]),
        "ECons" => {
            let mut rest = extract_expr_list(
                egraph,
                &egraph.eclasses[&egraph.enodes[node].1[1]].1[0],
                list_cache,
                expr_cache,
            )?;
            rest.insert(0, expr);
            list_cache.insert(node, rest.clone());
            Some(rest)
        }
        _ => unreachable!(),
    }
}

pub fn extract_expr<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> Option<Expression> {
    if let Some(e) = expr_cache.get(node) {
        return Some(*e);
    }

    fn extract_shortest<'a>(
        egraph: &'a SerializedEGraph,
        class: &'a ClassId,
        seen: &mut FxHashMap<&'a NodeId, usize>,
        cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
    ) -> Option<Vec<&'a NodeId>> {
        const MAX_CYCLES: usize = 1;
        egraph.eclasses[class]
            .1
            .iter()
            .filter_map(|en| {
                if *seen.get(en).unwrap_or(&0) >= MAX_CYCLES || egraph.enodes[en].0 == "[...]" {
                    return None;
                }
                if let Some(c) = cache.get(en) {
                    return c.clone();
                }
                *seen.entry(en).or_insert(0) += 1;
                let out = if egraph.enodes[en].1.is_empty() {
                    Some(vec![en])
                } else {
                    egraph.enodes[en]
                        .1
                        .iter()
                        .try_fold(vec![en], |mut acc, ch| {
                            extract_shortest(egraph, ch, seen, cache).map(|p| {
                                acc.extend(p);
                                acc
                            })
                        })
                };
                *seen.get_mut(en).unwrap() -= 1;
                cache.insert(en, out.clone());
                out
            })
            .min_by_key(|p| p.len())
    }

    let traj = extract_shortest(
        egraph,
        &egraph.node_to_class[node],
        &mut FxHashMap::default(),
        &mut FxHashMap::default(),
    )?;
    fn build_expression<'a>(
        egraph: &SerializedEGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
    ) -> Expression {
        let nid = trajectory[*current];
        let op = egraph.enodes[nid].0.as_str();
        match op {
            // unary math
            "MNeg" | "MRecip" => {
                *current += 1;
                let c0 = build_expression(egraph, trajectory, current);
                match op {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }
            // binary math
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" | "MCeilDiv" => {
                *current += 1;
                let lhs = build_expression(egraph, trajectory, current);
                *current += 1;
                let rhs = build_expression(egraph, trajectory, current);
                match op {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MGte" => lhs.gte(rhs),
                    "MLt" => lhs.lt(rhs),
                    "MCeilDiv" => lhs.ceil_div(rhs),
                    "MFloorTo" => lhs / rhs * rhs, // TODO: real floorto in Expression
                    _ => unreachable!(),
                }
            }
            // wrappers around a literal/var child
            "MNum" | "MVar" => {
                *current += 1;
                build_expression(egraph, trajectory, current)
            }
            "MIter" => Expression::from('z'),
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                Expression::from(name.chars().next().unwrap())
            }
            op => op
                .parse::<usize>()
                .map(Expression::from)
                .or_else(|_| op.replace('"', "").parse::<char>().map(Expression::from))
                .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
        }
    }
    let e = build_expression(egraph, &traj, &mut 0);
    expr_cache.insert(node, e);
    Some(e)
}

#[tracing::instrument(skip_all)]
pub fn egglog_to_llir(
    egraph: &SerializedEGraph,
    ops: &Vec<Arc<Box<dyn EgglogOp>>>,
    limit: usize,
) -> Vec<LLIRGraph> {
    // Get maps for all e-classes to e-node options
    let mut choices = vec![FxHashMap::default()];
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") {
            continue;
        }
        choices = enodes
            .iter()
            .flat_map(|enode| {
                choices.iter().cloned().map(move |mut choice_map| {
                    choice_map.insert(eclass, enode);
                    choice_map
                })
            })
            .rev()
            .take(limit)
            .collect();
    }

    // Create IR graphs
    let mut graphs = vec![];
    let mut c = FxHashMap::default();
    let mut lc = FxHashMap::default();
    for choice in choices {
        // Make reachability set from root
        let mut reachable = FxHashSet::default();
        reachable.insert(choice[&egraph.roots[0]]);
        let mut reachability_stack = vec![choice[&egraph.roots[0]]];
        while let Some(r) = reachability_stack.pop() {
            for ch in &egraph.enodes[r].1 {
                if egraph.eclasses[ch].0.contains("IR") {
                    let n = choice[ch];
                    if !reachable.contains(n) {
                        reachability_stack.push(n);
                        reachable.insert(n);
                    }
                }
            }
        }
        let mut graph = LLIRGraph::default();
        let mut edges_to_place = vec![];
        let mut enode_to_node = FxHashMap::default();
        'nodes: for (_, &node) in &choice {
            if !reachable.contains(node) {
                continue;
            }
            let ch = egraph.enodes[node]
                .1
                .iter()
                .map(|c| {
                    if egraph.eclasses[c].0.contains("IR") {
                        choice[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();
            match egraph.enodes[node].0.as_str() {
                "GMEM" => {
                    let index = egraph.enodes[ch[0]]
                        .0
                        .replace("\"", "")
                        .parse::<usize>()
                        .unwrap();
                    let label = egraph.enodes[ch[1]].0.replace("\"", "");
                    enode_to_node.insert(
                        node,
                        graph.add_node(LLIROp::new(Box::new(GMEM { node: index, label }))),
                    );
                }
                op => {
                    for phys_op in ops {
                        if op == phys_op.term().0 {
                            // Extract this op
                            let (op_instance, sources) =
                                phys_op.extract(egraph, &ch, &mut lc, &mut c);
                            let r = graph.add_node(op_instance);
                            enode_to_node.insert(node, r);
                            for source in sources {
                                edges_to_place.push((source, node));
                            }
                            continue 'nodes;
                        }
                    }
                    todo!("{op} extraction not implemented!");
                }
            }
        }
        for (src, dest) in edges_to_place {
            graph.add_edge(enode_to_node[src], enode_to_node[dest], ());
        }

        graphs.push(graph);
    }
    graphs
}

/// Helper: implement `$name` for a tuple arity `(A, B, C, ...)` by concatenation.
#[macro_export]
macro_rules! __impl_tuple_into_dyn_arcbox_concat_arity {
    ($tr:ident; $($T:ident),+ $(,)?) => {
        $crate::paste!{
        impl<$($T),+> [<Into $tr>] for ($($T,)+)
        where
            $(
                $T: [<Into $tr>],
            )+
        {
            #[inline]
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {
                $(
                    <$T as [<Into $tr>]>::append_into(out);
                )+
            }
        }
        }
    };
}

/// Define `$name` which builds `Vec<Arc<Box<dyn $tr + 'static>>>`
/// from leaf types (`$tr + Default + 'static`) and supports nested tuples.
///
/// Tuples work as long as every element implements `$name`, so you can nest:
/// `(Op1, (Op2, Op3), (), (Op4, (Op5,)))`.
#[macro_export]
macro_rules! impl_into_ops {
    ($tr:ident) => {
        $crate::paste!{
        pub trait [<Into $tr>] {
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            );

            #[inline]
            fn into_vec() -> ::std::vec::Vec<
                ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
            > {
                let mut out = ::std::vec::Vec::new();
                Self::append_into(&mut out);
                out
            }
        }

        // base
        impl [<Into $tr>] for () {
            #[inline]
            fn append_into(
                _out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {}
        }

        // leaf: any concrete op type
        impl<T> [<Into $tr>] for T
        where
            T: $tr + ::std::default::Default + 'static,
        {
            #[inline]
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {
                out.push(::std::sync::Arc::new(::std::boxed::Box::new(
                    <T as ::std::default::Default>::default(),
                )));
            }
        }
        }

        // tuple concatenation impls (extend arity list as needed)
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);
    };
}
