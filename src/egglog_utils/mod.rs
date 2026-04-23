use colored::Colorize;
use egglog::{ast::Span, prelude::RustSpan, var};
use itertools::Itertools;
use petgraph::{Direction, graph::NodeIndex};
use rand::Rng;
use regex::Regex;
use rustc_hash::FxHashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{str, sync::Arc, time::Duration};
use tracing::trace;

pub mod api;
pub mod base;

pub const RUN_SCHEDULE: &str = "(run-schedule
    (repeat 10
        (saturate expr)
        (saturate dtype_prop)
        (run)
    )
    (saturate expr)
    (saturate cleanup)
    (saturate base_cleanup)
)";

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    // Partition ops by sort class: IR-class (Input, Output) vs OpKind-class (everything else)
    let mut ir_variants = Vec::new();
    let mut opkind_variants = Vec::new();
    for o in ops {
        let s = o.sort();
        let variant_str = format!(
            "({} {})",
            s.name,
            s.fields.iter().map(|f| &f.sort).join(" ")
        );
        if s.class == "IR" {
            ir_variants.push(variant_str);
        } else if s.class == "OpKind" {
            opkind_variants.push(variant_str);
        } else {
            panic!("Unknown sort class '{}' for op '{}'", s.class, s.name);
        }
    }
    let ir_str = ir_variants.join("\n");
    let opkind_str = opkind_variants.join("\n");
    let extra_ir: FxHashSet<String> = ops.iter().flat_map(|o| o.ir_defs()).collect();
    let extra_ir_str = extra_ir.into_iter().join("\n");
    format!(
        "
    (datatype*
        (IR
            (OutputJoin IR IR)
            (Op OpKind IList)
            {extra_ir_str}
            {ir_str}
        )
        (OpKind
            {opkind_str}
        )
        (IList
            (ICons IR IList)
            (INil)
        )
    )
    (function dtype (IR) DType :merge new)
    "
    )
}

fn op_cleanups_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    format!(
        "
    {}
    ",
        ops.iter()
            .filter(|op| op.cleanup())
            .map(|o| {
                let s = o.sort();
                let body_terms = (0..s.fields.len())
                    .map(|i| (b'a' + i as u8) as char)
                    .join(" ");
                if s.class == "OpKind" {
                    // Normalized op: (Op (XxxKind ...) ?inputs)
                    format!(
                        "(rule
                ((= ?m (Op ({} {body_terms}) ?__cleanup_inputs)))
                ((delete (Op ({} {body_terms}) ?__cleanup_inputs)))
                :ruleset cleanup
            )",
                        s.name, s.name
                    )
                } else {
                    // Direct IR variant (Input, Output)
                    format!(
                        "(rule
                ((= ?m ({} {body_terms})))
                ((delete ({} {body_terms})))
                :ruleset cleanup
            )",
                        s.name, s.name
                    )
                }
            })
            .join("\n")
    )
}

pub fn early_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> String {
    [
        base::base_expression_egglog(),
        op_defs_string(ops),
        ops.iter()
            .flat_map(|o| o.early_rewrites())
            .map(|r| r.to_egglog_string())
            .join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        base::base_cleanup_egglog(),
        program.to_string(),
        format!(
            "(run-schedule
                (repeat 6
                    (saturate expr)
                    (run)
                )
                (saturate base_cleanup)
            )
            (extract {root})"
        ),
    ]
    .join("\n")
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    [
        base::base_expression_egglog(),
        op_defs_string(ops),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        base::base_cleanup_egglog(),
        ops.iter()
            .flat_map(|o| o.rewrites())
            .map(|r| r.to_egglog_string())
            .join("\n"),
        program.to_string(),
        RUN_SCHEDULE.to_string(),
    ]
    .join("\n")
}

use crate::{
    dtype::DType,
    graph::{Graph, LLIRGraph, SubgraphDescriptor},
    hlir::{Input, Output},
    op::{CustomOp, EgglogOp},
    prelude::FxHashMap,
    shape::Expression,
};
use egglog::{ArcSort, CommandOutput, EGraph, Value};
use egraph_serialize::{ClassId, NodeId};

#[derive(Debug)]
///  This is snapshot of an EGraph with Rust native hash maps and sets for enabling more native traversal / algorithm writing.
///  The name comes from the serialize egraph crates, which returns a ETermDAG, which caused issues, so this is a homebrew semi-static egraph
pub struct SerializedEGraph {
    pub enodes: FxHashMap<NodeId, (String, Vec<ClassId>)>,
    pub eclasses: FxHashMap<ClassId, (String, Vec<NodeId>)>,
    pub node_to_class: FxHashMap<NodeId, ClassId>,
    pub roots: Vec<ClassId>,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogStageReport {
    pub num_matches_per_rule: FxHashMap<String, usize>,
    pub search_and_apply_time_per_rule: FxHashMap<String, Duration>,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogRunReport {
    pub early: EgglogStageReport,
    pub full: EgglogStageReport,
    pub total_time: Duration,
}

impl SerializedEGraph {
    /// This is an opinionated function which does more than strictly take the state of the egglog object.
    /// It also filters out "[...]" nodes and then changes the structure from the e-termDAG that egraph-serialize
    /// produces to a strict egraph, where the children of e-classes are e-nodes.
    pub fn new(egraph: &EGraph, root_eclasses: Vec<(ArcSort, Value)>) -> Self {
        let s = egraph.serialize(egglog::SerializeConfig {
            root_eclasses,
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
        let mut s_egraph = SerializedEGraph {
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
        s_egraph.enodes.retain(|_, (label, _)| label != "[...]");
        loop {
            let mut to_remove = vec![];
            for (id, (_, children)) in &s_egraph.enodes {
                if children.iter().any(|c| {
                    !s_egraph.eclasses[c]
                        .1
                        .iter()
                        .any(|n| s_egraph.enodes.contains_key(n))
                }) {
                    to_remove.push(id.clone());
                }
            }
            for n in &to_remove {
                s_egraph.enodes.remove(n);
            }
            if to_remove.is_empty() {
                break;
            }
        }
        // Correct the eclass mapping
        for (_, enodes) in s_egraph.eclasses.values_mut() {
            enodes.retain(|n| s_egraph.enodes.contains_key(n));
        }
        s_egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
        s_egraph
            .node_to_class
            .retain(|n, _| s_egraph.enodes.contains_key(n));
        s_egraph
    }
}

/// Hash a SerializedEGraph by its structural content for dedup comparison.
/// Only considers IR/IList eclasses and enodes (not primitives like i64, String, DType
/// which contain per-chunk-specific values like node indices and weight labels).
pub fn hash_serialized_egraph(egraph: &SerializedEGraph) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Only count IR/IList eclasses (computation nodes, not primitives)
    let ir_eclasses: Vec<_> = egraph
        .eclasses
        .values()
        .filter(|(label, _)| label.contains("IR") || label.contains("IList"))
        .collect();
    ir_eclasses.len().hash(&mut hasher);
    let mut eclass_info: Vec<_> = ir_eclasses
        .iter()
        .map(|(label, enodes)| (label.clone(), enodes.len()))
        .collect();
    eclass_info.sort();
    eclass_info.hash(&mut hasher);
    // Only hash IR/IList enodes by op name and child count
    let mut enode_info: Vec<_> = egraph
        .enodes
        .iter()
        .filter(|(node_id, _)| {
            let eclass = &egraph.node_to_class[*node_id];
            if let Some((label, _)) = egraph.eclasses.get(eclass) {
                label.contains("IR") || label.contains("IList")
            } else {
                false
            }
        })
        .map(|(_, (op, children))| (op.clone(), children.len()))
        .collect();
    enode_info.sort();
    enode_info.hash(&mut hasher);
    hasher.finish()
}

/// Hash egglog text with normalization for structural dedup.
///
/// Structurally identical chunks (e.g. transformer layers) produce identical
/// egglog text except for:
/// - Input node indices and labels (differ per layer)
/// - Output node indices (differ per layer)
/// - CustomOpKind integer IDs (global custom_ops index, differs per layer)
///
/// This function hashes the text while normalizing those chunk-specific values:
/// - Input lines: only the dtype is hashed (not node index or label)
/// - Output lines: only the "OUTPUT" marker is hashed (not the node index)
/// - CustomOpKind lines: the integer ID is replaced with a constant
/// - All other lines (ops, shapes, strides): hashed verbatim
pub fn hash_egglog_normalized(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for line in text.lines() {
        if line.contains("(Input ") {
            // Format: (let tN (Input NODE "LABEL" (DTYPE)))
            // Strip the node index and label identity, but preserve whether this
            // is a synthetic boundary input or a real graph input.
            // The dtype is the last parenthesized token, e.g. "(F32)".
            if let Some(dtype_start) = line.rfind(" (") {
                let dtype = &line[dtype_start + 1..];
                let kind = if line.contains("\"boundary\"") {
                    "BOUNDARY_INPUT"
                } else {
                    "REAL_INPUT"
                };
                (kind, dtype).hash(&mut hasher);
            } else {
                line.hash(&mut hasher);
            }
        } else if line.contains("(Output ") && !line.contains("(OutputJoin ") {
            "OUTPUT".hash(&mut hasher);
        } else if line.contains("(CustomOpKind ") {
            // Format: (let tN (Op (CustomOpKind ID (DTYPE)) (ICons ...)))
            // The integer ID varies per layer. Replace it with a constant.
            normalize_custom_op_id(line).hash(&mut hasher);
        } else {
            line.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Replace the integer ID in a CustomOpKind egglog line with a constant "0".
fn normalize_custom_op_id(line: &str) -> String {
    if let Some(custom_start) = line.find("(CustomOpKind ") {
        let after = &line[custom_start + "(CustomOpKind ".len()..];
        // The ID is the first token (integer) after "CustomOpKind "
        if let Some(space_after_id) = after.find(' ') {
            let id_str = &after[..space_after_id];
            if id_str.chars().all(|c| c.is_ascii_digit()) {
                return format!(
                    "{}0{}",
                    &line[..custom_start + "(CustomOpKind ".len()],
                    &line[custom_start + "(CustomOpKind ".len() + space_after_id..]
                );
            }
        }
    }
    line.to_string()
}

pub fn hlir_to_egglog(graph: &Graph) -> (String, String) {
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
    let mut out = String::new();

    let mut curr_id = 0;
    for n in topo_order {
        let sources: Vec<(NodeIndex, String)> = graph
            .get_sources(n)
            .into_iter()
            .map(|src| (src, names[&src].clone()))
            .collect_vec();
        let code = graph[n].to_egglog(&sources);
        out.push_str(&format!("(let t{curr_id} {code})\n"));
        names.insert(n, format!("t{curr_id}"));
        curr_id += 1;
    }

    // Join outputs using dummy op
    let names = graph
        .externals(Direction::Outgoing)
        .map(|n| names.remove(&n).unwrap())
        .collect_vec();
    let mut root = names[0].clone();
    for node in names.into_iter().skip(1) {
        curr_id += 1;
        out.push_str(&format!("(let t{curr_id} (OutputJoin {root} {node}))\n"));
        root = format!("t{curr_id}");
    }
    (out.replace("(MVar \"z\")", "(MIter)"), root)
}

/// Convert a subgraph of the HLIR to egglog, injecting synthetic Input/Output
/// nodes at graph break boundaries.
pub fn hlir_subgraph_to_egglog(graph: &Graph, subgraph: &SubgraphDescriptor) -> (String, String) {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut out = String::new();
    let mut curr_id = 0;
    let mut boundary_output_names: Vec<String> = vec![];

    // Emit synthetic Input nodes for boundary inputs
    for boundary in &subgraph.boundary_inputs {
        let var_name = format!("t{curr_id}");
        let code = format!(
            "(Input {} \"boundary\" ({:?}))",
            boundary.break_node.index(),
            boundary.dtype
        );
        out.push_str(&format!("(let {var_name} {code})\n"));
        // Map the boundary node to this synthetic Input variable.
        // When downstream nodes reference the boundary source, they'll use this.
        names.insert(boundary.break_node, var_name);
        curr_id += 1;
    }

    // Topo-order only the nodes in this subgraph
    // Build sub-indeg map restricted to subgraph nodes
    let mut indeg: HashMap<NodeIndex, usize> = HashMap::new();
    for &n in &subgraph.nodes {
        let count = graph
            .graph
            .neighbors_directed(n, Direction::Incoming)
            .filter(|pred| subgraph.nodes.contains(pred))
            .count();
        indeg.insert(n, count);
    }

    let mut ready: BinaryHeap<(Reverse<usize>, NodeIndex)> = BinaryHeap::new();
    for (&n, &d) in &indeg {
        if d == 0 {
            ready.push((Reverse(n.index()), n));
        }
    }

    let mut topo_order: Vec<NodeIndex> = Vec::with_capacity(indeg.len());
    while let Some((_, n)) = ready.pop() {
        topo_order.push(n);
        for succ in graph.graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(e) = indeg.get_mut(&succ) {
                *e -= 1;
                if *e == 0 {
                    ready.push((Reverse(succ.index()), succ));
                }
            }
        }
    }

    // Convert each node in topological order to egglog
    for n in topo_order {
        let sources: Vec<(NodeIndex, String)> = graph
            .get_sources(n)
            .into_iter()
            .map(|src| {
                let name = names
                    .get(&src)
                    .cloned()
                    .unwrap_or_else(|| panic!("Missing egglog name for node {:?}", src));
                (src, name)
            })
            .collect_vec();
        let code = graph.graph[n].to_egglog(&sources);
        out.push_str(&format!("(let t{curr_id} {code})\n"));
        names.insert(n, format!("t{curr_id}"));
        curr_id += 1;
    }

    // Emit synthetic Output nodes for boundary outputs.
    // `boundary_outputs` stores producer nodes for virtual region boundaries.
    for &brk in &subgraph.boundary_outputs {
        let pred = if subgraph.nodes.contains(&brk) {
            brk
        } else {
            graph
                .graph
                .neighbors_directed(brk, Direction::Incoming)
                .next()
                .expect("Boundary output indirection must have exactly one input")
        };
        let pred_name = names.get(&pred).cloned().unwrap_or_else(|| {
            panic!(
                "Missing egglog name for boundary output predecessor {:?}",
                pred
            )
        });
        let code = format!("(Output {} {})", pred_name, brk.index());
        let boundary_name = format!("t{curr_id}");
        out.push_str(&format!("(let {boundary_name} {code})\n"));
        boundary_output_names.push(boundary_name);
        curr_id += 1;
    }

    // Join outputs: real outputs (nodes with no outgoing edges within the subgraph)
    // plus boundary outputs
    let mut output_names: Vec<String> = boundary_output_names;

    // Real outputs: only actual Output HLIR ops that exist in this subgraph
    // (not arbitrary nodes that happen to have no subgraph successors)
    for &n in &subgraph.nodes {
        if graph.try_get_op::<Output>(n).is_some() {
            if let Some(name) = names.get(&n) {
                output_names.push(name.clone());
            }
        }
    }

    if output_names.is_empty() {
        // Fallback: use the last node added
        output_names.push(format!("t{}", curr_id - 1));
    }

    // Join with OutputJoin
    let mut root = output_names[0].clone();
    for node in output_names.into_iter().skip(1) {
        curr_id += 1;
        out.push_str(&format!("(let t{curr_id} (OutputJoin {root} {node}))\n"));
        root = format!("t{curr_id}");
    }

    (out.replace("(MVar \"z\")", "(MIter)"), root)
}

pub fn elist_to_egglog(shape: &[Expression]) -> String {
    list_to_egglog(
        &shape.iter().map(|e| e.to_egglog()).collect_vec(),
        "ECons",
        "ENil",
    )
}

pub fn list_to_egglog(list: &[impl ToString], cons: &str, nil: &str) -> String {
    if list.is_empty() {
        format!("({nil})")
    } else {
        format!(
            "({cons} {} {})",
            list[0].to_string(),
            list_to_egglog(&list[1..], cons, nil)
        )
    }
}

fn termdag_to_egglog(td: &egglog::TermDag, root: egglog::TermId) -> (String, String) {
    let mut out = String::new();
    for id in 0..td.size() {
        let code = match td.get(id) {
            egglog::Term::Lit(lit) => format!("{lit}"),
            egglog::Term::Var(v) => v.clone(),
            egglog::Term::App(head, args) => format!(
                "({head} {})",
                args.iter().map(|s| format!("t{s}")).join(" ")
            ),
        };
        out.push_str(&format!("(let t{id} {code})\n"));
    }
    (out.replace("(MVar \"z\")", "(MIter)"), format!("t{root}"))
}

fn stage_report(egraph: &egglog::EGraph, total_time: Duration) -> EgglogStageReport {
    let run_report = egraph.get_overall_run_report();
    EgglogStageReport {
        num_matches_per_rule: run_report
            .num_matches_per_rule
            .iter()
            .map(|(name, matches)| (name.to_string(), *matches))
            .collect(),
        search_and_apply_time_per_rule: run_report
            .search_and_apply_time_per_rule
            .iter()
            .map(|(name, elapsed)| (name.to_string(), *elapsed))
            .collect(),
        total_time,
    }
}

fn trace_stage_report(header: &str, report: &EgglogStageReport) {
    trace!("{}", header.green());
    trace!(
        "{}",
        report
            .num_matches_per_rule
            .iter()
            .filter(|(k, _)| !k.contains("("))
            .map(|(k, v)| format!(
                "{k}: {v} ({})",
                pretty_duration::pretty_duration(&report.search_and_apply_time_per_rule[k], None)
            ))
            .join("\n")
            .green()
    );
    trace!(
        "{}",
        format!(
            "---- {} Took {} ----",
            header,
            pretty_duration::pretty_duration(&report.total_time, None).bold()
        )
        .green()
    );
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_report(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let total_start = std::time::Instant::now();

    let early_start = std::time::Instant::now();
    let code = early_egglog(program, root, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    let outputs = egraph.run_program(commands)?;
    let early_report = stage_report(&egraph, early_start.elapsed());

    let CommandOutput::ExtractBest(termdag, _cost, term) = outputs.last().unwrap() else {
        panic!();
    };
    let (program, root) = termdag_to_egglog(termdag, termdag.lookup(term));

    let full_start = std::time::Instant::now();
    let code = full_egglog(&program, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    trace!("{}", "Egglog running...".green());
    let _outputs = egraph.run_program(commands)?;
    let full_report = stage_report(&egraph, full_start.elapsed());
    trace_stage_report("---- Egglog Early Rule Matches ----", &early_report);
    trace_stage_report("---- Egglog Full Rule Matches ----", &full_report);

    let run_report = EgglogRunReport {
        early: early_report,
        full: full_report,
        total_time: total_start.elapsed(),
    };
    trace!(
        "{}",
        format!(
            "---- Egglog Total Took {} ----",
            pretty_duration::pretty_duration(&run_report.total_time, None).bold()
        )
        .green()
    );

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

    // Cascade: remove enodes whose children reference empty eclasses
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
    for (_, enodes) in egraph.eclasses.values_mut() {
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

    Ok((egraph, run_report))
}

#[tracing::instrument(skip_all)]
pub fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    run_egglog_with_report(program, root, ops, cleanup).map(|(egraph, _)| egraph)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_multi_roots(
    program: &str,
    roots: &[String],
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    // Two-stage pipeline mirroring run_egglog_with_report:
    //   1. Run early rewrites, extract best per root.
    //   2. Reassemble extracted termdags into one program, run full rewrites.
    //
    // Without this staging the regionalized path stuffs early + full rules
    // into one saturation, which makes pattern-matching rewrites like
    // GLUMoE dependent on saturation order. The two-stage approach commits
    // one rewrite choice per e-class before full rules run.
    //
    // Collect rule names declared in early rewrites (each raw egg file embeds
    // rule names via `:name "..."`). If stage 1 reports zero matches across
    // all of these, stage 1 did nothing useful — fall back to single-stage on
    // the original program. Running the extract round-trip when nothing fused
    // caused a measurable regression (observed ~40% TPOT on Llama).
    let name_re = Regex::new(r#":name\s+"([^"]+)""#).expect("valid name regex");
    let early_rule_names: FxHashSet<String> = ops
        .iter()
        .flat_map(|o| o.early_rewrites())
        .flat_map(|r| {
            let s = r.to_egglog_string();
            name_re
                .captures_iter(&s)
                .map(|c| c[1].to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    // ----- Stage 1: early rewrites, one extract per root -----
    let early_code = [
        base::base_expression_egglog(),
        op_defs_string(ops),
        ops.iter()
            .flat_map(|o| o.early_rewrites())
            .map(|r| r.to_egglog_string())
            .join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        base::base_cleanup_egglog(),
        program.to_string(),
        "(run-schedule
            (repeat 6
                (saturate expr)
                (run)
            )
            (saturate base_cleanup)
        )"
        .to_string(),
        roots.iter().map(|r| format!("(extract {r})")).join("\n"),
    ]
    .join("\n");

    let mut egraph = egglog::EGraph::default();
    let commands = egraph
        .parser
        .get_program_from_string(None, &early_code)?;
    let early_outputs = egraph.run_program(commands)?;

    // If no early-rewrite rule actually fired, skip the extract/reassemble
    // round-trip and run the original program through the single-stage path.
    let run_report = egraph.get_overall_run_report();
    let any_early_fired = run_report
        .num_matches_per_rule
        .iter()
        .any(|(name, count)| *count > 0 && early_rule_names.contains(&name.to_string()));
    if !any_early_fired {
        let code = full_egglog(program, ops, cleanup);
        let mut egraph = egglog::EGraph::default();
        let commands = egraph.parser.get_program_from_string(None, &code)?;
        let _ = egraph.run_program(commands)?;
        let mut root_eclasses = Vec::with_capacity(roots.len());
        for root in roots {
            let (sort, value) = egraph.eval_expr(&var!(root))?;
            root_eclasses.push((sort, value));
        }
        return Ok(SerializedEGraph::new(&egraph, root_eclasses));
    }

    let extracted: Vec<&CommandOutput> = early_outputs
        .iter()
        .filter(|o| matches!(o, CommandOutput::ExtractBest(..)))
        .collect();
    assert_eq!(
        extracted.len(),
        roots.len(),
        "early stage extract count ({}) must match root count ({})",
        extracted.len(),
        roots.len()
    );

    // Reassemble each extracted termdag into a prefixed let-binding block so
    // the next stage sees a single program with unique tN names per root.
    let token_re = Regex::new(r"\bt\d+\b").expect("valid egglog temp-var regex");
    let mut merged = String::new();
    let mut merged_roots = Vec::with_capacity(roots.len());
    for (idx, out) in extracted.iter().enumerate() {
        let CommandOutput::ExtractBest(termdag, _cost, term) = out else {
            unreachable!();
        };
        let (text, local_root) = termdag_to_egglog(termdag, termdag.lookup(term));
        let prefix = format!("s{idx}_");
        let renamed = token_re.replace_all(&text, |caps: &regex::Captures<'_>| {
            format!("{prefix}{}", &caps[0])
        });
        let renamed_root = token_re
            .replace_all(&local_root, |caps: &regex::Captures<'_>| {
                format!("{prefix}{}", &caps[0])
            })
            .to_string();
        merged.push_str(&renamed);
        if !merged.ends_with('\n') {
            merged.push('\n');
        }
        merged_roots.push(renamed_root);
    }

    // ----- Stage 2: full rewrites on the re-emitted program -----
    let code = full_egglog(&merged, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    trace!("{}", "Egglog running...".green());
    let _outputs = egraph.run_program(commands)?;

    let mut root_eclasses = Vec::with_capacity(merged_roots.len());
    for root in &merged_roots {
        let (sort, value) = egraph.eval_expr(&var!(root))?;
        root_eclasses.push((sort, value));
    }
    Ok(SerializedEGraph::new(&egraph, root_eclasses))
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
    if egraph.enodes[node].0 == "ENil" {
        return Some(vec![]);
    }
    let eclass = &egraph.enodes[node].1[0];
    let expr = extract_expr(egraph, &egraph.eclasses[eclass].1[0], expr_cache)?;
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

pub fn extract_dtype<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> DType {
    match egraph.enodes[node].0.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "Int" => DType::Int,
        "Bool" => DType::Bool,
        "F4E2M1" => DType::F4E2M1,
        "F8E4M3" => DType::F8E4M3,
        "F8UE8M0" => DType::F8UE8M0,
        "I4" => DType::I4,
        "TF32" => DType::TF32,
        other => panic!("unknown dtype {other}"),
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
    fn build_expression(
        egraph: &SerializedEGraph,
        trajectory: &[&NodeId],
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
                .parse::<i64>()
                .map(Expression::from)
                .or_else(|_| op.replace('"', "").parse::<char>().map(Expression::from))
                .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
        }
    }
    let e = build_expression(egraph, &traj, &mut 0);
    expr_cache.insert(node, e);
    Some(e)
}

pub type EGraphChoiceSet<'a> = FxHashMap<&'a ClassId, &'a NodeId>;

pub fn random_initial_choice<'a>(
    egraph: &'a SerializedEGraph,
    rng: &mut impl Rng,
) -> EGraphChoiceSet<'a> {
    let mut choices = FxHashMap::default();
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") && !label.contains("IList") {
            continue;
        }
        choices.insert(eclass, &enodes[rng.random_range(0..enodes.len())]);
    }
    choices
}

/// Validate that a choice set is complete and consistent.
/// Returns Ok(()) if valid, Err with description if invalid.
pub fn validate_choice_set<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
    ops: &[Arc<Box<dyn EgglogOp>>],
) -> Result<(), String> {
    // Check all IR/IList eclasses have a choice
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") && !label.contains("IList") {
            continue;
        }
        let Some(chosen) = choices.get(eclass) else {
            return Err(format!("Missing choice for eclass {}", eclass.as_ref()));
        };
        // Check chosen enode exists in the eclass
        if !enodes.contains(chosen) {
            return Err(format!(
                "Chosen enode {} not in eclass {}",
                chosen.as_ref(),
                eclass.as_ref()
            ));
        }
    }

    // Verify reachability from root
    let mut reachable = FxHashSet::default();
    let root_choice = choices
        .get(&egraph.roots[0])
        .ok_or_else(|| format!("No choice for root eclass {}", egraph.roots[0].as_ref()))?;
    reachable.insert(*root_choice);
    let mut stack = vec![*root_choice];
    while let Some(r) = stack.pop() {
        let (_, children) = egraph
            .enodes
            .get(r)
            .ok_or_else(|| format!("Enode {} not found in egraph", r.as_ref()))?;
        for ch in children {
            let (label, _) = egraph
                .eclasses
                .get(ch)
                .ok_or_else(|| format!("Eclass {} not found", ch.as_ref()))?;
            if label.contains("IR") || label.contains("IList") {
                let n = choices
                    .get(ch)
                    .ok_or_else(|| format!("No choice for reachable eclass {}", ch.as_ref()))?;
                if !reachable.contains(n) {
                    stack.push(n);
                    reachable.insert(n);
                }
            }
        }
    }

    // Check all reachable IR nodes have corresponding ops
    for node in &reachable {
        let (op_name, children) = &egraph.enodes[*node];
        let eclass = &egraph.node_to_class[*node];
        let (label, _) = &egraph.eclasses[eclass];
        if label != "IR" {
            continue; // Skip IList / OpKind nodes
        }
        if op_name == "OutputJoin" {
            continue;
        }
        if op_name == "Op" {
            // Normalized op — check OpKind child
            if let Some(kind_eclass) = children.first() {
                if let Some((_, kind_enodes)) = egraph.eclasses.get(kind_eclass) {
                    if let Some(kn) = kind_enodes.first() {
                        let kind_name = &egraph.enodes[kn].0;
                        if kind_name != "CustomOpKind"
                            && !ops.iter().any(|op| op.sort().name == *kind_name)
                        {
                            return Err(format!("No extractor for OpKind {kind_name}"));
                        }
                    }
                }
            }
            continue;
        }
        // Direct IR variant (Input, Output)
        if !ops.iter().any(|op| op.sort().name == *op_name) {
            return Err(format!("No extractor for op {op_name}"));
        }
    }

    Ok(())
}

/// Hash a choice set for uniqueness checking
pub fn hash_choice_set(choices: &EGraphChoiceSet) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Sort by ClassId for deterministic hashing
    let mut sorted: Vec<_> = choices.iter().collect();
    sorted.sort_by(|(k1, _), (k2, _)| k1.as_ref().cmp(k2.as_ref()));
    for (class_id, node_id) in sorted {
        class_id.hash(&mut hasher);
        node_id.hash(&mut hasher);
    }
    hasher.finish()
}

/// Extract a generation of mutated offspring from a base genome.
///
/// Takes a base `EGraphChoiceSet` and produces up to `generation_size` mutated offspring,
/// each with `mutations_per_generation` random mutations. Offspring are deduplicated
/// against `prev_selected` (which is updated with new hashes).
///
/// If the search space is exhausted, returns as many unique offspring as possible.
pub fn extract_generation<'a>(
    egraph: &'a SerializedEGraph,
    base: &EGraphChoiceSet<'a>,
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'a>> {
    // Get list of mutable eclasses (those with more than one enode option)
    let mutable_classes: Vec<&ClassId> = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .map(|(class_id, _)| class_id)
        .collect();

    // If there are no mutable classes, we can only return the base if it's unseen
    if mutable_classes.is_empty() {
        let h = hash_choice_set(base);
        if !prev_selected.contains(&h) {
            prev_selected.insert(h);
            return vec![base.clone()];
        }
        return vec![];
    }

    let mut offspring = Vec::with_capacity(generation_size);
    // Limit attempts to avoid infinite loops when search space is exhausted
    let max_attempts = generation_size * 100;
    let mut attempts = 0;

    while offspring.len() < generation_size && attempts < max_attempts {
        attempts += 1;

        // Create a mutated offspring from base
        let mut child = base.clone();

        for _ in 0..rng.random_range(1..=mutations_per_generation) {
            // Pick a random mutable eclass
            let class_id = mutable_classes[rng.random_range(0..mutable_classes.len())];
            let (_, enodes) = &egraph.eclasses[class_id];
            // Pick a random enode for this class
            child.insert(class_id, &enodes[rng.random_range(0..enodes.len())]);
        }

        // Hash and check if seen before
        let h = hash_choice_set(&child);
        if !prev_selected.contains(&h) {
            prev_selected.insert(h);
            offspring.push(child);
        }
    }
    offspring
}

/// Walk an IList in the egraph, returning the chosen IR enodes in order.
fn walk_ilist<'a>(
    egraph: &'a SerializedEGraph,
    ilist_eclass: &'a ClassId,
    choices: &EGraphChoiceSet<'a>,
) -> Vec<&'a NodeId> {
    let mut inputs = Vec::new();
    let mut current = choices[ilist_eclass];
    loop {
        if egraph.enodes[current].0 == "INil" {
            break;
        }
        // ICons: child[0] = IR eclass, child[1] = IList tail eclass
        let input_eclass = &egraph.enodes[current].1[0];
        let input_node = choices[input_eclass];
        inputs.push(input_node);
        let tail_eclass = &egraph.enodes[current].1[1];
        current = choices[tail_eclass];
    }
    inputs
}

#[tracing::instrument(skip_all)]
pub fn egglog_to_llir<'a>(
    egraph: &'a SerializedEGraph,
    choices: EGraphChoiceSet<'a>,
    ops: &'a Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    custom_op_id_remap: Option<&FxHashMap<usize, usize>>,
) -> LLIRGraph {
    egglog_to_llir_from_root(
        egraph,
        choices,
        ops,
        custom_ops,
        list_cache,
        expr_cache,
        custom_op_id_remap,
        &egraph.roots[0],
    )
}

pub fn egglog_to_llir_from_root<'a>(
    egraph: &'a SerializedEGraph,
    choices: EGraphChoiceSet<'a>,
    ops: &'a Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    custom_op_id_remap: Option<&FxHashMap<usize, usize>>,
    root_class: &ClassId,
) -> LLIRGraph {
    // Make reachability set from root
    let mut reachable = FxHashSet::default();
    reachable.insert(choices[root_class]);
    let mut reachability_stack = vec![choices[root_class]];
    while let Some(r) = reachability_stack.pop() {
        for ch in &egraph.enodes[r].1 {
            if egraph.eclasses[ch].0.contains("IR") || egraph.eclasses[ch].0.contains("IList") {
                let n = choices[ch];
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
    for &node in choices.values() {
        if !reachable.contains(node) {
            continue;
        }
        if egraph.eclasses[&egraph.node_to_class[node]].0 != "IR" {
            // Skip IList / OpKind
            continue;
        }
        let enode_label = egraph.enodes[node].0.as_str();
        if enode_label == "Op" {
            // Normalized op: (Op OpKind IList)
            // child[0] = OpKind eclass, child[1] = IList eclass
            let kind_eclass = &egraph.enodes[node].1[0];
            let ilist_eclass = &egraph.enodes[node].1[1];

            // Resolve OpKind enode
            let kind_enode = &egraph.eclasses[kind_eclass].1[0];
            let kind_label = &egraph.enodes[kind_enode].0;

            // Resolve kind's metadata children (shapes, strides, etc.)
            let kind_children: Vec<&NodeId> = egraph.enodes[kind_enode]
                .1
                .iter()
                .map(|c| {
                    if egraph.eclasses[c].0.contains("IR") || egraph.eclasses[c].0.contains("IList")
                    {
                        choices[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();

            // Walk IList to get IR inputs
            let input_enodes = walk_ilist(egraph, ilist_eclass, &choices);

            // Check for CustomOpKind first
            if kind_label == "CustomOpKind" {
                // kind_children: [id, dtype]
                let id: usize = egraph.enodes[kind_children[0]].0.parse().unwrap();
                let remapped_id = custom_op_id_remap
                    .and_then(|m| m.get(&id).copied())
                    .unwrap_or(id);
                let r = graph.add_node(custom_ops[remapped_id].to_llir_op());
                enode_to_node.insert(node, r);
                for source in input_enodes {
                    edges_to_place.push((source, node));
                }
            } else {
                // Find matching op by OpKind name
                let Some(op) = ops.iter().find(|op| kind_label.as_str() == op.sort().name) else {
                    todo!("{kind_label} extraction not implemented!");
                };
                let (op_instance, sources) =
                    op.extract(egraph, &kind_children, input_enodes, list_cache, expr_cache);
                let r = graph.add_node(op_instance);
                enode_to_node.insert(node, r);
                for source in sources {
                    edges_to_place.push((source, node));
                }
            }
        } else if enode_label != "OutputJoin" {
            // Direct IR variant (Input, Output) — skip unknown labels (backend IR wrappers)
            let Some(op) = ops.iter().find(|op| enode_label == op.sort().name) else {
                continue;
            };
            let ch = egraph.enodes[node]
                .1
                .iter()
                .map(|c| {
                    if egraph.eclasses[c].0.contains("IR") || egraph.eclasses[c].0.contains("IList")
                    {
                        choices[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();
            // Direct IR ops pass children as kind_children, empty input_enodes
            let (op_instance, sources) = op.extract(egraph, &ch, vec![], list_cache, expr_cache);
            let r = graph.add_node(op_instance);
            enode_to_node.insert(node, r);
            for source in sources {
                edges_to_place.push((source, node));
            }
        }
    }
    for (src, dest) in edges_to_place {
        let src_node_id = *enode_to_node.get(&src).unwrap_or_else(|| {
            panic!("Source enode {src:?} not found in enode_to_node map during edge placement")
        });
        let dest_node_id = *enode_to_node.get(&dest).unwrap_or_else(|| {
            panic!(
                "Destination enode {dest:?} not found in enode_to_node map during edge placement",
            )
        });

        graph.add_edge(src_node_id, dest_node_id, ());
    }
    // if enabled!(Level::TRACE) {
    //     fs::write(
    //         format!("llir_graphs/llir_{}.dot", i),
    //         graph.clone().to_dot().unwrap(),
    //     )
    //     .unwrap();
    // }
    crate::graph::unroll_loops_in_llir(&mut graph);
    graph
}

/// Merge multiple per-chunk LLIR graphs into a single LLIR graph,
/// resolving boundary Input/Output nodes at graph break boundaries.
pub fn stitch_llir_graphs(
    chunk_llirs: &[LLIRGraph],
    descriptors: &[SubgraphDescriptor],
) -> LLIRGraph {
    use petgraph::stable_graph::NodeIndex;

    let mut merged = LLIRGraph::default();

    // Collect the set of boundary break_node indices for matching
    let mut boundary_output_set: FxHashSet<usize> = FxHashSet::default();
    let mut boundary_input_set: FxHashSet<usize> = FxHashSet::default();
    for desc in descriptors {
        for brk in &desc.boundary_outputs {
            boundary_output_set.insert(brk.index());
        }
        for bi in &desc.boundary_inputs {
            boundary_input_set.insert(bi.break_node.index());
        }
    }

    // Per-chunk node mapping: old NodeIndex -> new NodeIndex in merged graph
    let mut node_maps: Vec<FxHashMap<NodeIndex, NodeIndex>> = Vec::with_capacity(chunk_llirs.len());

    // Track boundary producers: break_node_index -> new NodeIndex of the actual producer
    let mut boundary_producers: FxHashMap<usize, NodeIndex> = FxHashMap::default();

    // Track real Input node deduplication: Input.node -> new NodeIndex
    let mut real_inputs: FxHashMap<usize, NodeIndex> = FxHashMap::default();

    for (_chunk_idx, chunk_graph) in chunk_llirs.iter().enumerate() {
        let mut this_map: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
        let output_id_counts = chunk_graph
            .node_indices()
            .filter_map(|n| chunk_graph[n].to_op::<Output>().map(|op| op.node))
            .counts();

        // Pass 1: Add all non-boundary nodes
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];

            // Check if this is a boundary Output
            if let Some(output_op) = op.to_op::<Output>() {
                if boundary_output_set.contains(&output_op.node) {
                    let pred = chunk_graph
                        .neighbors_directed(old_node, petgraph::Direction::Incoming)
                        .next();
                    let is_boundary_wrapper = pred
                        .and_then(|p| chunk_graph[p].to_op::<Output>())
                        .is_some_and(|pred_out| pred_out.node == output_op.node);
                    let duplicated_output_id =
                        output_id_counts.get(&output_op.node).copied().unwrap_or(0) > 1;
                    if !duplicated_output_id || is_boundary_wrapper {
                        // Pure boundary output: skip and stitch through the producer in pass 2.
                        // If the same HLIR output id appears twice, keep the real top-level
                        // Output and only drop the synthetic boundary wrapper.
                        continue;
                    }
                }
            }

            // Check if this is a boundary Input
            if let Some(input_op) = op.to_op::<Input>() {
                if boundary_input_set.contains(&input_op.node) {
                    // Skip — will resolve in pass 2
                    continue;
                }

                // Check if this is a real Input that was already added (dedup)
                if let Some(&existing) = real_inputs.get(&input_op.node) {
                    this_map.insert(old_node, existing);
                    continue;
                }
            }

            let new_node = merged.add_node(op.clone());
            this_map.insert(old_node, new_node);

            // Track real inputs for deduplication
            if let Some(input_op) = op.to_op::<Input>() {
                real_inputs.insert(input_op.node, new_node);
            }
        }

        // Pass 2: Resolve boundary Output nodes (record the producer)
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];
            if let Some(output_op) = op.to_op::<Output>() {
                if boundary_output_set.contains(&output_op.node) {
                    // Find the predecessor (the actual producer)
                    let pred = chunk_graph
                        .neighbors_directed(old_node, petgraph::Direction::Incoming)
                        .next()
                        .expect("Boundary Output must have exactly one input");
                    if let Some(&producer_new) = this_map.get(&pred) {
                        boundary_producers.insert(output_op.node, producer_new);
                    } else {
                        eprintln!(
                            "[stitch] WARNING: chunk {}: boundary Output node={} predecessor {:?} not in this_map!",
                            _chunk_idx,
                            output_op.node,
                            pred.index()
                        );
                    }
                }
            }
        }

        // Pass 2b: Resolve boundary Input nodes (map to producer from prior chunk)
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];
            if let Some(input_op) = op.to_op::<Input>() {
                if boundary_input_set.contains(&input_op.node) {
                    if let Some(&producer) = boundary_producers.get(&input_op.node) {
                        this_map.insert(old_node, producer);
                    } else {
                        eprintln!(
                            "[stitch] WARNING: chunk {}: boundary Input node={} has no producer in boundary_producers!",
                            _chunk_idx, input_op.node
                        );
                        eprintln!(
                            "[stitch]   available producers: {:?}",
                            boundary_producers.keys().collect::<Vec<_>>()
                        );
                    }
                }
            }
        }

        // Pass 3: Add edges (preserving duplicate edges for ops like x*x)
        for edge in chunk_graph.edge_indices() {
            let (src, dst) = chunk_graph.edge_endpoints(edge).unwrap();
            if let (Some(&new_src), Some(&new_dst)) = (this_map.get(&src), this_map.get(&dst)) {
                if new_src != new_dst {
                    merged.add_edge(new_src, new_dst, ());
                }
            }
        }

        node_maps.push(this_map);
    }

    merged
}
