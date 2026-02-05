use colored::Colorize;
use egglog::{ast::Span, prelude::RustSpan, var};
use itertools::Itertools;
use petgraph::{Direction, graph::NodeIndex, visit::EdgeRef};
use rand::{Rng, rngs::ThreadRng};
use rustc_hash::FxHashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{str, sync::Arc};
use tracing::info;

pub const BASE: &str = include_str!("base.egg");
pub const BASE_CLEANUP: &str = include_str!("base_cleanup.egg");
pub const RUN_SCHEDULE: &str = include_str!("run_schedule.egg");

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    let ops_str = ops
        .iter()
        .map(|o| {
            let (name, body) = o.term();
            format!(
                "({name} {})",
                body.into_iter().map(|j| format!("{j:?}")).join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "
    (datatype*
        (IR
            (OutputJoin IR IR)
            {ops_str}
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
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.early_rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        format!(
            "(run-schedule
                (saturate expr)
                (run)
                (saturate base_cleanup)
            )
            (extract {root})"
        ),
    ]
    .join("\n")
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    [
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        RUN_SCHEDULE.to_string(),
    ]
    .join("\n")
}

use crate::{
    graph::{Graph, LLIRGraph},
    op::{CustomOp, DType, EgglogOp},
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
        let sources = graph
            .get_sources(n)
            .into_iter()
            .zip(
                graph
                    .edges_directed(n, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| names[&e.source()].clone()),
            )
            .map(|((n, sh), name)| (n, name, sh))
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

#[tracing::instrument(skip_all)]
pub fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    let start = std::time::Instant::now();
    let code = early_egglog(program, root, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    let outputs = egraph.run_program(commands)?;
    let CommandOutput::ExtractBest(termdag, _cost, term) = outputs.last().unwrap() else {
        panic!();
    };
    let (program, root) = termdag_to_egglog(termdag, termdag.lookup(term));
    let code = full_egglog(&program, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    info!("{}", "Egglog running...".green());
    let _outputs = egraph.run_program(commands)?;
    info!("{}", "---- Egglog Rule Matches ----".green());
    let run_report = egraph.get_overall_run_report();
    info!(
        "{}",
        run_report
            .num_matches_per_rule
            .iter()
            .filter(|(k, _)| !k.contains("("))
            .map(|(k, v)| format!(
                "{k}: {v} ({})",
                pretty_duration::pretty_duration(
                    &run_report.search_and_apply_time_per_rule[k],
                    None
                )
            ))
            .join("\n")
            .green()
    );
    info!(
        "{}",
        format!(
            "---- Egglog Took {} ----",
            pretty_duration::pretty_duration(&start.elapsed(), None).bold()
        )
        .green()
    );
    // if enabled!(Level::DEBUG) {
    //     let log_dir = Path::new("egraph");
    //     if log_dir.exists() {
    //         fs::remove_dir_all(log_dir).unwrap();
    //     }
    //     fs::create_dir(log_dir).unwrap();
    //     fs::write(log_dir.join("egraph.dot"), egraph.to_dot().unwrap()).unwrap();
    //     fs::write(log_dir.join("egraph.html"), egraph.to_html().unwrap()).unwrap();
    // }
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
                .parse::<i32>()
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
    rng: &mut ThreadRng,
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
        let (op_name, _) = &egraph.enodes[*node];
        let eclass = &egraph.node_to_class[*node];
        let (label, _) = &egraph.eclasses[eclass];
        if label != "IR" {
            continue; // Skip IList nodes
        }
        if op_name == "OutputJoin" || op_name == "CustomOpHLIR" {
            continue;
        }
        if !ops.iter().any(|op| op.term().0 == *op_name) {
            return Err(format!("No extractor for op {}", op_name));
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
    rng: &mut ThreadRng,
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

#[tracing::instrument(skip_all)]
pub fn egglog_to_llir<'a>(
    egraph: &'a SerializedEGraph,
    choices: EGraphChoiceSet<'a>,
    ops: &'a Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> LLIRGraph {
    // Get maps for all e-classes to e-node options
    // if enabled!(Level::DEBUG) {
    //     let log_dir = Path::new("llir_graphs");

    //     if log_dir.exists() {
    //         fs::remove_dir_all(log_dir).unwrap();
    //     }
    //     fs::create_dir(log_dir).unwrap();
    // }

    // Make reachability set from root
    let mut reachable = FxHashSet::default();
    reachable.insert(choices[&egraph.roots[0]]);
    let mut reachability_stack = vec![choices[&egraph.roots[0]]];
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
            // Skip IList
            continue;
        }
        let ch = egraph.enodes[node]
            .1
            .iter()
            .map(|c| {
                if egraph.eclasses[c].0.contains("IR") || egraph.eclasses[c].0.contains("IList") {
                    choices[c]
                } else {
                    &egraph.eclasses[c].1[0]
                }
            })
            .collect_vec();
        if egraph.enodes[node].0.as_str() == "CustomOpHLIR" {
            // Extract custom op inputs and id
            let mut inputs = vec![];
            // Walk through the IList to get inputs - use choices[] for IR/IList eclasses
            let ilist_eclass = &egraph.enodes[node].1[0];
            let mut ch = choices[ilist_eclass];
            loop {
                if egraph.enodes[ch].0 == "INil" {
                    break;
                } else {
                    // The first child of ICons is an IR node - use choices[] to get the chosen enode
                    let input_eclass = &egraph.enodes[ch].1[0];
                    inputs.push(choices[input_eclass]);
                    // The second child of ICons is the rest of the IList - use choices[] for the tail
                    ch = choices[&egraph.enodes[ch].1[1]];
                }
            }
            let id: usize = egraph.enodes[&egraph.eclasses[&egraph.enodes[node].1[1]].1[0]]
                .0
                .parse()
                .unwrap();
            let r = graph.add_node(custom_ops[id].to_llir_op());
            enode_to_node.insert(node, r);
            for source in inputs {
                edges_to_place.push((source, node));
            }
        } else if egraph.enodes[node].0.as_str() != "OutputJoin" {
            let Some(op) = ops
                .iter()
                .find(|op| egraph.enodes[node].0.as_str() == op.term().0)
            else {
                todo!("{} extraction not implemented!", egraph.enodes[node].0);
            };
            // Extract this op
            let (op_instance, sources) = op.extract(egraph, &ch, list_cache, expr_cache);
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
    graph
}
