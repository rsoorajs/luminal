//! Core analysis functions for egglog debugging.
//!
//! All functions in this module are backend-agnostic. Backend-specific
//! operations are passed as parameters or accessed through the `Runtime` trait.

use super::DTypeStatus;
use luminal::egglog_utils;
use luminal::hlir::HLIROps;
use luminal::op::{EgglogOp, IntoEgglogOp, Runtime};
use luminal::prelude::egglog;
use luminal::prelude::egglog::prelude::RustSpan;
use luminal::prelude::egglog_ast::span::Span;
use luminal::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Result of analyzing an Add operation without backend equivalent.
#[derive(Debug, Clone)]
pub struct UnmatchedAdd {
    pub class_id: String,
    pub a_labels: String,
    pub b_labels: String,
}

/// Analysis result for lowering.
#[derive(Debug, Default)]
pub struct LoweringAnalysis {
    pub label: String,
    pub root_labels: Vec<String>,
    pub output_input_labels: Vec<String>,
    pub all_add_have_backend: bool,
    pub unmatched_adds: Vec<UnmatchedAdd>,
    pub add_dtypes: BTreeMap<String, DTypeStatus>,
}

/// Evaluate dtype for a variable in an egraph.
pub fn eval_dtype(egraph: &mut egglog::EGraph, var: &str) -> DTypeStatus {
    let expr = egglog::call!("dtype", vec![egglog::var!(var.to_string())]);
    match egraph.eval_expr(&expr) {
        Ok((sort, value)) => match egraph.extract_value_to_string(&sort, value) {
            Ok((s, _)) => DTypeStatus::Resolved(s),
            Err(_) => DTypeStatus::Missing("extract-error".to_string()),
        },
        Err(err) => DTypeStatus::Missing(format!("{err}")),
    }
}

/// Parse program to find Add variables and their inputs.
pub fn find_add_variables(program: &str) -> (Vec<String>, Option<(String, String, String)>) {
    let mut add_vars: Vec<String> = Vec::new();
    let mut output_var: Option<String> = None;
    let mut add_inputs: Option<(String, String, String)> = None;

    for line in program.lines() {
        let line = line.trim();
        if !line.starts_with("(let ") {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() >= 3 && tokens[0] == "(let" {
            let var = tokens[1].to_string();
            let head = tokens[2].trim_start_matches('(');
            if head == "Add" {
                add_vars.push(var.clone());
            }
            if head == "Output" && tokens.len() >= 3 {
                output_var = Some(var.clone());
            }
        }
    }

    // Find Add inputs related to output
    if let Some(ref out_var) = output_var {
        for line in program.lines() {
            let line = line.trim();
            if !line.starts_with("(let ") || !line.contains("(Add ") || !line.contains(out_var) {
                continue;
            }
            let mut vars: Vec<String> = Vec::new();
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b't' {
                    let mut j = i + 1;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                    if j > i + 1 {
                        vars.push(line[i..j].to_string());
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
            if vars.len() >= 3 && vars[0] == *out_var {
                add_inputs = Some((vars[0].clone(), vars[1].clone(), vars[2].clone()));
                break;
            }
        }
    }

    (add_vars, add_inputs)
}

/// Analyze lowering with a specific set of ops.
///
/// This is the core analysis function that works with any set of ops.
/// Use `analyze_lowering` for convenience when working with a specific backend.
pub fn analyze_with_ops(
    program: &str,
    root: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    label: &str,
    add_vars: &[String],
    backend_add_name: Option<&str>, // e.g., "MetalAdd", "CudaAdd"
) -> LoweringAnalysis {
    let code = egglog_utils::full_egglog(program, &ops, false);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code).unwrap();
    let _outputs = egraph.run_program(commands).unwrap();

    let (sort, value) = egraph.eval_expr(&egglog::var!(root)).unwrap();
    let serialized = egglog_utils::SerializedEGraph::new(&egraph, vec![(sort, value)]);

    // Helper closure to get labels for a class
    let get_labels = |class_id_str: &str| -> Vec<String> {
        for (cid, (_, nodes)) in &serialized.eclasses {
            if format!("{:?}", cid) == class_id_str {
                let mut labels: Vec<String> = nodes
                    .iter()
                    .filter_map(|node_id| {
                        serialized.enodes.get(node_id).map(|(lbl, _)| lbl.clone())
                    })
                    .collect();
                labels.sort();
                labels.dedup();
                return labels;
            }
        }
        vec!["<missing>".to_string()]
    };

    // Get root labels
    let root_class_id = serialized.roots.first().unwrap();
    let root_labels = get_labels(&format!("{:?}", root_class_id));

    let mut analysis = LoweringAnalysis {
        label: label.to_string(),
        root_labels,
        ..Default::default()
    };

    // Check for backend Add presence (if backend_add_name is provided)
    if let Some(backend_add) = backend_add_name {
        let mut eclass_has_backend_add: FxHashSet<String> = FxHashSet::default();
        for (node_id, (lbl, _)) in &serialized.enodes {
            if lbl == backend_add {
                let class_id = &serialized.node_to_class[node_id];
                eclass_has_backend_add.insert(format!("{:?}", class_id));
            }
        }

        // Check Output inputs
        for (_node_id, (lbl, children)) in &serialized.enodes {
            if lbl != "Output" {
                continue;
            }
            let inp_class = &children[0];
            analysis.output_input_labels = get_labels(&format!("{:?}", inp_class));
        }

        // Find Adds without backend equivalent
        let mut missing = 0;
        for (node_id, (lbl, children)) in &serialized.enodes {
            if lbl != "Add" {
                continue;
            }
            let class_id = &serialized.node_to_class[node_id];
            let class_id_str = format!("{:?}", class_id);
            if eclass_has_backend_add.contains(&class_id_str) {
                continue;
            }
            missing += 1;

            let a_labels = get_labels(&format!("{:?}", &children[1])).join("|");
            let b_labels = get_labels(&format!("{:?}", &children[3])).join("|");

            analysis.unmatched_adds.push(UnmatchedAdd {
                class_id: class_id_str,
                a_labels,
                b_labels,
            });
        }

        analysis.all_add_have_backend = missing == 0;
    }

    // Collect dtype for all Add variables
    for add_var in add_vars {
        let dtype = eval_dtype(&mut egraph, add_var);
        analysis.add_dtypes.insert(add_var.clone(), dtype);
    }

    analysis
}

/// Run full lowering analysis comparing HLIR-only vs Backend+HLIR.
///
/// This is a generic function that works with any backend implementing `Runtime`.
///
/// # Type Parameters
/// * `R` - The runtime type (e.g., `MetalRuntime`, `CudaRuntime`)
///
/// # Arguments
/// * `program` - The egglog program string
/// * `root` - The root variable name
/// * `backend_add_name` - The name of the backend's Add operation (e.g., "MetalAdd")
pub fn analyze_lowering<R: Runtime>(
    program: &str,
    root: &str,
    backend_add_name: &str,
) -> (LoweringAnalysis, LoweringAnalysis)
where
    R::Ops: IntoEgglogOp,
{
    let (add_vars, _add_inputs) = find_add_variables(program);

    // HLIR-only analysis
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    let hlir_analysis = analyze_with_ops(program, root, hlir_ops, "HLIR", &add_vars, None);

    // Backend+HLIR analysis
    let mut backend_ops = R::Ops::into_vec();
    backend_ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
    let backend_label = format!("{}+HLIR", std::any::type_name::<R>().split("::").last().unwrap_or("Backend"));
    let backend_analysis = analyze_with_ops(
        program,
        root,
        backend_ops,
        &backend_label,
        &add_vars,
        Some(backend_add_name),
    );

    (hlir_analysis, backend_analysis)
}

/// Convenience function for HLIR-only analysis (no backend).
pub fn analyze_hlir_only(program: &str, root: &str) -> LoweringAnalysis {
    let (add_vars, _) = find_add_variables(program);
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    analyze_with_ops(program, root, hlir_ops, "HLIR", &add_vars, None)
}
