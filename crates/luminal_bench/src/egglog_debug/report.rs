//! Formatted output and reporting for egglog debug analysis.

use super::{DTypeStatus, LoweringAnalysis, TraceEntry, DTypeChainAnalysis};
use std::collections::BTreeMap;

/// Print HLIR node type summary.
pub fn print_hlir_summary(counts: &BTreeMap<String, usize>) {
    println!("-- HLIR node types --");
    for (k, v) in counts {
        println!("  {}: {}", k, v);
    }
}

/// Print egglog op head summary.
pub fn print_egglog_summary(counts: &BTreeMap<String, usize>) {
    println!("-- Egglog op heads --");
    for (k, v) in counts {
        println!("  {}: {}", k, v);
    }
}

/// Print lowering analysis results.
pub fn print_lowering_analysis(analysis: &LoweringAnalysis) {
    println!("-- {} Analysis --", analysis.label);
    println!("  Root eclass labels: {}", analysis.root_labels.join("|"));

    if !analysis.output_input_labels.is_empty() {
        println!("  Output input labels: {}", analysis.output_input_labels.join("|"));
    }

    // Only print backend Add check if we have unmatched adds info
    if !analysis.unmatched_adds.is_empty() || analysis.all_add_have_backend {
        if analysis.all_add_have_backend {
            println!("  ✅ All Add eclasses have backend equivalent");
        } else {
            println!("  ❌ {} Add eclasses missing backend equivalent:", analysis.unmatched_adds.len());
            for add in &analysis.unmatched_adds {
                println!("    - class={} a={} b={}", add.class_id, add.a_labels, add.b_labels);
            }
        }
    }

    // Print dtype status
    if !analysis.add_dtypes.is_empty() {
        println!("  Add dtypes:");
        for (var, dtype) in &analysis.add_dtypes {
            let status = match dtype {
                DTypeStatus::Resolved(s) => format!("✅ {}", s),
                DTypeStatus::Missing(e) => format!("❌ missing ({})", e),
            };
            println!("    {}: {}", var, status);
        }
    }
}

/// Print dependency trace as a tree.
pub fn print_trace_tree(trace: &[TraceEntry]) {
    println!("-- Dependency trace --");
    for entry in trace {
        println!("{}", entry.format_tree());
    }
}

/// Print dtype chain analysis.
pub fn print_dtype_chain(analysis: &DTypeChainAnalysis) {
    println!("-- DType chain analysis for {} --", analysis.target);

    if analysis.all_resolved {
        println!("  ✅ All nodes in chain have resolved dtype");
    } else if let Some(ref first) = analysis.first_missing {
        println!("  ❌ First missing dtype at: {}", first);
    }

    println!("  Chain:");
    for entry in &analysis.chain {
        let dtype_str = match &entry.dtype {
            Some(DTypeStatus::Resolved(s)) => format!("✅ {}", s),
            Some(DTypeStatus::Missing(_)) => "❌ missing".to_string(),
            None => "? unknown".to_string(),
        };
        let indent = "    ".repeat(entry.depth + 1);
        println!("{}{} ({}) {}", indent, entry.var, entry.op_type, dtype_str);
    }
}

/// Summary report for a debug session.
#[derive(Debug)]
pub struct DebugReport {
    pub case_name: String,
    pub size: usize,
    pub hlir_counts: BTreeMap<String, usize>,
    pub egglog_counts: BTreeMap<String, usize>,
    pub hlir_analysis: Option<LoweringAnalysis>,
    pub backend_analysis: Option<LoweringAnalysis>,
    pub build_succeeded: bool,
}

impl DebugReport {
    /// Print full report to stdout.
    pub fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Case: {} (size={})", self.case_name, self.size);
        println!("{}", "=".repeat(60));

        print_hlir_summary(&self.hlir_counts);
        println!();
        print_egglog_summary(&self.egglog_counts);

        if let Some(ref analysis) = self.hlir_analysis {
            println!();
            print_lowering_analysis(analysis);
        }

        if let Some(ref analysis) = self.backend_analysis {
            println!();
            print_lowering_analysis(analysis);
        }

        println!();
        if self.build_succeeded {
            println!("✅ build_search_space succeeded");
        } else {
            println!("❌ build_search_space failed");
        }
    }
}
