//! Debug script to locate which HLIR op(s) fail to lower to a backend dialect,
//! leading to `No valid graphs present in the e-graph!`.
//!
//! ## Design
//!
//! This tool is backend-agnostic. The specific backend is selected via feature flags.
//! All core analysis logic lives in `luminal_bench::egglog_debug` module.
//!
//! ## Usage
//!
//! ```bash
//! # Metal backend
//! cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu --analyze
//!
//! # Run all cases
//! cargo run -p luminal_bench --features metal --example debug_ops -- --all --analyze
//!
//! # Dump egglog program
//! cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu --dump-egglog target/gelu.egg
//! ```

use luminal::prelude::*;
use luminal_bench::egglog_debug::{
    self, analyze_lowering, print_lowering_analysis, summarize_egglog_ops, summarize_hlir_ops,
};

// ============================================================================
// Backend Configuration
// ============================================================================

/// Backend-specific configuration trait.
trait BackendConfig {
    type Runtime: luminal::op::Runtime;
    const NAME: &'static str;
    const ADD_OP_NAME: &'static str; // e.g., "MetalAdd", "CudaAdd"

    fn build_search_space(cx: &mut Graph);
}

#[cfg(feature = "metal")]
mod metal_backend {
    use super::*;
    use luminal_metal::runtime::MetalRuntime;

    pub struct MetalConfig;

    impl BackendConfig for MetalConfig {
        type Runtime = MetalRuntime;
        const NAME: &'static str = "Metal";
        const ADD_OP_NAME: &'static str = "MetalAdd";

        fn build_search_space(cx: &mut Graph) {
            cx.build_search_space::<MetalRuntime>();
        }
    }
}

#[cfg(feature = "metal")]
use metal_backend::MetalConfig as ActiveBackend;

// Future: Add CUDA backend
// #[cfg(feature = "cuda")]
// mod cuda_backend { ... }

// ============================================================================
// Test Cases
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum Case {
    Mul,
    Sigmoid,
    Tanh,
    GeluInner,
    Gelu,
}

impl Case {
    fn all() -> &'static [Case] {
        &[
            Case::Mul,
            Case::Sigmoid,
            Case::Tanh,
            Case::GeluInner,
            Case::Gelu,
        ]
    }

    fn from_str(s: &str) -> Option<Case> {
        match s {
            "mul" => Some(Case::Mul),
            "sigmoid" => Some(Case::Sigmoid),
            "tanh" => Some(Case::Tanh),
            "gelu-inner" => Some(Case::GeluInner),
            "gelu" => Some(Case::Gelu),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Case::Mul => "Mul",
            Case::Sigmoid => "Sigmoid",
            Case::Tanh => "Tanh",
            Case::GeluInner => "GeluInner",
            Case::Gelu => "Gelu",
        }
    }

    fn build(&self, cx: &mut Graph, size: usize) {
        let x = cx.tensor(size);
        let out = match self {
            Case::Mul => x.clone() * x,
            Case::Sigmoid => x.sigmoid(),
            Case::Tanh => x.tanh(),
            Case::GeluInner => {
                (0.797_884_560_8_f32 * x.clone() * (1. + 0.044_715_f32 * x.clone() * x)).tanh()
            }
            Case::Gelu => x.gelu(),
        };
        let _ = out.output();
    }
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

struct Args {
    case: Case,
    size: usize,
    dump_egglog: Option<std::path::PathBuf>,
    print_egglog: bool,
    analyze: bool,
    all: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        case: Case::Gelu,
        size: 262_144,
        dump_egglog: None,
        print_egglog: false,
        analyze: false,
        all: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--case" => {
                let val = iter.next().expect("Missing value for --case");
                args.case = Case::from_str(&val)
                    .unwrap_or_else(|| panic!("Unknown case: {}. Use: mul|sigmoid|tanh|gelu-inner|gelu", val));
            }
            "--size" => {
                let val = iter.next().expect("Missing value for --size");
                args.size = val.parse().expect("Invalid --size value");
            }
            "--dump-egglog" => {
                let val = iter.next().expect("Missing value for --dump-egglog");
                args.dump_egglog = Some(val.into());
            }
            "--print-egglog" => args.print_egglog = true,
            "--analyze" => args.analyze = true,
            "--all" => args.all = true,
            "--help" | "-h" => {
                println!(
                    "Usage: debug_ops [OPTIONS]\n\n\
                    Options:\n  \
                      --case <CASE>       Test case: mul|sigmoid|tanh|gelu-inner|gelu (default: gelu)\n  \
                      --size <N>          Tensor size (default: 262144)\n  \
                      --all               Run all test cases\n  \
                      --analyze           Run lowering analysis\n  \
                      --dump-egglog PATH  Write egglog program to file\n  \
                      --print-egglog      Print egglog program to stdout\n  \
                      --help              Show this help"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}. Use --help for usage.", other);
                std::process::exit(2);
            }
        }
    }

    args
}

// ============================================================================
// Main Logic
// ============================================================================

fn run_case<B: BackendConfig>(case: Case, size: usize, args: &Args)
where
    B::Runtime: luminal::op::Runtime,
    <B::Runtime as luminal::op::Runtime>::Ops: luminal::op::IntoEgglogOp,
{
    println!("\n=== Case: {} (size={}) [{}] ===", case.name(), size, B::NAME);

    // Build graph
    let mut cx = Graph::default();
    case.build(&mut cx, size);

    // Summarize HLIR
    let hlir_counts = summarize_hlir_ops(&cx);
    println!("-- HLIR node types --");
    for (k, v) in &hlir_counts {
        println!("  {}: {}", k, v);
    }

    // Get egglog program
    let (program, root) = hlir_to_egglog(&cx);

    // Summarize egglog ops
    let egglog_counts = summarize_egglog_ops(&program);
    println!("-- Egglog op heads --");
    for (k, v) in &egglog_counts {
        println!("  {}: {}", k, v);
    }
    println!("-- Egglog root: {} --", root);

    // Dump egglog if requested
    if let Some(ref base_path) = args.dump_egglog {
        let path = if args.all {
            let parent = base_path.parent().unwrap_or(std::path::Path::new("."));
            let stem = base_path.file_stem().and_then(|s| s.to_str()).unwrap_or("debug");
            parent.join(format!("{}-{}.egg", stem, case.name()))
        } else {
            base_path.clone()
        };

        let content = format!("; hlir_to_egglog dump\n; root: {}\n{}", root, program);
        std::fs::write(&path, content).expect("Failed to write egglog file");
        println!("Wrote egglog program to {}", path.display());
    }

    if args.print_egglog {
        println!("-- Egglog program --\n{}", program);
    }

    // Run analysis if requested
    if args.analyze {
        println!("-- Lowering analysis --");
        let (hlir_analysis, backend_analysis) =
            analyze_lowering::<B::Runtime>(&program, &root, B::ADD_OP_NAME);

        print_lowering_analysis(&hlir_analysis);
        print_lowering_analysis(&backend_analysis);
    }

    // Try to build search space
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        B::build_search_space(&mut cx);
    }));
    std::panic::set_hook(prev_hook);

    match result {
        Ok(()) => println!("✅ build_search_space succeeded"),
        Err(_) => println!("❌ build_search_space failed"),
    }
}

fn main() {
    let args = parse_args();

    println!("=== debug_ops ({}) ===", ActiveBackend::NAME);
    println!("Backend: {}", ActiveBackend::NAME);
    println!("Tip: Use --analyze for detailed lowering analysis.\n");

    if args.all {
        for case in Case::all() {
            run_case::<ActiveBackend>(*case, args.size, &args);
        }
    } else {
        run_case::<ActiveBackend>(args.case, args.size, &args);
    }
}
