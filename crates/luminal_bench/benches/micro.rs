//! Micro benchmark runner using criterion
//!
//! Run with:
//! ```bash
//! cargo bench -p luminal_bench --features metal --bench micro
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::time::Duration;

#[cfg(feature = "metal")]
use luminal_bench::{
    BenchmarkBackend, BenchmarkPattern, MetalBenchmark, all_micro_patterns,
};

#[cfg(feature = "metal")]
use luminal::prelude::*;

#[cfg(feature = "metal")]
fn run_metal_pattern_benchmark(c: &mut Criterion, pattern: &dyn BenchmarkPattern, hw: &luminal_bench::HardwareInfo) {
    use luminal::hlir::Input;
    use luminal::op::Runtime;
    use luminal_metal::runtime::MetalRuntime;
    use rand::Rng;

    let backend_name = MetalBenchmark::name();
    let pattern_name = pattern.name();
    let group_name = format!("{}/{}", backend_name, pattern_name);

    let mut group = c.benchmark_group(&group_name);

    for size in pattern.sizes() {
        group.bench_with_input(BenchmarkId::from_parameter(size.name), size, |b, size| {
            // Setup: build graph and compile once
            let mut cx = Graph::default();
            pattern.build_graph(&mut cx, *size);

            cx.build_search_space::<MetalRuntime>();
            let mut rt = MetalRuntime::initialize(());

            // Set up dummy input data
            let mut rng = rand::rng();
            for node in cx.graph.node_indices() {
                if let Some(Input { .. }) = (*cx.graph[node]).as_any().downcast_ref::<Input>() {
                    let data: Vec<f32> = (0..size.value).map(|_| rng.random::<f32>()).collect();
                    rt.set_data(node, &data);
                }
            }

            // Search for best implementation
            rt = cx.search(rt, 5);
            rt.allocate_intermediate_buffers(&cx.dyn_map);

            // Use iter_custom for precise timing with stats
            b.iter_custom(|iters| {
                let mut total_time = Duration::ZERO;
                let mut last_stats = None;

                for _ in 0..iters {
                    if let Some(stats) = rt.execute_with_stats(&cx.dyn_map) {
                        total_time += Duration::from_secs_f64(stats.execution_time_us / 1_000_000.0);
                        last_stats = Some(stats);
                    } else {
                        // Fallback to regular execute
                        let start = std::time::Instant::now();
                        rt.execute(&cx.dyn_map);
                        total_time += start.elapsed();
                    }
                }

                // Print MBU/MFU on first iteration (avoid spam)
                if iters == 1 {
                    if let Some(stats) = last_stats {
                        let peak_bw = hw.peak_bandwidth_gbps.unwrap_or(100.0);
                        let peak_tf = hw.peak_tflops.unwrap_or(1.0);
                        let mbu = stats.mbu(peak_bw);
                        let mfu = stats.mfu(peak_tf);
                        println!(
                            "\n  {} [{}]: {:.2} GB/s ({:.1}% MBU), {:.3} TFLOPS ({:.1}% MFU)",
                            pattern_name,
                            size.name,
                            stats.bandwidth_gbps(),
                            mbu,
                            stats.tflops(),
                            mfu
                        );
                    }
                }

                total_time
            });
        });
    }

    group.finish();
}

#[cfg(feature = "metal")]
fn metal_micro_benchmarks(c: &mut Criterion) {
    // Print hardware info
    let hw = MetalBenchmark::hardware_info();
    println!("\n=== Metal Benchmark ===");
    println!("Device: {}", hw.device_name);
    println!("Memory: {:.1} GB", hw.memory_gb);
    if let Some(bw) = hw.peak_bandwidth_gbps {
        println!("Peak Bandwidth: {:.0} GB/s", bw);
    }
    if let Some(tf) = hw.peak_tflops {
        println!("Peak Compute: {:.1} TFLOPS", tf);
    }
    println!();

    // Run all micro patterns
    for pattern in all_micro_patterns() {
        run_metal_pattern_benchmark(c, pattern.as_ref(), &hw);
    }
}

#[cfg(not(feature = "metal"))]
fn metal_micro_benchmarks(_c: &mut Criterion) {
    println!("Metal benchmarks disabled. Run with --features metal");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = metal_micro_benchmarks
}

criterion_main!(benches);
