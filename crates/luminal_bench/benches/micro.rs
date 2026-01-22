//! Micro benchmark runner using criterion
//!
//! Run with:
//! ```bash
//! cargo bench -p luminal_bench --features metal --bench micro
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[cfg(feature = "metal")]
use luminal_bench::{
    BenchmarkBackend, BenchmarkPattern, MetalBenchmark, all_micro_patterns, bytes_for_pattern,
};

#[cfg(feature = "metal")]
use luminal::prelude::*;

#[cfg(feature = "metal")]
fn run_metal_pattern_benchmark(c: &mut Criterion, pattern: &dyn BenchmarkPattern) {
    use luminal::hlir::Input;
    use luminal::op::Runtime;
    use luminal_metal::runtime::MetalRuntime;
    use rand::Rng;

    let backend_name = MetalBenchmark::name();
    let pattern_name = pattern.name();
    let group_name = format!("{}/{}", backend_name, pattern_name);

    let mut group = c.benchmark_group(&group_name);

    for size in pattern.sizes() {
        // Set throughput for bandwidth calculation
        let bytes = bytes_for_pattern(pattern_name, size.value);
        group.throughput(Throughput::Bytes(bytes as u64));

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

            // Benchmark only the execute phase
            b.iter(|| {
                rt.execute(&cx.dyn_map);
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
        run_metal_pattern_benchmark(c, pattern.as_ref());
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
