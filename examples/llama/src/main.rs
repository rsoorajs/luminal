mod model;

use itertools::Itertools;
use luminal::{
    graph::{Graph, Runtime},
    op::DType,
};
use luminal_cuda::{
    block::IntoBlockOp,
    runtime::{record_exec_timings_to_file, CudaRuntime, CustomState},
};
use model::*;
use rustc_hash::*;
use std::{fs::File, io::Write, time::Duration, time::Instant};
use tokenizers::Tokenizer;
use tracing::{span, Level};
use tracing_appender::non_blocking;
use tracing_perfetto_sdk_layer::NativeLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // Set up tracing
    let file = File::create("trace.pftrace").unwrap();
    let (writer, _guard) = non_blocking(file);
    let layer = NativeLayer::from_config(trace_config(), writer)
        .build()
        .unwrap();
    let filter = EnvFilter::builder()
        .parse(format!("{}=trace,luminal=trace", env!("CARGO_PKG_NAME")))
        .unwrap();
    let layer_handle = layer.clone();
    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .init();

    let max_seq_len = 4096;
    let gen_tokens = 5;
    let input_sentence = "Hello, how are you";

    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").expect("Failed to load tokenizer");
    let encoding = tokenizer.encode(input_sentence, true).unwrap();
    let mut sentence = encoding.get_ids().to_vec();

    println!("Building Graph...");
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let model = model::Llama::init(&mut cx);
    let logits = model.forward(input, token_ids).output();

    let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    println!("Allocating KV Cache...");
    let mut custom_state = FxHashMap::default();
    custom_state.insert(
        "kv_cache".to_string(),
        CustomState::KVCache(
            (0..LAYERS)
                .map(|_| {
                    (
                        stream
                            .alloc_zeros::<f32>((HIDDEN / KV_GROUPS) * max_seq_len)
                            .unwrap(),
                        stream
                            .alloc_zeros::<f32>((HIDDEN / KV_GROUPS) * max_seq_len)
                            .unwrap(),
                    )
                })
                .collect_vec(),
        ),
    );

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    println!("Compiling...");
    let mut runtime = cx.search(CudaRuntime::initialize((ctx, stream, custom_state)), 10_000);

    println!("Loading weights...");
    runtime.load_safetensors("setup/model_combined.safetensors");

    print!("{input_sentence}");
    std::io::stdout().flush().unwrap();

    let mut timings = vec![];
    let mut prev_seq = 0;
    let mut ttft = None;
    let mut decode_durations = Vec::with_capacity(gen_tokens.saturating_sub(1));
    let start_generation = Instant::now();
    let mut total_flops = 0u64;
    let mut total_bytes = 0u64;
    for i in 0..gen_tokens {
        let span = if i == 0 {
            span!(Level::INFO, "prefill")
        } else {
            span!(Level::INFO, "decode")
        };
        let _entered = span.enter();
        // Embed the tokenized sequence
        let seq_len = sentence.len();
        cx.set_dyn_dim('s', seq_len);
        cx.set_dyn_dim('p', prev_seq);

        runtime.set_data(
            input.id,
            Box::new(sentence.iter().map(|i| *i as i32).collect_vec()),
        );
        runtime.set_data(
            token_ids.id,
            Box::new(
                (prev_seq..seq_len + prev_seq)
                    .map(|i| i as i32)
                    .collect_vec(),
            ),
        );
        if i < 2 {
            // Re-allocate intermediate buffers
            runtime.allocate_intermediate_buffers(&cx.dyn_map);
        }

        let iter_start = Instant::now();
        let (iter_flops, iter_bytes) = estimate_flops_and_bytes(seq_len, prev_seq);
        total_flops += iter_flops;
        total_bytes += iter_bytes;
        timings.extend(runtime.execute(&cx.dyn_map));
        let logits_data = runtime.get_f32(logits.id);

        let sample_span = span!(Level::INFO, "sample");
        let _sample_entered = sample_span.enter();
        sentence = vec![*sample(&logits_data, VOCAB_SIZE).last().unwrap()];
        prev_seq += seq_len;
        print!("{}", tokenizer.decode(&sentence, true).unwrap());
        std::io::stdout().flush().unwrap();
        let iter_duration = iter_start.elapsed();
        if i == 0 {
            ttft = Some(iter_duration);
        } else {
            decode_durations.push(iter_duration);
        }
    }
    println!();

    let total_elapsed = start_generation.elapsed();
    let decode_total = decode_durations
        .iter()
        .fold(Duration::ZERO, |acc, value| acc + *value);
    let tpot = if decode_durations.is_empty() {
        None
    } else {
        Some(decode_total / decode_durations.len() as u32)
    };
    let achieved_tflops = total_flops as f64 / total_elapsed.as_secs_f64() / 1e12;
    let achieved_gbps = total_bytes as f64 / total_elapsed.as_secs_f64() / 1e9;
    let peak_tflops = std::env::var("LUMINAL_PEAK_TFLOPS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let peak_gbps = std::env::var("LUMINAL_PEAK_BW_GBPS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let mfu = if peak_tflops > 0.0 {
        Some(achieved_tflops / peak_tflops)
    } else {
        None
    };
    let mbu = if peak_gbps > 0.0 {
        Some(achieved_gbps / peak_gbps)
    } else {
        None
    };
    println!("Benchmark results:");
    if let Some(ttft) = ttft {
        println!("  TTFT: {:.2} ms", ttft.as_secs_f64() * 1e3);
    }
    if let Some(tpot) = tpot {
        println!("  TPOT: {:.2} ms", tpot.as_secs_f64() * 1e3);
    }
    println!(
        "  Achieved: {:.2} TFLOP/s, {:.2} GB/s",
        achieved_tflops, achieved_gbps
    );
    if let Some(mfu) = mfu {
        println!("  MFU (est): {:.1}%", mfu * 100.0);
    } else {
        println!("  MFU (est): N/A (set LUMINAL_PEAK_TFLOPS)");
    }
    if let Some(mbu) = mbu {
        println!("  MBU (est): {:.1}%", mbu * 100.0);
    } else {
        println!("  MBU (est): N/A (set LUMINAL_PEAK_BW_GBPS)");
    }

    layer_handle
        .flush(Duration::from_secs(5), Duration::from_secs(5))
        .unwrap();
    layer_handle.stop().unwrap();
    drop(_guard);
    record_exec_timings_to_file(
        &timings,
        &<luminal_cuda::block::Ops as IntoBlockOp>::into_vec(),
        "trace.pftrace",
    );
}

fn estimate_flops_and_bytes(seq_len: usize, prev_seq: usize) -> (u64, u64) {
    let total_seq = seq_len + prev_seq;
    let hidden = HIDDEN as u64;
    let intermediate = INTERMEDIATE as u64;
    let seq = seq_len as u64;
    let total_seq = total_seq as u64;
    let head_dim = HEAD_DIM as u64;
    let n_heads = hidden / head_dim;
    let kv_hidden = (HIDDEN / KV_GROUPS) as u64;
    let vocab = VOCAB_SIZE as u64;
    let bytes_per = std::mem::size_of::<f32>() as u64;

    let q_proj_flops = 2 * seq * hidden * hidden;
    let k_proj_flops = 2 * seq * hidden * kv_hidden;
    let v_proj_flops = 2 * seq * hidden * kv_hidden;
    let o_proj_flops = 2 * seq * hidden * hidden;
    let mlp_flops = 6 * seq * hidden * intermediate;
    let attn_flops = 4 * seq * total_seq * head_dim * n_heads;
    let lm_head_flops = 2 * seq * hidden * vocab;

    let per_layer_flops =
        q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops + mlp_flops + attn_flops;
    let total_flops = per_layer_flops * LAYERS as u64 + lm_head_flops;

    let q_bytes = bytes_per * (seq * hidden + hidden * hidden + seq * hidden);
    let k_bytes = bytes_per * (seq * hidden + hidden * kv_hidden + seq * kv_hidden);
    let v_bytes = bytes_per * (seq * hidden + hidden * kv_hidden + seq * kv_hidden);
    let o_bytes = bytes_per * (seq * hidden + hidden * hidden + seq * hidden);
    let mlp_bytes = bytes_per * (seq * hidden + hidden * intermediate + seq * intermediate) * 2
        + bytes_per * (seq * intermediate + intermediate * hidden + seq * hidden);
    let attn_bytes = bytes_per * (seq * hidden + total_seq * kv_hidden * 2 + seq * hidden);
    let lm_head_bytes = bytes_per * (seq * hidden + hidden * vocab + seq * vocab);

    let per_layer_bytes = q_bytes + k_bytes + v_bytes + o_bytes + mlp_bytes + attn_bytes;
    let total_bytes = per_layer_bytes * LAYERS as u64 + lm_head_bytes;

    (total_flops, total_bytes)
}

#[tracing::instrument(skip_all)]
fn sample(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .iter()
        .chunks(vocab_size)
        .into_iter()
        .map(|logits| {
            logits
                .into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0 as u32
        })
        .collect()
}

fn trace_config() -> tracing_perfetto_sdk_schema::TraceConfig {
    tracing_perfetto_sdk_schema::TraceConfig {
        buffers: vec![tracing_perfetto_sdk_schema::trace_config::BufferConfig {
            size_kb: Some(4096),
            ..Default::default()
        }],
        data_sources: vec![tracing_perfetto_sdk_schema::trace_config::DataSource {
            config: Some(tracing_perfetto_sdk_schema::DataSourceConfig {
                name: Some("rust_tracing".into()),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}
