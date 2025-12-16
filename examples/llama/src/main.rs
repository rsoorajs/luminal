mod model;

use half::bf16;
use itertools::Itertools;
use luminal::{
    graph::{Graph, Runtime},
    op::DType,
    utils::{display_graph, IntoEgglogOp},
};
use luminal_cuda::{
    block::{self, record_exec_timings_to_file, CudaRuntime, CustomState, IntoBlockOp},
    kernel, logical,
};
use rustc_hash::*;
use safetensors::SafeTensors;
use std::{
    fs::{self, File},
    io::Write,
    time::Duration,
};
use tokenizers::Tokenizer;
use tracing::{span, Level};
use tracing_appender::non_blocking;
use tracing_perfetto_sdk_layer::NativeLayer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn load_safetensor(st: &SafeTensors, name: &str) -> Vec<f32> {
    let tensor = st.tensor(name).unwrap();
    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(2)
        .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    data
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

fn main() {
    let file = File::create("trace.pftrace").unwrap();
    let (writer, _guard) = non_blocking(file);
    let layer = NativeLayer::from_config(trace_config(), writer)
        .build()
        .unwrap();
    let filter = EnvFilter::builder()
        .parse(format!("{}=trace", env!("CARGO_PKG_NAME")))
        .unwrap();
    let layer_handle = layer.clone();
    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .init();
    let (batch, hidden, intermediate, n_heads, n_kv_heads, vocab_size, max_seq_len) =
        ('s', 4096, 14336, 32, 8, 128256, 4096);
    let n_kv_groups = n_heads / n_kv_heads;
    let layers: usize = 32;
    let n_tokens: usize = 5;

    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer");
    let input_sentence = "Hello, how are you";
    let encoding = tokenizer
        .encode(input_sentence, true)
        .expect("Failed to encode");
    let mut sentence = encoding.get_ids().to_vec();

    // Load embedding weights from safetensors
    let embed_file = fs::read("model_combined.safetensors").unwrap();
    let st = SafeTensors::deserialize(&embed_file).expect("Failed to deserialize safetensors");
    let embed_data = load_safetensor(&st, "model.embed_tokens.weight");

    let mut cx = Graph::default();

    let input = cx.named_tensor("input", batch).as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", batch).as_dtype(DType::Int);
    let model = model::Llama::init(
        &mut cx,
        batch,
        hidden,
        intermediate,
        n_heads,
        n_kv_heads,
        vocab_size,
        layers,
    );
    let _logits = model.forward(input, token_ids);

    // compile
    println!("Building E-graph...");
    let ops = <(
        luminal::op::Ops,
        logical::Ops,
        kernel::ops::Ops,
        block::MKOps,
    ) as IntoEgglogOp>::into_vec();
    cx.build_search_space(&ops);
    let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    let mut custom_state = FxHashMap::default();
    custom_state.insert(
        "kv_cache".to_string(),
        CustomState::KVCache(
            (0..layers)
                .map(|_| {
                    (
                        stream
                            .alloc_zeros::<f32>((hidden / n_kv_groups) * max_seq_len)
                            .unwrap(),
                        stream
                            .alloc_zeros::<f32>((hidden / n_kv_groups) * max_seq_len)
                            .unwrap(),
                    )
                })
                .collect_vec(),
        ),
    );
    let mut runtime = cx.search(
        CudaRuntime::initialize((ctx, stream, custom_state)),
        &ops,
        10_000,
    );

    // load weights
    println!("Compiling...");
    println!("Loading weights...");
    runtime.load_safetensors("model_combined.safetensors");

    print!("{input_sentence}");
    std::io::stdout().flush().unwrap();

    let mut timings = vec![];
    let mut start_times = vec![];
    let mut prev_seq = 0;
    for i in 0..n_tokens {
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
        let mut embeddings = vec![0.0f32; seq_len * hidden];
        for (i, &token_id) in sentence.iter().enumerate() {
            let start = token_id as usize * hidden;
            let end = start + hidden;
            embeddings[i * hidden..(i + 1) * hidden].copy_from_slice(&embed_data[start..end]);
        }

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

        runtime.execute(&cx.dyn_map);
        // timings.push(new_timings);
        // start_times.push(new_start_time);
        let outs = runtime.dtoh_outputs();
        // if let Some(CustomState::DebugBuffers(debug_buffers)) = custom_state.remove("debug") {
        //     print_debug_buffers(debug_buffers);
        // }

        let sample_span = span!(Level::INFO, "sample");
        let _sample_entered = sample_span.enter();
        sentence = vec![*sample(&outs[0], vocab_size).last().unwrap()];
        prev_seq += seq_len;
        print!("{}", tokenizer.decode(&sentence, true).unwrap());
        std::io::stdout().flush().unwrap();
    }
    println!();

    layer_handle
        .flush(Duration::from_secs(5), Duration::from_secs(5))
        .unwrap();
    layer_handle.stop().unwrap();
    drop(_guard);
    record_exec_timings_to_file(
        &timings,
        &start_times,
        &<block::MKOps as IntoBlockOp>::into_vec(),
        "trace.pftrace",
    );
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
