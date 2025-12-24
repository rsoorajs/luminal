mod model;

use itertools::Itertools;
use luminal::{
    prelude::FxHashMap,
    graph::{Graph, Runtime},
    op::DType,
};
use luminal_cuda::{
    block::IntoBlockOp,
    runtime::{record_exec_timings_to_file, CudaRuntime, CustomState},
};
use model::*;
use std::{fs::File, io::Write, time::Duration};
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
    let _logits = model.forward(input, token_ids);

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

        timings.extend(runtime.execute(&cx.dyn_map));
        let outs = runtime.dtoh_outputs();

        let sample_span = span!(Level::INFO, "sample");
        let _sample_entered = sample_span.enter();
        sentence = vec![*sample(&outs[0], VOCAB_SIZE).last().unwrap()];
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
        &<luminal_cuda::block::Ops as IntoBlockOp>::into_vec(),
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
