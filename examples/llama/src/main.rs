mod benchmark;
mod model;

use benchmark::Benchmarker;
use itertools::Itertools;
use luminal::prelude::*;
use luminal_cuda::{cuda_bandwidth_gbps, cuda_compute_f32_tflops, runtime::CudaRuntime};
use model::*;
use std::io::Write;
use tokenizers::Tokenizer;
use tracing::{span, Level};

fn main() {
    let trace_session = luminal_tracing::subscriber()
        .perfetto("trace.pftrace")
        .env_filter(format!("{}=trace,luminal=trace", env!("CARGO_PKG_NAME")))
        .init();

    let ctx = luminal_cuda::cudarc::driver::CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    ctx.set_flags(luminal_cuda::cudarc::driver::sys::CUctx_flags::CU_CTX_SCHED_BLOCKING_SYNC)
        .unwrap();
    let stream = ctx.default_stream();

    let max_seq_len = 4096;
    let gen_tokens: usize = 5;
    let input_sentence = "Hello, how are you";

    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").expect("Failed to load tokenizer");
    let encoding = tokenizer.encode(input_sentence, true).unwrap();
    let mut sentence = encoding.get_ids().to_vec();
    let mut kv_cache = KVCache::new(&stream, max_seq_len);

    println!("Building Graph...");
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let model = model::Llama::init(&mut cx);
    let logits = model.forward(input, token_ids, &kv_cache).output();

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    let mut runtime = CudaRuntime::initialize(stream.clone());
    println!("Loading weights...");
    runtime.load_safetensors(&cx, "setup/model_combined.safetensors");
    stream.synchronize().unwrap();
    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    // inputs for search
    runtime.set_data(input, vec![1_i32]);
    runtime.set_data(token_ids, vec![0_i32]);
    runtime = cx.search(runtime, 5);
    kv_cache.reset();

    print!("{input_sentence}");
    std::io::stdout().flush().unwrap();

    let mut prev_seq = 0;
    let mut benchmarker = Benchmarker::new(); // H100 specs
    for i in 0..gen_tokens {
        let span = if i == 0 {
            span!(Level::INFO, "prefill")
        } else {
            span!(Level::INFO, "decode")
        };
        let _entered = span.enter();
        // Embed the tokenized sequence
        let seq_len = sentence.len();
        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        runtime.set_data(input, sentence.iter().map(|i| *i as i32).collect_vec());
        runtime.set_data(
            token_ids,
            (prev_seq..seq_len + prev_seq)
                .map(|i| i as i32)
                .collect_vec(),
        );
        if i < 2 {
            // Re-allocate intermediate buffers
            runtime.allocate_intermediate_buffers(&cx.dyn_map);
        }

        benchmarker.start_iteration(seq_len, prev_seq);
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        let sample_span = span!(Level::INFO, "sample");
        let _sample_entered = sample_span.enter();
        sentence = vec![*sample(&logits_data, VOCAB_SIZE).last().unwrap()];
        prev_seq += seq_len;
        print!("{}", tokenizer.decode(&sentence, true).unwrap());
        std::io::stdout().flush().unwrap();
        benchmarker.end_iteration(i);
    }
    println!();

    trace_session.stop();
    if let (Some(flops), Some(bandwidth)) =
        (cuda_compute_f32_tflops(&ctx), cuda_bandwidth_gbps(&ctx))
    {
        benchmarker.report(flops as f64, bandwidth as f64);
    }
    // Dump cuda trace to timeline
    if let Some(path) = trace_session.perfetto_path {
        runtime.record_cuda_perfetto_trace(path);
    }
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
