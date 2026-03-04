mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

const REPO_ID: &str = "Qwen/Qwen3-30B-A3B";

// This example compiles and runs Qwen3-30B-A3B (MoE) on CUDA.

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 30;
    let search_graphs = 50;
    let prompt = "The capital of France is";

    // Set up cuda context and stream
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Download model if needed and prepare weights (converts to FP32, stacks experts)
    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    // Tokenize prompt
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let sentence = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    // Allocate kv cache and norm weight buffers
    let mut kv_cache = KVCache::new(&stream, max_seq_len);
    let mut norm_bufs = NormWeightBuffers::new(&stream);

    // Create compute graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let model = model::Qwen3MoE::init(&mut cx);
    let logits = model.forward(input, &kv_cache, &norm_bufs).output();

    // Build search space
    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    // Load model weights from safetensors file
    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // Load QK-norm weights directly from safetensors into GPU buffers
    println!("Loading norm weights...");
    norm_bufs.load_from_safetensors(&weights_path);

    // Run search process
    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1]);
    runtime = cx.search(runtime, search_graphs);
    kv_cache.reset();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    // Prefill: process prompt tokens one at a time
    // (Batch prefill is unreliable — some e-graph search results produce graphs that
    //  don't handle seq>1 correctly. This is a pre-existing graph compilation issue.)
    let mut prev_seq = 0;
    let prefill_start = std::time::Instant::now();
    for &token in &sentence {
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![token as i32]);
        runtime.execute(&cx.dyn_map);
        prev_seq += 1;
    }
    let prefill_duration = prefill_start.elapsed();

    // Get logits from last prefill step and sample first new token
    let logits_data = runtime.get_f32(logits);
    let mut next_token = *sample(&logits_data, VOCAB_SIZE).last().unwrap();
    print!("{}", tokenizer.decode(&[next_token], true).unwrap());
    std::io::stdout().flush().unwrap();

    // Decode loop: generate remaining tokens
    let mut fwd_durations = vec![];
    for _ in 1..gen_tokens {
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);
        next_token = *sample(&logits_data, VOCAB_SIZE).last().unwrap();
        prev_seq += 1;
        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Report benchmarks
    println!(
        "  TTFT: {:.2} ms ({} prompt tokens)",
        prefill_duration.as_secs_f64() * 1e3,
        sentence.len()
    );
    if fwd_durations.len() > 1 {
        println!(
            "  TPOT: {:.2} ms",
            (fwd_durations.iter().skip(1).sum::<Duration>() / (fwd_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1_000.
        );
    }
}

fn sample(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as u32
        })
        .collect()
}
