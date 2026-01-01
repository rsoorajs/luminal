use std::time::{Duration, Instant};

use crate::model::{HEAD_DIM, HIDDEN, INTERMEDIATE, KV_GROUPS, LAYERS, VOCAB_SIZE};

/// Handles all benchmarking metrics and reporting for the LLM inference
pub struct Benchmarker {
    peak_tflops: f64,
    peak_gbps: f64,
    start_generation: Instant,
    ttft: Option<Duration>,
    decode_durations: Vec<Duration>,
    seq_lengths: Vec<(usize, usize)>,
    current_iter_start: Option<Instant>,
}

impl Benchmarker {
    /// Create a new benchmarker with the specified capacity for decode iterations
    pub fn new(peak_tflops: f64, peak_gbps: f64) -> Self {
        Self {
            peak_tflops,
            peak_gbps,
            start_generation: Instant::now(),
            ttft: None,
            decode_durations: vec![],
            seq_lengths: vec![],
            current_iter_start: None,
        }
    }

    /// Mark the start of an iteration (prefill or decode)
    pub fn start_iteration(&mut self, seq_len: usize, prev_seq: usize) {
        self.current_iter_start = Some(Instant::now());
        self.seq_lengths.push((seq_len, prev_seq));
    }

    /// Mark the end of an iteration and record timing
    pub fn end_iteration(&mut self, iteration: usize) {
        if let Some(start) = self.current_iter_start.take() {
            let duration = start.elapsed();
            if iteration == 0 {
                self.ttft = Some(duration);
            } else {
                self.decode_durations.push(duration);
            }
        }
    }

    /// Print the benchmark results to stdout
    pub fn report(&self) {
        let total_elapsed = self.start_generation.elapsed();
        let decode_total = self
            .decode_durations
            .iter()
            .fold(Duration::ZERO, |acc, value| acc + *value);
        let tpot = if self.decode_durations.is_empty() {
            None
        } else {
            Some(decode_total / self.decode_durations.len() as u32)
        };

        let (total_flops, total_bytes) = self
            .seq_lengths
            .iter()
            .map(|(seq_len, prev_seq)| llama_estimate_flops_and_bytes(*seq_len, *prev_seq))
            .fold((0u64, 0u64), |(acc_flops, acc_bytes), (flops, bytes)| {
                (acc_flops + flops, acc_bytes + bytes)
            });

        let achieved_tflops = total_flops as f64 / total_elapsed.as_secs_f64() / 1e12;
        let achieved_gbps = total_bytes as f64 / total_elapsed.as_secs_f64() / 1e9;
        let mfu = if self.peak_tflops > 0.0 {
            Some(achieved_tflops / self.peak_tflops)
        } else {
            None
        };
        let mbu = if self.peak_gbps > 0.0 {
            Some(achieved_gbps / self.peak_gbps)
        } else {
            None
        };
        println!("Benchmark results:");
        if let Some(ttft) = self.ttft {
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
    }
}

fn llama_estimate_flops_and_bytes(seq_len: usize, prev_seq: usize) -> (u64, u64) {
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
