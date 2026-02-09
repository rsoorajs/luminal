use half::bf16;
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor: either FP32 data or raw U8 bytes (for NvFp4 packed weights)
enum StoredTensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    U8 { shape: Vec<usize>, data: Vec<u8> },
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Try to download single shard model first
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    // Otherwise download sharded model
    let index_path = repo.get("model.safetensors.index.json")?;
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    // Download each shard
    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }
    Ok(model_dir)
}

/// Convert FP8 E4M3 byte to f32
fn fp8_e4m3_to_float(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF;
    let mant = bits & 0x7;
    let result = if exp == 0 {
        (mant as f32 / 8.0) * 2.0f32.powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    };
    if sign == 1 { -result } else { result }
}

/// Convert f32 to nearest FP8 E4M3 byte
fn float_to_fp8_e4m3(val: f32) -> u8 {
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    if abs_val == 0.0 {
        return 0;
    }
    let mut best_bits = 0u8;
    let mut best_err = f32::MAX;
    for bits in 0..=0x7Fu8 {
        let decoded = fp8_e4m3_to_float(bits);
        if decoded.is_nan() {
            continue;
        }
        let err = (decoded - abs_val).abs();
        if err < best_err {
            best_err = err;
            best_bits = bits;
        }
    }
    best_bits | (sign << 7)
}

/// Check if a tensor name corresponds to a quantized weight
/// Quantized weights have companion `weight_scale` and `weight_scale_2` tensors
fn is_quantized_weight(name: &str) -> bool {
    name.ends_with(".weight")
        && !name.contains("layernorm")
        && !name.contains("norm")
        && !name.contains("embed_tokens")
        && !name.contains("lm_head")
}

/// Pack HF NVFP4 weight tensors into our kernel's expected buffer format.
///
/// HF format:
///   - weight: U8 [N, K/2] — packed FP4 nibbles
///   - weight_scale: F8_E4M3 [N, K/16] — per-block scales
///   - weight_scale_2: F32 scalar — per-tensor scale
///
/// Our kernel format (per column, N columns):
///   [K/2 packed FP4 bytes][K/16 FP8 scale bytes]
///
/// We bake weight_scale_2 into the FP8 block scales so tensor_scale=1.0.
fn pack_nvfp4_weight(
    weight_data: &[u8],    // U8 [N, K/2]
    scale_data: &[u8],     // F8_E4M3 [N, K/16]
    tensor_scale: f32,     // scalar
    n: usize,
    k: usize,
) -> Vec<u8> {
    let packed_per_col = k / 2;
    let scales_per_col = k / 16;
    let col_stride = packed_per_col + scales_per_col;
    let mut buf = vec![0u8; n * col_stride];

    for col in 0..n {
        let col_offset = col * col_stride;

        // Copy packed FP4 data (K/2 bytes per row)
        let src_packed = &weight_data[col * packed_per_col..(col + 1) * packed_per_col];
        buf[col_offset..col_offset + packed_per_col].copy_from_slice(src_packed);

        // Copy and modify FP8 scales: bake in tensor_scale
        let src_scales = &scale_data[col * scales_per_col..(col + 1) * scales_per_col];
        for i in 0..scales_per_col {
            let original_fp8 = src_scales[i];
            let scale_f32 = fp8_e4m3_to_float(original_fp8) * tensor_scale;
            buf[col_offset + packed_per_col + i] = float_to_fp8_e4m3(scale_f32);
        }
    }

    buf
}

/// Combines sharded NVFP4 safetensors files into a single file.
///
/// This function:
/// 1. Loads tensors from shard(s)
/// 2. Converts non-quantized tensors (BF16) to FP32
/// 3. Packs quantized tensors into our NvFp4 kernel format (U8)
/// 4. Writes combined file
pub fn combine_safetensors_nvfp4(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

    // Determine which shard files to load
    let shard_files: Vec<PathBuf> = if single_shard_path.exists() && !index_path.exists() {
        vec![single_shard_path]
    } else if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();

        println!("Loading {} shard files...", files.len());
        files.into_iter().map(|f| model_dir.join(f)).collect()
    } else {
        return Err("No model.safetensors or model.safetensors.index.json found".into());
    };

    // First pass: collect all raw tensor data from shards
    // We need to access weight, weight_scale, and weight_scale_2 together,
    // so load everything first.
    struct RawTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    let mut raw_tensors: HashMap<String, RawTensor> = HashMap::new();

    for shard_path in &shard_files {
        println!(
            "  Loading {}...",
            shard_path.file_name().unwrap().to_string_lossy()
        );
        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            raw_tensors.insert(
                name.to_string(),
                RawTensor {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    data: tensor.data().to_vec(),
                },
            );
        }
    }

    println!("Loaded {} raw tensors", raw_tensors.len());

    // Second pass: convert tensors
    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    // Collect quantized weight base names (e.g., "model.layers.0.self_attn.q_proj")
    let quantized_bases: Vec<String> = raw_tensors
        .keys()
        .filter(|name| is_quantized_weight(name))
        .map(|name| name.strip_suffix(".weight").unwrap().to_string())
        .collect();

    for base in &quantized_bases {
        let weight_name = format!("{base}.weight");
        let scale_name = format!("{base}.weight_scale");
        let scale2_name = format!("{base}.weight_scale_2");

        let weight = raw_tensors.get(&weight_name)
            .unwrap_or_else(|| panic!("Missing {weight_name}"));
        let scale = raw_tensors.get(&scale_name)
            .unwrap_or_else(|| panic!("Missing {scale_name}"));
        let scale2 = raw_tensors.get(&scale2_name)
            .unwrap_or_else(|| panic!("Missing {scale2_name}"));

        // weight shape: [N, K/2], scale shape: [N, K/16]
        let n = weight.shape[0];
        let k_half = weight.shape[1];
        let k = k_half * 2;

        // tensor_scale_2 is a scalar F32
        assert_eq!(scale2.dtype, Dtype::F32, "{scale2_name} must be F32");
        let tensor_scale = f32::from_le_bytes([
            scale2.data[0], scale2.data[1], scale2.data[2], scale2.data[3],
        ]);

        println!(
            "  Packing {weight_name}: [{n}, {k}] (tensor_scale={tensor_scale:.6})"
        );

        let packed = pack_nvfp4_weight(&weight.data, &scale.data, tensor_scale, n, k);

        // Output shape for our format: [N, K/2 + K/16]
        let col_stride = k / 2 + k / 16;
        all_tensors.insert(
            weight_name,
            StoredTensor::U8 {
                shape: vec![n, col_stride],
                data: packed,
            },
        );
    }

    // Process non-quantized tensors
    let skip_suffixes = [".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale"];
    for (name, raw) in &raw_tensors {
        // Skip already-processed quantized weights
        if is_quantized_weight(name) {
            continue;
        }
        // Skip scale/activation tensors
        if skip_suffixes.iter().any(|s| name.ends_with(s)) {
            continue;
        }

        // Convert BF16/F16/F32 to F32
        let fp32_data = match raw.dtype {
            Dtype::F32 => bytemuck::cast_slice::<u8, f32>(&raw.data).to_vec(),
            Dtype::F16 => {
                let f16_slice: &[half::f16] = bytemuck::cast_slice(&raw.data);
                f16_slice.iter().map(|x| x.to_f32()).collect()
            }
            Dtype::BF16 => {
                let bf16_slice: &[bf16] = bytemuck::cast_slice(&raw.data);
                bf16_slice.iter().map(|x| x.to_f32()).collect()
            }
            other => {
                println!("  Skipping {name} (unsupported dtype {other:?})");
                continue;
            }
        };

        println!("  Converting {name}: {:?} ({:?} -> F32)", raw.shape, raw.dtype);
        all_tensors.insert(
            name.clone(),
            StoredTensor::F32 {
                shape: raw.shape.clone(),
                data: fp32_data,
            },
        );
    }

    println!("Prepared {} tensors for output", all_tensors.len());

    // Serialize to combined file
    println!("Saving combined model to {}...", output_path.display());

    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let view = match stored {
                StoredTensor::F32 { shape, data } => {
                    let data_bytes: &[u8] = bytemuck::cast_slice(data);
                    TensorView::new(Dtype::F32, shape.clone(), data_bytes).unwrap()
                }
                StoredTensor::U8 { shape, data } => {
                    TensorView::new(Dtype::U8, shape.clone(), data).unwrap()
                }
            };
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::serialize(&tensor_views, None)?;

    let mut file = File::create(&output_path)?;
    file.write_all(&serialized)?;

    println!("Combined model saved successfully!");
    Ok(output_path)
}

/// Downloads an NVFP4 model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - model_combined.safetensors (F32 for non-quantized, U8 for NvFp4 packed weights)
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_nvfp4(&model_dir)?;
    Ok(model_dir)
}
