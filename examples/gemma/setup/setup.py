#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "safetensors",
#     "torch",
#     "packaging",
#     "numpy",
#     "huggingface-hub"
# ]
# ///

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


def download_model_files(repo_id: str, output_dir: Path):
    print(f"Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id)

    files_to_download = []
    for file in all_files:
        if (
            file == "tokenizer.json"
            or file.endswith(".safetensors")
            or file == "model.safetensors.index.json"
        ):
            files_to_download.append(file)

    print(f"Found {len(files_to_download)} files to download")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in files_to_download:
        print(f"  Downloading {filename}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=None, local_dir=output_dir
        )
        print(f"    Saved to {downloaded_path}")

    print("All files downloaded successfully!")


def combine_and_convert_safetensors_to_fp32(model_dir: Path):
    """
    Combine sharded safetensors into a single file, converting tensors to FP32 on the fly.
    For multimodal Gemma 3, extract only language model weights and strip the 'language_model.' prefix.
    Outputs: model_combined.safetensors
    """
    output_path = model_dir / "model_combined.safetensors"
    if output_path.exists():
        print(f"FP32 combined safetensors file already exists at {output_path}")
        print("Skipping combine+convert step.")
        return

    index_path = model_dir / "model.safetensors.index.json"
    single_shard_path = model_dir / "model.safetensors"

    # Check if it's a single shard model
    if single_shard_path.exists() and not index_path.exists():
        print("Single shard model detected, converting to FP32...")
        all_tensors = {}
        with safe_open(single_shard_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                # For multimodal model, extract only language model weights
                if key.startswith("language_model."):
                    new_key = key[len("language_model."):]  # Strip prefix
                elif key.startswith("vision_tower.") or key.startswith("multi_modal_projector."):
                    continue  # Skip vision model weights
                else:
                    new_key = key

                t = sf.get_tensor(key)
                if torch.is_floating_point(t) and t.dtype != torch.float32:
                    t = t.to(dtype=torch.float32)
                all_tensors[new_key] = t

        print(f"Saving combined FP32 model to {output_path}...")
        save_file(all_tensors, output_path)
        print(f"Combined FP32 model saved successfully to {output_path}")
        return

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    # Stream through shards; convert tensors to fp32 as we read.
    all_tensors = {}

    print(f"Loading {len(shard_files)} shard files (converting to fp32)...")
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        print(f"  Loading {shard_file}...")
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                # For multimodal model, extract only language model weights
                if key.startswith("language_model."):
                    new_key = key[len("language_model."):]  # Strip prefix
                elif key.startswith("vision_tower.") or key.startswith("multi_modal_projector."):
                    continue  # Skip vision model weights
                else:
                    new_key = key

                t = sf.get_tensor(key)

                # Convert float dtypes to fp32; keep non-floats as-is (e.g., int tensors, masks).
                if torch.is_floating_point(t) and t.dtype != torch.float32:
                    t = t.to(dtype=torch.float32)

                all_tensors[new_key] = t

    print(f"Extracted {len(all_tensors)} language model tensors")

    # Gemma 3 scaling:
    # - Input embeddings are scaled by sqrt(hidden_size) = sqrt(2560) ≈ 50.596
    # - lm_head uses unscaled embedding weights (tie_word_embeddings)
    # We pre-scale embed_tokens and create a separate lm_head (unscaled copy)
    hidden_size = 2560
    embed_scale = hidden_size ** 0.5

    embed_key = "model.embed_tokens.weight"
    if embed_key in all_tensors:
        print(f"Pre-scaling embedding by sqrt({hidden_size}) = {embed_scale:.4f}...")
        # Save unscaled embedding as lm_head
        all_tensors["lm_head.weight"] = all_tensors[embed_key].clone()
        # Scale the embedding for input
        all_tensors[embed_key] = all_tensors[embed_key] * embed_scale

    # Gemma 3 RMSNorm uses (1 + weight) instead of just weight
    # Pre-add 1.0 to all norm weights so the model can use simple multiplication
    print("Pre-adding 1.0 to RMSNorm weights (Gemma uses 1+weight pattern)...")
    norm_keys = []
    for key in all_tensors.keys():
        if any(norm_type in key for norm_type in [
            'input_layernorm.weight',
            'post_attention_layernorm.weight',
            'pre_feedforward_layernorm.weight',
            'post_feedforward_layernorm.weight',
            'model.norm.weight',
            'q_norm.weight',
            'k_norm.weight'
        ]):
            norm_keys.append(key)

    for key in norm_keys:
        all_tensors[key] = all_tensors[key] + 1.0
    print(f"  Transformed {len(norm_keys)} norm weight tensors")

    print(f"Saving combined FP32 model to {output_path}...")
    save_file(all_tensors, output_path)
    print(f"Combined FP32 model saved successfully to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    # Use unsloth's ungated mirror of Gemma 3 4B instruct
    repo_id = "unsloth/gemma-3-4b-it"

    download_model_files(repo_id, script_dir)

    print("\nCombining + converting safetensors to FP32...")
    combine_and_convert_safetensors_to_fp32(script_dir)

    print("\nDone!")
