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

from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


def download_model_files(repo_id: str, output_dir: Path):
    """Download model files from Hugging Face Hub."""

    print(f"Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id)

    # Filter for files we need: tokenizer.json and all .safetensors files
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


def combine_safetensors(model_dir: Path):
    """Combine sharded safetensors files into a single file."""

    # Check if combined file already exists
    output_path = model_dir / "model_combined.safetensors"
    if output_path.exists():
        print(f"Combined safetensors file already exists at {output_path}")
        print("Skipping combination step.")
        return

    # Load the index
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    # Collect all tensors
    all_tensors = {}
    weight_map = index.get("weight_map", {})

    # Get unique shard files
    shard_files = sorted(set(weight_map.values()))

    print(f"Loading {len(shard_files)} shard files...")
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        print(f"  Loading {shard_file}...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    # Save combined file
    print(f"Saving combined model to {output_path}...")
    save_file(all_tensors, output_path)

    print(f"Combined model saved successfully to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    repo_id = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Download files from Hugging Face Hub
    download_model_files(repo_id, script_dir)

    # Combine safetensors files
    print("\nCombining safetensors files...")
    combine_safetensors(script_dir)

    print("\nDone!")
