import modal
import subprocess
import os

app = modal.App("luminal-ci-llama")

# Persistent volumes for caching
hf_cache = modal.Volume.from_name("luminal-hf-cache", create_if_missing=True)
cargo_cache = modal.Volume.from_name("luminal-cargo-cache", create_if_missing=True)

# Use existing GHCR CUDA image (already has Python, Rust, CUDA)
# add_local_dir copies source code into the image (replaces deprecated modal.Mount)
WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry("ghcr.io/luminal-ai/luminal-docker:cuda")
    .add_local_dir(".", remote_path=WORKDIR)
)


@app.function(
    image=cuda_image,
    gpu="A100-80GB",
    timeout=3600,  # 60 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/target": cargo_cache,
    },
)
def run_llama():
    """Build and run the Llama 3 8B example on an A100 GPU."""
    subprocess.run(["nvidia-smi"], check=True)

    subprocess.run(
        ["cargo", "run", "--release"],
        cwd=f"{WORKDIR}/examples/llama",
        env={**os.environ, "CUDA_COMPUTE_CAP": "80"},
        check=True,
    )

    # Persist caches for next run
    hf_cache.commit()
    cargo_cache.commit()


@app.local_entrypoint()
def main():
    run_llama.remote()
