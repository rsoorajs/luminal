import modal
import subprocess
import os

app = modal.App("luminal-ci-llama")

hf_cache = modal.Volume.from_name("luminal-hf-cache", create_if_missing=True)

WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry(
        "ghcr.io/luminal-ai/luminal-docker:cuda", force_build=True
    )
    .add_local_dir(".", remote_path=WORKDIR)
)


@app.function(
    image=cuda_image,
    gpu="A100-80GB",
    timeout=3600,  # 60 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache,
    },
)
def run_llama():
    """Build and run the Llama 3 8B example on an A100 GPU."""
    subprocess.run(["nvidia-smi"], check=True)

    subprocess.run(
        ["cargo", "run", "--release"],
        cwd=f"{WORKDIR}/examples/llama",
        env={**os.environ, "CUDA_COMPUTE_CAP": "80", "CUDARC_CUDA_VERSION": "13000"},
        check=True,
    )

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_llama.remote()
