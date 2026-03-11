import modal
import subprocess
import os

example = os.environ.get("EXAMPLE", "llama")
gpu_type = os.environ.get("GPU_TYPE", "A100-80GB")

app = modal.App(f"luminal-ci-{example}")

hf_cache = modal.Volume.from_name("luminal-hf-cache", create_if_missing=True)

WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry(
        "ghcr.io/luminal-ai/luminal-docker:cuda"
    )
    .add_local_dir(".", remote_path=WORKDIR)
)


@app.function(
    image=cuda_image,
    gpu=gpu_type,
    timeout=3600,  # 60 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache,
    },
)
def run_example():
    """Build and run a luminal example on a Modal GPU."""
    subprocess.run(["nvidia-smi"], check=True)

    subprocess.run(
        ["cargo", "run", "--release"],
        cwd=f"{WORKDIR}/examples/{example}",
        env={
            **os.environ,
            "HF_HOME": "/root/.cache/huggingface",
        },
        check=True,
    )

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_example.remote()
