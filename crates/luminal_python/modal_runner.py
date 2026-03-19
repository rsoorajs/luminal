"""Run pytest on Modal with a dynamically selected GPU.

Usage:
    modal run modal_runner.py --gpu A100 --test "tests/test_llama3.py::test_hf_llama3_full"
    modal run modal_runner.py --gpu T4 --test "tests/"
"""

import subprocess
import sys

import modal

app = modal.App("luminal-tests")

image = (
    modal.Image.from_registry("ghcr.io/luminal-ai/luminal-docker:cuda")
    .add_local_dir(
        ".",
        remote_path="/root/luminal_python",
        ignore=[
            ".venv",
            ".pytest_cache",
            "__pycache__",
            "luminal_artifacts",
            "rust/target",
        ],
    )
    .workdir("/root/luminal_python")
)

@app.cls(image=image, timeout=30 * 60)
class TestRunner:
    @modal.method()
    def run(self, test: str) -> int:
        cmd = (
            f"LUMINAL_BACKEND=cuda uv run --frozen "
            f"--reinstall-package luminal_python pytest {test} "
            f"-v -s"
        )
        return subprocess.run(cmd, shell=True).returncode


@app.local_entrypoint()
def main(gpu: str = "", test: str = "tests/"):
    if not gpu:
        print("ERROR: --gpu is required (e.g. --gpu A100, --gpu T4, --gpu H100)")
        sys.exit(1)

    runner = TestRunner.with_options(gpu=gpu)()
    sys.exit(runner.run.remote(test=test))
