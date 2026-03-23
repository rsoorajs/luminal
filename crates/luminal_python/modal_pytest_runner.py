"""Run pytest on Modal with a dynamically selected GPU.

Usage:
    uv run modal run modal_pytest_runner.py --gpu A100 tests/test_llama3.py::test_hf_llama3_full -v
    uv run modal run modal_pytest_runner.py --gpu T4 tests/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("luminal-tests")

LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = "/root/luminal/crates/luminal_python"
VENV_PATH = f"{PROJECT_DIR}/.venv"

image = (
    modal.Image.from_registry("ghcr.io/luminal-ai/luminal-docker:cuda")
    .uv_sync(
        str(LOCAL_PROJECT_DIR),
        frozen=False,
        groups=["dev"],
        env={"UV_PROJECT_ENVIRONMENT": VENV_PATH},
    )
    .workdir(PROJECT_DIR)
    .add_local_dir(
        str(LOCAL_PROJECT_DIR.parent.parent),
        remote_path="/root/luminal",
        ignore=[
            ".git",
            ".claude-project",
            ".cargo-local",
            "**/.venv",
            "**/.pytest_cache",
            "**/__pycache__",
            "**/luminal_artifacts",
            "**/target",
            "docs",
        ],
    )
)


@app.cls(image=image, timeout=30 * 60)
class TestRunner:
    @modal.method()
    def run(self, pytest_args: list[str], pytest_addopts: str = "") -> int:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"src:{existing}" if existing else "src"
        env["LUMINAL_BACKEND"] = "cuda"
        env["UV_PROJECT_ENVIRONMENT"] = VENV_PATH
        env["MATURIN_PEP517_ARGS"] = "--features cuda --profile release"
        if pytest_addopts:
            env["PYTEST_ADDOPTS"] = pytest_addopts

        cmd = [
            "uv",
            "run",
            "--group",
            "dev",
            "--reinstall-package",
            "luminal_python",
            "python",
            "-m",
            "pytest",
            *pytest_args,
        ]
        return subprocess.run(cmd, env=env).returncode


def _parse_cli_args(cli_args: tuple[str, ...]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        prog="modal run modal_pytest_runner.py",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help="GPU type to request from Modal (for example: A100, T4, H100).",
    )
    parsed, pytest_args = parser.parse_known_args(cli_args)

    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["tests/"]

    return parsed.gpu, pytest_args


@app.local_entrypoint()
def main(*cli_args: str):
    gpu, pytest_args = _parse_cli_args(cli_args)
    pytest_addopts = os.environ.get("PYTEST_ADDOPTS", "")
    runner = TestRunner.with_options(gpu=gpu)()
    sys.exit(
        runner.run.remote(
            pytest_args=pytest_args,
            pytest_addopts=pytest_addopts,
        )
    )
