"""Test configuration."""

# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook

    maturin_import_hook.install()
except ImportError:
    pass  # Hook not available, rebuilds will be manual

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch._dynamo


def pytest_addoption(parser):
    parser.addoption(
        "--modal", action="store_true", default=False,
        help="Dispatch tests to Modal for remote GPU execution",
    )
    parser.addoption(
        "--gpu", default="A100",
        help="GPU type when using --modal (A100, T4, H100)",
    )


def pytest_runtestloop(session):
    """Override the test loop to dispatch to Modal when --modal is passed."""
    if not session.config.getoption("modal"):
        return None  # Fall through to default local execution

    gpu = session.config.getoption("gpu")
    test_ids = [item.nodeid for item in session.items]

    # Pass test IDs in the order that local collection (including
    # pytest-randomly shuffling) already determined.  Disable
    # re-shuffling on the remote side so the order is preserved.
    test_arg = " ".join(test_ids) + " -p no:randomly"

    cmd = [
        sys.executable, "-m", "modal", "run",
        str(Path(session.config.rootdir) / "modal_runner.py"),
        "--gpu", gpu,
        "--test", test_arg,
    ]
    print(f"\n=== Dispatching {len(test_ids)} tests to Modal ({gpu}) ===\n")
    result = subprocess.run(cmd, cwd=session.config.rootdir)

    if result.returncode != 0:
        session.testsfailed = len(session.items)

    return True  # Prevents default runtestloop


@pytest.fixture
def device() -> torch.device:
    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    return torch.device("cuda") if backend == "cuda" else torch.device("cpu")


@pytest.fixture(autouse=True, scope="function")
def reset_torch_dynamo():
    # We need this for two reasons
    # 1. Some of our casts tests use the same model, but those graph have some state to them
    # and the cache will return old models
    # 2. The cache adds a large preformace hit to the test suite
    torch._dynamo.config.cache_size_limit = 1
    # Disable silent fallback to eager mode so backend errors surface as test failures
    torch._dynamo.config.suppress_errors = False
    """Reset PyTorch Dynamo state after each test to prevent state leakage.

    This fixture automatically runs after every test function to clear
    torch._dynamo's compilation cache, ensuring test isolation.
    """
    yield  # Test runs here
    torch._dynamo.reset()
