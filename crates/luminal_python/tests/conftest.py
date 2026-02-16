"""Test configuration."""
# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook
    maturin_import_hook.install()
except ImportError:
    pass  # Hook not available, rebuilds will be manual

import pytest
import torch._dynamo


@pytest.fixture(autouse=True, scope="function")
def reset_torch_dynamo():
    """Reset PyTorch Dynamo state after each test to prevent state leakage.

    This fixture automatically runs after every test function to clear
    torch._dynamo's compilation cache, ensuring test isolation.
    """
    yield  # Test runs here
    torch._dynamo.reset()
