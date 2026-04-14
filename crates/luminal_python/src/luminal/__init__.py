"""Luminal Python bindings - PyTorch backend using Luminal."""

# Import Python components
# Register DynamicCache pytree serialization once at import time
from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel

# Import Rust extension components (built by maturin)
# These are available directly in the package namespace
from .luminal import CompiledGraph, process_pt2, available_backends, _registry_capsule
from .main import luminal_backend

_register_cache_serialization()


def _discover_backends():
    """Auto-discover and register external backend plugins.

    External backends (e.g. luminal_tron, luminal_cuda) declare an entry point
    in their pyproject.toml under the ``luminal.backends`` group::

        [project.entry-points."luminal.backends"]
        tron = "luminal_tron:register"

    Each entry point should be a callable that accepts a PyCapsule containing
    the ``register_backend`` function pointer from the Rust registry.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    capsule = _registry_capsule()
    for ep in entry_points(group="luminal.backends"):
        try:
            register_fn = ep.load()
            register_fn(capsule)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to load luminal backend plugin '{ep.name}': {e}",
                stacklevel=2,
            )


_discover_backends()

# Re-export everything for clean package interface
__all__ = [
    "CompiledModel",
    "luminal_backend",
    "CompiledGraph",
    "process_pt2",
    "available_backends",
]
