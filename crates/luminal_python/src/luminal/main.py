import os
import tempfile

import torch
import torch._dynamo

import luminal

from .compiled_model import CompiledModel

# ---------------------------------------------------------------------------
# Shared helpers (used by pt2)
# ---------------------------------------------------------------------------


def _detect_backend(example_inputs):
    """Detect backend from input device. Returns 'cuda' or 'native'."""
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    return "cuda" if device.type == "cuda" else "native"


def _collect_weight_pointers(weights, backend):
    """Partition weight tensors into CUDA device pointers and CPU host pointers.

    Args:
        weights: dict of name -> torch.Tensor
        backend: "cuda", "gpu", "cpu", or "native"

    Returns:
        (keep_alive, device_ptrs, cpu_ptrs) where:
        - keep_alive: list[Tensor] to prevent GC of shared weight memory
        - device_ptrs: {name: (device_ptr, n_bytes)}
        - cpu_ptrs: {name: (host_ptr, n_elements)}
    """
    keep_alive = []
    device_ptrs = {}
    cpu_ptrs = {}
    for name, tensor in weights.items():
        t = tensor.detach().contiguous()
        if t.dtype != torch.float32:
            t = t.float()
        if backend in ("cuda", "gpu") and t.is_cuda:
            keep_alive.append(t)
            device_ptrs[name] = (t.data_ptr(), t.numel() * 4)
        else:
            t = t.cpu() if t.is_cuda else t
            keep_alive.append(t)
            cpu_ptrs[name] = (t.data_ptr(), t.numel())
    return keep_alive, device_ptrs, cpu_ptrs


def _load_cpu_weights(compiled_graph, cpu_weights):
    """Load CPU weight data into a compiled graph after Rust compilation."""
    for name, (ptr, n_elements) in cpu_weights.items():
        compiled_graph.set_weight_from_ptr(name, ptr, n_elements)


# ---------------------------------------------------------------------------
# torch.compile backend entry point
# ---------------------------------------------------------------------------


def luminal_backend(gm, example_inputs, options=None):
    """Luminal torch.compile backend.

    Usage:
        torch.compile(model, backend=luminal_backend)
    """
    backend = _detect_backend(example_inputs)
    return _compile_pt2(gm, example_inputs, backend)


# ---------------------------------------------------------------------------
# PT2 compilation path (delegates to pt2 module)
# ---------------------------------------------------------------------------


def _compile_pt2(gm, example_inputs, backend):
    """PT2/torch.export path — delegates to pt2.pt2_backend."""
    from .pt2 import pt2_backend

    return pt2_backend(gm, example_inputs, backend=backend)
