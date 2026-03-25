import os
import tempfile

import onnx
import torch
import torch._dynamo

import luminal

from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel


def luminal_backend(gm, example_inputs, options=None):
    """Luminal torch.compile backend.

    Usage:
        torch.compile(model, backend=luminal_backend)
        torch.compile(model, backend=luminal_backend, options={"export_mode": "pt2"})

    Options:
        export_mode: "onnx" (default) or "pt2"
        opset: ONNX opset version (default 20)
    """
    options = options or {}

    # Env var override
    env_mode = os.getenv("LUMINAL_EXPORT_MODE", "").lower()
    export_mode = (
        env_mode if env_mode in ("pt2", "onnx") else options.get("export_mode", "onnx")
    )
    opset = options.get("opset", 20)

    _register_cache_serialization()
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    backend = "cuda" if device.type == "cuda" else "native"

    if export_mode == "pt2":
        return _compile_pt2(gm, example_inputs, backend)
    return _compile_onnx(gm, example_inputs, backend, opset=opset)


def _compile_onnx(gm, example_inputs, backend, opset=20):
    """ONNX compilation path."""
    # Identify weight vs user inputs from FX graph placeholders.
    # torch.compile lifts model parameters into graph inputs — we detect them by name prefix.
    weight_map = {}  # onnx_name -> (placeholder_idx, tensor)
    user_indices = []
    ph_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            onnx_name = f"input_{ph_idx}"
            if node.name.startswith(("l_self_", "l_model_", "l__self_")):
                weight_map[onnx_name] = (ph_idx, example_inputs[ph_idx])
            else:
                user_indices.append(ph_idx)
            ph_idx += 1

    # Collect weight device pointers upfront so Rust can use them during search
    # (avoids allocating duplicate GPU buffers from ONNX file data)
    weight_refs = []
    weight_device_ptrs = {}  # onnx_name -> (device_ptr, n_bytes)
    cpu_weights = {}         # onnx_name -> (host_ptr, n_elements)
    for name, (_idx, tensor) in weight_map.items():
        if backend == "cuda" and tensor.is_cuda:
            t = tensor.detach().contiguous().float()
            weight_refs.append(t)
            weight_device_ptrs[name] = (t.data_ptr(), t.numel() * 4)
        else:
            t = tensor.detach().cpu().contiguous().float()
            weight_refs.append(t)
            cpu_weights[name] = (t.data_ptr(), t.numel())

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _ = gm.eval()
    try:
        _ = torch.onnx.export(
            gm,
            tuple(example_inputs),
            tmp_path,
            opset_version=opset,
            input_names=[f"input_{i}" for i in range(len(example_inputs))],
        )

        result = luminal.process_onnx(tmp_path, backend, weight_device_ptrs)
    finally:
        os.unlink(tmp_path)

    # Persist all weight inputs and set CPU weights after compilation
    for name in weight_map:
        result.persist_input(name)
    for name, (ptr, n_elements) in cpu_weights.items():
        result.set_input_from_ptr(name, ptr, n_elements)

    compiled = CompiledModel(result, weight_refs=weight_refs, user_indices=user_indices)
    return compiled


def _compile_pt2(gm, example_inputs, backend):
    """PT2/torch.export path — delegates to pt2.pt2_backend."""
    from .pt2 import pt2_backend

    return pt2_backend(gm, example_inputs, backend=backend)
