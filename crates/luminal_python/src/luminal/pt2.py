"""PT2 compilation pipeline for Luminal.

Provides:
  - compile(model, example_input, ...) — standalone PT2 path
  - pt2_backend(gm, example_inputs)    — torch.compile compatible backend
  - LuminalModel                       — wrapper for compiled models
"""

import atexit
import os
import shutil
import tempfile

import torch

from .luminal import compile_pt2 as _compile_pt2, Pt2CompiledModel


class LuminalModel:
    """Wrapper around compiled Luminal model for PyTorch-like interface."""

    def __init__(self, compiled: Pt2CompiledModel):
        self._compiled = compiled

    def __call__(self, x):
        input_device = x.device
        arr = x.detach().cpu().contiguous().float().numpy()
        data = arr.flatten().tolist()
        shape = list(arr.shape)
        result_data, result_shape = self._compiled.execute(data, shape)
        return torch.tensor(result_data, dtype=torch.float32).reshape(result_shape).to(input_device)


def _get_cache_dict(cache):
    """Convert DynamicCache to a dict for pytree flattening."""
    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [
            layer.values for layer in cache.layers if layer.values is not None
        ],
    }


def flatten_dynamic_cache(cache):
    return torch.utils._pytree._dict_flatten(_get_cache_dict(cache))


def unflatten_dynamic_cache(values, context):
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        key = key_list[idx] if idx < len(key_list) else None
        value = value_list[idx] if idx < len(value_list) else None
        cache.update(key, value, idx)
    return cache


def flatten_with_keys_dynamic_cache(cache):
    return torch.utils._pytree._dict_flatten_with_keys(_get_cache_dict(cache))


def _register_cache_serialization(verbose: int = 0):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    if DynamicCache is not None and DynamicCache not in torch.utils._pytree.SUPPORTED_NODES:
        if verbose:
            print("[luminal.pt2] register DynamicCache pytree serialization")
        torch.utils._pytree.register_pytree_node(
            DynamicCache,
            flatten_dynamic_cache,
            unflatten_dynamic_cache,
            serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
        )
        torch.fx._pytree.register_pytree_flatten_spec(
            DynamicCache,
            lambda cache, spec: torch.fx._pytree._dict_flatten_spec(
                _get_cache_dict(cache), spec
            ),
        )


def compile(
    model,
    example_input,
    search_iterations=25,
    backend=None,
    export_kwargs=None,
    dynamic_dim=None,
):
    """Compile a PyTorch model to run on Luminal via PT2 pipeline.

    Args:
        model: A PyTorch nn.Module.
        example_input: Example input tensor(s) for tracing.
        search_iterations: Number of optimization search iterations.
        backend: "cpu" or "cuda". Auto-detected if None.
        export_kwargs: Extra kwargs passed to torch.export.export.
        dynamic_dim: Which input dimension to make dynamic.

    Returns:
        A LuminalModel callable.
    """
    import inspect

    from safetensors.torch import save_file

    _register_cache_serialization()

    if dynamic_dim is None:
        dynamic_dim = "auto"

    if backend is None:
        backend = os.environ.get("LUMINAL_BACKEND", None)
        if backend is None:
            backend = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = export_kwargs or {}

    export_kwargs_extra = dict(strict=False)
    if (
        "prefer_deferred_runtime_asserts_over_guards"
        in inspect.signature(torch.export.export).parameters
    ):
        export_kwargs_extra["prefer_deferred_runtime_asserts_over_guards"] = True

    ep = None

    candidate_dims = []
    if isinstance(dynamic_dim, int):
        candidate_dims = [dynamic_dim]
    elif dynamic_dim == "auto" and example_input.dim() >= 2:
        if not example_input.is_floating_point():
            candidate_dims = [example_input.dim() - 1]

    if candidate_dims:
        from torch.export import Dim

        for dim_idx in candidate_dims:
            try:
                seq = Dim("seq", min=2)
                arg_shapes = {dim_idx: seq}
                kwarg_shapes = {k: None for k in kwargs}
                dynamic_shapes = (
                    (arg_shapes,) + tuple(kwarg_shapes.values())
                    if kwarg_shapes
                    else (arg_shapes,)
                )
                ep = torch.export.export(
                    model,
                    (example_input,),
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    **export_kwargs_extra,
                )
                break
            except Exception:
                continue

    if ep is None:
        ep = torch.export.export(
            model,
            (example_input,),
            kwargs=kwargs,
            dynamic_shapes=None,
            **export_kwargs_extra,
        )

    tmpdir = tempfile.mkdtemp(prefix="luminal_")
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
    pt2_path = os.path.join(tmpdir, "model.pt2")
    weights_path = os.path.join(tmpdir, "weights.safetensors")

    torch.export.save(ep, pt2_path)

    state_dict = {k: v.float().clone() for k, v in ep.state_dict.items()}
    if state_dict:
        save_file(state_dict, weights_path)
    else:
        weights_path = ""

    compiled = _compile_pt2(pt2_path, weights_path, backend, search_iterations)
    return LuminalModel(compiled)


def pt2_backend(gm: torch.fx.GraphModule, example_inputs: list):
    """torch.compile backend using PT2 pipeline.

    Usage: torch.compile(model, backend=luminal.pt2.pt2_backend)
    """
    _register_cache_serialization()
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    backend_name = "cuda" if device.type == "cuda" else "cpu"

    gm = gm.eval()

    # Classify each placeholder as a lifted param or user input.
    # Lifted params may be interleaved with user inputs in the FX graph,
    # so we track positions explicitly rather than assuming they come first.
    buffer_indices = []  # positions in example_inputs that are lifted params
    user_indices = []    # positions in example_inputs that are user inputs
    buffer_nodes = []    # FX nodes to replace with get_attr
    placeholder_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            if name.startswith("l_self_") or name.startswith("l_model_"):
                buffer_indices.append(placeholder_idx)
                buffer_nodes.append(node)
            else:
                user_indices.append(placeholder_idx)
            placeholder_idx += 1

    if buffer_nodes:
        # Re-internalize lifted params: register as buffers, replace placeholders
        # with get_attr nodes so torch.export sees them as model state, not inputs.
        for i, node in enumerate(buffer_nodes):
            attr_name = f"_luminal_param_{i}"
            gm.register_buffer(attr_name, example_inputs[buffer_indices[i]].detach().clone())
            with gm.graph.inserting_before(node):
                new_node = gm.graph.create_node("get_attr", attr_name)
                new_node.meta = node.meta.copy()
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()

    user_inputs = [example_inputs[i] for i in user_indices] if user_indices else list(example_inputs)

    import inspect

    from safetensors.torch import save_file

    export_kwargs_extra = dict(strict=False)
    if (
        "prefer_deferred_runtime_asserts_over_guards"
        in inspect.signature(torch.export.export).parameters
    ):
        export_kwargs_extra["prefer_deferred_runtime_asserts_over_guards"] = True

    ep = torch.export.export(gm, tuple(user_inputs), **export_kwargs_extra)

    tmpdir = tempfile.mkdtemp(prefix="luminal_backend_")
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
    pt2_path = os.path.join(tmpdir, "model.pt2")
    weights_path = os.path.join(tmpdir, "weights.safetensors")

    torch.export.save(ep, pt2_path)

    state_dict = {k: v.float().clone() for k, v in ep.state_dict.items()}
    if state_dict:
        save_file(state_dict, weights_path)
    else:
        weights_path = ""

    compiled_inner = _compile_pt2(pt2_path, weights_path, backend_name, 0)
    lm = LuminalModel(compiled_inner)

    def wrapper(*args):
        # After re-internalizing lifted params, dynamo only passes user inputs
        # to the wrapper at runtime (lifted params are no longer part of the
        # calling convention).
        return (lm(args[0]),)

    return wrapper
