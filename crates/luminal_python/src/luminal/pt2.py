"""PT2 compilation pipeline for Luminal.

Provides:
  - compile(model, example_input, ...) — standalone PT2 path
  - pt2_backend(gm, example_inputs)    — torch.compile compatible backend
  - LuminalModel                       — wrapper for compiled models
"""

import os
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


_pytree_registered = False


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

    # Detect buffer/param inputs lifted by torch.compile
    n_buffer = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            if name.startswith("l_self_") or name.startswith("l_model_"):
                n_buffer += 1

    if n_buffer > 0:
        captured_buffers = [inp.detach().clone() for inp in example_inputs[:n_buffer]]

        def wrapper(*args):
            return gm(*captured_buffers, args[n_buffer])

        return wrapper
    else:
        import inspect

        from safetensors.torch import save_file

        export_kwargs_extra = dict(strict=False)
        if (
            "prefer_deferred_runtime_asserts_over_guards"
            in inspect.signature(torch.export.export).parameters
        ):
            export_kwargs_extra["prefer_deferred_runtime_asserts_over_guards"] = True

        ep = torch.export.export(gm, tuple(example_inputs), **export_kwargs_extra)

        tmpdir = tempfile.mkdtemp(prefix="luminal_backend_")
        pt2_path = os.path.join(tmpdir, "model.pt2")
        torch.export.save(ep, pt2_path)

        compiled_inner = _compile_pt2(pt2_path, "", backend_name, 0)
        lm = LuminalModel(compiled_inner)

        def wrapper(*args):
            return (lm(args[0]),)

        return wrapper
