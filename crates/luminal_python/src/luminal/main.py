import os
import tempfile
from typing import Callable, List

import onnx
import torch
import torch._dynamo

import luminal

from .compiled_model import CompiledModel


def luminal_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _ = gm.eval()
    try:
        _ = torch.onnx.export(
            gm,
            tuple(example_inputs),
            tmp_path,
            input_names=[f"input_{i}" for i in range(len(example_inputs))],
        )

        result = luminal.process_onnx(tmp_path)
    finally:
        os.unlink(tmp_path)
    compiled = CompiledModel(result)
    return compiled
