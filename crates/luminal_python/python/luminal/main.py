import os
import tempfile
from typing import Callable, List

import luminal
import onnx
import torch
import torch._dynamo
from compiled_model import CompiledModel


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


class AddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return self.weight + x


class AddAddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_1: torch.Tensor = torch.rand((5, 5))
        self.weight_2: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return self.weight_1 + x + self.weight_2


class AddConstantTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x + 10


def add_test():
    add_test_model: torch.nn.Module = AddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def add_add_test():
    add_test_model: torch.nn.Module = AddAddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def add_broadcast_test():
    add_test_model: torch.nn.Module = AddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand(5)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def add_constant_test():
    add_test_model: torch.nn.Module = AddConstantTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand(5)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


class SubTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((10, 10))

    def forward(self, x: torch.Tensor):
        return self.weight - x


def sub_test():
    sub_test_model: torch.nn.Module = SubTestModel()
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10))
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def sub_broadcast_test():
    sub_test_model: torch.nn.Module = SubTestModel()
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10))
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def test_cleanup():
    torch._dynamo.reset()


def main():

    tests = [
        add_add_test,
        add_constant_test,
        sub_test,
        sub_broadcast_test,
        add_test,
        add_broadcast_test,
    ]

    for test in tests:
        test_cleanup()
        test()


if __name__ == "__main__":
    main()
