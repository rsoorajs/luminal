import os
import tempfile
from typing import Callable, List

import onnx
import pytest
import torch
import torch._dynamo

import luminal
from luminal import luminal_backend


class AddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return self.weight + x


class MulTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return self.weight * x


class DivTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return self.weight / x


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


class LinearLayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5))
        self.bias: torch.Tensor = torch.rand((5, 5))

    def forward(self, x: torch.Tensor):
        return (self.weight @ x) + self.bias


class SqrtTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sqrt()


def test_add():
    add_test_model: torch.nn.Module = AddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_linear_layer():
    add_test_model: torch.nn.Module = LinearLayerModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_mul():
    add_test_model: torch.nn.Module = MulTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )

    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_div():
    div_test_model: torch.nn.Module = DivTestModel()
    div_test_mode_compiled: Callable = torch.compile(
        div_test_model, backend=luminal_backend
    )

    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = div_test_model(x)
    output: torch.Tensor = div_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_add_add():
    add_test_model: torch.nn.Module = AddAddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_add_broadcast():
    add_test_model: torch.nn.Module = AddTestModel()
    add_test_mode_compiled: Callable = torch.compile(
        add_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand(5)
    original: torch.Tensor = add_test_model(x)
    output: torch.Tensor = add_test_mode_compiled(x)
    assert torch.allclose(output, original)


def test_add_constant():
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


def test_sub():
    sub_test_model: torch.nn.Module = SubTestModel()
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10))
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def test_sub_broadcast():
    sub_test_model: torch.nn.Module = SubTestModel()
    sub_test_mode_compiled = torch.compile(sub_test_model, backend=luminal_backend)
    x = torch.rand((10, 10))
    output = sub_test_mode_compiled(x)
    original = sub_test_model(x)
    assert torch.allclose(output, original)


def test_sqrt():
    sqrt_test_model: torch.nn.Module = SqrtTestModel()
    sqrt_test_model_compiled = torch.compile(sqrt_test_model, backend=luminal_backend)
    x = torch.rand(100)
    output = sqrt_test_model_compiled(x)
    original = sqrt_test_model(x)
    assert torch.allclose(output, original)
