import os
import tempfile
from typing import Callable, List

import onnx
import pytest
import torch
import torch._dynamo
from test_models import (
    AddAddTestModel,
    AddConstantTestModel,
    AddTestModel,
    CosTestModel,
    DivTestModel,
    LinearLayerModel,
    MulTestModel,
    SinTestModel,
    SqrtTestModel,
    SubTestModel,
    TransposeTestModel,
    Transpose3DTestModel,
    Transpose4DTestModel,
    TransposeReverseTestModel,
    TransposeInExpressionModel,
)

import luminal
from luminal import luminal_backend


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


def test_sin():
    sin_test_model: torch.nn.Module = SinTestModel()
    sin_test_model_compiled = torch.compile(sin_test_model, backend=luminal_backend)
    x = torch.rand(100)
    output = sin_test_model_compiled(x)
    original = sin_test_model(x)
    assert torch.allclose(output, original)


def test_cos():
    cos_test_model: torch.nn.Module = CosTestModel()
    cos_test_model_compiled = torch.compile(cos_test_model, backend=luminal_backend)
    x = torch.rand(100)
    output = cos_test_model_compiled(x)
    original = cos_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_2d():
    """Test basic 2D matrix transpose."""
    transpose_test_model: torch.nn.Module = TransposeTestModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 10))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_3d():
    """Test 3D transpose with dimension permutation."""
    transpose_test_model: torch.nn.Module = Transpose3DTestModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((2, 3, 4))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_4d():
    """Test 4D transpose (common in conv nets: NCHW -> NHWC)."""
    transpose_test_model: torch.nn.Module = Transpose4DTestModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((1, 3, 224, 224))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_reverse():
    """Test default transpose (reverse all dimensions)."""
    transpose_test_model: torch.nn.Module = TransposeReverseTestModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((2, 3, 4, 5))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_in_expression():
    """Test transpose within a larger computational graph."""
    transpose_test_model: torch.nn.Module = TransposeInExpressionModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 10))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)


def test_transpose_square_matrix():
    """Test transpose of square matrix (edge case)."""
    transpose_test_model: torch.nn.Module = TransposeTestModel()
    transpose_test_model_compiled = torch.compile(
        transpose_test_model, backend=luminal_backend
    )
    x: torch.Tensor = torch.rand((5, 5))
    output: torch.Tensor = transpose_test_model_compiled(x)
    original: torch.Tensor = transpose_test_model(x)
    assert torch.allclose(output, original)
