from typing import Callable

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
    # Constant models
    ConstantScalarFloatModel,
    Constant1DArrayFloatModel,
    Constant2DMatrixFloatModel,
    ConstantRawDataFloatModel,
    ConstantInt32ConversionModel,
    ConstantInt64ConversionModel,
    ConstantFloat64ConversionModel,
    ConstantBoolConversionModel,
    ConstantInt64RawDataModel,
    ConstantNegativeValuesModel,
    ConstantZeroValueModel,
    ConstantMultipleInGraphModel,
    # Cast models
    CastDoubleToFloatModel,
    CastInt32ToFloatModel,
    CastInt64ToFloatModel,
    CastBoolToFloatModel,
    CastInComputationGraphModel,
    CastWith2DTensorModel,
    CastNegativeValuesModel,
    CastScalarValueModel,
)

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


# ========== ONNX Constant Node Tests ==========
# These tests verify the parse_constant_node function in ops_parse.rs
# which handles ONNX Constant nodes (nodes with embedded data in attributes)


def test_constant_scalar_float():
    """Test scalar constant (broadcasts to input shape)."""
    model: torch.nn.Module = ConstantScalarFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_1d_array_float():
    """Test 1D array constant."""
    model: torch.nn.Module = Constant1DArrayFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_2d_matrix_float():
    """Test 2D matrix constant."""
    model: torch.nn.Module = Constant2DMatrixFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_raw_data_float():
    """Test raw binary data format (chunks_exact(4) code path)."""
    model: torch.nn.Module = ConstantRawDataFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int32_conversion():
    """Test INT32 -> f32 conversion."""
    model: torch.nn.Module = ConstantInt32ConversionModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int64_conversion():
    """Test INT64 -> f32 conversion."""
    model: torch.nn.Module = ConstantInt64ConversionModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([2.0, 3.0, 4.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_float64_conversion():
    """Test FLOAT64 -> f32 conversion."""
    model: torch.nn.Module = ConstantFloat64ConversionModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_bool_conversion():
    """Test BOOL -> f32 conversion (0.0/1.0)."""
    model: torch.nn.Module = ConstantBoolConversionModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_int64_raw_data():
    """Test raw binary format for INT64 (chunks_exact(8) code path)."""
    model: torch.nn.Module = ConstantInt64RawDataModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10.0, 20.0, 30.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_negative_values():
    """Test negative constants."""
    model: torch.nn.Module = ConstantNegativeValuesModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([100.0, 200.0, 300.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_zero_value():
    """Test all-zero constant."""
    model: torch.nn.Module = ConstantZeroValueModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_constant_multiple_in_graph():
    """Test multiple Constant nodes in one graph."""
    model: torch.nn.Module = ConstantMultipleInGraphModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([5.0, 6.0, 7.0])
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


# ========== ONNX Cast Node Tests ==========
# These tests verify the parse_cast_node function in ops_parse.rs
# which handles ONNX Cast nodes (type conversion operations)


def test_cast_double_to_float():
    """Test downcast: Double (FLOAT64) -> Float."""
    model: torch.nn.Module = CastDoubleToFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1.123456789, 2.987654321, 3.555555555, 4.111111111], dtype=torch.float64)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_int32_to_float():
    """Test INT32 -> Float conversion."""
    model: torch.nn.Module = CastInt32ToFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_int64_to_float():
    """Test INT64 -> Float conversion."""
    model: torch.nn.Module = CastInt64ToFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([100, 200, 300, 400], dtype=torch.int64)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_bool_to_float():
    """Test BOOL -> Float conversion (non-zero -> 1.0, zero -> 0.0)."""
    model: torch.nn.Module = CastBoolToFloatModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_in_computation_graph():
    """Test Cast node followed by an operation (Cast + Add)."""
    model: torch.nn.Module = CastInComputationGraphModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([10, 20, 30], dtype=torch.int32)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_with_2d_tensor():
    """Test Cast with 2D tensor (matrix)."""
    model: torch.nn.Module = CastWith2DTensorModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_negative_values():
    """Test Cast with negative integer values."""
    model: torch.nn.Module = CastNegativeValuesModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([-10, -5, 0, 5, 10], dtype=torch.int32)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)


def test_cast_scalar_value():
    """Test Cast with scalar (single element)."""
    model: torch.nn.Module = CastScalarValueModel()
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.tensor([42.123456], dtype=torch.float64)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original)
