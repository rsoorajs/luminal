"""Test models for HLIR ops testing."""
import torch


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


class SinTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class CosTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class SubTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((10, 10))

    def forward(self, x: torch.Tensor):
        return self.weight - x


class TransposeTestModel(torch.nn.Module):
    """Test basic 2D transpose (matrix transpose)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.t()  # 2D transpose


class Transpose3DTestModel(torch.nn.Module):
    """Test 3D transpose with explicit permutation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(2, 0, 1)  # Rotate dimensions


class Transpose4DTestModel(torch.nn.Module):
    """Test 4D transpose (NCHW -> NHWC)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)  # Common in CNNs


class TransposeReverseTestModel(torch.nn.Module):
    """Test reverse permutation (default transpose behavior)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.ndim))
        return x.permute(*reversed(dims))


class TransposeInExpressionModel(torch.nn.Module):
    """Test transpose as part of larger expression."""
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((10, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (5, 10), transpose to (10, 5), add weight
        return x.t() + self.weight


# ========== Constant Node Test Models ==========
# These models test ONNX Constant node handling via inline tensor literals


class ConstantScalarFloatModel(torch.nn.Module):
    """Test scalar constant (broadcasts to input shape)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor(10.5)
        return x + constant


class Constant1DArrayFloatModel(torch.nn.Module):
    """Test 1D array constant."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        return x * constant


class Constant2DMatrixFloatModel(torch.nn.Module):
    """Test 2D matrix constant."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return x + constant


class ConstantRawDataFloatModel(torch.nn.Module):
    """Test constant with specific values (tests raw data format)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([7.5, 8.5, 9.5])
        return x + constant


class ConstantInt32ConversionModel(torch.nn.Module):
    """Test INT32 constant values (PyTorch exports as integers)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        return x + constant.float()


class ConstantInt64ConversionModel(torch.nn.Module):
    """Test INT64 constant values."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([100, 200, 300], dtype=torch.int64)
        return x * constant.float()


class ConstantFloat64ConversionModel(torch.nn.Module):
    """Test FLOAT64 (double) constant values."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        return x * constant.float()


class ConstantBoolConversionModel(torch.nn.Module):
    """Test boolean constant values (converted to 0.0/1.0)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([True, False, True, False, True], dtype=torch.bool)
        return x * constant.float()


class ConstantInt64RawDataModel(torch.nn.Module):
    """Test INT64 constant with large values (tests raw data path)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1000, 2000, 3000], dtype=torch.int64)
        return x + constant.float()


class ConstantNegativeValuesModel(torch.nn.Module):
    """Test negative constant values."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([-5.0, -10.0, -15.0])
        return x + constant


class ConstantZeroValueModel(torch.nn.Module):
    """Test all-zero constant."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.0, 0.0, 0.0, 0.0])
        return x * constant


class ConstantMultipleInGraphModel(torch.nn.Module):
    """Test multiple constants in one graph."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        const1 = torch.tensor([10.0, 20.0, 30.0])
        const2 = torch.tensor([1.0, 2.0, 3.0])
        return x + const1 + const2


# ========== Cast Node Test Models ==========
# These models test ONNX Cast node handling via .to(dtype) method


class CastDoubleToFloatModel(torch.nn.Module):
    """Test downcast: Double (FLOAT64) -> Float."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input will be float64, cast to float32
        return x.to(torch.float32)


class CastInt32ToFloatModel(torch.nn.Module):
    """Test INT32 -> Float conversion."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastInt64ToFloatModel(torch.nn.Module):
    """Test INT64 -> Float conversion."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastBoolToFloatModel(torch.nn.Module):
    """Test BOOL -> Float conversion (non-zero -> 1.0, zero -> 0.0)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastInComputationGraphModel(torch.nn.Module):
    """Test Cast node followed by an operation (Cast + Add)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        casted = x.to(torch.float32)
        constant = torch.tensor([2.0, 2.0, 2.0])
        return casted + constant


class CastWith2DTensorModel(torch.nn.Module):
    """Test Cast with 2D tensor (matrix)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastNegativeValuesModel(torch.nn.Module):
    """Test Cast with negative integer values."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


class CastScalarValueModel(torch.nn.Module):
    """Test Cast with scalar (single element)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


# ========== Mod Node Test Models ==========


class ModTestModel(torch.nn.Module):
    """Tests element-wise modulo with a weight tensor."""
    def __init__(self) -> None:
        super().__init__()
        self.weight: torch.Tensor = torch.rand((5, 5)) + 1.0  # ensure non-zero divisor

    def forward(self, x: torch.Tensor):
        return x % self.weight


class ModByConstantModel(torch.nn.Module):
    """Tests modulo with an inline constant tensor (ONNX Constant node)."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([3.0, 4.0, 5.0])
        return x % constant
