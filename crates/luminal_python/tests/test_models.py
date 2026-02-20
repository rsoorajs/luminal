"""Test models for HLIR ops testing."""

import torch


class AddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight + x


class MulTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight * x


class DivTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor):
        return self.weight / x


class AddAddTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight_1", torch.rand((5, 5)))
        self.register_buffer("weight_2", torch.rand((5, 5)))

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
        self.register_buffer("weight", torch.rand((5, 5)))
        self.register_buffer("bias", torch.rand((5, 5)))

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
        self.register_buffer("weight", torch.rand((10, 10)))

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
        self.register_buffer("weight", torch.rand((10, 5)))

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
        constant = torch.tensor(10.5).to(x.device)
        return x + constant


class Constant1DArrayFloatModel(torch.nn.Module):
    """Test 1D array constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(x.device)
        return x * constant


class Constant2DMatrixFloatModel(torch.nn.Module):
    """Test 2D matrix constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(x.device)
        return x + constant


class ConstantRawDataFloatModel(torch.nn.Module):
    """Test constant with specific values (tests raw data format)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([7.5, 8.5, 9.5]).to(x.device)
        return x + constant


class ConstantInt32ConversionModel(torch.nn.Module):
    """Test INT32 constant values (PyTorch exports as integers)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).to(x.device)
        return x + constant.float()


class ConstantInt64ConversionModel(torch.nn.Module):
    """Test INT64 constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([100, 200, 300], dtype=torch.int64).to(x.device)
        return x * constant.float()


class ConstantFloat64ConversionModel(torch.nn.Module):
    """Test FLOAT64 (double) constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64).to(x.device)
        return x * constant.float()


class ConstantBoolConversionModel(torch.nn.Module):
    """Test boolean constant values (converted to 0.0/1.0)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([True, False, True, False, True], dtype=torch.bool).to(
            x.device
        )
        return x * constant.float()


class ConstantInt64RawDataModel(torch.nn.Module):
    """Test INT64 constant with large values (tests raw data path)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1000, 2000, 3000], dtype=torch.int64).to(x.device)
        return x + constant.float()


class ConstantNegativeValuesModel(torch.nn.Module):
    """Test negative constant values."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([-5.0, -10.0, -15.0]).to(x.device)
        return x + constant


class ConstantZeroValueModel(torch.nn.Module):
    """Test all-zero constant."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(x.device)
        return x * constant


class ConstantMultipleInGraphModel(torch.nn.Module):
    """Test multiple constants in one graph."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        const1 = torch.tensor([10.0, 20.0, 30.0]).to(x.device)
        const2 = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
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
        constant = torch.tensor([2.0, 2.0, 2.0]).to(x.device)
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
        self.register_buffer(
            "weight", torch.rand((5, 5)) + 1.0
        )  # ensure non-zero divisor

    def forward(self, x: torch.Tensor):
        return x.fmod(self.weight)


class ModByConstantModel(torch.nn.Module):
    """Tests modulo with an inline constant tensor (ONNX Constant node)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([3.0, 4.0, 5.0]).to(x.device)
        return x.fmod(constant)


# ========== Floor Node Test Models ==========


class FloorTestModel(torch.nn.Module):
    """Tests element-wise floor operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class FloorNegativeModel(torch.nn.Module):
    """Tests floor with negative values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class FloorInExpressionModel(torch.nn.Module):
    """Tests floor as part of a larger expression (floor + mul)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x) * 2.0


# ========== Reshape Node Test Models ==========
# These models test ONNX Reshape node handling in ops_parse.rs


class ReshapeToFlatModel(torch.nn.Module):
    """Reshape 2D tensor to 1D (full flatten)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)  # (3, 4) -> (12,)


class ReshapeToMatrixModel(torch.nn.Module):
    """Reshape 1D tensor to 2D matrix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(3, 4)  # (12,) -> (3, 4)


class ReshapeTo3DModel(torch.nn.Module):
    """Reshape 1D tensor to 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, 3, 4)  # (24,) -> (2, 3, 4)


class ReshapeInferLastDimModel(torch.nn.Module):
    """Reshape with -1 to infer last dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(3, -1)  # (12,) -> (3, 4)


class ReshapeInferFirstDimModel(torch.nn.Module):
    """Reshape with -1 to infer first dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, 4)  # (12,) -> (3, 4)


class Reshape3Dto2DModel(torch.nn.Module):
    """Reshape 3D tensor to 2D (common in networks before linear layer)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(2, -1)  # (2, 3, 4) -> (2, 12)


class ReshapeInExpressionModel(torch.nn.Module):
    """Reshape followed by element-wise addition (Reshape in graph)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((2, 6)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = x.reshape(2, 6)  # (12,) -> (2, 6)
        return reshaped + self.weight


class ReshapeRoundtripModel(torch.nn.Module):
    """Reshape to different shape and back (two Reshape nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)  # (3, 4) -> (12,)
        return flat.reshape(3, 4)  # (12,) -> (3, 4)


class ReshapeAfterOpsModel(torch.nn.Module):
    """Apply operation then reshape (op -> Reshape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = x * 2.0
        return doubled.reshape(-1)  # (2, 3, 4) -> (24,)


class ShapeReshapeBatchFlattenModel(torch.nn.Module):
    """Batch flatten using dynamic shape: preserves first dim, flattens rest."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)  # (2, 3, 4) -> (2, 12)


class ShapeReshapeKeepBatchModel(torch.nn.Module):
    """Use shape of input to reshape, keeping batch dimension dynamic."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)  # (2, 3, 4) -> (2, 12)


# ========== Less Node Test Models ==========
# These models test ONNX Less node handling in ops_parse.rs


class LessTestModel(torch.nn.Module):
    """Tests element-wise less-than comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x < self.weight).to(torch.float32)


class LessBroadcastModel(torch.nn.Module):
    """Tests less-than with broadcasting (1D input vs 2D weight)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x < self.weight).to(torch.float32)


class LessWithConstantModel(torch.nn.Module):
    """Tests less-than against an inline constant (ONNX Constant + Less nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x < constant).to(torch.float32)


# ========== Gather Node Test Models ==========
# These models test ONNX Gather node handling in ops_parse.rs


class Gather1DModel(torch.nn.Module):
    """Tests Gather on 1D float data with constant integer indices (1D path)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([2, 0, 4, 1])
        return x[indices]


class GatherEmbeddingModel(torch.nn.Module):
    """Tests Gather via nn.Embedding (axis=0 gather on 2D weight, integer input)."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding: torch.nn.Embedding = torch.nn.Embedding(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Gather2DAxis0Model(torch.nn.Module):
    """Tests Gather on 2D weight along axis=0 with runtime integer indices."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(6, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class Gather2DAxis1Model(torch.nn.Module):
    """Tests Gather on 2D float input along axis=1 (exercises permute-gather-permute strategy)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor([1, 3, 0])
        return x[:, indices]


class GatherNegativeIndicesModel(torch.nn.Module):
    """Tests Gather with negative runtime indices (-1 for last element, etc.)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(6, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class GatherConstantFoldModel(torch.nn.Module):
    """Tests Gather constant folding: both data and indices are ONNX Constant nodes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).to(x.device)
        indices = torch.tensor([4, 1, 3])
        return x + data[indices]


# ========== Squeeze Node Test Models ==========
# These models test ONNX Squeeze node handling in ops_parse.rs


class SqueezeAxisModel(torch.nn.Module):
    """Tests squeezing a single size-1 leading dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0)  # (1, 3, 4) -> (3, 4)


class SqueezeAllDimsModel(torch.nn.Module):
    """Tests squeezing all size-1 dimensions (no explicit axes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()  # (1, 3, 1, 4) -> (3, 4)


class SqueezeMultipleAxesModel(torch.nn.Module):
    """Tests squeezing multiple explicit size-1 dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0).squeeze(1)  # (1, 3, 1, 4) -> (3, 4)


class SqueezeNegativeAxisModel(torch.nn.Module):
    """Tests squeezing a size-1 dimension using a negative axis index."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1)  # (3, 4, 1) -> (3, 4)


class SqueezeInExpressionModel(torch.nn.Module):
    """Tests squeeze as part of a larger expression (squeeze then add)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.squeeze(0)  # (1, 5) -> (5,)
        return squeezed + self.weight


# ========== ReduceSum Node Test Models ==========


class ReduceSumAxis0Model(torch.nn.Module):
    """Tests sum reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=0)  # (4, 5) -> (5,)


class ReduceSumAxis1Model(torch.nn.Module):
    """Tests sum reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)  # (4, 5) -> (4,)


class ReduceSumKeepDimsModel(torch.nn.Module):
    """Tests sum reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceSumAllAxesModel(torch.nn.Module):
    """Tests sum reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()  # (3, 4) -> scalar


class ReduceSum3DAxis1Model(torch.nn.Module):
    """Tests sum reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceSumMultiAxisModel(torch.nn.Module):
    """Tests sum reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceSumMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests sum reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceSumNegativeAxisModel(torch.nn.Module):
    """Tests sum reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1)  # (3, 4) -> (3,)


class ReduceSumInExpressionModel(torch.nn.Module):
    """Tests ReduceSum used in a larger expression (mean via sum/n)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True) / x.shape[1]  # (3, 4) -> (3, 1) mean


# ========== ReduceMax Node Test Models ==========


class ReduceMaxAxis0Model(torch.nn.Module):
    """Tests max reduction along axis 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=0)  # (4, 5) -> (5,)


class ReduceMaxAxis1Model(torch.nn.Module):
    """Tests max reduction along axis 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1)  # (4, 5) -> (4,)


class ReduceMaxKeepDimsModel(torch.nn.Module):
    """Tests max reduction along axis 1 with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1, keepdim=True)  # (4, 5) -> (4, 1)


class ReduceMaxAllAxesModel(torch.nn.Module):
    """Tests max reduction over all axes (scalar result)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max()  # (3, 4) -> scalar


class ReduceMax3DAxis1Model(torch.nn.Module):
    """Tests max reduction along axis 1 for 3D tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1)  # (2, 3, 4) -> (2, 4)


class ReduceMaxMultiAxisModel(torch.nn.Module):
    """Tests max reduction along multiple axes (0 and 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(0, 2))  # (2, 3, 4) -> (3,)


class ReduceMaxMultiAxisKeepDimsModel(torch.nn.Module):
    """Tests max reduction along axes (0, 2) with keepdim=True."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=(0, 2), keepdim=True)  # (2, 3, 4) -> (1, 3, 1)


class ReduceMaxNegativeAxisModel(torch.nn.Module):
    """Tests max reduction along axis -1 (last axis)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=-1)  # (3, 4) -> (3,)


class ReduceMaxInExpressionModel(torch.nn.Module):
    """Tests ReduceMax used in a larger expression (max * 2.0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1, keepdim=True) * 2.0  # (3, 4) -> (3, 1)


# ========== Sigmoid Node Test Models ==========


class SigmoidTestModel(torch.nn.Module):
    """Tests sigmoid on a 2D tensor (values in range [0,1] after sigmoid)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class SigmoidInExpressionModel(torch.nn.Module):
    """Tests sigmoid followed by an element-wise multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * 2.0


# ========== Tanh Node Test Models ==========


class TanhTestModel(torch.nn.Module):
    """Tests tanh on a 2D tensor (outputs in [-1,1])."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class TanhInExpressionModel(torch.nn.Module):
    """Tests tanh followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) + 1.0


# ========== Relu Node Test Models ==========


class ReluTestModel(torch.nn.Module):
    """Tests relu with mixed positive/negative inputs (negatives clipped to 0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReluAllNegativeModel(torch.nn.Module):
    """Tests relu with all-negative inputs (entire output should be zeros)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ReluInExpressionModel(torch.nn.Module):
    """Tests relu followed by an element-wise multiply."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * 2.0


# ========== Abs Node Test Models ==========


class AbsTestModel(torch.nn.Module):
    """Tests abs with mixed positive/negative inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class AbsAllNegativeModel(torch.nn.Module):
    """Tests abs with all-negative inputs (output equals negation of input)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class AbsInExpressionModel(torch.nn.Module):
    """Tests abs followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x) + 1.0


# ========== Neg Node Test Models ==========


class NegTestModel(torch.nn.Module):
    """Tests neg with mixed positive/negative inputs (sign is flipped)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x)


class NegAllPositiveModel(torch.nn.Module):
    """Tests neg with all-positive inputs (output should be all negative)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x)


class NegInExpressionModel(torch.nn.Module):
    """Tests neg followed by an element-wise add."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x) + 1.0


# ========== Equal Node Test Models ==========


class EqualTestModel(torch.nn.Module):
    """Tests element-wise equality against a stored weight with discrete values."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randint(0, 3, (5, 5)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.weight).to(torch.float32)


class EqualBroadcastModel(torch.nn.Module):
    """Tests equality with broadcasting (1D input broadcasts against 2D weight)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randint(0, 3, (5, 5)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.weight).to(torch.float32)


class EqualWithConstantModel(torch.nn.Module):
    """Tests equality against an inline constant (exercises Equal + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
        return (x == constant).to(torch.float32)


# ========== Clip Node Test Models ==========


class ClipTestModel(torch.nn.Module):
    """Tests clip (clamp) with both min and max bounds."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=-0.5, max=0.5)


class ClipMinOnlyTestModel(torch.nn.Module):
    """Tests clip with only a min bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


class ClipMaxOnlyTestModel(torch.nn.Module):
    """Tests clip with only a max bound."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=0.5)


# ========== Pow Node Test Models ==========


class PowTestModel(torch.nn.Module):
    """Tests element-wise power against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)) + 1.0)  # positive exponents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x ** self.weight


class PowByConstantModel(torch.nn.Module):
    """Tests power by an inline constant tensor (exercises Pow + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([2.0, 3.0, 0.5]).to(x.device)
        return x ** constant


# ========== Where Node Test Models ==========


# ========== Max Node Test Models ==========


class MaxTestModel(torch.nn.Module):
    """Tests element-wise maximum against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, self.weight)


class MaxWithConstantModel(torch.nn.Module):
    """Tests element-wise maximum against an inline constant (ONNX Max + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(x.device)
        return torch.maximum(x, constant)


# ========== Min Node Test Models ==========


class MinTestModel(torch.nn.Module):
    """Tests element-wise minimum against a stored weight tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x, self.weight)


class MinWithConstantModel(torch.nn.Module):
    """Tests element-wise minimum against an inline constant (ONNX Min + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0]).to(x.device)
        return torch.minimum(x, constant)


class WhereTestModel(torch.nn.Module):
    """Tests element-wise where with condition derived from input comparison, selecting from weight buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("true_vals", torch.rand((5, 5)) + 1.0)
        self.register_buffer("false_vals", torch.rand((5, 5)) - 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, self.true_vals, self.false_vals)


class WhereSelfSelectModel(torch.nn.Module):
    """Tests where selecting between input x and its negation (abs-like pattern)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, x, -x)


class WhereWithConstantModel(torch.nn.Module):
    """Tests where with inline constant tensors for both branches (exercises Where + Constant nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.tensor([1.0, 2.0, 3.0]).to(x.device)
        neg = torch.tensor([-1.0, -2.0, -3.0]).to(x.device)
        return torch.where(x > 0.0, pos, neg)


# ========== Concat Node Test Models ==========


class ConcatAxis0Model(torch.nn.Module):
    """Tests concatenation along axis 0 (stacking rows)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=0)


class ConcatAxis1Model(torch.nn.Module):
    """Tests concatenation along axis 1 (stacking columns)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=1)


class ConcatThreeTensorsModel(torch.nn.Module):
    """Tests concatenation of three tensors along axis 0."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight1", torch.rand((2, 4)))
        self.register_buffer("weight2", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight1, self.weight2, x], dim=0)


class ConcatNegativeAxisModel(torch.nn.Module):
    """Tests concatenation with negative axis (-1 = last axis)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=-1)


class ConcatInExpressionModel(torch.nn.Module):
    """Tests concat followed by an element-wise operation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((3, 4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.weight, x], dim=0) * 2.0


# ========== Softmax Node Test Models ==========


class SoftmaxTestModel(torch.nn.Module):
    """Tests softmax along the last dimension (default axis=-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)


class SoftmaxDim0TestModel(torch.nn.Module):
    """Tests softmax along dim=0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=0)


# ========== LessOrEqual Node Test Models ==========


class LessOrEqualTestModel(torch.nn.Module):
    """Tests element-wise less-than-or-equal comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x <= self.weight).to(torch.float32)


class LessOrEqualWithConstantModel(torch.nn.Module):
    """Tests less-than-or-equal against an inline constant (ONNX Constant + LessOrEqual nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x <= constant).to(torch.float32)


# ========== GreaterOrEqual Node Test Models ==========


class GreaterOrEqualTestModel(torch.nn.Module):
    """Tests element-wise greater-than-or-equal comparison against a stored weight."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x >= self.weight).to(torch.float32)


class GreaterOrEqualWithConstantModel(torch.nn.Module):
    """Tests greater-than-or-equal against an inline constant (ONNX Constant + GreaterOrEqual nodes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constant = torch.tensor([0.25, 0.5, 0.75]).to(x.device)
        return (x >= constant).to(torch.float32)


# ========== Not Node Test Models ==========


class NotTestModel(torch.nn.Module):
    """Tests logical NOT on a boolean tensor derived from comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(x > 0.5).float()


# ========== And Node Test Models ==========


class AndTestModel(torch.nn.Module):
    """Tests logical AND between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(x > 0.3, self.weight > 0.3).float()


# ========== Or Node Test Models ==========


class OrTestModel(torch.nn.Module):
    """Tests logical OR between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_or(x > 0.7, self.weight > 0.7).float()


# ========== Xor Node Test Models ==========


class XorTestModel(torch.nn.Module):
    """Tests logical XOR between two boolean tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.rand((5, 5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logical_xor(x > 0.5, self.weight > 0.5).float()


# ========== Trilu Node Test Models ==========


class TrilTestModel(torch.nn.Module):
    """Tests lower triangular selection (tril, diagonal=0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x)


class TriuTestModel(torch.nn.Module):
    """Tests upper triangular selection (triu, diagonal=0)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu(x)


class TrilDiagonalTestModel(torch.nn.Module):
    """Tests tril with a positive diagonal offset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril(x, diagonal=1)


class TriuDiagonalTestModel(torch.nn.Module):
    """Tests triu with a negative diagonal offset."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu(x, diagonal=-1)
