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
