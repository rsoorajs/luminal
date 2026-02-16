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
