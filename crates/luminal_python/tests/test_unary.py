from typing import Callable
import torch
import torch._dynamo
from test_models import (
    SigmoidTestModel, SigmoidInExpressionModel,
    TanhTestModel, TanhInExpressionModel,
    ReluTestModel, ReluAllNegativeModel, ReluInExpressionModel,
    AbsTestModel, AbsAllNegativeModel, AbsInExpressionModel,
)
from luminal import luminal_backend

# ── Sigmoid ──────────────────────────────────────────────────────────────────

def test_sigmoid(device):
    model = SigmoidTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1   # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_sigmoid_in_expression(device):
    model = SigmoidInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

# ── Tanh ─────────────────────────────────────────────────────────────────────

def test_tanh(device):
    model = TanhTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_tanh_in_expression(device):
    model = TanhInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

# ── Relu ─────────────────────────────────────────────────────────────────────

def test_relu(device):
    model = ReluTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1   # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_relu_all_negative(device):
    model = ReluAllNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = -torch.rand((5, 5), device=device)           # all negative -> output all zeros
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_relu_in_expression(device):
    model = ReluInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

# ── Abs ──────────────────────────────────────────────────────────────────────

def test_abs(device):
    model = AbsTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1   # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_abs_all_negative(device):
    model = AbsAllNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = -torch.rand((5, 5), device=device)           # all negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)

def test_abs_in_expression(device):
    model = AbsInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)
