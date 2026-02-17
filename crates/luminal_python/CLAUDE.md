A couple of short things to keep in mind
1. If you want to run tests:
   - `./run_test.sh` - runs tests with the native backend
   - `./run_tests_cuda.sh` - runs tests with the CUDA backend

## Testing Best Practices

### Overview
The luminal_python crate provides a bridge between PyTorch models and the luminal library via ONNX. Tests should verify this integration end-to-end by testing the actual user workflow: PyTorch model → torch.compile → luminal backend.

### Test Pattern (CORRECT)

All tests should follow this standard pattern:

```python
def test_operation():
    """Brief description of what operation is being tested."""
    # 1. Instantiate PyTorch model
    model: torch.nn.Module = OperationTestModel()

    # 2. Compile with luminal backend
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)

    # 3. Create test input
    x: torch.Tensor = torch.tensor([...])  # or torch.rand(...)

    # 4. Run both original and compiled versions
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)

    # 5. Verify outputs match
    assert torch.allclose(output, original)
```

### Test Models

- Define test model classes in `tests/test_models.py`
- Each model should be a simple `torch.nn.Module` that demonstrates one operation or pattern
- Use clear, descriptive class names (e.g., `AddTestModel`, `TransposeTestModel`)
- Include docstrings explaining what the model tests

Example:
```python
class AddTestModel(torch.nn.Module):
    """Tests element-wise addition."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x
```

### What NOT to Do

**❌ DO NOT create ONNX files directly in tests:**
```python
# WRONG - bypasses the PyTorch integration
model_path = create_onnx_model(...)
graph_result = luminal.process_onnx(model_path, backend='native')
```

**✓ DO create PyTorch models and use torch.compile:**
```python
# CORRECT - tests actual user workflow
model: torch.nn.Module = MyTestModel()
model_compiled = torch.compile(model, backend=luminal_backend)
```

### Rationale

- **End-to-end testing**: Tests verify the complete PyTorch → ONNX → luminal pipeline
- **User-facing API**: Tests use the same API that users will use (torch.compile)
- **Correctness**: Comparing compiled vs original PyTorch output ensures correctness
- **Maintainability**: Consistent pattern across all tests makes the codebase easier to understand
- **Simplicity**: No manual ONNX file creation, no tempfile cleanup, no numpy comparisons

### Special Cases

**Testing constants:**
Use inline tensor literals in the forward method - PyTorch exports these as ONNX Constant nodes:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    constant = torch.tensor([1.0, 2.0, 3.0])
    return x + constant
```

**Testing type casts:**
Use `.to(dtype)` method - PyTorch exports these as ONNX Cast nodes:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)
```

**Testing complex operations:**
Chain operations naturally in PyTorch - ONNX export handles the conversion:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    transposed = x.transpose(0, 1)
    scaled = transposed * 2.0
    return scaled + 1.0
```
