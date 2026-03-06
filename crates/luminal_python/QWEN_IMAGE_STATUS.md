# Qwen-Image Diffusion Model Support Status

## What Works

### Transformer (QwenImageTransformer2DModel) — FULLY WORKING
- Tiny config (1 layer, 4 heads, dim=64): PASS, max_diff < 5e-7
- Small config (2 layers, 8 heads, dim=256): PASS
- Works on native backend; untested on CUDA but no expected blockers

### ONNX Export Approach
The transformer uses complex-valued RoPE (`torch.view_as_complex`) which is not ONNX-exportable. The solution is `TransformerONNXWrapper` in `tests/test_qwen_image.py` which:
1. Pre-computes RoPE frequencies as real cos/sin buffers
2. Replaces the attention processor (`QwenDoubleStreamAttnProcessor2_0`) with `RealRoPEAttnProcessor` that applies RoPE using real-valued rotation
3. Inlines the forward pass to avoid the complex RoPE code path

This wrapper is functionally equivalent to the original model for fixed image/text shapes.

### New ONNX Ops Added
| Op | File | Notes |
|---|---|---|
| Conv | `rust/src/ops_parse/convolution.rs` | N-dim, group=1 only, unfold+matmul approach, with bias |
| Pad | `rust/src/ops_parse/movement.rs` | Constant mode, optional axes |
| Resize | `rust/src/ops_parse/movement.rs` | Nearest-neighbor, integer scale factors only |
| Tile | `rust/src/ops_parse/movement.rs` | Repeat via expand_dim+merge_dims |
| Gelu | `rust/src/dispatch.rs` | Maps to `GraphTensor::gelu()` |
| GroupNormalization | `rust/src/ops_parse/unary.rs` | Reshape→LayerNorm→reshape→scale+bias |
| ReduceL2 | `rust/src/dispatch.rs` | Decomposed to `sqrt(sum(x^2))` |

## What Doesn't Work

### VAE (AutoencoderKLQwenImage) — BLOCKED

The VAE decoder uses **Conv3d** (5D tensors: `[B, C, T, H, W]`). This triggers a panic:

```
thread panicked at src/shape/tracker.rs:256:
not yet implemented: Need CuTE-style nested dims for this!
```

#### Root Cause
`ShapeTracker::merge_dims` cannot handle the dimension manipulations that the Conv unfold+permute+merge sequence produces on 5D tensors. The unfold creates a 10D tensor (5 window dims + 5 kernel dims), and the subsequent permute+merge sequence hits the unimplemented CuTE-style nested dim path in the ShapeTracker.

This is the **same fundamental limitation** documented in MEMORY.md: `ShapeTracker::merge_dims` has a `todo!()` for certain non-trivial dim combinations.

#### What Would Fix It
Resolving the `todo!()` at `src/shape/tracker.rs:256` to handle nested/non-contiguous dimension merging. Once that works, the Conv3d path should function because:
- The ONNX export itself works fine (produces valid simplified ONNX with ops: Add, Clip, Conv, Div, Expand, MatMul, Mul, ReduceL2, Reshape, Sigmoid, Slice, Softmax, Squeeze, Tile, Transpose, Unsqueeze)
- All these ops are now implemented in the parser
- The Conv parser logic is correct for N-dim (verified working on 2D via the transformer)

#### Additional VAE Issues (Lower Priority)
1. **`nn.Upsample` not ONNX-exportable** — Already solved: `_OnnxFriendlyUpsample` in `test_qwen_image.py` replaces it with `repeat_interleave`
2. **Conv group > 1** — The Conv parser asserts `group=1`. If the VAE uses grouped convolutions, this would need extending. Implementation approach: split input channels into groups, convolve each group separately, concatenate.
3. **ConvTranspose** — If the VAE decoder uses transposed convolution, that's a separate op not yet implemented.

### Full Pipeline Test — BLOCKED BY VAE
The pipeline test (text encoder → transformer → scheduler → VAE decode) can't run because the VAE is blocked. The transformer portion works, and the scheduler runs in pure Python, so once the VAE works the pipeline should be straightforward.

## File Inventory

| File | Action | Description |
|---|---|---|
| `pyproject.toml` | Modified | Added `diffusers>=0.35.0`, `onnxsim` to dev deps |
| `rust/src/ops_parse/convolution.rs` | Created | Conv ONNX op parser |
| `rust/src/ops_parse/mod.rs` | Modified | Added `pub mod convolution` |
| `rust/src/dispatch.rs` | Modified | Added Conv, Pad, Resize, Tile, Gelu, GroupNormalization, ReduceL2 dispatch |
| `rust/src/ops_parse/movement.rs` | Modified | Added parse_pad_node, parse_resize_node, parse_tile_node |
| `rust/src/ops_parse/unary.rs` | Modified | Added parse_group_norm_node |
| `tests/test_qwen_image.py` | Created | Transformer tests (pass) + VAE test (xfail) |

## Running Tests
```bash
# All Qwen-Image tests
uv run pytest tests/test_qwen_image.py -v

# Just the working transformer tests
uv run pytest tests/test_qwen_image.py -v -k "transformer"

# Full existing test suite (should still pass)
uv run pytest tests/test_hlir_ops.py tests/test_unary.py -v
```
