# ONNX Op Difficulty Analysis

This document estimates implementation difficulty for ONNX ops not yet supported in
`crates/luminal_python/rust/src/ops_parse/`.

Difficulty categories:
- **Easy** — Luminal frontend API already has the primitives; bridging is <~30 lines, straightforward mapping
- **Medium** — Primitives mostly exist but require non-trivial logic (constant folding for runtime inputs, multi-output handling, attribute parsing, broadcasting gymnastics)
- **Hard** — No direct primitive, requires new backend ops, complex parsing, or fundamentally scatter semantics which luminal doesn't have

---

## Summary Table

| Op              | Difficulty    | Notes |
|-----------------|---------------|-------|
| Ceil            | ✅ Done       | Already in `ops_parse/unary.rs` |
| GatherElements  | Easy          | 1:1 map to `gather_elements()` |
| LayerNorm       | Easy          | `layer_norm()` + scale/bias |
| Gemm            | Easy          | `matmul` + optional transpose + alpha/beta/C |
| IsNaN           | Easy          | `x.ne(x)` (IEEE 754 property) |
| Expand          | Easy          | `broadcast_to()` util; shape is a constant-folded input |
| Split           | Medium        | Multi-output; slicing per segment |
| Slice           | Medium        | Constant-fold runtime inputs; step≠1 tricky |
| TopK            | Medium        | Two outputs; `k` constant-folded; two existing frontend fns |
| Erf             | Medium        | Polynomial approximation with existing ops |
| OneHot          | Medium        | arange + broadcast + cond; values constant-folded |
| Conv            | Hard          | Grouped conv; auto_pad; unfold-based bridge |
| Einsum          | Hard          | Equation parser + arbitrary composition |
| ScatterElements | Hard          | No scatter primitive; workaround limited |
| ScatterND       | Hard          | As above, but harder |
| Resize          | Hard          | No interpolation primitive; many modes |

---

## Already Implemented

### Ceil
Already present in `ops_parse/unary.rs`. No work needed.

---

## Easy

### GatherElements
- **Mapping**: `GraphTensor::gather_elements(indexes, axis)`
- **Why Easy**: 1:1 mapping; extract the `axis` attribute, get the two inputs, call the function.
- **Caveats**: Negative axis normalization needed.

### LayerNorm
- **Mapping**: `GraphTensor::layer_norm(axes, epsilon)` exists in the luminal frontend; scale and bias are multiply + add.
- **Why Easy**: `output = input.layer_norm(axes, eps) * scale + bias`; both scale and bias are optional.
- **Caveats**: Need to derive which axes to normalize from the `axis` attribute (normalize over `[axis, axis+1, ...]`).

### Gemm
- **Mapping**: Composes `matmul` + optional `permute` + scalar multiply + bias add.
- **Why Easy**: `Y = alpha * A' * B' + beta * C`; all building blocks exist (`matmul`, `permute`, scalar `*`).
- **Caveats**: `transA`, `transB`, `alpha`, `beta` attributes; `C` (bias) is optional; `alpha`/`beta` default to 1.0.

### IsNaN
- **Mapping**: IEEE 754 guarantees `NaN ≠ NaN`; implement as `x.ne(x)` returning a Bool tensor.
- **Why Easy**: One-liner once you know the trick.
- **Caveats**: Result is `DType::Bool`; caller may need to cast to F32.

### Expand
- **Mapping**: `broadcast_to(tensor, target_shape)` utility exists in `util.rs`.
- **Why Easy**: The broadcast utility already handles numpy-style broadcasting.
- **Caveats**: Target shape comes as a runtime **input tensor** (not an attribute); requires constant folding via the `known_values` map. If shape is dynamic at graph-build time it won't work.

---

## Medium

### Split
- **Mapping**: `GraphTensor::slice_along(range, axis)` for each segment.
- **Why Medium**: Produces **multiple output tensors** (not just one); split sizes come from an optional input tensor or attribute; must iterate and slice each output separately, inserting all of them into the `tensors` map.
- **Caveats**: `split` input may be absent (equal split); negative axis; unequal splits.

### Slice
- **Mapping**: `GraphTensor::slice_along(range, axis)` applied per axis.
- **Why Medium**: All of `starts`/`ends`/`axes`/`steps` are **runtime input tensors**, requiring constant folding for all of them; step ≠ 1 is not directly supported by `slice_along` (would need `unfold` or manual gather); negative indices need normalization.
- **Caveats**: Steps other than 1 are hard and rare; most real-world usage has constant `starts`/`ends`.

### TopK
- **Mapping**: `topk_indexes(k, axis, largest)` and `topk_values(k, axis, largest)` exist in the luminal frontend.
- **Why Medium**: ONNX TopK produces **two outputs** (values + indices); `k` is an input tensor requiring constant folding; need to handle `largest` and `sorted` attributes; two-output handling in the dispatch function.
- **Caveats**: `topk_values` is newly added and needs verification; two-output wiring in the op parser.

### Erf
- **Mapping**: No native `erf` primitive in luminal; must approximate with existing ops.
- **Why Medium**: A standard polynomial approximation (e.g., Abramowitz & Stegun) or tanh-based approximation is implementable with `tanh`, `pow`, `mul`, `add`; there is an accuracy tradeoff to consider.
- **Implementation sketch**: `erf(x) ≈ tanh(√(2/π) * (x + 0.044715·x³))` or a 7-term Chebyshev polynomial.
- **Caveats**: No single primitive; must pick an approximation and test accuracy vs. the PyTorch reference.

### OneHot
- **Mapping**: Compose `arange(depth)` + broadcasting + comparison + `cond`.
- **Why Medium**: `indices` can be multi-dimensional; `values` input gives `(off_value, on_value)`; need careful shape manipulation to broadcast `arange` against `indices`.
- **Implementation sketch**: `arange(depth).expand(...) == indices.unsqueeze(-1).expand(...)` → bool mask → `.cond(on_value, off_value)`.
- **Caveats**: `values` is a 2-element input tensor (needs constant folding); the `axis` attribute controls where the one-hot dimension is inserted.

---

## Hard

### Conv
- **Mapping**: `unfold(kernel, strides, dilation)` exists; `luminal_nn::ConvND` exists but lives in a separate crate.
- **Why Hard**: ONNX Conv has `group` (depthwise/grouped convolution); `auto_pad` modes (`SAME_UPPER`, `SAME_LOWER`, `VALID`); asymmetric padding via the `pads` attribute; optional bias. Manually bridging all of this with `unfold` is complex. `group > 1` requires splitting input/weight channels and doing separate matmuls.
- **Caveats**: `group = 1` (standard conv) is Medium; `group > 1` bumps it to Hard.

### Einsum
- **Mapping**: No `einsum` primitive; must parse the equation string and compose luminal ops.
- **Why Hard**: Requires implementing a mini-parser for Einstein summation notation; must handle arbitrary contractions, batch dims, and implicit output. Can in principle be decomposed into `permute` + `matmul` + `sum`, but the general case requires significant engineering, plus edge cases (ellipsis, repeated indices, etc.).
- **Caveats**: A partial implementation covering common 2-operand cases is Medium; the general case is Hard.

### ScatterElements
- **Mapping**: No scatter primitive in luminal; this is the inverse of `gather_elements`.
- **Why Hard**: Scatter requires writing values to indexed positions — fundamentally a different access pattern from gather. Luminal's graph is built on read-side indexing (Iota-based). Implementing scatter requires either: (1) a new backend kernel, or (2) a mask-and-sum workaround (`where` + broadcast + reduce) that is O(output_size × index_size) and only correct for non-overlapping indices.
- **Caveats**: The mask-based workaround may work for small tensors with non-overlapping indices; production-quality scatter needs a new primitive.

### ScatterND
- **Mapping**: Generalization of ScatterElements to N-dimensional indices.
- **Why Hard**: Even harder than ScatterElements; multi-dim index tensor; same lack of a scatter primitive; index shape can have arbitrary batch dims.
- **Caveats**: Same workaround applies but is even more complex to implement correctly.

### Resize
- **Mapping**: No interpolation primitive in luminal.
- **Why Hard**: ONNX Resize supports nearest, linear, and cubic interpolation; many coordinate transformation modes (`half_pixel`, `pytorch_half_pixel`, `align_corners`, `asymmetric`, `tf_crop_and_resize`); scales vs. sizes input modes. Would require implementing full interpolation logic from scratch using `gather`, `arange`, and arithmetic.
- **Caveats**: Nearest-neighbor-only support with a fixed coordinate transformation mode is achievable at Medium difficulty; linear/cubic is Hard.
