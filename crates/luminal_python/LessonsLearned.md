# Lessons Learned

This file documents hard bugs encountered in this codebase, their root causes, and principles
to prevent similar issues in the future.

---

## 2026-02-24 — Intermittent CUDA Backend Failures: Embed False Match + Batched Matmul Dimension Drop

### Background: Why the Failures Were Intermittent

Both bugs only appeared on roughly 50% of test runs. The source of non-determinism is
`FxHashMap` (a fixed-seed hash map). The egglog optimizer's `SerializedEGraph::new` builds
`Vec<NodeId>` orderings for each e-class by iterating a `FxHashMap`, producing non-deterministic
node orderings. `random_initial_choice()` in `src/egglog_utils/mod.rs` then randomly picks one
e-node per e-class as the starting representation for the profiling phase. The combination means
some runs pick a correct kernel and some pick a broken one from the same e-class.

**Lesson**: When a test fails intermittently at a roughly 50% rate, suspect the egglog extractor
choosing between two e-nodes in the same e-class — one correct, one broken. The fix is always in
the broken e-node's rewrite rule.

---

### Bug 1: `test_gather_elements` — KernelEmbed and RowEmbed False Match

**Files changed**:
- `crates/luminal_cuda/src/kernel/hlir.rs` (KernelEmbed, 4 rules)
- `crates/luminal_cuda/src/block/ops.rs` (RowEmbed, 4 rules)

#### What happened

`gather_elements` (axis-aware gather) decomposes into a flat gather by computing:

```
flat_idx = Add(
    Mul(indices, stride[axis]),
    Mul(Expand(Iota(dim_size)), stride[non_axis])
)
```

`KernelEmbed` and `RowEmbed` are optimized embedding lookup kernels. A genuine embedding
lookup produces:

```
flat_idx = Add(
    Mul(Cast(token_ids), embed_dim),
    Iota(embed_dim)              ← bare Iota, the position within an embedding row
)
```

The egglog rewrite rules for both ops matched `Add(?mul_result, ?iota_result)` where
`?iota_result` was **unconstrained** — it could bind to anything, including
`Mul(Expand(Iota(n)), stride)` from `gather_elements`. This created a `KernelEmbed`/`RowEmbed`
node in the same e-class as the `Gather` node. When the extractor picked it, `build_payload`
called `flatten_mul_strides(range, token_stride)` which asserted `range.len() == token_stride.len()`:
- `range` came from `RemoveNthFromEnd(idx_shape, 0)` → length 1
- `token_stride` came from the indices strides → length 2
- Assertion failed → panic.

#### The fix

Add `(= ?iota_result (Iota ?iota_expr ?iota_range))` to all 8 rules, requiring the positional
component to be a bare `Iota` node:

```egglog
(= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
(= ?iota_result (Iota ?iota_expr ?iota_range))   ← added
(= ?mul_result (Mul ...))
```

#### Investigation note

The initial plan correctly identified `KernelEmbed` as faulty, but missed `RowEmbed`. The two
ops are structurally identical but live in different parts of the codebase (`kernel/` vs
`block/`). The second bug was only discovered when the backtrace pointed to
`RowEmbed::build_payload` instead of `KernelEmbed::compile`. Always search for sibling
implementations when fixing a pattern-matching bug in one op.

---

### Bug 2: `test_matmul_batched` — CuBlasLt Drops Batch Dimension

**Files changed**:
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_RmRm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_RmCm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_CmRm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_CmCm_rewrite.egg`

#### What happened

The luminal frontend decomposes `(2,3,4) @ (2,4,5)` into:

```rust
let w = rhs.permute((0, 2, 1));       // (2,4,5) → (2,5,4)
let mul = self.expand_dim(2, d)        // (2,3,4) → (2,3,5,4)
        * w.expand_dim(1, b);          // (2,5,4) → (2,3,5,4)
mul.sum(3)                             // → (2,3,5), correct out_shape
```

All four cublaslt rewrite rules extracted `m` and `n` from the output shape using
`nth_from_end`, which succeeds for any rank:

```egglog
(= ?m (nth_from_end ?out_shape 1))
(= ?n (nth_from_end ?out_shape 0))
```

For `out_shape = [2, 3, 5]`: `?m = 3`, `?n = 5`. The batch dim `2` is never extracted or
stored. The rules also validated stride patterns using `nth_from_end` on the stride arrays —
but for this batched case, **all stride checks coincidentally passed** because the last three
strides of the 4D expanded tensors happened to satisfy the 2D row/column-major patterns.

The resulting `CuBlasLt` node had `output_size() = m * n = 15`. The batch dimension was
silently discarded. The runtime allocated a 15-element output buffer, cuBLAS wrote a 3×5
result, and the test got back 15 values instead of 30.

#### The fix

Add `(= (len ?out_shape) 2)` to all 4 rules:

```egglog
(= (len ?out_shape) 2)    ← added: cuBLAS is 2D only
(= ?m (nth_from_end ?out_shape 1))
(= ?n (nth_from_end ?out_shape 0))
```

`len` counts elements in the `ECons`-list shape. With this constraint, any `Sum` node with a
3D+ output shape (batched matmul) is not matched by cuBLAS rules and falls through to
`KernelSumReduce + KernelMul` (or the tiling block ops), which correctly use
`out_shape.iter().product()` for their output sizes.

Note: `TileMatmulSplitK` and `TileMatmulFullSplit` do NOT need this fix — their `output_size()`
already returns `untiled_range.iter().product()` which includes all dimensions.

---

### General Principle: Always Constrain Shape Rank in Egglog Rules

Both bugs share the same structural cause: **egglog rewrite rules that used `nth_from_end` to
extract dimensions from a shape list without constraining the list's length.** Since
`nth_from_end` silently succeeds for any list with enough trailing elements, rules written for
2D tensors accidentally matched higher-rank tensors.

**Rule for writing egglog rewrite rules in this codebase**:

> If a rule is designed for a specific tensor rank, always add an explicit
> `(= (len ?shape) N)` constraint. If a rule is designed to handle arbitrary ranks but an
> op's output only covers a subset of dimensions (like cuBLAS covering only the last 2),
> that is a correctness bug — either implement strided batched cuBLAS or add the rank
> constraint and fall back to a kernel that handles all dimensions.

---

### Debugging Intermittent CUDA Failures: Effective Approach

The investigation used extensive `eprintln!` debug logging to trace which kernels were compiled
vs. skipped. Key observations:

1. **In the passing case**: `KernelSumReduce::compile()` was called, kernels were allocated.
2. **In the failing case**: `KernelSumReduce::compile()` was never called, yet output was produced.

This asymmetry pointed to a `HostOp` path (cuBLAS) executing instead of the `KernelOp` path,
which narrowed the search to cublaslt rewrite rules. The HLIR-level `SumReduce::to_egglog` log
confirmed the correct HLIR node existed — the bug was in the e-graph optimization choosing
a different (broken) e-node from the same e-class.

**Effective debug strategy for egglog non-determinism bugs**:
1. Add logging at compile time for each kernel type (`KernelFoo::compile`, `HostFoo::execute`)
2. Compare passing vs. failing runs to see which kernels are/aren't invoked
3. The missing kernel's e-class contains a broken alternative — find it via the egglog rewrite rules
4. Check the op that *is* executing — its `output_size()` reveals what's wrong with the false match

---

## 2026-02-25 — OneHot Test Panic: Cast(Int→F32) Produces Int Output

### What the symptom was

`test_onehot` panicked at `src/hlir.rs:1625` in `get_f32()`: the output buffer was
`NativeData::Int` instead of the expected `NativeData::F32`.

### What the actual root cause was

The Cast parser's `* 1.0` workaround for `Int → F32` casts used `input * one_expanded`
(Int GraphTensor on the left, F32 constant on the right). However, `Mul for GraphTensor`
always uses `self.dtype` (the **left** operand's dtype) for the result, and the native
runtime's `Mul::execute` dispatches on the **first** input's `NativeData` variant. So
`Int * F32` produced `DType::Int` / `NativeData::Int` — the exact opposite of the intended
F32 output.

### Why it was hard to find

1. **The OneHot parser was a red herring**: The initial plan assumed the OneHot ONNX node
   was being parsed, but `torch.onnx.export` decomposes `one_hot` into
   `Unsqueeze → Equal → Cast(Bool→Int) → Cast(Int→F32)`. The OneHot parser was never called.
2. **The `* 1.0` workaround looked correct**: It was used successfully in many other parsers,
   but those all had F32 inputs (where `F32 * F32 = F32`). The Int→F32 case was the only
   path where the left operand was Int.
3. **Operand order matters silently**: Nothing warns about mixed-dtype Mul — it just takes
   the left operand's dtype.

### The fix

In `ops_parse/unary.rs` `parse_cast_node`, split the combined condition into two cases:
- **No-op cast** (`cast_result.id == input.id`): `input * one_expanded` — preserves dtype
- **Int source** (`input.dtype == DType::Int`): `one_expanded * input` — F32 on the left
  ensures F32 output

### General principle

**In luminal, binary op dtype is always the LEFT operand's dtype.** When constructing
`GraphTensor * constant_float(1.0)` for type materialization, always put the operand
whose dtype you want to preserve on the LEFT side. When converting Int→F32, the F32
constant must be the left operand.
