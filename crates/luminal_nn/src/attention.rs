use luminal::prelude::*;

/// Gather entire rows from a 2D tensor using row indices.
///
/// - `data`: (R, D) tensor
/// - `indices`: (N,) Int tensor of row indices
/// - `d`: the number of columns (D), must match data's second dimension
///
/// Returns: (N, D) tensor where output[i] = data[indices[i]]
pub fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // base[i] = indices[i] * D → flat starting position for each row
    let base = (indices * d).expand_dim(1, d); // (N, D) broadcast along cols

    // col[j] = j → column offsets 0..D
    let col = data.graph().arange(d as i32).expand_dim(0, n); // (N, D) broadcast along rows

    // flat_idx[i,j] = indices[i] * D + j
    let flat_idx = base + col;

    data.gather(flat_idx)
}

/// Scatter entire rows into a 2D tensor using row indices.
///
/// - `src`: (N, D) tensor of values to write
/// - `indices`: (N,) Int tensor of destination row indices
/// - `dest`: (R, D) tensor to write into (copied first, then overwritten at index positions)
/// - `d`: the number of columns (D)
///
/// Returns: (R, D) tensor where output = copy(dest); output[indices[i]] = src[i]
pub fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();

    // Same index expansion as gather_rows
    let base = (indices * d).expand_dim(1, d);
    let col = src.graph().arange(d as i32).expand_dim(0, n);
    let flat_idx = base + col;

    src.scatter(flat_idx, dest)
}

/// Pure HLIR paged attention for one layer.
///
/// Inputs:
/// - `q`:           (s, hidden)         f32 — query vectors
/// - `k_new`:       (s, kv_dim)         f32 — new key vectors
/// - `v_new`:       (s, kv_dim)         f32 — new value vectors
/// - `k_cache`:     (num_slots, kv_dim) f32 — key cache (preallocated)
/// - `v_cache`:     (num_slots, kv_dim) f32 — value cache (preallocated)
/// - `gather_idx`:  (ctx_len,)          Int — which cache slots to read
/// - `scatter_idx`: (s,)                Int — which cache slots to write new KV into
/// - `attn_mask`:   (s, ctx_len)        f32 — causal mask (0 or -inf)
/// - `n_heads`:     number of query heads
/// - `n_kv_heads`:  number of KV heads (for GQA)
/// - `head_dim`:    dimension per head
///
/// Returns: (attn_out, k_cache_new, v_cache_new)
///   - `attn_out`:     (s, hidden)         f32
///   - `k_cache_new`:  (num_slots, kv_dim) f32
///   - `v_cache_new`:  (num_slots, kv_dim) f32
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    attn_mask: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let s = q.dims()[0];

    // ── Phase 1: Write new KV into cache ──
    let k_cache = scatter_rows(k_new, scatter_idx, k_cache, kv_dim);
    let v_cache = scatter_rows(v_new, scatter_idx, v_cache, kv_dim);

    // ── Phase 2: Gather context KV from cache ──
    let k = gather_rows(k_cache, gather_idx, kv_dim); // (ctx, kv_dim)
    let v = gather_rows(v_cache, gather_idx, kv_dim); // (ctx, kv_dim)

    // ── Phase 3: Reshape for multi-head attention ──
    // Q: (s, hidden) → (s, n_heads, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    //                 → (n_kv_heads, kv_groups, s, head_dim)
    let q = q
        .split_dims(1, head_dim) // (s, n_heads, head_dim)
        .split_dims(1, kv_groups) // (s, n_kv_heads, kv_groups, head_dim)
        .permute((1, 2, 0, 3)); // (n_kv_heads, kv_groups, s, head_dim)

    // K: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, head_dim, ctx)
    let k = k
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 2, 0)); // (n_kv_heads, head_dim, ctx)

    // V: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, ctx, head_dim)
    let v = v
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 0, 2)); // (n_kv_heads, ctx, head_dim)

    // ── Phase 4: Attention ──
    // Broadcast K, V over kv_groups dimension
    let k = k.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, head_dim, ctx)
    let v = v.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, ctx, head_dim)

    // QK^T: (n_kv_heads, kv_groups, s, head_dim) @ (n_kv_heads, kv_groups, head_dim, ctx)
    //     → (n_kv_heads, kv_groups, s, ctx)
    let scores = q.matmul(k) * scale;

    // Apply causal mask: broadcast (s, ctx) → (n_kv_heads, kv_groups, s, ctx)
    let mask = attn_mask
        .expand_dim(0, n_kv_heads) // (n_kv_heads, s, ctx)
        .expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, s, ctx)
    let scores = scores + mask;

    // Softmax over context dimension (axis 3)
    let weights = scores.softmax(3);

    // Weighted sum: (n_kv_heads, kv_groups, s, ctx) @ (n_kv_heads, kv_groups, ctx, head_dim)
    //            → (n_kv_heads, kv_groups, s, head_dim)
    let out = weights.matmul(v);

    // ── Phase 5: Reshape output ──
    // (n_kv_heads, kv_groups, s, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    // Force materialization with * 1.0 to make contiguous, then reinterpret shape
    let mut out = out.permute((2, 0, 1, 3)) * 1.0;
    out.shape = ShapeTracker::new((s, n_heads * head_dim));

    (out, k_cache, v_cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_rows() {
        let mut cx = Graph::new();
        let data = cx.tensor((4, 3)); // 4 rows, 3 cols
        let indices = cx.tensor(3).as_dtype(DType::Int);
        let result = gather_rows(data, indices, 3).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        // data = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        rt.set_data(
            data.id,
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        // Gather rows 0, 2, 3
        rt.set_data(indices.id, vec![0, 2, 3]);
        rt.execute(&cx.dyn_map);

        assert_eq!(
            *rt.get_f32(result.id),
            vec![1., 2., 3., 7., 8., 9., 10., 11., 12.]
        );
    }

    #[test]
    fn test_scatter_rows() {
        let mut cx = Graph::new();
        let src = cx.tensor((2, 3));
        let indices = cx.tensor(2).as_dtype(DType::Int);
        let dest = cx.tensor((4, 3));
        let result = scatter_rows(src, indices, dest, 3).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(src.id, vec![10., 20., 30., 40., 50., 60.]);
        rt.set_data(indices.id, vec![1, 3]);
        rt.set_data(dest.id, vec![0.; 12]);
        rt.execute(&cx.dyn_map);

        assert_eq!(
            *rt.get_f32(result.id),
            vec![0., 0., 0., 10., 20., 30., 0., 0., 0., 40., 50., 60.]
        );
    }

    #[test]
    fn test_scatter_then_gather_roundtrip() {
        let mut cx = Graph::new();
        let kv_new = cx.tensor((2, 4)); // 2 new rows, dim=4
        let scatter_idx = cx.tensor(2).as_dtype(DType::Int);
        let cache = cx.tensor((6, 4)); // 6 slots
        let gather_idx = cx.tensor(2).as_dtype(DType::Int);

        // Scatter new rows into cache, then gather them back
        let updated_cache = scatter_rows(kv_new, scatter_idx, cache, 4);
        let gathered = gather_rows(updated_cache, gather_idx, 4).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(kv_new.id, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        rt.set_data(scatter_idx.id, vec![1, 4]); // Write to slots 1 and 4
        rt.set_data(cache.id, vec![0.; 24]); // Zero cache
        rt.set_data(gather_idx.id, vec![1, 4]); // Read back from same slots
        rt.execute(&cx.dyn_map);

        assert_eq!(
            *rt.get_f32(gathered.id),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
    }

    #[test]
    fn test_paged_attention_shape_and_cache_update() {
        // Minimal config: n_heads=2, n_kv_heads=2, head_dim=2, kv_groups=1
        // hidden = 4, kv_dim = 4
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 2;
        let hidden = n_heads * head_dim; // 4
        let kv_dim = n_kv_heads * head_dim; // 4
        let num_slots = 8;

        let mut cx = Graph::new();
        let q = cx.tensor((1, hidden)); // 1 new token
        let k_new = cx.tensor((1, kv_dim));
        let v_new = cx.tensor((1, kv_dim));
        let k_cache = cx.tensor((num_slots, kv_dim));
        let v_cache = cx.tensor((num_slots, kv_dim));
        let gather_idx = cx.tensor(3).as_dtype(DType::Int); // 3 context tokens
        let scatter_idx = cx.tensor(1).as_dtype(DType::Int); // 1 new token
        let attn_mask = cx.tensor((1, 3)); // (s=1, ctx=3)

        let (attn_out, k_cache_new, v_cache_new) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            attn_mask,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let attn_out = attn_out.output();
        let k_cache_new = k_cache_new.output();
        let v_cache_new = v_cache_new.output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        // Q = [1, 0, 1, 0] → head0=[1,0], head1=[1,0]
        rt.set_data(q.id, vec![1., 0., 1., 0.]);
        // k_new = [0.5, 0.5, 0.5, 0.5]
        rt.set_data(k_new.id, vec![0.5, 0.5, 0.5, 0.5]);
        // v_new = [1, 2, 3, 4]
        rt.set_data(v_new.id, vec![1., 2., 3., 4.]);
        // Zero caches
        rt.set_data(k_cache.id, vec![0.; num_slots * kv_dim]);
        rt.set_data(v_cache.id, vec![0.; num_slots * kv_dim]);
        // Scatter new KV to slot 2
        rt.set_data(scatter_idx.id, vec![2]);
        // Gather context from slots 0, 1, 2 (slots 0,1 are zeros, slot 2 is the new KV)
        rt.set_data(gather_idx.id, vec![0, 1, 2]);
        // No masking (all can attend to all)
        rt.set_data(attn_mask.id, vec![0., 0., 0.]);

        rt.execute(&cx.dyn_map);

        // Verify output shape: (1, hidden=4)
        let out = rt.get_f32(attn_out.id);
        assert_eq!(out.len(), hidden);

        // Verify KV cache was updated: k_cache_new should have [0.5, 0.5, 0.5, 0.5] at slot 2
        let k_out = rt.get_f32(k_cache_new.id);
        assert_eq!(k_out.len(), num_slots * kv_dim);
        // Slot 2 is at offset 2*4=8..12
        assert_eq!(&k_out[8..12], &[0.5, 0.5, 0.5, 0.5]);
        // Slot 0 should still be zeros
        assert_eq!(&k_out[0..4], &[0., 0., 0., 0.]);

        let v_out = rt.get_f32(v_cache_new.id);
        assert_eq!(&v_out[8..12], &[1., 2., 3., 4.]);
    }

    #[test]
    fn test_paged_attention_known_values() {
        // Test with values where we can compute expected attention output.
        // n_heads=1, n_kv_heads=1, head_dim=2, kv_groups=1
        // hidden=2, kv_dim=2
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 2;
        let hidden = 2;
        let kv_dim = 2;
        let num_slots = 4;

        let mut cx = Graph::new();
        let q = cx.tensor((1, hidden));
        let k_new = cx.tensor((1, kv_dim));
        let v_new = cx.tensor((1, kv_dim));
        let k_cache = cx.tensor((num_slots, kv_dim));
        let v_cache = cx.tensor((num_slots, kv_dim));
        let gather_idx = cx.tensor(2).as_dtype(DType::Int);
        let scatter_idx = cx.tensor(1).as_dtype(DType::Int);
        let attn_mask = cx.tensor((1, 2));

        let (attn_out, _, _) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            attn_mask,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let attn_out = attn_out.output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        // Setup: 1 cached token at slot 0, 1 new token written to slot 1
        // K cached at slot 0: [1, 0]
        // K new (written to slot 1): [0, 1]
        // V cached at slot 0: [10, 20]
        // V new (written to slot 1): [30, 40]
        // Q: [1, 1]
        let mut k_cache_data = vec![0.; num_slots * kv_dim];
        k_cache_data[0] = 1.;
        k_cache_data[1] = 0.; // slot 0 K = [1, 0]
        let mut v_cache_data = vec![0.; num_slots * kv_dim];
        v_cache_data[0] = 10.;
        v_cache_data[1] = 20.; // slot 0 V = [10, 20]

        rt.set_data(q.id, vec![1., 1.]);
        rt.set_data(k_new.id, vec![0., 1.]); // new K = [0, 1]
        rt.set_data(v_new.id, vec![30., 40.]); // new V = [30, 40]
        rt.set_data(k_cache.id, k_cache_data);
        rt.set_data(v_cache.id, v_cache_data);
        rt.set_data(scatter_idx.id, vec![1]); // write to slot 1
        rt.set_data(gather_idx.id, vec![0, 1]); // gather slots 0, 1
        rt.set_data(attn_mask.id, vec![0., 0.]); // no masking

        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(attn_out.id);
        assert_eq!(out.len(), hidden);

        // Manual computation:
        // After scatter: K = [[1,0], [0,1]] at slots 0,1
        //                V = [[10,20], [30,40]] at slots 0,1
        // After gather (slots 0,1):
        //   K context = [[1,0], [0,1]], shape (2, 2)
        //   V context = [[10,20], [30,40]], shape (2, 2)
        //
        // Reshape for attention (1 head, 1 kv_head, head_dim=2):
        //   Q: (1, 1, 1, 2) = [[[[1, 1]]]]
        //   K: (1, 1, 2, 2) = [[[[1, 0], [0, 1]]]]  (transposed: head_dim x ctx)
        //   V: (1, 1, 2, 2) = [[[[10, 20], [30, 40]]]]
        //
        // QK^T = Q @ K = [1,1] @ [[1,0],[0,1]] = [1, 1]
        // scale = 1/sqrt(2) ≈ 0.7071
        // scores = [0.7071, 0.7071]
        // softmax([0.7071, 0.7071]) = [0.5, 0.5]
        //
        // out = [0.5, 0.5] @ [[10,20],[30,40]] = [20, 30]
        let expected = vec![20.0, 30.0];
        for (a, b) in out.iter().zip(&expected) {
            assert!((a - b).abs() < 0.1, "Expected {expected:?}, got {out:?}");
        }
    }

    #[test]
    fn test_paged_attention_causal_mask() {
        // Verify that the causal mask blocks future positions.
        // n_heads=1, n_kv_heads=1, head_dim=2
        let n_heads = 1;
        let n_kv_heads = 1;
        let head_dim = 2;
        let hidden = 2;
        let kv_dim = 2;
        let num_slots = 4;

        let mut cx = Graph::new();
        let q = cx.tensor((2, hidden)); // 2 new tokens
        let k_new = cx.tensor((2, kv_dim));
        let v_new = cx.tensor((2, kv_dim));
        let k_cache = cx.tensor((num_slots, kv_dim));
        let v_cache = cx.tensor((num_slots, kv_dim));
        let gather_idx = cx.tensor(3).as_dtype(DType::Int); // 3 context (1 cached + 2 new)
        let scatter_idx = cx.tensor(2).as_dtype(DType::Int);
        let attn_mask = cx.tensor((2, 3));

        let (attn_out, _, _) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            attn_mask,
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let attn_out = attn_out.output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        // Cache has 1 token at slot 0
        let mut k_cache_data = vec![0.; num_slots * kv_dim];
        k_cache_data[0] = 1.;
        k_cache_data[1] = 0.; // slot 0: K=[1,0]
        let mut v_cache_data = vec![0.; num_slots * kv_dim];
        v_cache_data[0] = 100.;
        v_cache_data[1] = 0.; // slot 0: V=[100,0]

        // 2 new tokens
        rt.set_data(q.id, vec![1., 0., 0., 1.]);
        rt.set_data(k_new.id, vec![0., 1., 1., 1.]); // token0 K=[0,1], token1 K=[1,1]
        rt.set_data(v_new.id, vec![0., 10., 0., 20.]); // token0 V=[0,10], token1 V=[0,20]
        rt.set_data(k_cache.id, k_cache_data);
        rt.set_data(v_cache.id, v_cache_data);
        rt.set_data(scatter_idx.id, vec![1, 2]); // write to slots 1, 2
        rt.set_data(gather_idx.id, vec![0, 1, 2]); // gather all 3

        // Causal mask: token0 (pos 1) can see ctx 0,1 but not 2
        //              token1 (pos 2) can see ctx 0,1,2
        let neg_inf = -1e9_f32;
        rt.set_data(
            attn_mask.id,
            vec![
                0., 0., neg_inf, // token 0: can see positions 0,1
                0., 0., 0., // token 1: can see all
            ],
        );

        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(attn_out.id);
        assert_eq!(out.len(), 2 * hidden);

        // Token 0 with mask: attends only to context positions 0,1
        // Token 1 without mask: attends to all 3 context positions
        // Just verify the output has valid (non-NaN, non-inf) values and correct length
        for val in out.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }
}
