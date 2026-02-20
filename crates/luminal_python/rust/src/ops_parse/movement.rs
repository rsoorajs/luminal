use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, get_int_attr};

/// Handle Transpose node: output = permute(input, perm)
///
/// The perm attribute specifies the permutation of dimensions.
/// If perm is not specified, reverses all dimensions (ONNX default).
pub fn parse_transpose_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Transpose Node");

    // Validate node structure
    assert!(
        node.input.len() == 1,
        "Transpose nodes must have exactly one input, {} present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Transpose nodes must have exactly one output, {} present",
        node.output.len(),
    );

    // Get input tensor
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Transpose: missing input tensor '{}'", node.input[0]))?;

    let input_rank = input.dims().len();

    // Extract perm attribute or use default (reverse all dims)
    let perm: Vec<usize> = if let Some(attr) = node.attribute.iter().find(|a| a.name == "perm") {
        let perm_i64: &Vec<i64> = &attr.ints;

        // Validate perm length
        if perm_i64.len() != input_rank {
            return Err(format!(
                "Transpose: perm length {} does not match input rank {}",
                perm_i64.len(),
                input_rank
            ));
        }

        // Convert to usize and validate range
        let mut perm_usize: Vec<usize> = Vec::with_capacity(perm_i64.len());
        for &axis in perm_i64.iter() {
            if axis < 0 || axis >= input_rank as i64 {
                return Err(format!(
                    "Transpose: perm axis {} out of range [0, {})",
                    axis, input_rank
                ));
            }
            perm_usize.push(axis as usize);
        }

        // Validate uniqueness (check it's a valid permutation)
        let mut sorted = perm_usize.clone();
        sorted.sort();
        for (i, &val) in sorted.iter().enumerate() {
            if val != i {
                return Err(format!(
                    "Transpose: perm {:?} is not a valid permutation",
                    perm_i64
                ));
            }
        }

        perm_usize
    } else {
        // Default: reverse all dimensions
        (0..input_rank).rev().collect()
    };

    // Apply permute operation
    let permuted = input.permute(perm);

    // Force materialization by multiplying by 1.0
    // This is necessary because permute is a view operation that doesn't
    // rearrange data in memory. The multiplication adds a graph node that
    // will read according to the permuted shape and write in contiguous order.
    let result = permuted * 1.0;

    // Store output
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    trace!("Finished parse: Transpose Node");
    Ok(())
}

/// Handle Reshape node: change the tensor's shape without modifying data.
///
/// The target shape is read from the second input (must be a known constant).
/// Supports -1 (infer from total elements) and 0 (copy from input dimension).
/// Non-contiguous tensors (e.g., from Expand) are materialized before reshaping.
pub fn parse_reshape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Started parse: Reshape");
    assert!(
        node.input.len() == 2,
        "Reshape should have exactly 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Reshape: missing input tensor '{}'", node.input[0]))?;

    let shape_data = known_values.get(&node.input[1]).ok_or_else(|| {
        format!(
            "Reshape: shape input '{}' must be a known constant",
            node.input[1]
        )
    })?;

    // Compute total elements for resolving -1
    let input_dims = input.dims();
    let total_elements: usize = input_dims
        .iter()
        .map(|d| d.to_usize().expect("Reshape: input dims must be concrete"))
        .product();

    // Resolve target shape, handling -1 (infer) and 0 (copy from input)
    let mut target_shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
    // First pass: resolve 0 (copy from input at same position)
    for i in 0..target_shape.len() {
        if target_shape[i] == 0 {
            target_shape[i] = input_dims[i].to_usize().unwrap_or(1) as i64;
        }
    }
    // Second pass: resolve -1 (infer from total elements)
    let known_product: i64 = target_shape.iter().filter(|&&d| d > 0).product();
    for d in target_shape.iter_mut() {
        if *d == -1 {
            *d = total_elements as i64 / known_product;
        }
    }
    let final_shape: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();

    let mut result = input;
    // If tensor is not contiguous (e.g., has broadcast strides from Expand),
    // materialize it before reshaping by multiplying by 1.0.
    // This forces a contiguous copy through the binary op mechanism.
    if !result.shape.is_contiguous() {
        let one = result.graph().constant_float(1.0);
        let src_dims = result.dims();
        let broadcast_shape: Vec<usize> = src_dims
            .iter()
            .map(|d| d.to_usize().expect("dim must be concrete"))
            .collect();
        let one_expanded = broadcast_to(one, &broadcast_shape);
        result *= one_expanded;
    }
    result.shape = ShapeTracker::new(final_shape.clone());

    // Force materialization to create a distinct graph node for the CUDA backend.
    // Without this, the output shares the same NodeIndex as the input tensor,
    // and CudaRuntime::get_f32 cannot retrieve data for input-aliased outputs.
    // (Same pattern as parse_transpose_node's `permuted * 1.0`.)
    let one = result.graph().constant_float(1.0);
    let one_expanded = broadcast_to(one, &final_shape);
    result *= one_expanded;

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values (reshape doesn't change data, just layout)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    trace!("Finished parse: Reshape");
    Ok(())
}

/// Handle Squeeze node: remove size-1 dimensions from a tensor.
///
/// Opset 13+: axes come from the second input. Opset 11: from the "axes" attribute.
/// If no axes are specified, all dimensions of size 1 are removed.
/// Axes are processed in reverse order to avoid index shifting.
pub fn parse_squeeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Squeeze Node");
    assert!(
        !node.input.is_empty(),
        "Squeeze should have at least 1 input"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Squeeze: missing input tensor '{}'", node.input[0]))?;

    // Opset 13+: axes come from second input; opset 11: from attribute
    let axes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_data = known_values
            .get(&node.input[1])
            .ok_or_else(|| format!("Squeeze: axes '{}' must be known", node.input[1]))?;
        let ndim = input.dims().len();
        axes_data
            .iter()
            .map(|&v| {
                let a = v as i64;
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        let ndim = input.dims().len();
        attr.ints
            .iter()
            .map(|&v| {
                if v < 0 {
                    (ndim as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        // No axes specified: squeeze all dims of size 1
        input
            .dims()
            .iter()
            .enumerate()
            .filter(|(_, d)| d.to_usize() == Some(1))
            .map(|(i, _)| i)
            .collect()
    };

    // Sort in reverse order so removing earlier axes doesn't shift later ones
    let mut sorted_axes = axes.clone();
    sorted_axes.sort();
    sorted_axes.reverse();

    let mut result = input;
    for &axis in &sorted_axes {
        result = result.squeeze(axis);
    }

    // Force materialization to create a distinct graph node for the CUDA backend.
    // Without this, squeeze (a pure shape op) produces no kernels and the output
    // cannot be retrieved from the runtime.
    // (Same pattern as parse_transpose_node, parse_reshape_node, parse_identity.)
    let output_dims = result.dims();
    // If squeezing produced a 0-dim scalar, represent as [1] to avoid CUDA launching
    // kernels with grid (0, 1, 1). luminal's Expression::product() returns 0 for empty
    // iterators, making any kernel on a 0-dim tensor invalid on CUDA.
    let shape: Vec<usize> = if output_dims.is_empty() {
        result.shape = ShapeTracker::new(vec![1usize]);
        vec![1usize]
    } else {
        output_dims
            .iter()
            .map(|d| d.to_usize().expect("Squeeze: dim must be concrete"))
            .collect()
    };
    let one = result.graph().constant_float(1.0);
    let one_expanded = broadcast_to(one, &shape);
    result *= one_expanded;

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    trace!("Finished parse: Squeeze Node");
    Ok(())
}

/// Handle Unsqueeze node: insert size-1 dimensions at the specified axes.
///
/// Axes are read from the second input (must be a known constant).
/// Negative axis values are resolved relative to the output rank.
pub fn parse_unsqueeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Unsqueeze Node");
    assert!(
        node.input.len() == 2,
        "Unsqueeze should have exactly 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Unsqueeze: missing input tensor '{}'", node.input[0]))?;

    let axes_data = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Unsqueeze: axes '{}' must be known", node.input[1]))?;

    let ndim = input.dims().len();
    let out_ndim = ndim + axes_data.len();
    let mut axes: Vec<usize> = axes_data
        .iter()
        .map(|&v| {
            let a = v as i64;
            if a < 0 {
                (out_ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    axes.sort();

    let mut result = input;
    for &axis in &axes {
        result = result.unsqueeze(axis);
    }

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    trace!("Finished parse: Unsqueeze Node");
    Ok(())
}

/// Handle Concat node: concatenate multiple tensors along a given axis.
///
/// The axis attribute specifies which dimension to concatenate along.
/// Negative axis values are resolved relative to the input rank.
/// All inputs must have the same shape except along the concat axis.
pub fn parse_concat_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Concat Node");
    assert!(
        !node.input.is_empty(),
        "Concat needs at least 1 input, {} present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Concat must have exactly one output, {} present",
        node.output.len()
    );

    let output_name = &node.output[0];

    let first = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Concat: missing input tensor '{}'", node.input[0]))?;

    let ndim = first.dims().len();

    // Extract axis attribute, supporting negative indexing
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // Iteratively concat all inputs along the axis
    let mut result = first;
    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("Concat: missing input tensor '{}'", input_name))?;
        result = result.concat_along(rhs, axis);
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Concat Node");
    Ok(())
}

/// Handle Gather node: index into a tensor along a specified axis.
///
/// For 1D data, performs simple element-wise gathering. For ND data, gathers
/// slices along the specified axis. Supports arbitrary axis values.
/// Supports constant folding when both data and indices are known.
pub fn parse_gather_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Gather Node");
    assert!(node.input.len() == 2, "Gather should have 2 inputs");

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gather: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let data_ndim = data_dims.len();

    // Handle axis (support negative indexing)
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (data_ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // If both inputs are known, fully constant-fold
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let output_name = &node.output[0];

        // Get concrete data dimensions
        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: data dims must be concrete"))
            .collect();

        // Get indices tensor shape
        let indices_tensor = tensors.get(&node.input[1]);
        let indices_shape: Vec<usize> = if let Some(idx_tensor) = indices_tensor {
            idx_tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("Gather: indices dims must be concrete"))
                .collect()
        } else {
            vec![vidx.len()]
        };

        // Compute output shape: data_shape[0..axis] + indices_shape + data_shape[axis+1..]
        let mut output_shape: Vec<usize> = data_shape[..axis].to_vec();
        output_shape.extend(&indices_shape);
        output_shape.extend(&data_shape[axis + 1..]);

        // Gather with arbitrary axis
        let output_size: usize = output_shape.iter().product();
        let mut folded = vec![0.0f32; output_size.max(1)];

        if output_size > 0 {
            let pre_size: usize = data_shape[..axis].iter().product::<usize>().max(1);
            let post_size: usize = data_shape[axis + 1..].iter().product::<usize>().max(1);
            let indices_size: usize = vidx.len();
            let axis_dim = data_shape[axis] as i64;

            for pre in 0..pre_size {
                for (idx_pos, &idx_val) in vidx.iter().enumerate() {
                    // Normalize negative indices: ONNX allows -1 for last element, etc.
                    let idx_raw = idx_val as i64;
                    let idx = (((idx_raw % axis_dim) + axis_dim) % axis_dim) as usize;
                    for post in 0..post_size {
                        let data_flat =
                            pre * (data_shape[axis] * post_size) + idx * post_size + post;
                        let out_flat =
                            pre * (indices_size * post_size) + idx_pos * post_size + post;
                        folded[out_flat] = vdata[data_flat];
                    }
                }
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        trace!("Finished parse: Gather Node (constant folded)");
        return Ok(());
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gather: missing indices tensor '{}'", node.input[1]))?;

    // Normalize negative indices: ONNX allows -1 for last element, -2 for second-to-last, etc.
    // Use conditional normalization instead of modulo to avoid floating-point precision loss:
    // if index < 0 then index + axis_dim else index
    let axis_dim = data_dims[axis].to_usize().ok_or_else(|| {
        "Gather: axis dimension must be concrete for index normalization".to_string()
    })?;
    let axis_dim_f32 = axis_dim as f32;

    // Cast to F32 for normalization arithmetic (ONNX indices may be INT64 → luminal Int,
    // or constant F32 weights). This avoids Bool * F32 panics when lt() returns Bool dtype.
    let indices_f32 = indices_raw.cast(DType::F32);
    let zero = indices_f32
        .graph()
        .constant_float(0.0)
        .expand_rhs(indices_f32.shape);
    let adjustment = indices_f32
        .graph()
        .constant_float(axis_dim_f32)
        .expand_rhs(indices_f32.shape);
    // lt() returns Bool; cast to F32 before arithmetic (same pattern as parse_gathernd_node)
    let is_negative = indices_f32.lt(zero).cast(DType::F32);
    let indices = indices_f32 + (is_negative * adjustment);

    let result = if data_ndim == 1 {
        // 1D data: simple flat element-wise gather
        data.gather(indices.cast(DType::Int))
    } else if axis == 0 {
        // ND data with axis=0: gather entire slices along first dimension
        gather_axis0(data, indices, &data_dims)?
    } else {
        // General case: axis != 0
        // Strategy: permute to bring axis to front, gather, permute back

        // Build permutation to move axis to position 0
        let mut perm: Vec<usize> = (0..data_ndim).collect();
        perm.remove(axis);
        perm.insert(0, axis);

        // Permute data
        let permuted_data = data.permute(perm);
        let permuted_dims = permuted_data.dims();

        // Gather on axis 0 of permuted data
        let gathered = gather_axis0(permuted_data, indices, &permuted_dims)?;

        // Compute gathered shape
        let idx_dims = indices.dims();
        let mut gathered_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &permuted_dims[1..] {
            gathered_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(gathered_shape.clone());

        // Build inverse permutation to restore original axis order
        let indices_rank = idx_dims.len();
        let mut inv_perm: Vec<usize> = Vec::with_capacity(reshaped.dims().len());

        // Dimensions before original axis (they're after indices in gathered result)
        for i in 0..axis {
            inv_perm.push(indices_rank + i);
        }
        // The indices dimensions
        for i in 0..indices_rank {
            inv_perm.push(i);
        }
        // Dimensions after original axis
        for i in axis..(data_ndim - 1) {
            inv_perm.push(indices_rank + i);
        }

        reshaped.permute(inv_perm)
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    trace!("Finished parse: Gather Node");
    Ok(())
}

/// Handle GatherND node: gather elements or slices using multi-dimensional indices.
///
/// The last dimension of `indices` specifies the coordinate depth `q`.
/// For `q=1` (most common from PyTorch's dynamo exporter), degenerates to Gather(axis=0).
/// Supports constant folding when both data and indices are known at graph-build time.
pub fn parse_gathernd_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: GatherND Node");
    assert!(
        node.input.len() >= 2,
        "GatherND should have at least 2 inputs, {} present",
        node.input.len()
    );

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("GatherND: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let n = data_dims.len();

    let b = get_int_attr(node, "batch_dims", 0) as usize;

    let output_name = &node.output[0];

    // --- Constant folding path ---
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let indices_tensor = *tensors
            .get(&node.input[1])
            .ok_or_else(|| format!("GatherND: missing indices tensor '{}'", node.input[1]))?;

        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("GatherND: data dims must be concrete"))
            .collect();
        let idx_shape: Vec<usize> = indices_tensor
            .dims()
            .iter()
            .map(|d| d.to_usize().expect("GatherND: index dims must be concrete"))
            .collect();

        // K = number of outer index dims (excluding coordinate depth dim q)
        let k = idx_shape.len() - 1;
        // q = last dim of indices = coordinate depth
        let q = idx_shape[k];

        // output_shape = data_shape[..b] + idx_shape[b..k] + data_shape[b+q..]
        let mut output_shape: Vec<usize> = data_shape[..b].to_vec();
        output_shape.extend_from_slice(&idx_shape[b..k]);
        output_shape.extend_from_slice(&data_shape[b + q..]);

        let batch_size: usize = data_shape[..b].iter().product::<usize>().max(1);
        let outer_size: usize = idx_shape[b..k].iter().product::<usize>().max(1);
        let slice_size: usize = data_shape[b + q..].iter().product::<usize>().max(1);
        let data_elements_per_batch: usize = data_shape[b..].iter().product::<usize>().max(1);

        // Strides for dims b..b+q of data: stride[j] = product(data_shape[b+j+1..b+q]) * slice_size
        let data_strides: Vec<usize> = (0..q)
            .map(|j| data_shape[b + j + 1..b + q].iter().product::<usize>() * slice_size)
            .collect();

        let total_size = batch_size * outer_size * slice_size;
        let mut folded = vec![0.0f32; total_size.max(1)];

        for batch in 0..batch_size {
            for p in 0..outer_size {
                let idx_offset = (batch * outer_size + p) * q;
                let mut flat_base = batch * data_elements_per_batch;
                for j in 0..q {
                    let coord_raw = vidx[idx_offset + j] as i64;
                    let dim = data_shape[b + j] as i64;
                    let coord = (((coord_raw % dim) + dim) % dim) as usize;
                    flat_base += coord * data_strides[j];
                }
                let out_base = (batch * outer_size + p) * slice_size;
                folded[out_base..(slice_size + out_base)]
                    .copy_from_slice(&vdata[flat_base..(slice_size + flat_base)]);
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        trace!("Finished parse: GatherND Node (constant folded)");
        return Ok(());
    }

    // --- Dynamic path ---
    if b != 0 {
        return Err(format!(
            "GatherND: batch_dims={} > 0 is not yet supported in dynamic mode",
            b
        ));
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("GatherND: missing indices tensor '{}'", node.input[1]))?;

    let data_shape: Vec<usize> = data_dims
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "GatherND: data dims must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let idx_shape: Vec<usize> = indices_raw
        .dims()
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "GatherND: index dims must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let k = idx_shape.len() - 1;
    let q = idx_shape[k];

    if q == 1 {
        // Sub-case A: q=1 — GatherND degenerates to Gather(axis=0).
        // Reshape indices from [..., 1] to [...] by dropping the last size-1 dim.
        let squeezed_shape: Vec<usize> = if k == 0 {
            vec![1]
        } else {
            idx_shape[..k].to_vec()
        };
        let mut idx_squeezed = indices_raw;
        idx_squeezed.shape = ShapeTracker::new(squeezed_shape);

        // Cast to F32 for normalization arithmetic (ONNX indices are INT64 → luminal Int)
        let idx_float = idx_squeezed.cast(DType::F32);

        // Normalize negative indices: if index < 0 then index + axis_dim else index
        let axis_dim_f32 = data_shape[0] as f32;
        let zero = idx_float
            .graph()
            .constant_float(0.0)
            .expand_rhs(idx_float.shape);
        let adjustment = idx_float
            .graph()
            .constant_float(axis_dim_f32)
            .expand_rhs(idx_float.shape);
        // lt() returns Bool; cast to F32 before arithmetic (same pattern as parse_floor_node)
        let is_negative = idx_float.lt(zero).cast(DType::F32);
        let idx_normalized = idx_float + (is_negative * adjustment);

        let result = if n == 1 {
            // 1D data: flat element-wise gather
            data.gather(idx_normalized.cast(DType::Int))
        } else {
            // ND data: gather slices along axis 0
            gather_axis0(data, idx_normalized, &data_dims)?
        };

        tensors.insert(output_name.clone(), result);
    } else {
        // Sub-case B: q > 1 — compute flat indices by summing coord_j * stride_j.
        let outer_size: usize = idx_shape[..k].iter().product::<usize>().max(1);
        let slice_size: usize = data_shape[q..].iter().product::<usize>().max(1);

        // Flatten indices to [outer_size * q] for column extraction
        let mut indices_flat = indices_raw;
        indices_flat.shape = ShapeTracker::new(vec![outer_size * q]);

        // Stride of dim j in data's first q dimensions
        let data_strides: Vec<usize> = (0..q)
            .map(|j| data_shape[j + 1..q].iter().product::<usize>() * slice_size)
            .collect();

        // Build flat_indices (Int) = sum over j of (coord_j * stride_j)
        let mut flat_indices: Option<GraphTensor> = None;
        for j in 0..q {
            // Column j positions in flattened indices: [j, q+j, 2q+j, ...]
            let arange = indices_flat.graph().arange(outer_size); // shape [outer_size], Int
            let q_expanded = indices_flat
                .graph()
                .constant(q as i32)
                .expand_rhs(arange.shape);
            let j_expanded = indices_flat
                .graph()
                .constant(j as i32)
                .expand_rhs(arange.shape);
            let positions = arange * q_expanded + j_expanded; // Int, shape [outer_size]

            // indices_flat may be Int (ONNX INT64); cast to F32 for arithmetic
            let coord_j_f32 = indices_flat.gather(positions).cast(DType::F32); // f32, shape [outer_size]

            // Normalize negative coordinates
            let dim_j_f32 = data_shape[j] as f32;
            let zero = coord_j_f32
                .graph()
                .constant_float(0.0)
                .expand_rhs(coord_j_f32.shape);
            let adj = coord_j_f32
                .graph()
                .constant_float(dim_j_f32)
                .expand_rhs(coord_j_f32.shape);
            // lt() returns Bool; cast to F32 before arithmetic
            let is_neg = coord_j_f32.lt(zero).cast(DType::F32);
            let coord_normalized = coord_j_f32 + (is_neg * adj);

            // Use Int arithmetic to avoid f32 precision loss for large indices
            let coord_int = coord_normalized.cast(DType::Int);
            let stride_expanded = coord_int
                .graph()
                .constant(data_strides[j] as i32)
                .expand_rhs(coord_int.shape);
            let contribution = coord_int * stride_expanded;

            flat_indices = Some(match flat_indices {
                None => contribution,
                Some(acc) => acc + contribution,
            });
        }

        let flat_indices = flat_indices.ok_or_else(|| "GatherND: q must be > 0".to_string())?;

        // Reshape data to [prefix_size, slice_size] for gather_axis0
        let prefix_size: usize = data_shape[..q].iter().product::<usize>().max(1);
        let mut data_reshaped = data;
        data_reshaped.shape = ShapeTracker::new(vec![prefix_size, slice_size]);

        // Reshape flat_indices to the outer shape expected by gather_axis0
        let outer_shape: Vec<usize> = if k == 0 {
            vec![1]
        } else {
            idx_shape[..k].to_vec()
        };
        let mut flat_indices_shaped = flat_indices;
        flat_indices_shaped.shape = ShapeTracker::new(outer_shape.clone());

        // Gather slices: result shape = [outer_shape..., slice_size]
        let data_reshaped_dims = data_reshaped.dims();
        let result = gather_axis0(data_reshaped, flat_indices_shaped, &data_reshaped_dims)?;

        // Reshape result to [outer_shape..., d_q, ..., d_{N-1}]
        let mut output_shape = outer_shape;
        output_shape.extend_from_slice(&data_shape[q..]);
        let mut result_shaped = result;
        result_shaped.shape = ShapeTracker::new(output_shape);

        tensors.insert(output_name.clone(), result_shaped);
    }

    trace!("Finished parse: GatherND Node");
    Ok(())
}

/// Handle Trilu node: apply lower or upper triangular mask to a tensor.
///
/// `upper=1` (default) → upper triangle (triu), `upper=0` → lower triangle (tril).
/// Optional `k` input specifies the diagonal offset (default 0).
/// Luminal's tril/triu require a square matrix; the last two dims must be equal.
pub fn parse_trilu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Trilu Node");

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Trilu: missing input '{}'", node.input[0]))?;

    // Optional k (diagonal offset) — provided as a constant tensor input
    let k: i32 = if node.input.len() > 1 && !node.input[1].is_empty() {
        known_values
            .get(&node.input[1])
            .and_then(|v| v.first())
            .map(|&f| f as i32)
            .unwrap_or(0)
    } else {
        0
    };

    let upper = get_int_attr(node, "upper", 1);

    // Validate that the last two dims form a square matrix
    let dims = input.dims();
    let ndim = dims.len();
    assert!(ndim >= 2, "Trilu: input must have at least 2 dimensions");
    let rows = dims[ndim - 2]
        .to_usize()
        .ok_or_else(|| "Trilu: could not resolve row dimension".to_string())?;
    let cols = dims[ndim - 1]
        .to_usize()
        .ok_or_else(|| "Trilu: could not resolve col dimension".to_string())?;
    assert_eq!(
        rows,
        cols,
        "Trilu: luminal only supports square matrices, got {}x{}",
        rows,
        cols
    );
    let size = cols;

    // Generate triangular mask (square, size x size), returns Bool dtype
    let mask = if upper == 0 {
        cx.tril(size, k)
    } else {
        cx.triu(size, k)
    };

    // Cast Bool mask to F32 and broadcast to match input shape (e.g. batch dims)
    let target_shape: Vec<usize> = dims
        .iter()
        .map(|e| e.to_usize().unwrap_or(1))
        .collect();
    let mask_f32 = broadcast_to(mask.cast(DType::F32), &target_shape);

    let result = input * mask_f32;
    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Trilu Node");
    Ok(())
}

/// Helper function for gathering along axis 0
fn gather_axis0(
    data: GraphTensor,
    indices: GraphTensor,
    data_dims: &[Expression],
) -> Result<GraphTensor, String> {
    let inner_dim: usize = data_dims[1..]
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "Gather: inner dimensions must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product();

    // Compute flat_indices = indices * inner_dim + offsets
    // IMPORTANT: Use Int arithmetic to avoid f32 precision loss for large indices.
    // f32 only has 24-bit mantissa, so indices > 16.7M lose precision.
    // For vocab_size=50304, embed_dim=768: max flat index = 38.6M
    let indices_int = indices.cast(DType::Int);
    let inner_dim_tensor = indices
        .graph()
        .constant(inner_dim as i32)
        .expand_rhs(indices_int.shape);
    let scaled = indices_int * inner_dim_tensor;
    let idx_ndim = indices.dims().len();
    let scaled_expanded = scaled.expand_dim(idx_ndim, inner_dim);

    // Create column offsets [0, 1, ..., inner_dim-1] and broadcast to indices shape
    // Keep as Int for precise arithmetic
    let offsets = data.graph().arange(inner_dim); // arange returns Int
    let idx_dims = indices.dims();
    let mut offsets_expanded = offsets;
    for i in (0..idx_dims.len()).rev() {
        offsets_expanded = offsets_expanded.expand_dim(0, idx_dims[i]);
    }

    let flat_indices = scaled_expanded + offsets_expanded; // Both Int, result is Int

    // Gather using flat indices into contiguous data buffer
    let gathered = data.gather(flat_indices);

    // Reshape from [..., inner_dim] to [..., D1, D2, ..., Dk]
    if data_dims.len() > 2 {
        let mut output_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &data_dims[1..] {
            output_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(output_shape);
        Ok(reshaped)
    } else {
        Ok(gathered)
    }
}
