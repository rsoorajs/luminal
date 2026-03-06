use std::collections::HashMap;

use luminal::{
    prelude::{tracing::trace, *},
    shape::Expression,
};
use onnx_protobuf::NodeProto;

use crate::util::get_int_attr;

/// Get an integer-list attribute from a node, with a default value applied per element.
fn get_ints_attr(node: &NodeProto, name: &str, default_elem: i64, spatial: usize) -> Vec<usize> {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.ints.iter().map(|&v| v as usize).collect();
        }
    }
    vec![default_elem as usize; spatial]
}

/// Parse an ONNX Conv node.
///
/// Supports N-dimensional convolution (1D, 2D, 3D) with group=1.
/// Uses the unfold-based approach from `luminal_nn::ConvND`.
///
/// Input layout: [batch, C_in, spatial...]
/// Weight layout: [C_out, C_in/group, kernel...]
/// Optional bias: [C_out]
pub fn parse_conv_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Conv Node");

    assert!(
        node.input.len() >= 2,
        "Conv needs at least 2 inputs (X, W), got {}",
        node.input.len()
    );

    let x = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Conv: missing input X '{}'", node.input[0]))?;
    let w = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Conv: missing weight W '{}'", node.input[1]))?;
    let bias = if node.input.len() > 2 && !node.input[2].is_empty() {
        Some(
            *tensors
                .get(&node.input[2])
                .ok_or_else(|| format!("Conv: missing bias B '{}'", node.input[2]))?,
        )
    } else {
        None
    };

    let x_dims = x.dims();
    let w_dims = w.dims();
    let rank = x_dims.len();
    assert!(rank >= 3, "Conv: input must be at least 3D (batch, channels, spatial...), got {rank}D");

    let spatial = rank - 2; // number of spatial dimensions

    // Parse attributes
    let kernel_shape = get_ints_attr(node, "kernel_shape", 1, spatial);
    let strides = get_ints_attr(node, "strides", 1, spatial);
    let dilations = get_ints_attr(node, "dilations", 1, spatial);
    let group = get_int_attr(node, "group", 1) as usize;

    // Parse pads: ONNX format is [begin_0, begin_1, ..., end_0, end_1, ...]
    let pads_flat = get_ints_attr(node, "pads", 0, 2 * spatial);
    let mut pads_begin = vec![0usize; spatial];
    let mut pads_end = vec![0usize; spatial];
    if pads_flat.len() == 2 * spatial {
        for i in 0..spatial {
            pads_begin[i] = pads_flat[i];
            pads_end[i] = pads_flat[spatial + i];
        }
    }

    assert_eq!(
        group, 1,
        "Conv: only group=1 is currently supported, got {group}"
    );

    // Get channel dimensions
    let ch_out = w_dims[0]
        .to_usize()
        .ok_or("Conv: weight C_out must be concrete")?;
    let ch_in = x_dims[1]
        .to_usize()
        .ok_or("Conv: input C_in must be concrete")?;

    let kernel_product: usize = kernel_shape.iter().product();

    // Reshape weight from ONNX [C_out, C_in, *kernel] to [C_out, C_in * kernel_product]
    let w_reshaped = {
        let mut wt = w * 1.0;
        wt.shape = ShapeTracker::new(vec![ch_out, ch_in * kernel_product]);
        wt
    };

    // Pad spatial dimensions
    let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
    for i in 0..spatial {
        let axis = 2 + i; // batch=0, channel=1, spatial starts at 2
        padding[axis] = (Expression::from(pads_begin[i]), Expression::from(pads_end[i]));
    }
    let padded = x.pad(padding, 0.0);

    // Build unfold parameters (ones for batch/channel, actual for spatial)
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    for i in 0..spatial {
        let axis = 2 + i;
        kernel_full[axis] = kernel_shape[i];
        stride_full[axis] = strides[i];
        dilation_full[axis] = dilations[i];
    }

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);

    // unfolded shape: [win_batch, win_channel, win_spatial..., k_batch, k_channel, k_spatial...]
    // We need to rearrange for matmul.

    // Move window dimensions to front
    let mut order: Vec<usize> = (rank..2 * rank).collect();
    order.extend(0..rank);
    let unfolded = unfolded.permute(order);
    let unfolded_dims = unfolded.dims();

    // Capture output spatial dimensions
    let output_spatial_dims: Vec<Expression> =
        unfolded_dims[3..3 + spatial].to_vec();

    // batch_len = 1 (single batch dim for standard conv input)
    let batch_len = 1;

    // Reorder: [win_batch, win_spatial..., win_channel, k_spatial..., k_batch, k_channel]
    let mut order2 = Vec::with_capacity(2 * rank);
    // win batch dims
    order2.extend(0..batch_len); // [0]
    // win spatial dims (outputs)
    order2.extend(batch_len + 1..batch_len + 1 + spatial); // [2..2+spatial]
    // win channel dim
    order2.push(batch_len); // [1]
    // k spatial dims
    order2.extend(rank + batch_len + 1..rank + batch_len + 1 + spatial);
    // k batch + k channel
    order2.extend(rank..rank + batch_len + 1);
    let mut patches = unfolded.permute(order2);

    // Drop kernel axes for batch + channel by merging them into the previous dimension
    for _ in 0..=batch_len {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }

    // Flatten channel and kernel spatial dims together
    for _ in 0..spatial {
        let channel_axis = batch_len + spatial;
        patches = patches.merge_dims(channel_axis, channel_axis + 1);
    }

    // Collapse output spatial dims into one for matmul
    for _ in 1..spatial {
        patches = patches.merge_dims(1, 2);
    }

    // patches shape: [batch, out_spatial_product, C_in * kernel_product]
    // weight shape: [C_out, C_in * kernel_product]
    // matmul: patches @ weight^T => [batch, out_spatial_product, C_out]
    let mut out = patches.matmul(w_reshaped.permute((1, 0)));

    // Restore spatial dimensions
    for dim in output_spatial_dims.iter().rev() {
        out = out.split_dims(batch_len, *dim);
    }

    // Move channel dimension ahead of spatial: [batch, C_out, spatial...]
    let mut final_order: Vec<usize> = (0..batch_len).collect();
    final_order.push(batch_len + spatial); // C_out
    final_order.extend(batch_len..batch_len + spatial); // spatial dims
    out = out.permute(final_order);

    // Add bias if present: bias shape [C_out], broadcast to [1, C_out, 1, 1, ...]
    if let Some(b) = bias {
        let mut bias_expanded = b;
        // Expand to [1, C_out, 1, 1, ...]
        bias_expanded = bias_expanded.expand_dim(0, 1); // batch dim
        for i in 0..spatial {
            let out_dims = out.dims();
            let spatial_size = out_dims[2 + i].clone();
            bias_expanded = bias_expanded.expand_dim(2 + i, spatial_size);
        }
        out = out + bias_expanded;
    }

    let result = out * 1.0;
    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Conv Node");
    Ok(())
}
