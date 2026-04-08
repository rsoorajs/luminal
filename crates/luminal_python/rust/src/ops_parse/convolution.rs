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

/// Parse an ONNX Conv node (1D/2D/3D, with grouped/depthwise support).
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
    assert!(
        rank >= 3,
        "Conv: input must be at least 3D (batch, channels, spatial...), got {rank}D"
    );

    let spatial = rank - 2;

    if let Some(attr) = node.attribute.iter().find(|a| a.name == "auto_pad")
        && !attr.s.is_empty()
    {
        let auto_pad = String::from_utf8_lossy(&attr.s);
        if auto_pad != "NOTSET" {
            return Err(format!(
                "Conv: auto_pad='{auto_pad}' is not supported; export explicit pads instead"
            ));
        }
    }

    // ONNX Conv: kernel_shape is optional and should be inferred from W when absent.
    // Spec: https://onnx.ai/onnx/operators/onnx__Conv.html
    let kernel_shape: Vec<usize> = {
        let attr_ks = get_ints_attr(node, "kernel_shape", 0, 0);
        if attr_ks.is_empty() {
            w_dims[2..]
                .iter()
                .map(|d| d.to_usize().expect("Conv: kernel dims must be concrete"))
                .collect()
        } else {
            attr_ks
        }
    };
    let strides = get_ints_attr(node, "strides", 1, spatial);
    let dilations = get_ints_attr(node, "dilations", 1, spatial);
    let group = get_int_attr(node, "group", 1) as usize;

    let pads_flat = get_ints_attr(node, "pads", 0, 2 * spatial);
    let mut pads_begin = vec![0usize; spatial];
    let mut pads_end = vec![0usize; spatial];
    if pads_flat.len() == 2 * spatial {
        pads_begin[..spatial].copy_from_slice(&pads_flat[..spatial]);
        pads_end[..spatial].copy_from_slice(&pads_flat[spatial..]);
    }

    let ch_out = w_dims[0]
        .to_usize()
        .ok_or("Conv: weight C_out must be concrete")?;
    let ch_in = x_dims[1]
        .to_usize()
        .ok_or("Conv: input C_in must be concrete")?;
    if kernel_shape.len() != spatial {
        return Err(format!(
            "Conv: kernel rank {} does not match input spatial rank {spatial}",
            kernel_shape.len()
        ));
    }
    if strides.len() != spatial || dilations.len() != spatial {
        return Err(format!(
            "Conv: stride/dilation rank mismatch for spatial rank {spatial} (got strides={}, dilations={})",
            strides.len(),
            dilations.len()
        ));
    }
    if group == 0 || ch_in % group != 0 || ch_out % group != 0 {
        return Err(format!(
            "Conv: invalid group configuration (C_in={ch_in}, C_out={ch_out}, groups={group})"
        ));
    }
    let ch_per_group = ch_in / group;
    let kernel_product: usize = kernel_shape.iter().product();

    // Grouped conv
    if group > 1 {
        let group_out = ch_out / group;
        let out = if ch_per_group == 1 {
            // Depthwise (including channel multiplier > 1): avoid per-group slicing by
            // applying all per-channel kernels over the full channel view.
            depthwise_conv(
                x,
                w,
                &kernel_shape,
                &strides,
                &dilations,
                &pads_begin,
                &pads_end,
                ch_in,
                group_out,
                kernel_product,
                spatial,
            )
        } else {
            // General grouped conv: pre-pad the full input, then slice per group.
            // Pre-padding ensures each group's spatial slice is within the padded buffer.
            let padded_x = {
                let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
                for i in 0..spatial {
                    padding[2 + i] = (pads_begin[i].into(), pads_end[i].into());
                }
                x.pad(padding, 0.0)
            };

            let no_pad = vec![0usize; spatial];
            let mut group_outputs = Vec::with_capacity(group);
            for g in 0..group {
                let x_g = slice_channel_group(padded_x, g, ch_per_group, spatial);
                let w_g = slice_weight_group(w, g, group_out, ch_per_group * kernel_product);
                group_outputs.push(conv_unfold(
                    x_g,
                    w_g,
                    &kernel_shape,
                    &strides,
                    &dilations,
                    &no_pad,
                    &no_pad,
                    ch_per_group,
                    group_out,
                    spatial,
                ));
            }

            let mut result = group_outputs[0];
            for g_out in &group_outputs[1..] {
                result = result.concat_along(*g_out, 1);
            }
            result
        };

        let out = if let Some(b) = bias {
            add_conv_bias(out, b, spatial)
        } else {
            out
        };
        tensors.insert(node.output[0].clone(), out);
        trace!("Finished parse: Conv Node (grouped, groups={group})");
        return Ok(());
    }

    // Standard group=1 path
    let w_reshaped = {
        let mut wt = w;
        wt.shape = ShapeTracker::new_with_element_bits(
            vec![ch_out, ch_in * kernel_product],
            wt.dtype.bits(),
        );
        wt
    };

    let mut out = conv_unfold(
        x,
        w_reshaped,
        &kernel_shape,
        &strides,
        &dilations,
        &pads_begin,
        &pads_end,
        ch_in,
        ch_out,
        spatial,
    );

    if let Some(b) = bias {
        out = add_conv_bias(out, b, spatial);
    }

    tensors.insert(node.output[0].clone(), out);

    trace!("Finished parse: Conv Node");
    Ok(())
}

/// Core unfold-based convolution for a single group.
///
/// `x`: [batch, ch_in, spatial...]
/// `w_flat`: [ch_out, ch_in * kernel_product] (already reshaped)
/// Returns: [batch, ch_out, out_spatial...]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_unfold(
    x: GraphTensor,
    w_flat: GraphTensor,
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads_begin: &[usize],
    pads_end: &[usize],
    _ch_in: usize,
    _ch_out: usize,
    spatial: usize,
) -> GraphTensor {
    let rank = 2 + spatial;

    // Pad spatial dimensions (skip if all padding is zero)
    let needs_pad = pads_begin.iter().any(|&p| p > 0) || pads_end.iter().any(|&p| p > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
        for i in 0..spatial {
            padding[2 + i] = (pads_begin[i].into(), pads_end[i].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Build full-rank unfold parameters (1 for batch/channel, actual for spatial)
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    kernel_full[2..(spatial + 2)].copy_from_slice(&kernel_shape[..spatial]);
    stride_full[2..(spatial + 2)].copy_from_slice(&strides[..spatial]);
    dilation_full[2..(spatial + 2)].copy_from_slice(&dilations[..spatial]);

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
    // Shape: [win_N, win_C, win_spatial..., k_N=1, k_C=1, k_spatial...]

    // Permute to [N, win_spatial..., C_in, k_N, k_C, k_spatial...]
    let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
    perm.push(0);
    perm.extend(2..2 + spatial);
    perm.push(1);
    perm.extend(rank..2 * rank);
    let permuted = unfolded.permute(perm);

    let output_spatial_dims: Vec<Expression> = permuted.dims()[1..1 + spatial].to_vec();

    // Merge all channel+kernel dims into [N, spatial..., ch_in * kernel_product]
    let mut patches = permuted;
    let target = 2 + spatial;
    while patches.dims().len() > target {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }

    // Merge spatial dims into one
    for _ in 1..spatial {
        patches = patches.merge_dims(1, 2);
    }
    // patches: [N, spatial_product, ch_in * kernel_product]

    let mut out = patches.matmul(w_flat.permute((1, 0)));
    // out: [N, spatial_product, ch_out]

    // Restore spatial dimensions
    for i in (1..spatial).rev() {
        out = out.split_dims(1, output_spatial_dims[i]);
    }

    // Move ch_out from last to position 1: [N, ch_out, spatial...]
    let mut final_order: Vec<usize> = Vec::with_capacity(2 + spatial);
    final_order.push(0);
    final_order.push(1 + spatial);
    final_order.extend(1..1 + spatial);
    out.permute(final_order)
}

/// Depthwise convolution: groups == in_channels, ch_per_group == 1.
///
/// Processes all channels simultaneously using element-wise multiply + reduce,
/// avoiding per-channel input slicing which can cause index-expression bugs in luminal.
///
/// out[n, c, oh, ow] = sum_k patches[n, c, oh, ow, k] * weight[c, k]
#[allow(clippy::too_many_arguments)]
pub(crate) fn depthwise_conv(
    x: GraphTensor,
    w: GraphTensor, // [C, 1, *kernel]
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads_begin: &[usize],
    pads_end: &[usize],
    ch: usize,
    group_out: usize,
    kernel_product: usize,
    spatial: usize,
) -> GraphTensor {
    let rank = 2 + spatial;

    let needs_pad = pads_begin.iter().any(|&p| p > 0) || pads_end.iter().any(|&p| p > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
        for i in 0..spatial {
            padding[2 + i] = (pads_begin[i].into(), pads_end[i].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Unfold the full [N, C, H+2p, W+2p] with kernel [1, 1, kH, kW]
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    kernel_full[2..(spatial + 2)].copy_from_slice(&kernel_shape[..spatial]);
    stride_full[2..(spatial + 2)].copy_from_slice(&strides[..spatial]);
    dilation_full[2..(spatial + 2)].copy_from_slice(&dilations[..spatial]);

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
    // Shape: [N, C, out_H, out_W, 1, 1, kH, kW]

    // Permute to [N, C, out_spatial..., k_all...]
    let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
    perm.push(0); // N
    perm.push(1); // C
    perm.extend(2..2 + spatial); // win_spatial
    perm.extend(rank..2 * rank); // all kernel dims
    let permuted = unfolded.permute(perm);

    let out_spatial_dims: Vec<Expression> = permuted.dims()[2..2 + spatial].to_vec();

    // Merge all kernel dims (including 1-size k_N, k_C) into kernel_product
    let target = 3 + spatial; // [N, C, spatial..., K]
    let mut patches = permuted;
    while patches.dims().len() > target {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    // patches: [N, C, out_H, ..., out_W, kernel_product]

    // Merge spatial into one: [N, C, out_spatial_product, kernel_product]
    for _ in 1..spatial {
        patches = patches.merge_dims(2, 3);
    }

    // Weight [C * group_out, 1, *kernel] -> [C, group_out, kernel_product]
    let mut w_flat = w;
    w_flat.shape =
        ShapeTracker::new_with_element_bits(vec![ch, group_out, kernel_product], w.dtype.bits());

    // patches: [N, C, out_spatial_product, kernel_product]
    // Expand to [N, C, group_out, out_spatial_product, kernel_product]
    let patches = patches.expand_dim(2, group_out);

    // Expand weight for broadcast: [1, C, group_out, out_spatial_product, kernel_product]
    let w_expanded = w_flat.expand_dim(0, 1).expand_dim(3, patches.dims()[3]);

    // Element-wise multiply and sum over kernel dim
    let product = patches * w_expanded;
    let mut out = product.sum(vec![4]).merge_dims(1, 2);
    // out: [N, C * group_out, out_spatial_product]

    // Restore spatial dimensions
    for i in (1..spatial).rev() {
        out = out.split_dims(2, out_spatial_dims[i]);
    }
    // out: [N, C, out_spatial_0, ..., out_spatial_{s-1}]

    out
}

/// Slice input channels for one group.
/// Caller must ensure `x` is already padded so no further padding is needed.
fn slice_channel_group(
    x: GraphTensor,
    g: usize,
    ch_per_group: usize,
    spatial: usize,
) -> GraphTensor {
    let start = g * ch_per_group;
    let end = start + ch_per_group;
    let rank = 2 + spatial;
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(rank);
    let dims = x.dims();
    slices.push((0.into(), dims[0]));
    slices.push((start.into(), end.into()));
    for dim in dims.iter().take(rank).skip(2) {
        slices.push((0.into(), *dim));
    }
    x.slice(slices)
}

/// Slice and flatten weight for one group: [C_out, C_in/g, *kernel] -> [group_out, flat_inner].
fn slice_weight_group(
    w: GraphTensor,
    g: usize,
    group_out: usize,
    flat_inner: usize,
) -> GraphTensor {
    let start = g * group_out;
    let end = start + group_out;
    let w_dims = w.dims();
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(w_dims.len());
    slices.push((start.into(), end.into()));
    for dim in w_dims.iter().skip(1) {
        slices.push((0.into(), *dim));
    }
    // Materialize through Add: binary op outputs are contiguous in Luminal, which makes the
    // following flatten safe for the sliced weight buffer.
    let w_sliced = w.slice(slices) + 0.0;
    let mut w_flat = w_sliced;
    w_flat.shape =
        ShapeTracker::new_with_element_bits(vec![group_out, flat_inner], w_sliced.dtype.bits());
    w_flat
}

/// Add bias [C_out] broadcast to [batch, C_out, spatial...]
fn add_conv_bias(out: GraphTensor, bias: GraphTensor, spatial: usize) -> GraphTensor {
    let out_dims = out.dims();
    let mut b = bias.expand_dim(0, 1); // [1, C_out]
    for i in 0..spatial {
        b = b.expand_dim(2 + i, out_dims[2 + i]);
    }
    out + b
}
