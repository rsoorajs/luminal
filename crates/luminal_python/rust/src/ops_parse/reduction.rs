use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::get_int_attr;

/// Handle ReduceSum node: reduce tensor by summing along specified axes.
///
/// Supports multi-axis reduction, keepdims, and noop_with_empty_axes.
/// Bridges ONNX spec to luminal's single-axis .sum() by iterating axis-by-axis.
/// Opset 13+: axes come from second input; Opset 11: from "axes" attribute.
pub fn parse_reduce_sum_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    parse_reduce_op(
        node,
        tensors,
        known_values,
        "ReduceSum",
        |t, ax| t.sum(ax),
        |flat, _n| flat.sum(1),
    )
}

/// Handle ReduceMax node: computes the maximum along specified axes.
///
/// Supports multi-axis reduction, keepdims, and noop_with_empty_axes.
/// Bridges ONNX spec to luminal's single-axis .max() by iterating axis-by-axis.
/// Opset 13+: axes come from second input; Opset 11: from "axes" attribute.
pub fn parse_reduce_max_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    parse_reduce_op(
        node,
        tensors,
        known_values,
        "ReduceMax",
        |t, ax| t.max(ax),
        |flat, _n| flat.max(1),
    )
}

/// Handle ReduceMin node: computes the minimum along specified axes.
///
/// Supports multi-axis reduction, keepdims, and noop_with_empty_axes.
/// Bridges ONNX spec to luminal's single-axis .min() by iterating axis-by-axis.
/// Opset 13+: axes come from second input; Opset 11: from "axes" attribute.
pub fn parse_reduce_min_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    parse_reduce_op(
        node,
        tensors,
        known_values,
        "ReduceMin",
        |t, ax| t.min(ax),
        |flat, _n| flat.min(1),
    )
}

/// Handle ReduceMean node: computes the mean along specified axes.
///
/// Supports multi-axis reduction, keepdims, and noop_with_empty_axes.
/// Bridges ONNX spec to luminal's single-axis .mean() by iterating axis-by-axis.
/// Opset 13+: axes come from second input; Opset 11: from "axes" attribute.
pub fn parse_reduce_mean_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    // all_axes path uses sum(1)/N instead of mean() to avoid the Div<Expression>
    // path that creates an Iota+Cast chain triggering CUDA_ERROR_INVALID_VALUE.
    parse_reduce_op(
        node,
        tensors,
        known_values,
        "ReduceMean",
        |t, ax| t.mean(ax),
        |flat, n| flat.sum(1) / n as f32,
    )
}

fn parse_reduce_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    op_name: &str,
    per_axis_op: impl Fn(GraphTensor, usize) -> GraphTensor,
    all_axes_op: impl Fn(GraphTensor, usize) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        !node.input.is_empty(),
        "{} should have at least 1 input",
        op_name
    );
    assert!(
        node.output.len() == 1,
        "{} should have exactly 1 output",
        op_name
    );

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;

    let keepdims = get_int_attr(node, "keepdims", 1) != 0;
    let noop_with_empty_axes = get_int_attr(node, "noop_with_empty_axes", 0) != 0;

    let ndim = input.dims().len();

    // Resolve axes from second input (opset 13+) or from attribute (opset 11)
    let raw_axes: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_vals = known_values.get(&node.input[1]).ok_or_else(|| {
            format!(
                "{}: axes input '{}' must be a known constant",
                op_name, node.input[1]
            )
        })?;
        axes_vals.iter().map(|&v| v as i64).collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        attr.ints.clone()
    } else {
        vec![]
    };

    let output_name = &node.output[0];

    // Handle empty axes: noop or reduce all
    let raw_axes: Vec<i64> = if raw_axes.is_empty() {
        if noop_with_empty_axes {
            tensors.insert(output_name.clone(), input);
            trace!("Finished parse: {} Node (noop)", op_name);
            return Ok(());
        } else {
            (0..ndim as i64).collect()
        }
    } else {
        raw_axes
    };

    // Normalize negative axes and convert to usize
    let mut normalized_axes: Vec<usize> = raw_axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    normalized_axes.sort();
    normalized_axes.dedup();

    // Save original sorted axes for keepdims unsqueeze bookkeeping
    let sorted_axes = normalized_axes.clone();

    let input_dims = input.dims();

    if normalized_axes.len() == ndim {
        // All-axes reduction: flatten to [1, N] and reduce axis 1 → [1].
        // luminal's Expression::product() returns 0 for empty iterators, so a reduce
        // producing a 0-dim tensor causes CUDA to launch with grid (0,1,1), which is
        // invalid. Using [1, N] → reduce(1) → [1] avoids this entirely.
        let total: usize = input_dims
            .iter()
            .map(|d| d.to_usize().expect("reduce: dim must be concrete"))
            .product();
        let mut flat = input;
        flat.shape = ShapeTracker::new(vec![1, total]);
        let mut result = all_axes_op(flat, total);

        if keepdims {
            // Insert (ndim-1) additional size-1 dims to produce [1]*ndim
            for i in 1..ndim {
                result = result.unsqueeze(i);
            }
        }

        tensors.insert(output_name.clone(), result);
        trace!("Finished parse: {} Node (all-axes)", op_name);
        return Ok(());
    }

    // Partial reduction: iterative single-axis reduction
    let mut result = input;
    let mut current_axes = normalized_axes;
    for i in 0..current_axes.len() {
        let axis = current_axes[i];
        result = per_axis_op(result, axis);
        // Each reduction removes a dimension; shift subsequent axis indices down
        for item in current_axes.iter_mut().skip(i + 1) {
            if *item > axis {
                *item -= 1;
            }
        }
    }

    // Re-insert size-1 dims at original positions (ascending order keeps positions correct)
    if keepdims {
        for &axis in &sorted_axes {
            result = result.unsqueeze(axis);
        }
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}
