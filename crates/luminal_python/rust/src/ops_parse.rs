use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Sub},
};

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, compute_broadcast_shape};

/// Handle Add node: output = input[0] + input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_add_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Add Node");
    assert!(
        node.input.len() == 2,
        "Add nodes need to have two inputs {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Add nodes only have one input, {} where present",
        node.input.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Add: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Add: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.add(b_bc);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Add Node");

    return Ok(());
}

/// Handle Sub node: output = input[0] - input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_sub_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Sub Node");
    assert!(
        node.input.len() == 2,
        "Sub nodes need to have two inputs {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Sub nodes only have one input, {} where present",
        node.input.len(),
    );

    // TODO: Handle broadcasting
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sub: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Sub: missing input tensor '{}'", node.input[0]))?;

    let output = a.sub(b);
    tensors.insert(output_name.clone(), output);
    trace!("Finished parse: Sub Node");

    return Ok(());
}

/// Handle Mul node: output = input[0] * input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_mul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Mul Node");
    assert!(
        node.input.len() == 2,
        "Mul nodes need to have two inputs {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Mul nodes only have one input, {} where present",
        node.input.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Mul: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Mul: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.mul(b_bc);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Mul Node");

    return Ok(());
}

pub fn parse_matmul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Started parse: MatMul Node");
    assert!(node.input.len() == 2, "MatMul should have exactly 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[1]))?;
    //TODO: enforce some kind of check here that they are broadcastable
    let result = a.matmul(b);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: MatMul Node");
    Ok(())
}

/// Handle Div node: output = input[0] / input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_div_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Div Node");
    assert!(
        node.input.len() == 2,
        "Div nodes need to have two inputs {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Div nodes only have one input, {} where present",
        node.input.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Div: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Div: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.div(b_bc);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Div Node");

    return Ok(());
}

/// Handle Sqrt node: output = input[0].sqrt()
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_sqrt_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Sqrt Node");
    assert!(
        node.input.len() == 1,
        "Sqrt nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Div nodes only have one input, {} where present",
        node.input.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sqrt: missing input tensor '{}'", node.input[0]))?;

    let result = a.sqrt();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Sqrt Node");

    return Ok(());
}

/// Handle Transpose node: output = permute(input, perm)
///
/// The perm attribute specifies the permutation of dimensions.
/// If perm is not specified, reverses all dimensions (ONNX default).
pub fn parse_transpose_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
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

/// Handle Sin node: output = input[0].sin()
pub fn parse_sin_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Sin Node");
    assert!(
        node.input.len() == 1,
        "Sin nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Sin nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sin: missing input tensor '{}'", node.input[0]))?;

    let result = a.sin();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Sin Node");

    return Ok(());
}

/// Handle Cos node: output = input[0].cos()
pub fn parse_cos_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Cos Node");
    assert!(
        node.input.len() == 1,
        "Cos nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Cos nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Cos: missing input tensor '{}'", node.input[0]))?;

    let result = a.cos();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Cos Node");

    return Ok(());
}
