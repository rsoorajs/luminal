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

/// Handle Mod node: output = input[0] % input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_mod_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Mod Node");
    assert!(
        node.input.len() == 2,
        "Mod nodes need to have two inputs {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Mod nodes only have one input, {} where present",
        node.input.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Mod: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Mod: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc % b_bc;
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Mod Node");

    return Ok(());
}

/// Handle Sub node: output = input[0] - input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_sub_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
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

    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sub: missing input tensor '{}'", node.input[0]))?;

    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Sub: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let output = a_bc.sub(b_bc);
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

/// Handle Div node: output = input[0] / input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_div_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
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

/// Parse Less node (ONNX element-wise less-than comparison).
///
/// Outputs 1.0 where a < b, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_less_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Less Node");
    assert!(node.input.len() == 2, "Less should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Less: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Less: missing input tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    let result = a_bc.lt(b_bc);

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Less Node");
    Ok(())
}
