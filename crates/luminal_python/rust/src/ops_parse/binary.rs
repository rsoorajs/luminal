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

    Ok(())
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

    Ok(())
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

    Ok(())
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

    Ok(())
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

    Ok(())
}

/// Parse Less node (ONNX element-wise less-than comparison).
///
/// Outputs 1.0 where a < b, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_less_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    _known_values: &mut HashMap<String, Vec<f32>>,
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

/// Handle Pow node: input[0].pow(input[1])
///
pub fn parse_pow_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Pow Node");
    assert!(node.input.len() == 2, "Pow should have 2 inputs");
    assert!(
        node.output.len() == 1,
        "Pow nodes only have one output, got {}",
        node.output.len()
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Pow: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Pow: missing input tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    let result = a_bc.pow(b_bc);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Pow Node");
    Ok(())
}

/// Handle Equal node: element-wise equality comparison.
///
/// Outputs 1.0 where inputs are equal, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Equal should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Equal: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Equal: missing input tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    let result = a_bc.eq(b_bc);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    Ok(())
}

/// Parse Greater node (ONNX element-wise greater-than comparison).
///
/// Outputs 1.0 where a > b, 0.0 otherwise. Supports broadcasting.
pub fn parse_greater_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Greater Node");
    assert!(node.input.len() == 2, "Greater should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Greater: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Greater: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    // a > b  ≡  b < a
    let result = b_bc.lt(a_bc);

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Greater Node");
    Ok(())
}

/// Handle Min node: element-wise minimum over 2+ inputs with numpy-style broadcasting.
pub fn parse_min_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Min Node");
    assert!(
        node.input.len() >= 2,
        "Min nodes need at least two inputs, got {}",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Min nodes only have one output, got {}",
        node.output.len()
    );

    let output_name = &node.output[0];
    let mut result = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Min: missing input tensor '{}'", node.input[0]))?;

    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("Min: missing input tensor '{}'", input_name))?;
        let broadcast_shape = compute_broadcast_shape(&result.dims(), &rhs.dims());
        let lhs_bc = broadcast_to(result, &broadcast_shape);
        let rhs_bc = broadcast_to(rhs, &broadcast_shape);
        result = lhs_bc.minimum(rhs_bc);
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Min Node");
    Ok(())
}

/// Handle Max node: element-wise maximum over 2+ inputs with numpy-style broadcasting.
pub fn parse_max_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Max Node");
    assert!(
        node.input.len() >= 2,
        "Max nodes need at least two inputs, got {}",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Max nodes only have one output, got {}",
        node.output.len()
    );

    let output_name = &node.output[0];
    let mut result = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Max: missing input tensor '{}'", node.input[0]))?;

    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("Max: missing input tensor '{}'", input_name))?;
        let broadcast_shape = compute_broadcast_shape(&result.dims(), &rhs.dims());
        let lhs_bc = broadcast_to(result, &broadcast_shape);
        let rhs_bc = broadcast_to(rhs, &broadcast_shape);
        result = lhs_bc.maximum(rhs_bc);
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Max Node");
    Ok(())
}

/// Handle LessOrEqual node: output = input[0] <= input[1]
pub fn parse_less_or_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: LessOrEqual Node");
    assert!(node.input.len() == 2, "LessOrEqual should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("LessOrEqual: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("LessOrEqual: missing input tensor '{}'", node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.le(b_bc);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: LessOrEqual Node");
    Ok(())
}

/// Handle GreaterOrEqual node: output = input[0] >= input[1]
pub fn parse_greater_or_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: GreaterOrEqual Node");
    assert!(node.input.len() == 2, "GreaterOrEqual should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("GreaterOrEqual: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("GreaterOrEqual: missing input tensor '{}'", node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.ge(b_bc);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: GreaterOrEqual Node");
    Ok(())
}

/// Handle And node: logical AND — output = input[0] * input[1]  (boolean: 1×1=1, else 0)
pub fn parse_and_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: And Node");
    assert!(node.input.len() == 2, "And should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("And: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("And: missing input tensor '{}'", node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a.cast(DType::F32), &broadcast_shape);
    let b_bc = broadcast_to(b.cast(DType::F32), &broadcast_shape);
    let result = a_bc.mul(b_bc);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: And Node");
    Ok(())
}

/// Handle Or node: logical OR — output = min(input[0] + input[1], 1.0)
pub fn parse_or_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Or Node");
    assert!(node.input.len() == 2, "Or should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Or: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Or: missing input tensor '{}'", node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a.cast(DType::F32), &broadcast_shape);
    let b_bc = broadcast_to(b.cast(DType::F32), &broadcast_shape);
    let result = a_bc.add(b_bc).minimum_f32(1.0);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: Or Node");
    Ok(())
}

/// Handle Xor node: logical XOR — XOR on boolean tensors equals not-equal
pub fn parse_xor_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Xor Node");
    assert!(node.input.len() == 2, "Xor should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Xor: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Xor: missing input tensor '{}'", node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.ne(b_bc);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: Xor Node");
    Ok(())
}

/// Handle Where node: conditional select — output[i] = condition[i] ? x[i] : y[i]
pub fn parse_where_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 3, "Where should have 3 inputs");
    let condition = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Where: missing condition tensor '{}'", node.input[0]))?;
    let x = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Where: missing X tensor '{}'", node.input[1]))?;
    let y = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("Where: missing Y tensor '{}'", node.input[2]))?;

    let output_name = &node.output[0];

    let result = x.cond(condition, y);
    tensors.insert(output_name.clone(), result);
    Ok(())
}
