use std::collections::HashMap;

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
    parse_binary_broadcast_op(node, tensors, "Add", |a, b| a + b)
}

/// Handle Mod node: output = input[0] % input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_mod_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Mod", |a, b| a % b)
}

/// Handle Sub node: output = input[0] - input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_sub_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Sub", |a, b| a - b)
}

/// Handle Mul node: output = input[0] * input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_mul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Mul", |a, b| a * b)
}

/// Handle Div node: output = input[0] / input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_div_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Div", |a, b| a / b)
}

/// Parse Less node (ONNX element-wise less-than comparison).
///
/// Outputs 1.0 where a < b, 0.0 otherwise. Supports broadcasting.
pub fn parse_less_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Less", |a, b| a.lt(b))
}

/// Handle Pow node: input[0].pow(input[1])
pub fn parse_pow_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Pow", |a, b| a.pow(b))
}

/// Handle Equal node: element-wise equality comparison.
///
/// Outputs 1.0 where inputs are equal, 0.0 otherwise. Supports broadcasting.
pub fn parse_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Equal", |a, b| a.eq(b))
}

/// Parse Greater node (ONNX element-wise greater-than comparison).
///
/// Outputs 1.0 where a > b, 0.0 otherwise. Supports broadcasting.
pub fn parse_greater_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    // a > b  ≡  b < a
    parse_binary_broadcast_op(node, tensors, "Greater", |a, b| b.lt(a))
}

/// Handle LessOrEqual node: output = input[0] <= input[1]
pub fn parse_less_or_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "LessOrEqual", |a, b| a.le(b))
}

/// Handle GreaterOrEqual node: output = input[0] >= input[1]
pub fn parse_greater_or_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "GreaterOrEqual", |a, b| a.ge(b))
}

/// Handle And node: logical AND — output = input[0] * input[1]  (boolean: 1×1=1, else 0)
pub fn parse_and_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "And", |a, b| {
        a.cast(DType::F32) * b.cast(DType::F32)
    })
}

/// Handle Or node: logical OR — output = min(input[0] + input[1], 1.0)
pub fn parse_or_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Or", |a, b| {
        (a.cast(DType::F32) + b.cast(DType::F32)).minimum_f32(1.0)
    })
}

/// Handle Xor node: logical XOR — XOR on boolean tensors equals not-equal
pub fn parse_xor_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_binary_broadcast_op(node, tensors, "Xor", |a, b| a.ne(b))
}

/// Handle Min node: element-wise minimum over 2+ inputs with numpy-style broadcasting.
pub fn parse_min_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_variadic_broadcast_op(node, tensors, "Min", |a, b| a.minimum(b))
}

/// Handle Max node: element-wise maximum over 2+ inputs with numpy-style broadcasting.
pub fn parse_max_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_variadic_broadcast_op(node, tensors, "Max", |a, b| a.maximum(b))
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

fn parse_binary_broadcast_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() == 2,
        "{} should have 2 inputs, got {}",
        op_name,
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "{} should have 1 output, got {}",
        op_name,
        node.output.len()
    );
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input '{}'", op_name, node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("{}: missing input '{}'", op_name, node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = op(a_bc, b_bc);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}

fn parse_variadic_broadcast_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() >= 2,
        "{} needs at least two inputs, got {}",
        op_name,
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "{} nodes only have one output, got {}",
        op_name,
        node.output.len()
    );

    let mut result = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;

    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, input_name))?;
        let broadcast_shape = compute_broadcast_shape(&result.dims(), &rhs.dims());
        let lhs_bc = broadcast_to(result, &broadcast_shape);
        let rhs_bc = broadcast_to(rhs, &broadcast_shape);
        result = op(lhs_bc, rhs_bc);
    }

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}
