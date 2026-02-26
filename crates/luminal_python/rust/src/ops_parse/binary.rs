use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, compute_broadcast_shape};

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

pub fn parse_binary_broadcast_op(
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

pub fn parse_variadic_broadcast_op(
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
