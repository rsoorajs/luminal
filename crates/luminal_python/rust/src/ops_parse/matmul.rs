use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

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
