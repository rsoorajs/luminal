use std::collections::HashMap;

use luminal::prelude::*;
use onnx_protobuf::NodeProto;

use crate::ops_parse::*;

pub fn process_onnx_nodes(
    nodes: &[NodeProto],
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    for node in nodes {
        match node.op_type.as_str() {
            "Add" => parse_add_node(node, tensors, cx, weight_data, known_values)?,
            "Sub" => parse_sub_node(node, tensors, cx, weight_data, known_values)?,
            "Mul" => parse_mul_node(node, tensors, cx, weight_data, known_values)?,
            "Div" => parse_div_node(node, tensors, cx, weight_data, known_values)?,
            "MatMul" => parse_matmul_node(node, tensors)?,
            _ => {
                panic!("Missing Node {}", node.op_type)
            }
        }
    }

    Ok(())
}
