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
            "Add" => parse_add_node(node, tensors)?,
            "Mod" => parse_mod_node(node, tensors)?,
            "Sub" => parse_sub_node(node, tensors)?,
            "Mul" => parse_mul_node(node, tensors)?,
            "Div" => parse_div_node(node, tensors)?,
            "Sqrt" => parse_sqrt_node(node, tensors)?,
            "Transpose" => parse_transpose_node(node, tensors)?,
            "Floor" => parse_floor_node(node, tensors)?,
            "Sin" => parse_sin_node(node, tensors)?,
            "Neg" => parse_neg_node(node, tensors)?,
            "Cos" => parse_cos_node(node, tensors)?,
            "Pow" => parse_pow_node(node, tensors)?,
            "Sigmoid" => parse_sigmoid_node(node, tensors)?,
            "Tanh" => parse_tanh_node(node, tensors)?,
            "Relu" => parse_relu_node(node, tensors)?,
            "Abs" => parse_abs_node(node, tensors)?,
            "Clip" => parse_clip_node(node, tensors, known_values)?,
            "Equal" => parse_equal_node(node, tensors)?,
            "Where" => parse_where_node(node, tensors)?,
            "Constant" => parse_constant_node(node, tensors, cx, weight_data, known_values)?,
            "Cast" => parse_cast_node(node, tensors, weight_data, known_values)?,
            "MatMul" => parse_matmul_node(node, tensors)?,
            "Reshape" => parse_reshape_node(node, tensors, known_values)?,
            "Shape" => parse_shape_node(node, tensors, cx, weight_data, known_values)?,
            "Gather" => parse_gather_node(node, tensors, cx, weight_data, known_values)?,
            "GatherND" => parse_gathernd_node(node, tensors, cx, weight_data, known_values)?,
            "Less" => parse_less_node(node, tensors, known_values)?,
            "Greater" => parse_greater_node(node, tensors)?,
            "Min" => parse_min_node(node, tensors)?,
            "Max" => parse_max_node(node, tensors)?,
            "Identity" => parse_identity(node, tensors, known_values)?,
            "Unsqueeze" => parse_unsqueeze_node(node, tensors, known_values)?,
            "Squeeze" => parse_squeeze_node(node, tensors, known_values)?,
            "ReduceSum" => parse_reduce_sum_node(node, tensors, known_values)?,
            "ReduceMax" => parse_reduce_max_node(node, tensors, known_values)?,
            _ => {
                panic!("Missing Node {}", node.op_type)
            }
        }
    }

    Ok(())
}
