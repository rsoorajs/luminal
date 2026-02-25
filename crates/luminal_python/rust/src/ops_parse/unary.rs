use std::{collections::HashMap, ops::Neg};

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, get_float_attr, get_int_attr};

pub fn parse_sqrt_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Sqrt", |a| a.sqrt())
}

pub fn parse_sin_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Sin", |a| a.sin())
}

pub fn parse_neg_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Neg", |a| a.neg())
}

pub fn parse_cos_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Cos", |a| a.cos())
}

pub fn parse_sigmoid_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Sigmoid", |a| a.sigmoid())
}

pub fn parse_tanh_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Tanh", |a| a.tanh())
}

pub fn parse_relu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Relu", |a| a.relu())
}

pub fn parse_abs_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Abs", |a| a.abs())
}

pub fn parse_reciprocal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Reciprocal", |a| a.reciprocal())
}

/// Handle Softmax node: output = softmax(input[0], axis)
///
/// ONNX axis attribute defaults to -1 (last dimension, opset 13+).
/// Negative axis is normalized against the input rank.
pub fn parse_softmax_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Softmax Node");
    assert!(
        node.input.len() == 1,
        "Softmax nodes need to have one input, {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Softmax nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Softmax: missing input tensor '{}'", node.input[0]))?;

    let ndim = a.dims().len();
    let raw_axis = get_int_attr(node, "axis", -1);
    let axis = if raw_axis < 0 {
        (ndim as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };

    let result = a.softmax(axis);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Softmax Node");
    Ok(())
}

/// Handle Not node: logical NOT — output = 1.0 - input[0]
pub fn parse_not_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Not Node");
    assert!(
        node.input.len() == 1,
        "Not nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Not nodes only have one output, {} where present",
        node.output.len()
    );
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Not: missing input tensor '{}'", node.input[0]))?;
    let a_f32 = a.cast(DType::F32);
    let result = 1.0_f32 - a_f32;
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: Not Node");
    Ok(())
}

/// Handle Clip node: output = clip(input[0], min, max)
///
/// Equivalent to torch.clamp. min and max are optional tensor inputs
/// (typically constants) residing in known_values.
pub fn parse_clip_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Clip Node");
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Clip: missing input tensor '{}'", node.input[0]))?;

    // input[1] = min (optional), input[2] = max (optional)
    let min_name = node.input.get(1).map(String::as_str).unwrap_or("");
    let max_name = node.input.get(2).map(String::as_str).unwrap_or("");

    let min_val = if min_name.is_empty() {
        None
    } else {
        known_values.get(min_name).map(|v| v[0])
    };
    let max_val = if max_name.is_empty() {
        None
    } else {
        known_values.get(max_name).map(|v| v[0])
    };

    let result = match (min_val, max_val) {
        (Some(lo), Some(hi)) => a.clip(lo, hi),
        (Some(lo), None) => a.maximum_f32(lo),
        (None, Some(hi)) => a.minimum_f32(hi),
        (None, None) => a,
    };

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Clip Node");
    Ok(())
}

/// Handle Floor node: output = floor(input[0])
///
/// Implemented as: trunc(x) - (x < trunc(x) ? 1 : 0)
/// where trunc is truncation toward zero via cast to Int then back to F32.
/// This correctly handles negative non-integer values (e.g. floor(-1.5) = -2).
pub fn parse_floor_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Floor Node");
    assert!(
        node.input.len() == 1,
        "Floor nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Floor nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Floor: missing input tensor '{}'", node.input[0]))?;

    // trunc(x): truncation toward zero
    let trunc = a.cast(DType::Int).cast(DType::F32);
    // For negative non-integers, x < trunc(x), so subtract 1
    // Cast lt result (Bool) to F32 before arithmetic
    let adjustment = a.lt(trunc).cast(DType::F32);
    let result = trunc - adjustment;
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Floor Node");

    Ok(())
}

/// Handle Ceil node: output = ceil(input[0])
///
/// Implemented as: trunc(x) + (x > trunc(x) ? 1 : 0)
/// where trunc is truncation toward zero via cast to Int then back to F32.
/// This correctly handles positive non-integer values (e.g. ceil(1.5) = 2).
pub fn parse_ceil_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Ceil Node");
    assert!(
        node.input.len() == 1,
        "Ceil nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Ceil nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Ceil: missing input tensor '{}'", node.input[0]))?;

    // trunc(x): truncation toward zero
    let trunc = a.cast(DType::Int).cast(DType::F32);
    // For positive non-integers, x > trunc(x), so add 1
    let adjustment = a.gt(trunc).cast(DType::F32);
    let result = trunc + adjustment;
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Ceil Node");

    Ok(())
}

pub fn parse_cast_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Cast Node");
    assert!(node.input.len() == 1, "Cast should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Cast: missing input tensor '{}'", node.input[0]))?;

    // ONNX data type enum → luminal DType
    let to = get_int_attr(node, "to", 1);
    let dtype = match to {
        1 => DType::F32,     // FLOAT
        10 => DType::F16,    // FLOAT16
        16 => DType::Bf16,   // BFLOAT16
        6 | 7 => DType::Int, // INT32, INT64
        9 => DType::F32,     // BOOL → treat as F32 (0.0/1.0)
        11 => DType::F32,    // DOUBLE → F32 (downcast)
        _ => DType::F32,     // fallback
    };

    let cast_result = input.cast(dtype);
    let output_name = &node.output[0];

    // Use the *1.0 workaround when:
    // 1. cast() was a no-op (input already has target dtype — same node returned), OR
    // 2. source dtype is Int (e.g., ONNX INT32/INT64 → F32):
    //    the CUDA backend lacks a Cast(Int→F32) kernel; since all runtime data is
    //    already stored as F32 (Python converts inputs via .float()), this cast is
    //    semantically a no-op and *1.0 produces a CUDA-executable Mul node instead.
    let result = if cast_result.id == input.id || input.dtype == DType::Int {
        let src_dims = input.dims();
        let shape: Vec<usize> = src_dims
            .iter()
            .map(|d| d.to_usize().expect("cast no-op: dim must be concrete"))
            .collect();
        let one = input.graph().constant_float(1.0);
        let one_expanded = broadcast_to(one, &shape);
        input * one_expanded
    } else {
        cast_result
    };

    tensors.insert(output_name.clone(), result);

    // Propagate known values (cast is a no-op for our f32 storage)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded = if to == 9 {
            // Bool cast: non-zero → 1.0, zero → 0.0
            vals.iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect()
        } else if to == 6 || to == 7 {
            // Int cast: truncate
            vals.iter().map(|&v| (v as i64) as f32).collect()
        } else {
            vals
        };
        known_values.insert(output_name.clone(), folded.clone());
        // Register constant-folded result for CUDA initialization
        weight_data.push((output_name.clone(), folded));
    }

    trace!("Finished parse: Cast Node");
    Ok(())
}

fn parse_unary_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() == 1,
        "{} should have 1 input, got {}",
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
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;
    let result = op(a);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}

/// Handle IsNaN node: return 1.0 where input is NaN, 0.0 otherwise.
///
/// Note: luminal's ne(x, x) returns 0 for normal floats and may not correctly
/// detect actual NaN values (hardware-dependent). For inference with non-NaN
/// inputs this is correct.
pub fn parse_isnan_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "IsNaN", |a| a.ne(a))
}

/// Handle Erf node: output = erf(input[0])
///
/// Uses the Abramowitz & Stegun 7.1.26 polynomial approximation (max error < 1.5e-7):
///   For x ≥ 0: erf(x) ≈ 1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) · exp(-x²)
///   where t = 1 / (1 + 0.3275911·x)
///     a1 =  0.254829592
///     a2 = -0.284496736
///     a3 =  1.421413741
///     a4 = -1.453152027
///     a5 =  1.061405429
/// Extended to all x via odd symmetry: erf(-x) = -erf(x).
pub fn parse_erf_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Erf", |x| {
        let a = x.abs();
        let t = (1.0_f32 + 0.3275911_f32 * a).reciprocal();
        // Horner evaluation of a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        // poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))))
        let h = t * 1.061405429_f32 - 1.453152027_f32;  // a4 + a5*t
        let h = t * h + 1.421413741_f32;
        let h = t * h - 0.284496736_f32;
        let h = t * h + 0.254829592_f32;
        let poly = t * h;
        let erf_abs = 1.0_f32 - poly * (-a * a).exp();
        x.sign() * erf_abs
    })
}

/// Handle LayerNormalization node (opset 17).
///
/// Inputs: X (required), scale (required), bias (optional)
/// Attributes: axis (default -1), epsilon (default 1e-5)
/// Normalizes over axes [axis, axis+1, ..., rank-1], then applies scale and bias.
/// Only output 0 (the normalized result) is wired; outputs 1/2 (mean, inv_std_var)
/// are training-only and not supported for inference.
pub fn parse_layernorm_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: LayerNormalization Node");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("LayerNorm: missing input '{}'", node.input[0]))?;
    let scale = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("LayerNorm: missing scale '{}'", node.input[1]))?;

    let ndim = input.dims().len();
    let axis_raw = get_int_attr(node, "axis", -1);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };
    let epsilon = get_float_attr(node, "epsilon", 1e-5);
    let axes: Vec<usize> = (axis..ndim).collect();

    let mut result = input.layer_norm(axes, epsilon);

    // Apply scale (broadcast to input shape)
    let input_shape: Vec<usize> = input.dims().iter().map(|d| d.to_usize().unwrap()).collect();
    result = result * broadcast_to(scale, &input_shape);

    // Apply optional bias
    if node.input.len() > 2 && !node.input[2].is_empty() {
        let bias = *tensors
            .get(&node.input[2])
            .ok_or_else(|| format!("LayerNorm: missing bias '{}'", node.input[2]))?;
        result = result + broadcast_to(bias, &input_shape);
    }

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: LayerNormalization Node");
    Ok(())
}
