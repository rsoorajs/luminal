use std::{collections::HashMap, ops::Neg};

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, get_int_attr};

/// Handle Sqrt node: output = input[0].sqrt()
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_sqrt_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
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

    Ok(())
}

/// Handle Sin node: output = input[0].sin()
pub fn parse_sin_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
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

    Ok(())
}

/// Handle Neg node: output = -input[0]
pub fn parse_neg_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Neg Node");
    assert!(
        node.input.len() == 1,
        "Neg nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Neg nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Neg: missing input tensor '{}'", node.input[0]))?;

    let result = a.neg();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Neg Node");

    Ok(())
}

/// Handle Cos node: output = input[0].cos()
pub fn parse_cos_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
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

    Ok(())
}

pub fn parse_sigmoid_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Sigmoid Node");
    assert!(
        node.input.len() == 1,
        "Sigmoid nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Sigmoid nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sigmoid: missing input tensor '{}'", node.input[0]))?;

    let result = a.sigmoid();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Sigmoid Node");

    Ok(())
}

pub fn parse_tanh_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Tanh Node");
    assert!(
        node.input.len() == 1,
        "Tanh nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Tanh nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Tanh: missing input tensor '{}'", node.input[0]))?;

    let result = a.tanh();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Tanh Node");

    Ok(())
}

pub fn parse_relu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Relu Node");
    assert!(
        node.input.len() == 1,
        "Relu nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Relu nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Relu: missing input tensor '{}'", node.input[0]))?;

    let result = a.relu();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Relu Node");

    Ok(())
}

pub fn parse_abs_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Abs Node");
    assert!(
        node.input.len() == 1,
        "Abs nodes need to have one input {} where present",
        node.input.len()
    );

    assert!(
        node.output.len() == 1,
        "Abs nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Abs: missing input tensor '{}'", node.input[0]))?;

    let result = a.abs();
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Abs Node");

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
