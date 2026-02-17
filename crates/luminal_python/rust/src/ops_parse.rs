use std::{
    collections::HashMap,
    ops::{Add, Div, Mul, Sub},
};

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to, compute_broadcast_shape, get_int_attr};

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

/// Handle Mod node: output = input[0] % input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_mod_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
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

    return Ok(());
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

/// Handle Constant node: creates a tensor from embedded data in the node attributes.
///
/// Supports FLOAT, INT64, INT32, and FLOAT64 data types (all converted to f32).
/// The resulting tensor is registered as a known constant for downstream folding.
pub fn parse_constant_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Constant Node");
    assert!(
        node.output.len() == 1,
        "Constant should have exactly one output"
    );

    // Find the "value" attribute (type TENSOR)
    let value_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| "Constant node missing 'value' attribute".to_string())?;

    let tensor_proto = value_attr
        .t
        .as_ref()
        .ok_or_else(|| "Constant 'value' attribute has no TensorProto".to_string())?;

    // Determine shape: empty dims = scalar = [1] for luminal
    let shape: Vec<usize> = if tensor_proto.dims.is_empty() {
        vec![1]
    } else {
        tensor_proto.dims.iter().map(|&d| d as usize).collect()
    };

    // Extract float data based on data_type
    let floats: Vec<f32> = match tensor_proto.data_type {
        1 => {
            // FLOAT (f32)
            if !tensor_proto.float_data.is_empty() {
                tensor_proto.float_data.clone()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        7 => {
            // INT64
            // There is a cast from Int64 -> f32 here because Luminal does not support f32
            if !tensor_proto.int64_data.is_empty() {
                tensor_proto.int64_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()
            }
        }
        6 => {
            // INT32
            // There is a cast from Int32 -> f32 here because Luminal does not support f32
            if !tensor_proto.int32_data.is_empty() {
                tensor_proto.int32_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()
            }
        }
        9 => {
            // Bool
            // Bools are stored as bytes in raw_data or as int32 in int32_data
            if !tensor_proto.int32_data.is_empty() {
                tensor_proto
                    .int32_data
                    .iter()
                    .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                tensor_proto
                    .raw_data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect()
            }
        }
        11 => {
            // FLOAT64 (f64)
            // There is a cast from f64 -> f32 here because Luminal does not support f32
            // TODO: add f64 as this will loss information, this is a bad approach
            tensor_proto
                .raw_data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        dt => return Err(format!("Constant node: unsupported data_type {}", dt)),
    };

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), shape);
    tensors.insert(output_name.clone(), tensor);
    known_values.insert(output_name.clone(), floats.clone());
    weight_data.push((output_name.clone(), floats));

    trace!("Finished parse: Constant Node");
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

    let result = input.cast(dtype);
    let output_name = &node.output[0];
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

/// Handle Reshape node: change the tensor's shape without modifying data.
///
/// The target shape is read from the second input (must be a known constant).
/// Supports -1 (infer from total elements) and 0 (copy from input dimension).
/// Non-contiguous tensors (e.g., from Expand) are materialized before reshaping.
pub fn parse_reshape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Started parse: Reshape");
    assert!(
        node.input.len() == 2,
        "Reshape should have exactly 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Reshape: missing input tensor '{}'", node.input[0]))?;

    let shape_data = known_values.get(&node.input[1]).ok_or_else(|| {
        format!(
            "Reshape: shape input '{}' must be a known constant",
            node.input[1]
        )
    })?;

    // Compute total elements for resolving -1
    let input_dims = input.dims();
    let total_elements: usize = input_dims
        .iter()
        .map(|d| d.to_usize().expect("Reshape: input dims must be concrete"))
        .product();

    // Resolve target shape, handling -1 (infer) and 0 (copy from input)
    let mut target_shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
    // First pass: resolve 0 (copy from input at same position)
    for i in 0..target_shape.len() {
        if target_shape[i] == 0 {
            target_shape[i] = input_dims[i].to_usize().unwrap_or(1) as i64;
        }
    }
    // Second pass: resolve -1 (infer from total elements)
    let known_product: i64 = target_shape.iter().filter(|&&d| d > 0).product();
    for d in target_shape.iter_mut() {
        if *d == -1 {
            *d = total_elements as i64 / known_product;
        }
    }
    let final_shape: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();

    let mut result = input;
    // If tensor is not contiguous (e.g., has broadcast strides from Expand),
    // materialize it before reshaping by multiplying by 1.0.
    // This forces a contiguous copy through the binary op mechanism.
    if !result.shape.is_contiguous() {
        let one = result.graph().constant_float(1.0);
        let src_dims = result.dims();
        let broadcast_shape: Vec<usize> = src_dims
            .iter()
            .map(|d| d.to_usize().expect("dim must be concrete"))
            .collect();
        let one_expanded = broadcast_to(one, &broadcast_shape);
        result *= one_expanded;
    }
    result.shape = ShapeTracker::new(final_shape);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values (reshape doesn't change data, just layout)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    trace!("Finished parse: Reshape");
    Ok(())
}

/// Handle Shape node: extract the shape of the input tensor as a 1D constant.
///
/// All dimensions must be statically known. The shape values are stored as
/// known constants for downstream operations (Reshape, Expand, etc.).
pub fn parse_shape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Started parse: Shape");
    assert!(node.input.len() == 1, "Shape should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Shape: missing input tensor '{}'", node.input[0]))?;

    let dims = input.dims();
    let shape_values: Vec<f32> = dims
        .iter()
        .map(|d| {
            d.to_usize()
                .expect("Shape: all dimensions must be concrete") as f32
        })
        .collect();

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), vec![shape_values.len()]);
    tensors.insert(output_name.clone(), tensor);
    known_values.insert(output_name.clone(), shape_values.clone());
    weight_data.push((output_name.clone(), shape_values));
    trace!("Finished parse: Shape");
    Ok(())
}

/// Parse Lessnode (ONNX element-wise less-tha comparison).
///
/// Outputs 1.0 where a < b, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_less_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
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
    Ok(())
}

/// Handle Gather node: index into a tensor along a specified axis.
///
/// For 1D data, performs simple element-wise gathering. For ND data, gathers
/// slices along the specified axis. Supports arbitrary axis values.
/// Supports constant folding when both data and indices are known.
pub fn parse_gather_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Gather should have 2 inputs");

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gather: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let data_ndim = data_dims.len();

    // Handle axis (support negative indexing)
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (data_ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // If both inputs are known, fully constant-fold
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let output_name = &node.output[0];

        // Get concrete data dimensions
        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: data dims must be concrete"))
            .collect();

        // Get indices tensor shape
        let indices_tensor = tensors.get(&node.input[1]);
        let indices_shape: Vec<usize> = if let Some(idx_tensor) = indices_tensor {
            idx_tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("Gather: indices dims must be concrete"))
                .collect()
        } else {
            vec![vidx.len()]
        };

        // Compute output shape: data_shape[0..axis] + indices_shape + data_shape[axis+1..]
        let mut output_shape: Vec<usize> = data_shape[..axis].to_vec();
        output_shape.extend(&indices_shape);
        output_shape.extend(&data_shape[axis + 1..]);

        // Gather with arbitrary axis
        let output_size: usize = output_shape.iter().product();
        let mut folded = vec![0.0f32; output_size.max(1)];

        if output_size > 0 {
            let pre_size: usize = data_shape[..axis].iter().product::<usize>().max(1);
            let post_size: usize = data_shape[axis + 1..].iter().product::<usize>().max(1);
            let indices_size: usize = vidx.len();
            let axis_dim = data_shape[axis] as i64;

            for pre in 0..pre_size {
                for (idx_pos, &idx_val) in vidx.iter().enumerate() {
                    // Normalize negative indices: ONNX allows -1 for last element, etc.
                    let idx_raw = idx_val as i64;
                    let idx = (((idx_raw % axis_dim) + axis_dim) % axis_dim) as usize;
                    for post in 0..post_size {
                        let data_flat =
                            pre * (data_shape[axis] * post_size) + idx * post_size + post;
                        let out_flat =
                            pre * (indices_size * post_size) + idx_pos * post_size + post;
                        folded[out_flat] = vdata[data_flat];
                    }
                }
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gather: missing indices tensor '{}'", node.input[1]))?;

    // Normalize negative indices: ONNX allows -1 for last element, -2 for second-to-last, etc.
    // Use conditional normalization instead of modulo to avoid floating-point precision loss:
    // if index < 0 then index + axis_dim else index
    let axis_dim = data_dims[axis].to_usize().ok_or_else(|| {
        "Gather: axis dimension must be concrete for index normalization".to_string()
    })?;
    let axis_dim_f32 = axis_dim as f32;

    // This is adding other nodes to the graph other then just a gather
    let zero = indices_raw
        .graph()
        .constant_float(0.0)
        .expand_rhs(indices_raw.shape);
    let adjustment = indices_raw
        .graph()
        .constant_float(axis_dim_f32)
        .expand_rhs(indices_raw.shape);
    let is_negative = indices_raw.lt(zero); // Returns 1.0 for negative, 0.0 for non-negative
    let indices = indices_raw + (is_negative * adjustment);

    let result = if data_ndim == 1 {
        // 1D data: simple flat element-wise gather
        data.gather(indices.cast(DType::Int))
    } else if axis == 0 {
        // ND data with axis=0: gather entire slices along first dimension
        gather_axis0(data, indices, &data_dims)?
    } else {
        // General case: axis != 0
        // Strategy: permute to bring axis to front, gather, permute back

        // Build permutation to move axis to position 0
        let mut perm: Vec<usize> = (0..data_ndim).collect();
        perm.remove(axis);
        perm.insert(0, axis);

        // Permute data
        let permuted_data = data.permute(perm);
        let permuted_dims = permuted_data.dims();

        // Gather on axis 0 of permuted data
        let gathered = gather_axis0(permuted_data, indices, &permuted_dims)?;

        // Compute gathered shape
        let idx_dims = indices.dims();
        let mut gathered_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &permuted_dims[1..] {
            gathered_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(gathered_shape.clone());

        // Build inverse permutation to restore original axis order
        let indices_rank = idx_dims.len();
        let mut inv_perm: Vec<usize> = Vec::with_capacity(reshaped.dims().len());

        // Dimensions before original axis (they're after indices in gathered result)
        for i in 0..axis {
            inv_perm.push(indices_rank + i);
        }
        // The indices dimensions
        for i in 0..indices_rank {
            inv_perm.push(i);
        }
        // Dimensions after original axis
        for i in axis..(data_ndim - 1) {
            inv_perm.push(indices_rank + i);
        }

        reshaped.permute(inv_perm)
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    Ok(())
}

/// Helper function for gathering along axis 0
fn gather_axis0(
    data: GraphTensor,
    indices: GraphTensor,
    data_dims: &[Expression],
) -> Result<GraphTensor, String> {
    let inner_dim: usize = data_dims[1..]
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "Gather: inner dimensions must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product();

    // Compute flat_indices = indices * inner_dim + offsets
    // IMPORTANT: Use Int arithmetic to avoid f32 precision loss for large indices.
    // f32 only has 24-bit mantissa, so indices > 16.7M lose precision.
    // For vocab_size=50304, embed_dim=768: max flat index = 38.6M
    let indices_int = indices.cast(DType::Int);
    let inner_dim_tensor = indices
        .graph()
        .constant(inner_dim as i32)
        .expand_rhs(indices_int.shape);
    let scaled = indices_int * inner_dim_tensor;
    let idx_ndim = indices.dims().len();
    let scaled_expanded = scaled.expand_dim(idx_ndim, inner_dim);

    // Create column offsets [0, 1, ..., inner_dim-1] and broadcast to indices shape
    // Keep as Int for precise arithmetic
    let offsets = data.graph().arange(inner_dim); // arange returns Int
    let idx_dims = indices.dims();
    let mut offsets_expanded = offsets;
    for i in (0..idx_dims.len()).rev() {
        offsets_expanded = offsets_expanded.expand_dim(0, idx_dims[i]);
    }

    let flat_indices = scaled_expanded + offsets_expanded; // Both Int, result is Int

    // Gather using flat indices into contiguous data buffer
    let gathered = data.gather(flat_indices);

    // Reshape from [..., inner_dim] to [..., D1, D2, ..., Dk]
    if data_dims.len() > 2 {
        let mut output_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &data_dims[1..] {
            output_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(output_shape);
        Ok(reshaped)
    } else {
        Ok(gathered)
    }
}
