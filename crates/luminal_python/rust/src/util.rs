use std::{fs, path::Path};

use luminal::{
    prelude::{DType, GraphTensor},
    shape::Expression,
};
use onnx_protobuf::{NodeProto, ValueInfoProto};

// Given a Value from the Onnx proto return its tensor Shape, if it exists
// Note: some times pytorch will create tensors with a 0 shape
// we might want to handle, 0 shape and No shape as seperate ideas
pub fn get_shape_for_onnx_value(value: &onnx_protobuf::ValueInfoProto) -> Vec<usize> {
    if let Some(type_proto) = value.type_.as_ref()
        && let Some(onnx_protobuf::type_proto::Value::TensorType(tensor)) = &type_proto.value
        && let Some(shape) = tensor.shape.as_ref()
    {
        // Scalar (0-dim) tensors have an empty dim list; represent as [1] in luminal
        if shape.dim.is_empty() {
            return vec![1];
        }
        return shape
            .dim
            .iter()
            .map(|dimension| {
                if let Some(onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(v)) =
                    &dimension.value
                {
                    *v as usize
                } else {
                    1
                }
            })
            .collect();
    }

    vec![]
}

/// Extract DType from ONNX ValueInfoProto
pub fn get_dtype_for_onnx_value(value: &ValueInfoProto) -> DType {
    if let Some(type_proto) = value.type_.as_ref() {
        if let Some(tensor_type) = type_proto.value.as_ref().and_then(|v| {
            if let onnx_protobuf::type_proto::Value::TensorType(tt) = v {
                Some(tt)
            } else {
                None
            }
        }) {
            // ONNX data type enum to luminal DType
            return match tensor_type.elem_type {
                1 => DType::F32,   // FLOAT
                10 => DType::F16,  // FLOAT16
                16 => DType::Bf16, // BFLOAT16
                6 => DType::Int,   // INT32
                7 => DType::Int,   // INT64
                9 => DType::Bool,  // BOOL
                11 => DType::F32,  // DOUBLE (downcast to F32, same as Cast does)
                _ => DType::F32,   // Default fallback
            };
        }
    }
    DType::F32 // Fallback if no type information
}

/// Compute the broadcast output shape for two tensors (numpy rules: element-wise max).
pub fn compute_broadcast_shape(a: &[Expression], b: &[Expression]) -> Vec<usize> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1usize; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < max_rank - a.len() {
            1
        } else {
            a[i - (max_rank - a.len())]
                .to_usize()
                .expect("broadcast: dim must be concrete")
        };
        let b_dim = if i < max_rank - b.len() {
            1
        } else {
            b[i - (max_rank - b.len())]
                .to_usize()
                .expect("broadcast: dim must be concrete")
        };
        result[i] = a_dim.max(b_dim);
    }
    result
}

/// Broadcast a tensor's shape to match a target shape (numpy-style broadcasting).
/// Left-pads with size-1 dims, then expands dims that are 1 to match target.
pub fn broadcast_to(mut tensor: GraphTensor, target_shape: &[usize]) -> GraphTensor {
    let src_dims = tensor.dims();
    let src_len = src_dims.len();
    let tgt_len = target_shape.len();

    if src_len == tgt_len {
        // Same rank: just expand dims that are 1
        tensor.shape.expand(target_shape.to_vec());
        return tensor;
    }

    // Left-pad with size-1 dims using expand_dim (adds dim with stride 0)
    for _ in 0..(tgt_len - src_len) {
        tensor = tensor.expand_dim(0, 1);
    }

    // Now expand each dim that is 1 to match target
    tensor.shape.expand(target_shape.to_vec());
    tensor
}

/// Load float data from a TensorProto, handling inline (float_data/raw_data) and external storage.
pub fn load_tensor_floats(init: &onnx_protobuf::TensorProto, model_dir: &Path) -> Option<Vec<f32>> {
    // Try inline data based on data_type
    match init.data_type {
        1 => {
            // FLOAT
            if !init.float_data.is_empty() {
                return Some(init.float_data.clone());
            }
            if !init.raw_data.is_empty() {
                return Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
        }
        7 => {
            // INT64 — raw_data stores 8 bytes per element (little-endian int64)
            if !init.int64_data.is_empty() {
                return Some(init.int64_data.iter().map(|&v| v as f32).collect());
            }
            if !init.raw_data.is_empty() {
                return Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                );
            }
        }
        6 => {
            // INT32
            if !init.int32_data.is_empty() {
                return Some(init.int32_data.iter().map(|&v| v as f32).collect());
            }
            if !init.raw_data.is_empty() {
                return Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect(),
                );
            }
        }
        _ => {
            // Fallback: try float_data or interpret raw_data as F32
            if !init.float_data.is_empty() {
                return Some(init.float_data.clone());
            }
            if !init.raw_data.is_empty() {
                return Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            }
        }
    }
    // Try external data (data_location == EXTERNAL = 1)
    if !init.external_data.is_empty() {
        let mut location: Option<&str> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;
        for entry in &init.external_data {
            match entry.key.as_str() {
                "location" => location = Some(&entry.value),
                "offset" => offset = entry.value.parse().unwrap_or(0),
                "length" => length = entry.value.parse().ok(),
                _ => {}
            }
        }
        if let Some(loc) = location {
            let ext_path = model_dir.join(loc);
            match fs::read(&ext_path) {
                Ok(file_data) => {
                    let start = offset as usize;
                    let end = match length {
                        Some(len) => start + len as usize,
                        None => file_data.len(),
                    };
                    if end > file_data.len() {
                        return None;
                    }
                    let slice = &file_data[start..end];
                    let floats: Vec<f32> = slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    return Some(floats);
                }
                Err(_) => {
                    return None;
                }
            }
        }
    }
    None
}

/// Load initializer data as f32 values, handling multiple ONNX data types.
/// Used to seed known_values with small constant initializers for constant folding.
pub fn load_initializer_as_f32(init: &onnx_protobuf::TensorProto) -> Option<Vec<f32>> {
    match init.data_type {
        1 => {
            // FLOAT
            if !init.float_data.is_empty() {
                Some(init.float_data.clone())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            } else {
                None
            }
        }
        7 => {
            // INT64
            if !init.int64_data.is_empty() {
                Some(init.int64_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        6 => {
            // INT32
            if !init.int32_data.is_empty() {
                Some(init.int32_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect(),
                )
            } else {
                None
            }
        }
        11 => {
            // FLOAT64
            if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Transpose weight data from [rows, cols] to [cols, rows] row-major layout
pub fn transpose_weight_data(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

/// Get an integer attribute from a node, with a default value
pub fn get_int_attr(node: &NodeProto, name: &str, default: i64) -> i64 {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.i;
        }
    }
    default
}
