mod compiled_graph;
mod dispatch;
mod ops_parse;
mod runtime;
mod util;

// PT2 modules
mod pt2_compiled_model;
mod pt2_parser;
mod pt2_schema;
mod pt2_util;
mod translator;

use compiled_graph::{CompiledGraph, translate_onnx};
use onnx_protobuf::ModelProto;
use protobuf::Message;
use pt2_compiled_model::process_pt2;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[pyfunction]
#[pyo3(signature = (path, backend="native", weight_device_ptrs=None))]
fn process_onnx(
    path: &str,
    backend: &str,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
) -> PyResult<CompiledGraph> {
    if backend != "native" && backend != "cuda" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid backend '{}'. Must be 'native' or 'cuda'",
            backend
        )));
    }

    parse_onnx(path, backend, weight_device_ptrs.unwrap_or_default())
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

fn parse_onnx(
    path: &str,
    backend: &str,
    weight_device_ptrs: HashMap<String, (u64, usize)>,
) -> Result<CompiledGraph, String> {
    let data = fs::read(path)
        .map_err(|e| format!("Failed to read file: {}", e))
        .unwrap();
    let model_directory = Path::new(path).parent().unwrap_or(Path::new("."));
    let model = ModelProto::parse_from_bytes(&data)
        .map_err(|e| format!("Failed to parse Onnx Model: {}", e))
        .unwrap();

    let opset_version = model
        .opset_import
        .iter()
        .find(|entry| entry.domain.is_empty())
        .map(|entry| entry.version);

    match opset_version {
        Some(20) => {}
        Some(v) => {
            return Err(format!(
                "Unsupported ONNX opset version {v}. Only opset 20 is supported."
            ));
        }
        None => {
            return Err(
                "No ONNX opset version found in model. Only opset 20 is supported.".to_string(),
            );
        }
    }

    let (translation, weights) = translate_onnx(model, model_directory)?;
    CompiledGraph::parse_graph(translation, weights, backend, 10, weight_device_ptrs)
}

#[pymodule]
fn luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(process_pt2, m)?)?;
    m.add_class::<CompiledGraph>()?;
    Ok(())
}
