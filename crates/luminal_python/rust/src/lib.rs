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

use compiled_graph::CompiledGraph;
use onnx_protobuf::ModelProto;
use protobuf::Message;
use pt2_compiled_model::compile_pt2;
use pyo3::prelude::*;
use std::fs;
use std::path::Path;

fn validate_backend(backend: &str) -> PyResult<()> {
    match backend {
        "native" => Ok(()),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(()),
        #[cfg(not(feature = "cuda"))]
        "cuda" => Err(pyo3::exceptions::PyValueError::new_err(
            "CUDA backend requested, but this luminal extension was built without the `cuda` feature. Rebuild with `maturin develop --features cuda -r` or use backend='native'.",
        )),
        _ => {
            #[cfg(feature = "cuda")]
            {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid backend '{}'. Must be 'native' or 'cuda'",
                    backend
                )))
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid backend '{}'. This build only supports 'native'. Rebuild with the `cuda` feature to enable 'cuda'.",
                    backend
                )))
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (path, backend="native"))]
fn process_onnx(path: &str, backend: &str) -> PyResult<crate::compiled_graph::OnnxGraphResult> {
    validate_backend(backend)?;

    parse_onnx(path, backend).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

fn parse_onnx(path: &str, backend: &str) -> Result<crate::compiled_graph::OnnxGraphResult, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
    let model_directory = Path::new(path).parent().unwrap_or(Path::new("."));
    let model = ModelProto::parse_from_bytes(&data)
        .map_err(|e| format!("Failed to parse Onnx Model: {}", e))?;

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

    CompiledGraph::parse_graph(model, model_directory, backend)
}

#[pymodule]
fn luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(compile_pt2, m)?)?;
    m.add_class::<CompiledGraph>()?;
    Ok(())
}
