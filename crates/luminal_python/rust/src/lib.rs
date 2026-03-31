mod compiled_graph;
mod dispatch;
mod onnx_translator;
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
use pt2_compiled_model::process_pt2;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (path, backend="native", search_iters=10, weight_device_ptrs=None))]
fn process_onnx(
    path: &str,
    backend: &str,
    search_iters: usize,
    weight_device_ptrs: Option<HashMap<String, (u64, usize)>>,
) -> PyResult<CompiledGraph> {
    onnx_translator::compile_onnx(
        path,
        backend,
        weight_device_ptrs.unwrap_or_default(),
        search_iters,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pymodule]
fn luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(process_pt2, m)?)?;
    m.add_class::<CompiledGraph>()?;
    Ok(())
}
