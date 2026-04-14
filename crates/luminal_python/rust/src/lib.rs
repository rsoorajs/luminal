mod compiled_graph;
mod runtime;
pub mod typed_data;

// PT2 modules
mod pt2_compiled_model;
mod pt2_parser;
mod pt2_schema;
mod pt2_util;
mod translator;

use compiled_graph::CompiledGraph;
use pt2_compiled_model::process_pt2;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pymodule]
fn luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register built-in backends
    ::luminal::dyn_backend::register_native_backend();

    #[cfg(feature = "cuda")]
    luminal_cuda_lite::dyn_backend::register();

    #[cfg(feature = "cuda_heavy")]
    luminal_cuda::dyn_backend::register();

    m.add_function(wrap_pyfunction!(process_pt2, m)?)?;
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(_registry_capsule, m)?)?;
    Ok(())
}

/// List all registered backend names.
#[pyfunction]
fn available_backends() -> Vec<String> {
    ::luminal::dyn_backend::available_backends()
}

/// Export the backend registry's `register_backend` function pointer as a PyCapsule.
///
/// External backend plugins (e.g. luminal_tron) import this capsule and use it
/// to register their backend factory without needing compile-time linkage to
/// luminal_python. See the plugin discovery system in `luminal/__init__.py`.
/// Wrapper to make a function pointer Send-safe for PyCapsule.
#[allow(dead_code)]
struct FnPtrWrapper(pub *const std::ffi::c_void);
// Safety: the wrapped pointer is a function pointer (code, not data), valid for the process lifetime.
unsafe impl Send for FnPtrWrapper {}

#[pyfunction]
fn _registry_capsule<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let fptr = ::luminal::dyn_backend::register_backend as *const std::ffi::c_void;
    let name = std::ffi::CString::new("luminal._registry").unwrap();
    PyCapsule::new(py, FnPtrWrapper(fptr), Some(name))
}
