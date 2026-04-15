# Luminal Backend Plugin System

luminal uses a factory-capsule system for backends. Each backend is a Rust
`BackendFactory` function wrapped in a PyCapsule and passed directly through
the compilation chain — no string registry or global state.

## Using a backend

```python
import luminal

# Auto-detect (picks cuda_lite if GPU available, native otherwise)
compiled = torch.compile(model, backend=luminal.luminal_backend)

# External plugin (e.g. luminal_cuda for the cuda_heavy backend)
import luminal_cuda
compiled = torch.compile(model, backend=luminal.register_backend(luminal_cuda.luminal_backend))
```

`register_backend(capsule)` wraps a factory PyCapsule into a
`torch.compile`-compatible callable. The returned callable passes the factory
through the shared compilation pipeline (graph export, weight handling, search).

## Creating a backend plugin

A backend plugin is a Python package built with [maturin](https://www.maturin.rs/)
that wraps a Rust crate implementing `DynBackend`.

### Python side

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "luminal-mybackend"
dependencies = ["luminal"]

[tool.maturin]
module-name = "luminal_mybackend._native"
python-source = "python"
features = ["python"]
```

```python
# python/luminal_mybackend/__init__.py
from luminal_mybackend._native import _factory_capsule

# PyCapsule wrapping the backend factory
luminal_backend = _factory_capsule()
```

### Rust side

```rust
// In Cargo.toml, add pyo3 as an optional dependency gated by a "python" feature.
// Set crate-type = ["cdylib", "rlib"].

// In src/lib.rs:
#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _native(m: &pyo3::prelude::Bound<'_, pyo3::prelude::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_function(wrap_pyfunction!(_factory_capsule, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn _factory_capsule(
    py: pyo3::prelude::Python<'_>,
) -> pyo3::PyResult<pyo3::prelude::Bound<'_, pyo3::types::PyCapsule>> {
    use pyo3::types::PyCapsule;
    #[repr(transparent)]
    struct FnPtrWrapper(*const std::ffi::c_void);
    unsafe impl Send for FnPtrWrapper {}
    let fptr = my_crate::dyn_backend::my_factory as *const std::ffi::c_void;
    let name = std::ffi::CString::new("luminal.backend_factory").unwrap();
    PyCapsule::new(py, FnPtrWrapper(fptr), Some(name))
}
```

The factory function has the signature:

```rust
pub fn my_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String>
```

Use `luminal::dyn_backend::compile_backend` to handle the boilerplate
(build search space, init runtime, set dummy data, search, load weights).

## Requirements

Both the plugin and luminal_python must be compiled with:
- The **same Rust compiler version**
- The **same luminal crate version** (use matching git revs)

This is because the PyCapsule passes a Rust function pointer between two
separate cdylibs. The types (`Graph`, `BackendCompileArgs`, `BackendFactory`)
must have identical memory layouts on both sides.
