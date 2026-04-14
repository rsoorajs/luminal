# External Backend Plugin Example

luminal supports external backends that register themselves at runtime through
Python's [entry point](https://packaging.python.org/en/latest/specifications/entry-points/)
mechanism. This means you can install a backend as a separate pip package and
luminal will discover it automatically on `import luminal`.

## How it works

1. **luminal_python** exports a PyCapsule containing the `register_backend`
   function pointer from its Rust registry.
2. On `import luminal`, the `_discover_backends()` function scans all installed
   packages for entry points in the `luminal.backends` group.
3. Each entry point's `register(capsule)` function is called with the PyCapsule.
4. The plugin extracts the function pointer, creates a `BackendFactory`, and
   registers itself under a name (e.g. `"cuda_heavy"`, `"tron"`).

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
name = "luminal-walrus"
dependencies = ["luminal"]

[project.entry-points."luminal.backends"]
my_backend = "luminal_walrus:register"

[tool.maturin]
module-name = "luminal_walrus._native"
python-source = "python"
features = ["python"]
```

```python
# python/luminal_walrus/__init__.py
def register(capsule):
    from luminal_walrus._native import _register_via_capsule
    _register_via_capsule(capsule)
```

### Rust side

```rust
// In Cargo.toml, add pyo3 as an optional dependency gated by a "python" feature.
// Set crate-type = ["cdylib", "rlib"] so the crate works as both a Python
// extension and a normal Rust library.

// In src/lib.rs:
#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _native(m: &pyo3::prelude::Bound<'_, pyo3::prelude::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_function(wrap_pyfunction!(_register_via_capsule, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
fn _register_via_capsule(
    capsule: &pyo3::prelude::Bound<'_, pyo3::types::PyCapsule>,
) -> pyo3::PyResult<()> {
    use pyo3::types::PyCapsuleMethods;
    unsafe {
        let fptr = capsule.pointer();
        let register_fn: fn(&str, luminal::dyn_backend::BackendFactory) =
            std::mem::transmute(fptr);
        register_fn("my_backend", my_crate::dyn_backend::my_factory);
    }
    Ok(())
}
```

The `my_factory` function has the signature:

```rust
pub fn my_factory(
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String>
```

Use [`luminal::dyn_backend::compile_backend`] to handle the boilerplate
(build search space, init runtime, set dummy data, search, load weights).

## Usage

```bash
# Install luminal and the external backend
pip install luminal luminal-walrus

# The backend is auto-discovered on import
python example.py
```

## Requirements

Both the plugin and luminal_python must be compiled with:
- The **same Rust compiler version**
- The **same luminal crate version** (use matching git revs)

This is because the PyCapsule passes a Rust function pointer between two
separate cdylibs. The types (`Graph`, `BackendCompileArgs`, `BackendFactory`)
must have identical memory layouts on both sides.
