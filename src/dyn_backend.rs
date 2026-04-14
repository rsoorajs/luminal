//! Dynamic backend trait and plugin registry.
//!
//! This module provides:
//! - [`DynBackend`]: an object-safe trait that wraps a concrete `Runtime` for dynamic dispatch
//! - [`BackendFactory`] / [`BackendCompileArgs`]: the factory pattern for backend creation
//! - A global registry for backend discovery (`register_backend`, `create_backend`)
//! - [`NativeDynBackend`]: the reference `DynBackend` implementation for CPU

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use half::{bf16, f16};
use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

use crate::dtype::DType;
use crate::graph::Graph;
use crate::hlir::{Input, NativeData, NativeRuntime, Output};
use crate::op::Runtime;

// ---------------------------------------------------------------------------
// DynBackend trait
// ---------------------------------------------------------------------------

/// Object-safe backend trait for dynamic dispatch.
///
/// Wraps a concrete [`Runtime`] implementor, providing a uniform interface
/// for `luminal_python` (and other dynamic consumers) without requiring
/// generic type parameters.
pub trait DynBackend {
    /// Human-readable backend name (e.g. `"cuda_lite"`, `"native"`, `"tron"`).
    fn name(&self) -> &str;

    // --- Data management ---------------------------------------------------

    /// Set input data as raw bytes with dtype metadata.
    /// The backend converts to its native format internally.
    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType);

    /// Set input data from an f32 vector (convenience / backward compat).
    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>);

    /// Get output tensor data as f32 (copies to host if needed).
    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32>;

    // --- Execution ---------------------------------------------------------

    /// Execute the compiled graph with the given dynamic dimension map.
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>);

    // --- Device pointer support (GPU backends) -----------------------------

    /// Whether this backend supports direct device pointer operations.
    fn supports_device_ptrs(&self) -> bool {
        false
    }

    /// Set an input tensor from a device pointer (zero-copy).
    ///
    /// # Safety
    /// The device pointer must be valid and point to at least `n_bytes` bytes
    /// on the same device as this runtime.
    unsafe fn set_device_ptr(&mut self, _node: NodeIndex, _ptr: u64, _n_bytes: usize) {
        panic!(
            "set_device_ptr not supported by backend '{}'",
            self.name()
        );
    }

    /// Register an external device pointer for an output tensor (zero-copy output).
    ///
    /// # Safety
    /// The device pointer must be valid through the next `execute()` call.
    unsafe fn set_output_device_ptr(&mut self, _node: NodeIndex, _ptr: u64, _n_bytes: usize) {
        panic!(
            "set_output_device_ptr not supported by backend '{}'",
            self.name()
        );
    }

    /// Check whether an output was written directly to the registered pointer.
    fn output_is_zero_copy(&self, _node: NodeIndex) -> bool {
        false
    }

    /// Copy output data directly to a device pointer (device-to-device).
    ///
    /// # Safety
    /// `dest_ptr` must be a valid device allocation with at least `n_bytes`.
    unsafe fn copy_output_to_device_ptr(
        &self,
        _node: NodeIndex,
        _dest_ptr: u64,
        _n_bytes: usize,
    ) {
        panic!(
            "copy_output_to_device_ptr not supported by backend '{}'",
            self.name()
        );
    }
}

// ---------------------------------------------------------------------------
// BackendCompileArgs + BackendFactory
// ---------------------------------------------------------------------------

/// Arguments passed to a [`BackendFactory`] during compilation.
pub struct BackendCompileArgs {
    /// Number of search iterations for graph optimization.
    pub search_iters: usize,
    /// Weights as `(label, raw_bytes, dtype)`.
    pub weights: Vec<(String, Vec<u8>, DType)>,
    /// `label → element_count` for ALL Input nodes (for sizing dummy data during search).
    pub tensor_sizes: HashMap<String, usize>,
    /// `label → (device_ptr, n_bytes)` for zero-copy GPU weight sharing.
    pub device_ptrs: HashMap<String, (u64, usize)>,
}

/// A factory function that compiles a [`Graph`] into a ready-to-execute [`DynBackend`].
///
/// The factory is responsible for the full compilation pipeline:
/// 1. `graph.build_search_space_with_ops(ops, cleanup_hlir)`
/// 2. Initialize the concrete `Runtime`
/// 3. Load dummy / real data for search profiling
/// 4. `graph.search(runtime, search_iters)`
/// 5. Load real weights post-search
/// 6. Return a `Box<dyn DynBackend>` wrapper
// Note: BackendFactory itself must be Send+Sync for the global registry,
// but the DynBackend it produces need not be Send (e.g., CudaRuntime is !Send).
pub type BackendFactory = Arc<
    dyn Fn(&mut Graph, BackendCompileArgs) -> Result<Box<dyn DynBackend>, String> + Send + Sync,
>;

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

static REGISTRY: OnceLock<Mutex<HashMap<String, BackendFactory>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, BackendFactory>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a backend factory under the given name.
///
/// Names are case-insensitive (stored lowercase).
/// Replaces any previously registered factory with the same name.
pub fn register_backend(name: &str, factory: BackendFactory) {
    let name = name.to_lowercase();
    let mut map = registry().lock().unwrap();
    map.insert(name, factory);
}

/// Create a backend by name.
///
/// Returns `Err` if the name is not registered or if the factory fails.
pub fn create_backend(
    name: &str,
    graph: &mut Graph,
    args: BackendCompileArgs,
) -> Result<Box<dyn DynBackend>, String> {
    let name_lower = name.to_lowercase();
    let map = registry().lock().unwrap();
    let factory = map.get(&name_lower).ok_or_else(|| {
        let available: Vec<&String> = map.keys().collect();
        format!(
            "Unknown backend '{}'. Available: {:?}",
            name, available
        )
    })?;
    let factory = Arc::clone(factory);
    drop(map); // release lock before calling factory
    factory(graph, args)
}

/// List all registered backend names.
pub fn available_backends() -> Vec<String> {
    let map = registry().lock().unwrap();
    map.keys().cloned().collect()
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Build a `label → NodeIndex` map for all `Input` nodes in the graph.
///
/// Used by backend factories for weight loading by label matching.
pub fn build_label_map(graph: &Graph) -> HashMap<String, NodeIndex> {
    graph
        .graph
        .node_indices()
        .filter_map(|node_id| {
            (*graph.graph[node_id])
                .as_any()
                .downcast_ref::<Input>()
                .map(|input| (input.label.clone(), node_id))
        })
        .collect()
}

/// Convert raw bytes + [`DType`] to [`NativeData`].
///
/// Handles the common dtypes. Exotic types (F8, F6, etc.) fall back to
/// an empty F32 buffer — these are not expected in practice.
pub fn bytes_to_native_data(bytes: Vec<u8>, dtype: DType) -> NativeData {
    match dtype {
        DType::F32 | DType::TF32 => {
            let data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            NativeData::F32(data)
        }
        DType::F64 => {
            // Downcast f64 → f32
            let data: Vec<f32> = bytes
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            NativeData::F32(data)
        }
        DType::F16 => {
            let data: Vec<f16> = bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            NativeData::F16(data)
        }
        DType::Bf16 => {
            let data: Vec<bf16> = bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                .collect();
            NativeData::Bf16(data)
        }
        DType::Int => {
            let data: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            NativeData::Int(data)
        }
        DType::Bool => {
            let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            NativeData::Bool(data)
        }
        DType::I8 => {
            let data: Vec<i32> = bytes.iter().map(|&b| b as i8 as i32).collect();
            NativeData::Int(data)
        }
        DType::U8 => {
            let data: Vec<i32> = bytes.iter().map(|&b| b as i32).collect();
            NativeData::Int(data)
        }
        DType::I16 => {
            let data: Vec<i32> = bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                .collect();
            NativeData::Int(data)
        }
        DType::U16 => {
            let data: Vec<i32> = bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]) as i32)
                .collect();
            NativeData::Int(data)
        }
        _ => {
            // F8/F6/F4/I4/U4 — not expected in practice for native runtime
            NativeData::F32(vec![])
        }
    }
}

/// Create a byte buffer of `n_elements` ones for the given dtype.
///
/// IMPORTANT: Must use 1, NOT 0 — zero inputs cause NaN in many ops
/// (fmod, recip, log, etc.) during search profiling.
pub fn make_ones_bytes(n_elements: usize, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 | DType::TF32 => {
            let data = vec![1.0f32; n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
            }
        }
        DType::F64 => {
            let data = vec![1.0f64; n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8).to_vec()
            }
        }
        DType::F16 => {
            let data = vec![f16::from_f32(1.0); n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
            }
        }
        DType::Bf16 => {
            let data = vec![bf16::from_f32(1.0); n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
            }
        }
        DType::Int => {
            let data = vec![1i32; n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4).to_vec()
            }
        }
        DType::I8 | DType::U8 | DType::Bool => vec![1u8; n_elements],
        DType::I16 => {
            let data = vec![1i16; n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
            }
        }
        DType::U16 => {
            let data = vec![1u16; n_elements];
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2).to_vec()
            }
        }
        _ => {
            // Sub-byte types (F8, F6, F4, etc.) — just fill with 1 bytes
            vec![1u8; n_elements]
        }
    }
}

// ---------------------------------------------------------------------------
// NativeDynBackend
// ---------------------------------------------------------------------------

/// [`DynBackend`] wrapper for the native (CPU) runtime.
pub struct NativeDynBackend {
    pub runtime: NativeRuntime,
}

impl DynBackend for NativeDynBackend {
    fn name(&self) -> &str {
        "native"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType) {
        let native = bytes_to_native_data(bytes, dtype);
        self.runtime.set_data(node, native);
    }

    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }

    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        // Find the Output node in the LLIR graph that points to this HLIR node.
        let output_id = self
            .runtime
            .graph
            .node_indices()
            .find(|n| {
                if let Some(out) = (**self.runtime.graph[*n])
                    .as_any()
                    .downcast_ref::<Output>()
                {
                    out.node == node.index()
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("No output node found for {:?}", node));

        let data = self
            .runtime
            .buffers
            .get(&output_id)
            .unwrap_or_else(|| panic!("No buffer data for output {:?}", node));

        // Convert any NativeData variant to f32
        (0..data.len()).map(|i| data.f32(i)).collect()
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }
}

/// Register the native (CPU) backend in the global registry.
///
/// Registers under the names `"native"` and `"cpu"`.
pub fn register_native_backend() {
    let factory: BackendFactory = Arc::new(|graph, args| {
        // NativeRuntime has Ops = () — only HLIROps are used.
        let ops = Vec::new(); // build_search_space_with_ops appends HLIROps automatically
        graph.build_search_space_with_ops(ops, false); // cleanup_hlir=false for native

        let rt = NativeRuntime::default();
        let mut rt = graph.search(rt, args.search_iters);

        // Load weights after search, preserving native dtype.
        let label_map = build_label_map(graph);
        for (label, bytes, dtype) in args.weights {
            if let Some(&node_id) = label_map.get(&label) {
                let native = bytes_to_native_data(bytes, dtype);
                rt.set_data(node_id, native);
            }
        }

        Ok(Box::new(NativeDynBackend { runtime: rt }) as Box<dyn DynBackend>)
    });

    register_backend("native", Arc::clone(&factory));
    register_backend("cpu", factory);
}
