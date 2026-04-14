//! [`DynBackend`] implementation for the CUDA lite runtime.

use std::collections::HashSet;
use std::sync::Arc;

use luminal::dyn_backend::{
    BackendFactory, DynBackend, build_label_map, make_ones_bytes, register_backend,
};
use luminal::dtype::DType;
use luminal::hlir::Input;
use luminal::op::IntoEgglogOp;
use luminal::prelude::*;

use crate::cudarc::driver::CudaContext;
use crate::runtime::CudaRuntime;

/// [`DynBackend`] wrapper for [`CudaRuntime`].
pub struct CudaLiteDynBackend {
    pub runtime: CudaRuntime,
}

impl DynBackend for CudaLiteDynBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, _dtype: DType) {
        // CudaRuntime's set_data accepts Vec<u8> directly (raw bytes uploaded to GPU).
        self.runtime.set_data(node, bytes);
    }

    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }

    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        self.runtime.get_f32(node)
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }

    fn supports_device_ptrs(&self) -> bool {
        true
    }

    unsafe fn set_device_ptr(&mut self, node: NodeIndex, ptr: u64, n_bytes: usize) {
        unsafe { self.runtime.set_device_ptr(node, ptr, n_bytes) }
    }

    unsafe fn set_output_device_ptr(&mut self, node: NodeIndex, ptr: u64, n_bytes: usize) {
        unsafe { self.runtime.set_output_device_ptr(node, ptr, n_bytes) }
    }

    fn output_is_zero_copy(&self, node: NodeIndex) -> bool {
        self.runtime.output_is_zero_copy(node)
    }

    unsafe fn copy_output_to_device_ptr(&self, node: NodeIndex, dest_ptr: u64, n_bytes: usize) {
        unsafe { self.runtime.copy_output_to_device_ptr(node, dest_ptr, n_bytes) }
    }
}

/// Register the CUDA lite backend in the global registry.
///
/// Registers under the names `"cuda_lite"`, `"cuda"`, and `"gpu"`.
pub fn register() {
    let factory: BackendFactory = Arc::new(|graph, args| {
        // 1. Build search space
        let ops = <CudaRuntime as Runtime>::Ops::into_vec();
        graph.build_search_space_with_ops(ops, true); // cleanup_hlir=true for non-native

        // 2. Initialize CUDA
        let cuda_ctx =
            CudaContext::new(0).map_err(|e| format!("CUDA context init failed: {e}"))?;
        let stream = cuda_ctx.default_stream();
        let mut rt = CudaRuntime::initialize(stream);

        // 3. Set device pointers for zero-copy weights
        let label_map = build_label_map(graph);
        let mut device_ptr_nodes: HashSet<NodeIndex> = HashSet::new();
        for (label, &(ptr, n_bytes)) in &args.device_ptrs {
            if let Some(&node_id) = label_map.get(label) {
                unsafe { rt.set_device_ptr(node_id, ptr, n_bytes) };
                device_ptr_nodes.insert(node_id);
            }
        }

        // 4. Set dummy ones data for remaining Input nodes (required for search profiling).
        //    IMPORTANT: Must use 1.0, NOT 0.0 — zero causes NaN in many ops.
        for node_id in graph.graph.node_indices() {
            if device_ptr_nodes.contains(&node_id) {
                continue;
            }
            if let Some(input) = (*graph.graph[node_id])
                .as_any()
                .downcast_ref::<Input>()
            {
                if let Some(&n) = args.tensor_sizes.get(&input.label) {
                    if n > 0 {
                        rt.set_data(node_id, make_ones_bytes(n, input.dtype));
                    }
                }
            }
        }

        // 5. Search
        let mut rt = graph.search(rt, args.search_iters);

        // 6. Load real weight data post-search (skip device-ptr weights)
        for (label, bytes, _dtype) in &args.weights {
            if !args.device_ptrs.contains_key(label) {
                if let Some(&node_id) = label_map.get(label) {
                    rt.set_data(node_id, bytes.clone());
                }
            }
        }

        Ok(Box::new(CudaLiteDynBackend { runtime: rt }) as Box<dyn DynBackend>)
    });

    register_backend("cuda_lite", Arc::clone(&factory));
    register_backend("cuda", Arc::clone(&factory));
    register_backend("gpu", factory);
}
