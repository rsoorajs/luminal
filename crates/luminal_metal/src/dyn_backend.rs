//! [`DynBackend`] implementation for the Metal runtime.

use std::sync::Arc;

use luminal::dyn_backend::{
    BackendFactory, DynBackend, build_label_map, bytes_to_native_data, register_backend,
};
use luminal::dtype::DType;
use luminal::op::IntoEgglogOp;
use luminal::prelude::*;

use crate::runtime::MetalRuntime;

/// [`DynBackend`] wrapper for [`MetalRuntime`].
pub struct MetalDynBackend {
    pub runtime: MetalRuntime,
}

impl DynBackend for MetalDynBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn set_data_bytes(&mut self, node: NodeIndex, bytes: Vec<u8>, dtype: DType) {
        // Metal uses host-side NativeData storage (input_data map).
        let native = bytes_to_native_data(bytes, dtype);
        self.runtime.set_data(node, native);
    }

    fn set_data_f32(&mut self, node: NodeIndex, data: Vec<f32>) {
        self.runtime.set_data(node, data);
    }

    fn get_output_f32(&self, node: NodeIndex) -> Vec<f32> {
        // MetalRuntime::get_f32 handles Output node lookup + dtype conversion internally.
        self.runtime.get_f32(node)
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        self.runtime.execute(dyn_map);
    }
}

/// Register the Metal backend in the global registry.
///
/// Registers under the name `"metal"`.
pub fn register() {
    let factory: BackendFactory = Arc::new(|graph, args| {
        let ops = <MetalRuntime as Runtime>::Ops::into_vec();
        graph.build_search_space_with_ops(ops, true); // cleanup_hlir=true for non-native

        let rt = MetalRuntime::initialize(());
        let mut rt = graph.search(rt, args.search_iters);

        // Load weights after search
        let label_map = build_label_map(graph);
        for (label, bytes, dtype) in args.weights {
            if let Some(&node_id) = label_map.get(&label) {
                let native = bytes_to_native_data(bytes, dtype);
                rt.set_data(node_id, native);
            }
        }

        Ok(Box::new(MetalDynBackend { runtime: rt }) as Box<dyn DynBackend>)
    });

    register_backend("metal", factory);
}
