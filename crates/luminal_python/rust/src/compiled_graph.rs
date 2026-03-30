use luminal::{
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
    shape::Expression,
    visualization::ToDot,
};
use onnx_protobuf::ModelProto;
use pyo3::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{
    dispatch::process_onnx_nodes,
    runtime::RuntimeBackend,
    util::{
        DimParamMap, get_shape_for_onnx_value, get_shape_for_onnx_value_expr,
        load_all_tensor_floats, load_initializer_as_f32,
    },
};

/// Common intermediate result from translating a model graph (ONNX or FX).
pub struct GraphTranslation {
    pub graph: Graph,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

/// Pre-loaded weight data from any model format.
pub struct WeightData {
    /// (Input node label, f32 data) for weights and constants.
    pub weights: Vec<(String, Vec<f32>)>,
    /// label → element count for ALL Input nodes (for CUDA dummy data sizing).
    pub tensor_sizes: HashMap<String, usize>,
}

#[pyclass(unsendable)]
pub struct CompiledGraph {
    pub graph: Graph,
    pub runtime: RuntimeBackend,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_shape_exprs: Vec<Vec<Expression>>,
    pub input_shape_exprs: Vec<Vec<Expression>>,
    pub dim_param_map: DimParamMap,
}

impl CompiledGraph {
    /// Shared compilation pipeline for both ONNX and FX/PT2 graphs.
    ///
    /// Takes a format-neutral `GraphTranslation` (produced by `translate_onnx` or
    /// `translate_pt2`) and `WeightData`, builds the backend, loads weights, and
    /// returns a ready-to-execute `CompiledGraph`.
    pub fn parse_graph(
        translation: GraphTranslation,
        weight_data: WeightData,
        backend: &str,
        search_iters: usize,
        #[allow(unused_variables)] weight_device_ptrs: HashMap<String, (u64, usize)>,
    ) -> Result<CompiledGraph, String> {
        let GraphTranslation {
            mut graph,
            tensor_ids,
            input_names,
            output_names,
            output_shape_exprs,
            input_shape_exprs,
            dim_param_map,
        } = translation;

        let rt = match backend {
            #[cfg(feature = "cuda")]
            "cuda" | "gpu" => CompiledGraph::build_cuda_backend(
                &mut graph,
                &weight_data,
                search_iters,
                &weight_device_ptrs,
            )?,
            "native" | "cpu" => {
                CompiledGraph::build_native_backend(&mut graph, &weight_data, search_iters)?
            }
            _ => {
                return Err(format!(
                    "Invalid backend '{}'. Must be 'native' or 'cuda'",
                    backend
                ));
            }
        };

        // Resolve concrete output shapes from expressions
        let output_shapes: Vec<Vec<usize>> = output_shape_exprs
            .iter()
            .map(|exprs| exprs.iter().map(|e| e.to_usize().unwrap_or(1)).collect())
            .collect();

        Ok(CompiledGraph {
            graph,
            runtime: rt,
            tensor_ids,
            input_names,
            output_names,
            output_shapes,
            output_shape_exprs,
            input_shape_exprs,
            dim_param_map,
        })
    }

    /// Build a label → NodeIndex map for all Input nodes in the graph.
    /// Used for efficient weight loading by label matching.
    fn build_label_map(graph: &Graph) -> HashMap<String, NodeIndex> {
        graph
            .graph
            .node_indices()
            .filter_map(|node_id| {
                (*graph.graph[node_id])
                    .as_any()
                    .downcast_ref::<luminal::hlir::Input>()
                    .map(|input| (input.label.clone(), node_id))
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn build_cuda_backend(
        graph: &mut Graph,
        weight_data: &WeightData,
        search_iters: usize,
        device_ptrs: &HashMap<String, (u64, usize)>,
    ) -> Result<RuntimeBackend, String> {
        use luminal_cuda_lite::cudarc::driver::CudaContext;
        use luminal_cuda_lite::runtime::CudaRuntime;

        let cuda_ctx = CudaContext::new(0).map_err(|e| format!("CUDA context init failed: {e}"))?;
        let stream = cuda_ctx.default_stream();

        graph.build_search_space::<CudaRuntime>();

        let mut rt = CudaRuntime::initialize(stream);

        // Build label → NodeIndex map for device pointer matching.
        let label_map = CompiledGraph::build_label_map(graph);

        // For weights with device pointers: use them directly (zero-copy).
        // This avoids allocating ~N GB of dummy data during search.
        // The pointers survive search because profiling mode skips buffer consumption,
        // and persist_hlir_node ensures they survive post-search execution too.
        let mut device_ptr_nodes: HashSet<NodeIndex> = HashSet::new();
        let mut matched_count = 0usize;
        let mut missed_labels: Vec<String> = Vec::new();
        for (label, &(ptr, n_bytes)) in device_ptrs {
            if let Some(&node_id) = label_map.get(label) {
                unsafe { rt.set_device_ptr(node_id, ptr, n_bytes) };
                rt.persist_hlir_node(node_id);
                device_ptr_nodes.insert(node_id);
                matched_count += 1;
            } else {
                missed_labels.push(label.clone());
            }
        }
        let total_device_bytes: usize = device_ptrs.values().map(|(_, n)| *n).sum();
        trace!(
            "[CUDA BUILD] Device pointers: {} matched, {} missed out of {} total ({:.3} GiB)",
            matched_count,
            missed_labels.len(),
            device_ptrs.len(),
            total_device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if !missed_labels.is_empty() {
            trace!(
                "[CUDA BUILD] Missed device-ptr labels (first 10): {:?}",
                &missed_labels[..missed_labels.len().min(10)]
            );
            let available: Vec<&String> = label_map.keys().take(10).collect();
            trace!(
                "[CUDA BUILD] Available label_map keys (first 10): {:?}",
                available
            );
        }

        // Set dummy 1.0 data for remaining Input nodes (user inputs, constants without
        // device pointers) for safe search profiling.
        // IMPORTANT: Must use 1.0, NOT 0.0. Zero inputs cause NaN in many ops:
        //   - fmod(0, 0) = NaN  (Mod)
        //   - recip(0) = inf → weight * inf = NaN  (Div)
        //   - log(0) = -inf  (Pow)
        //   - chain ops with zero produce NaN  (Erf)
        let mut dummy_total_elements = 0usize;
        let mut dummy_count = 0usize;
        for node_id in graph.graph.node_indices() {
            if device_ptr_nodes.contains(&node_id) {
                continue;
            }
            if let Some(input) = (*graph.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
            {
                if let Some(&n) = weight_data.tensor_sizes.get(&input.label) {
                    if n > 0 {
                        dummy_total_elements += n;
                        dummy_count += 1;
                        rt.set_data(node_id, vec![1.0f32; n]);
                    }
                }
            }
        }
        trace!(
            "[CUDA BUILD] Dummy data: {} nodes, {} elements ({:.3} GiB as f32)",
            dummy_count,
            dummy_total_elements,
            (dummy_total_elements * 4) as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Search (device-pointer weights are used directly; dummy data for the rest)
        let mut rt = graph.search(rt, search_iters);

        // Load real weight data for non-device-ptr weights (constants from PT2 archive, etc.)
        let mut loaded_weight_elements = 0usize;
        let mut loaded_weight_count = 0usize;
        for (label, data) in &weight_data.weights {
            if !device_ptrs.contains_key(label) {
                if let Some(&node_id) = label_map.get(label) {
                    loaded_weight_elements += data.len();
                    loaded_weight_count += 1;
                    rt.set_data(node_id, data.clone());
                }
            }
        }
        trace!(
            "[CUDA BUILD] Post-search weight load: {} weights, {} elements ({:.3} GiB as f32)",
            loaded_weight_count,
            loaded_weight_elements,
            (loaded_weight_elements * 4) as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        Ok(RuntimeBackend::Cuda(Box::new(rt)))
    }

    fn build_native_backend(
        graph: &mut Graph,
        weight_data: &WeightData,
        search_iters: usize,
    ) -> Result<RuntimeBackend, String> {
        graph.build_search_space::<NativeRuntime>();
        let mut rt = graph.search(NativeRuntime::default(), search_iters);

        // Load weight data after search
        let label_map = CompiledGraph::build_label_map(graph);
        for (label, data) in &weight_data.weights {
            if let Some(&node_id) = label_map.get(label) {
                rt.set_data(node_id, data.clone());
            }
        }

        Ok(RuntimeBackend::Native(rt))
    }
}

/// Translate an ONNX model into a format-neutral GraphTranslation + WeightData.
pub fn translate_onnx(
    model: ModelProto,
    model_directory: &Path,
) -> Result<(GraphTranslation, WeightData), String> {
    let _span = span!(Level::TRACE, "ONNX Graph Translation").entered();
    let onnx_graph = &model.graph;
    let mut cx = Graph::new();
    let mut tensors: HashMap<String, GraphTensor> = HashMap::new();

    // Dynamic dimension tracking
    let mut dim_param_map: DimParamMap = HashMap::new();
    let mut next_char = 'a';

    // Separate initializers (weights) from true user inputs
    let initializer_names: HashSet<&str> = onnx_graph
        .initializer
        .iter()
        .map(|t| t.name.as_str())
        .collect();

    let input_names: Vec<String> = onnx_graph
        .input
        .iter()
        .filter(|inp| !initializer_names.contains(inp.name.as_str()))
        .map(|inp| inp.name.clone())
        .collect();

    // Create input tensors with dynamic dimension support
    for input in &onnx_graph.input {
        let shape_exprs = get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
        if shape_exprs.is_empty() {
            let shape = get_shape_for_onnx_value(input);
            if shape.is_empty() {
                trace!("Input {} skipped because it is empty", input.name.clone());
                continue;
            }
            let tensor = cx.named_tensor(input.name.clone(), shape);
            trace!("Input {} added to tensors", input.name.clone());
            tensors.insert(input.name.clone(), tensor);
            continue;
        }
        let tensor = cx.named_tensor(input.name.clone(), shape_exprs);
        trace!("Input {} added to tensors", input.name.clone());
        tensors.insert(input.name.clone(), tensor);
    }

    // Create initializer (weight) tensors
    for init in &onnx_graph.initializer {
        if !tensors.contains_key(&init.name) {
            let mut shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
            if shape.is_empty() {
                shape = vec![1];
            }
            let tensor = cx.named_tensor(init.name.clone(), shape);
            tensors.insert(init.name.clone(), tensor);
        }
    }

    // Load small constants for constant folding
    let mut known_values: HashMap<String, Vec<f32>> = HashMap::new();
    for init in &onnx_graph.initializer {
        let n_elements: usize = init
            .dims
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        if n_elements <= 32 {
            if let Some(floats) = load_initializer_as_f32(init) {
                known_values.insert(init.name.clone(), floats);
            } else {
                panic!("Unable to load initializer values for {:?}", init.name);
            }
        }
    }

    // Shape expressions for propagating symbolic shapes through ONNX graphs
    let mut shape_exprs: HashMap<String, Vec<Expression>> = HashMap::new();

    // Accumulates constant node data from process_onnx_nodes
    let mut constant_data: Vec<(String, Vec<f32>)> = Vec::new();

    // Process computation nodes
    process_onnx_nodes(
        &onnx_graph.node,
        &mut tensors,
        &mut cx,
        &mut constant_data,
        &mut known_values,
        &mut shape_exprs,
    )
    .map_err(|e| format!("process_onnx_nodes failed: {}", e))?;

    // Mark weight/constant tensors as persistent so their buffers survive execute()
    for (name, gt) in &tensors {
        if !input_names.contains(name) {
            gt.persist();
        }
    }

    // Mark graph outputs (must happen before build_search_space)
    let mut output_names = Vec::new();
    let mut output_shape_exprs = Vec::new();
    for output_vi in &onnx_graph.output {
        if let Some(&gt) = tensors.get(&output_vi.name) {
            // Force contiguous if the shape tracker is a non-contiguous view
            let gt = if gt.shape != gt.shape.contiguous() {
                let contiguous = gt * 1.0;
                tensors.insert(output_vi.name.clone(), contiguous);
                contiguous
            } else {
                gt
            };
            gt.output();
            let dims = gt.dims();
            output_shape_exprs.push(dims.clone());

            let shape: Vec<usize> = dims.iter().map(|d| d.to_usize().unwrap_or(1)).collect();
            if shape.is_empty() {
                return Err(format!(
                    "Output tensor '{}' has no shape information in the ONNX model",
                    output_vi.name
                ));
            }
            output_names.push(output_vi.name.clone());
        }
    }

    // Set initial dynamic dimension values from example input shapes
    let has_dynamic = !dim_param_map.is_empty();
    if has_dynamic {
        for input in &onnx_graph.input {
            if initializer_names.contains(input.name.as_str()) {
                continue;
            }
            let concrete_shape = get_shape_for_onnx_value(input);
            let expr_shape =
                get_shape_for_onnx_value_expr(input, &mut dim_param_map, &mut next_char);
            for (expr, concrete) in expr_shape.iter().zip(concrete_shape.iter()) {
                if expr.to_usize().is_none() {
                    if let Some(ch) = dim_param_map
                        .values()
                        .find(|&&ch| Expression::from(ch) == *expr)
                    {
                        cx.set_dim(*ch, *concrete);
                    }
                }
            }
        }
    }

    // Build weight data: initializers + constants from process_onnx_nodes
    let mut weights: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, floats) in load_all_tensor_floats(&onnx_graph.initializer, model_directory) {
        if let Some(f) = floats {
            weights.push((name, f));
        }
    }
    weights.extend(constant_data);

    // Build tensor sizes for CUDA dummy data allocation
    let mut tensor_sizes: HashMap<String, usize> = HashMap::new();
    for input in &onnx_graph.input {
        if !initializer_names.contains(input.name.as_str()) {
            let shape = get_shape_for_onnx_value(input);
            let n: usize = shape.iter().product::<usize>().max(1);
            tensor_sizes.insert(input.name.clone(), n);
        }
    }
    for init in &onnx_graph.initializer {
        let n: usize = init
            .dims
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1);
        tensor_sizes.insert(init.name.clone(), n);
    }
    // Also include sizes from weight_data (constants created by process_onnx_nodes)
    for (name, data) in &weights {
        if !tensor_sizes.contains_key(name) {
            tensor_sizes.insert(name.clone(), data.len());
        }
    }

    // Collect tensor name → NodeIndex mapping
    let tensor_ids: HashMap<String, NodeIndex> = tensors
        .iter()
        .map(|(name, gt)| (name.clone(), gt.id))
        .collect();

    // Build input_shape_exprs for user inputs (needed for auto-dim detection)
    let input_shape_exprs: Vec<Vec<Expression>> = input_names
        .iter()
        .map(|name| {
            if let Some(&gt) = tensors.get(name) {
                gt.dims()
            } else {
                vec![]
            }
        })
        .collect();

    let translation = GraphTranslation {
        graph: cx,
        tensor_ids,
        input_names,
        output_names,
        output_shape_exprs,
        input_shape_exprs,
        dim_param_map,
    };

    let weight_data = WeightData {
        weights,
        tensor_sizes,
    };

    Ok((translation, weight_data))
}

#[pymethods]
impl CompiledGraph {
    /// Get the list of input tensor names.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Get the list of output tensor names.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Get the output shapes.
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_shapes.clone()
    }

    /// Get all tensor names in the graph.
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_ids.keys().cloned().collect()
    }

    /// Get the name of the active backend (native or cuda).
    #[getter]
    fn backend(&self) -> &'static str {
        self.runtime.name()
    }

    /// Whether this graph has dynamic (symbolic) dimensions.
    #[getter]
    fn has_dynamic_dims(&self) -> bool {
        !self.dim_param_map.is_empty()
    }

    /// Get the dynamic dimension parameter names (e.g. ["seq_len"]).
    #[getter]
    fn dim_params(&self) -> Vec<String> {
        self.dim_param_map.keys().cloned().collect()
    }

    /// Set a dynamic dimension value by its param name (e.g. "seq_len").
    fn set_dim(&mut self, param_name: &str, value: usize) -> PyResult<()> {
        let ch = self.dim_param_map.get(param_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown dim param '{}'. Available: {:?}",
                param_name,
                self.dim_param_map.keys().collect::<Vec<_>>()
            ))
        })?;
        self.graph.set_dim(*ch, value);
        Ok(())
    }

    /// Auto-detect and set dynamic dimensions from input tensor shapes.
    /// For each user input, matches the concrete shape against its symbolic
    /// shape expressions and sets the corresponding dyn_map entries.
    fn auto_set_dims_from_input_shapes(&mut self, input_shapes: Vec<Vec<usize>>) {
        for (shape_exprs, shape) in self.input_shape_exprs.iter().zip(input_shapes.iter()) {
            for (dim_expr, &dim_val) in shape_exprs.iter().zip(shape.iter()) {
                // Check if this expression is a bare symbolic variable
                let terms = dim_expr.terms.read();
                if terms.len() == 1
                    && let luminal::shape::Term::Var(c) = terms[0]
                {
                    self.graph.set_dim(c, dim_val);
                }
            }
        }
    }

    /// Resolve output shapes using current dynamic dimension values.
    /// Returns concrete shapes after substituting all symbolic dims.
    fn resolve_output_shapes(&self) -> PyResult<Vec<Vec<usize>>> {
        let dyn_map = &self.graph.dyn_map;
        let mut result = Vec::new();
        for shape_exprs in &self.output_shape_exprs {
            let shape: Vec<usize> = shape_exprs
                .iter()
                .map(|e| {
                    e.exec(dyn_map).ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Cannot resolve dimension expression {:?}. Set all dynamic dims first.",
                            e
                        ))
                    })
                })
                .collect::<PyResult<Vec<usize>>>()?;
            result.push(shape);
        }
        Ok(result)
    }

    /// Set input tensor data by name.
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        self.runtime.set_data(*node_id, data);
        Ok(())
    }

    /// Set input tensor data from a CPU host memory pointer (avoids Python list conversion).
    /// The pointer must point to contiguous f32 data (from tensor.data_ptr() on a CPU float32 tensor).
    fn set_input_from_ptr(&mut self, name: &str, ptr: u64, n_elements: usize) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ptr as *const f32, n_elements).to_vec() };
        self.runtime.set_data(*node_id, data);
        Ok(())
    }

    /// Set input from a CUDA device pointer. Zero-copy on device.
    /// The pointer must be a valid CUDA device allocation with at least n_bytes bytes.
    #[cfg(feature = "cuda")]
    fn set_input_device_ptr(
        &mut self,
        name: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        match &mut self.runtime {
            RuntimeBackend::Cuda(rt) => unsafe { rt.set_device_ptr(*node_id, device_ptr, n_bytes) },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "set_input_device_ptr requires CUDA backend",
                ));
            }
        }
        Ok(())
    }

    /// Mark an input tensor as persistent (survives execute() calls).
    /// Call this for weight tensors that should not be consumed after each execution.
    fn persist_input(&mut self, name: &str) -> PyResult<()> {
        let _node_id = *self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        match &mut self.runtime {
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.persist_hlir_node(_node_id),
            RuntimeBackend::Native(_) => {} // Native: persist is handled at graph level
        }
        Ok(())
    }

    /// Set a weight tensor from a CUDA device pointer, matching by Input node label.
    /// Also marks the weight as persistent. For PT2 weights (e.g. "fc1.weight").
    #[cfg(feature = "cuda")]
    fn set_weight_device_ptr(
        &mut self,
        label: &str,
        device_ptr: u64,
        n_bytes: usize,
    ) -> PyResult<()> {
        for node_id in self.graph.graph.node_indices() {
            if let Some(input) = (*self.graph.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
                && input.label == label
            {
                match &mut self.runtime {
                    RuntimeBackend::Cuda(rt) => {
                        unsafe { rt.set_device_ptr(node_id, device_ptr, n_bytes) };
                        rt.persist_hlir_node(node_id);
                        return Ok(());
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "set_weight_device_ptr requires CUDA backend",
                        ));
                    }
                }
            }
        }
        Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "No Input node with label: {}",
            label
        )))
    }

    /// Set a weight tensor from a CPU host pointer, matching by Input node label.
    fn set_weight_from_ptr(&mut self, label: &str, ptr: u64, n_elements: usize) -> PyResult<()> {
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(ptr as *const f32, n_elements).to_vec() };
        for node_id in self.graph.graph.node_indices() {
            if let Some(input) = (*self.graph.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
                && input.label == label
            {
                self.runtime.set_data(node_id, data);
                return Ok(());
            }
        }
        Err(pyo3::exceptions::PyKeyError::new_err(format!(
            "No Input node with label: {}",
            label
        )))
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.runtime.execute(&self.graph.dyn_map);
    }

    /// Return the HLIR graph as a DOT string for visualization.
    fn to_dot(&self) -> PyResult<String> {
        self.graph.graph.to_dot().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("DOT generation failed: {e}"))
        })
    }

    /// Get output tensor data by name.
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Unknown output tensor: {}",
                name
            ))
        })?;
        Ok(self.runtime.get_f32(*node_id))
    }
}
