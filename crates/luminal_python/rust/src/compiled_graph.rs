use luminal::prelude::{
    tracing::{Level, span, trace},
    *,
};
use onnx_protobuf::{GraphProto, ModelProto};
use pyo3::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

#[cfg(feature = "cuda")]
use crate::util::transpose_weight_data;
use crate::{
    dispatch::process_onnx_nodes,
    runtime::*,
    util::{get_shape_for_onnx_value, load_initializer_as_f32, load_tensor_floats},
};

#[pyclass(unsendable)]
pub struct OnnxGraphResult {
    pub context: Graph,
    pub runtime: RuntimeBackend,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
}

impl OnnxGraphResult {
    pub fn parse_graph(
        model: ModelProto,
        model_directory: &Path,
        backend: &str,
    ) -> Result<OnnxGraphResult, String> {
        let _span = span!(Level::TRACE, "Onnx Graphing Parsing").entered();
        let onnx_graph = &model.graph;
        let mut context = Graph::new();
        // We will need to track the tensors we allocate so we can match up inputs and outputs in the graph
        let mut tensors: HashMap<String, GraphTensor> = HashMap::new();

        // This is the name of all of the tensors we will need to fill in parameters for
        let initializer_names: HashSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        // Input is an overloaded term in Onnx, it both means the inputs into the model, like the next token
        // and the parameters of the layers, for this we don't want any of the parameters
        // Input here is in the straightforward meaning, those tensors you feed into the network for a
        // forward passd
        let input_names: Vec<String> = onnx_graph
            .input
            .iter()
            .filter(|inp| !initializer_names.contains(inp.name.as_str()))
            .map(|inp| inp.name.clone())
            .collect();

        // Create "holding" tensors for the input
        // this way they can be considered in the graph computation, and later as we do mutiple runs we can target them and swap out the values
        // in them and not need to recompile the network
        for input in &onnx_graph.input {
            let shape = get_shape_for_onnx_value(input);
            if shape.is_empty() {
                trace!("Input {} skipped because it is empty", input.name.clone());
                continue;
            }
            // Always F32: Python runtime always sends float32 data via .float().numpy()
            let tensor = context.named_tensor(input.name.clone(), shape);
            trace!("Input {} added to tensors", input.name.clone());
            tensors.insert(input.name.clone(), tensor);
        }

        for init in &onnx_graph.initializer {
            if !tensors.contains_key(&init.name) {
                let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                let tensor = context.named_tensor(init.name.clone(), shape);
                tensors.insert(init.name.clone(), tensor);
            }
        }

        let mut weight_data = Vec::new();

        let mut known_values: HashMap<String, Vec<f32>> = HashMap::new();

        for init in &onnx_graph.initializer {
            let n_elements: usize = init
                .dims
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            // MAGIC_NUMBER:
            if n_elements <= 32 {
                if let Some(floats) = load_initializer_as_f32(init) {
                    known_values.insert(init.name.clone(), floats);
                } else {
                    // Questions
                    // Should this be fatal
                    // Should this be a print or a log
                    panic!("Unable to initializer values for {:?}", init.name);
                }
            }
        }
        // Added by claude, still unsure of its purpose

        // Process computation nodes (Constant nodes add to weight_data)
        process_onnx_nodes(
            &onnx_graph.node,
            &mut tensors,
            &mut context,
            &mut weight_data,
            &mut known_values,
        )
        .unwrap();

        // Mark graph outputs (must happen before build_search_space)
        let mut output_names = Vec::new();
        let mut output_shapes = Vec::new();

        for output_vi in &onnx_graph.output {
            if let Some(gt) = tensors.get(&output_vi.name) {
                gt.output();
                let shape = get_shape_for_onnx_value(output_vi);
                if shape.is_empty() {
                    return Err(format!(
                        "Output tensor '{}' has no shape information in the ONNX model",
                        output_vi.name
                    ));
                }
                output_names.push(output_vi.name.clone());
                output_shapes.push(shape);
            }
        }

        // Extract weight data from initializers (handles inline + external storage)
        for init in &onnx_graph.initializer {
            let floats = match load_tensor_floats(init, model_directory) {
                Some(f) => f,
                None => {
                    continue;
                }
            };
            weight_data.push((init.name.clone(), floats.clone()));
        }

        // Handle _kn tensors from Identity-aliased weights (e.g., layer 1 sharing layer 0 weights).
        // For each _kn tensor without weight_data, find data via the known_values propagation.
        let weight_data_names: HashSet<String> =
            weight_data.iter().map(|(n, _)| n.clone()).collect();
        let kn_tensors: Vec<String> = tensors
            .keys()
            .filter(|name| name.ends_with("_kn") && !weight_data_names.contains(*name))
            .cloned()
            .collect();
        if !kn_tensors.is_empty() {
            panic!("OTHER KNOW NAME THING");
        }

        // Collect tensor name -> NodeIndex mapping
        let tensor_ids: HashMap<String, NodeIndex> = tensors
            .iter()
            .map(|(name, gt)| (name.clone(), gt.id))
            .collect();

        // Track which tensor names are Input nodes (includes those created during process_onnx_nodes)
        let input_tensor_names: HashSet<String> = tensors.keys().cloned().collect();
        //        context.build_search_space::<NativeRuntime>();

        let rt = match backend {
            #[cfg(feature = "cuda")]
            "cuda" => OnnxGraphResult::build_cuda_backend(
                onnx_graph,
                model_directory,
                &mut tensors,
                &mut weight_data,
                &mut context,
                &input_tensor_names,
            )?,
            "native" => OnnxGraphResult::build_native_backend(
                onnx_graph,
                model_directory,
                &mut tensors,
                &mut weight_data,
                &mut context,
                &input_tensor_names,
            )?,
            _ => {
                return Err(format!(
                    "Invalid backend '{}'. Must be 'native' or 'cuda'",
                    backend
                ));
            }
        };

        Ok(OnnxGraphResult {
            context,
            runtime: rt,
            tensor_ids,
            input_names,
            output_names,
            output_shapes,
        })
    }

    #[cfg(feature = "cuda")]
    fn build_cuda_backend(
        onnx_graph: &protobuf::MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
        context: &mut Graph,
        input_tensor_names: &HashSet<String>,
    ) -> Result<RuntimeBackend, String> {
        let compute_n_elements = |name: &str| -> usize {
            if let Some(vi) = onnx_graph.input.iter().find(|i| i.name == name) {
                let shape = get_shape_for_onnx_value(vi);
                shape.iter().product::<usize>()
            } else if let Some(init) = onnx_graph.initializer.iter().find(|i| i.name == name) {
                init.dims.iter().map(|&d| d as usize).product::<usize>()
            } else if let Some((_, data)) = weight_data.iter().find(|(n, _)| n == name) {
                data.len()
            } else {
                0
            }
        };

        // CUDA: Two-phase - set data BEFORE search for profiling
        let (mut cuda_rt, _stream) = prepare_cuda(context)?;

        // Set dummy zero data for ALL input tensors
        for (name, gt) in &mut *tensors {
            if !input_tensor_names.contains(name) {
                continue;
            }
            let n_elements = compute_n_elements(name);
            if n_elements > 0 {
                cuda_rt.set_data(gt.id, vec![0.0f32; n_elements]);
            }
        }

        // Overwrite with real initializer data (for accurate profiling)
        for init in &onnx_graph.initializer {
            let floats = match load_tensor_floats(init, model_directory) {
                Some(f) => f,
                None => continue,
            };
            if let Some(gt) = tensors.get(&init.name) {
                cuda_rt.set_data(gt.id, floats.clone());
            }
            let kn_name = format!("{}_kn", &init.name);
            if let Some(gt_kn) = tensors.get(&kn_name) {
                let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let transposed = transpose_weight_data(&floats, dims[0], dims[1]);
                    cuda_rt.set_data(gt_kn.id, transposed);
                }
            }
        }

        // Load constant node data
        for (name, floats) in weight_data {
            if let Some(gt) = tensors.get(name) {
                cuda_rt.set_data(gt.id, floats.clone());
            }
        }

        // Now finalize (search with profiling, data is available)
        let cuda_rt = finalize_cuda(context, cuda_rt);

        Ok(cuda_rt)
    }

    fn build_native_backend(
        onnx_graph: &protobuf::MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
        context: &mut Graph,
        _input_tensor_names: &HashSet<String>,
    ) -> Result<RuntimeBackend, String> {
        let mut rt = initialize_native(context)?;
        context.search(NativeRuntime::default(), 1);

        // Set initializer data - these MUST exist after optimization (they're weights)
        // Skip _kn variants - they might be optimized away
        for init in &onnx_graph.initializer {
            let floats = match load_tensor_floats(init, model_directory) {
                Some(f) => f,
                None => continue,
            };
            if let Some(gt) = tensors.get(&init.name) {
                rt.set_data(gt.id, floats.clone());
            }
            // NOTE: Skip _kn transposed variants - might be optimized away
        }

        // Load constant node data, but skip _kn transposed variants
        for (name, floats) in weight_data {
            // Skip _kn transposed variants - might be optimized away
            if name.ends_with("_kn") {
                continue;
            }
            if let Some(gt) = tensors.get(name) {
                rt.set_data(gt.id, floats.clone());
            }
        }
        Ok(rt)
    }
}

#[pymethods]
impl OnnxGraphResult {
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

    /// Set input tensor data by name.
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self.tensor_ids.get(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name))
        })?;
        self.runtime.set_data(*node_id, data);
        Ok(())
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.runtime.execute(&self.context.dyn_map);
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
