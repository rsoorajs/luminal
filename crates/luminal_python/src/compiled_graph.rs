use luminal::prelude::{
    tracing::{Level, span, trace},
    *,
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
        println!("{:?}", initializer_names);

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
        println!("{:?}", input_names);

        // Create "holding" tensors for the input
        // this way they can be considered in the graph computation, and later as we do mutiple runs we can target them and swap out the values
        // in them and not need to recompile the network
        for input in &onnx_graph.input {
            let shape = get_shape_for_onnx_value(input);
            if shape.is_empty() {
                println!("Input {} skipped because it is empty", input.name.clone());
                continue;
            }
            let tensor = context.named_tensor(input.name.clone(), shape);
            println!("Input {} added to tensors", input.name.clone());
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
                    println!("Unable to initializer values for {:?}", init.name);
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
            // Also store transposed data for _kn tensors
            let kn_name = format!("{}_kn", &init.name);
            if tensors.contains_key(&kn_name) {
                panic!("THere is a know name thing")
                /*
                let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let transposed = transpose_weight_data(&floats, dims[0], dims[1]);
                    weight_data.push((kn_name, transposed));
                }
                */
            }
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
        for kn_name in kn_tensors {
            panic!("OTHER KNOW NAME THING");
            /*
            // Strip "_kn" to get the base weight name
            let base_name = &kn_name[..kn_name.len() - 3];
            // The base tensor's GraphTensor may point to an initializer (via Identity alias).
            // Look for the data by checking which initializer this tensor's node ID matches.
            if let Some(base_gt) = tensors.get(base_name) {
                // Find which weight_data entry has this same node ID (the aliased initializer)
                for (wd_name, wd_data) in &weight_data.clone() {
                    if let Some(wd_gt) = tensors.get(wd_name)
                        && wd_gt.id == base_gt.id
                    {
                        // Found the source data, generate transposed version
                        if let Some(kn_gt) = tensors.get(&kn_name) {
                            let kn_dims = kn_gt.dims();
                            if kn_dims.len() == 2 {
                                let k = kn_dims[0].to_usize().unwrap();
                                let n = kn_dims[1].to_usize().unwrap();
                                // The _kn shape is [K, N] where original is [N, K]
                                let transposed = transpose_weight_data(wd_data, n, k);
                                weight_data.push((kn_name.clone(), transposed));
                            }
                        }
                        break;
                    }
                }
            }
            */
        }

        // Collect tensor name -> NodeIndex mapping
        let tensor_ids: HashMap<String, NodeIndex> = tensors
            .iter()
            .map(|(name, gt)| (name.clone(), gt.id))
            .collect();

        // Track which tensor names are Input nodes (includes those created during process_onnx_nodes)
        let input_tensor_names: HashSet<String> = tensors.keys().cloned().collect();
        context.build_search_space::<NativeRuntime>();

        let mut rt = OnnxGraphResult::build_cuda_backend(
            onnx_graph,
            model_directory,
            &mut tensors,
            &mut weight_data,
        );

        Ok(OnnxGraphResult {
            context,
            runtime: rt,
            tensor_ids,
            input_names,
            output_names,
            output_shapes,
        })
    }

    fn build_cuda_backend(
        onnx_graph: MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
    ) -> RuntimeBackend {
        // CUDA: Two-phase - set data BEFORE search for profiling
        let (mut cuda_rt, _stream) = prepare_cuda(&mut context)?;

        // Set dummy zero data for ALL input tensors
        for (name, gt) in &tensors {
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
        for (name, floats) in &weight_data {
            if let Some(gt) = tensors.get(name) {
                cuda_rt.set_data(gt.id, floats.clone());
            }
        }

        // Now finalize (search with profiling, data is available)
        finalize_cuda(&mut context, cuda_rt);

        return cuda_rt;
    }

    fn build_native_backend(
        onnx_graph: MessageField<GraphProto>,
        model_directory: &Path,
        tensors: &mut HashMap<String, GraphTensor>,
        weight_data: &mut Vec<(String, Vec<f32>)>,
    ) -> RuntimeBackend {
        context.search(NativeRuntime::default(), 1);
        /*
        // This is zero init inputs tensors... I don't think we need to do that
        for name in &input_names {
            if let Some(gt) = tensors.get(name) {
                let n_elements = compute_n_elements(name);
                if n_elements > 0 {
                    rt.set_data(gt.id, vec![0.0f32; n_elements]);
                }
            }
        }
        */

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
        for (name, floats) in &weight_data {
            // Skip _kn transposed variants - might be optimized away
            if name.ends_with("_kn") {
                continue;
            }
            if let Some(gt) = tensors.get(name) {
                rt.set_data(gt.id, floats.clone());
            }
        }
        return RuntimeBackend::Native(rt);
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
