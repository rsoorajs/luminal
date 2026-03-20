use std::sync::Mutex;

use luminal::graph::Graph as LuminalGraph;
use luminal::prelude::*;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[cfg(feature = "cuda")]
use luminal_cuda_lite::cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use luminal_cuda_lite::runtime::CudaRuntime;

use crate::pt2_parser;
use crate::pt2_schema;
use crate::runtime::RuntimeBackend;
use crate::translator;

pub(crate) struct Pt2CompiledModelInner {
    runtime: RuntimeBackend,
    graph: LuminalGraph,
    user_input_ids: Vec<NodeIndex>,
    user_input_shapes: Vec<Vec<Expression>>,
    output_ids: Vec<NodeIndex>,
    output_shapes: Vec<Vec<Expression>>,
}

// Safety: Send is safe because all fields are logically Send (Graph, RuntimeBackend, Vec,
// etc.) and all access is serialized through the Mutex in Pt2CompiledModel.
unsafe impl Send for Pt2CompiledModelInner {}

#[pyclass]
pub struct Pt2CompiledModel {
    inner: Mutex<Pt2CompiledModelInner>,
}

#[pymethods]
impl Pt2CompiledModel {
    pub fn execute(
        &self,
        input_data: Vec<f32>,
        input_shape: Vec<usize>,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let mut inner = self.inner.lock().unwrap();

        if inner.user_input_ids.len() > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Model has {} user inputs, but execute() only supports single-input models. \
                 Use a wrapper to handle multiple inputs.",
                inner.user_input_ids.len()
            )));
        }
        if inner.output_ids.len() > 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Model has {} outputs, but execute() only returns the first. \
                 Multi-output support is not yet implemented.",
                inner.output_ids.len()
            )));
        }

        let input_id = *inner
            .user_input_ids
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No user inputs"))?;

        if let Some(input_shape_exprs) = inner.user_input_shapes.first().cloned() {
            set_dyn_dims_from_input(&mut inner.graph, &input_shape_exprs, &input_shape);
        }

        let dyn_map = inner.graph.dyn_map.clone();
        let output_id = *inner
            .output_ids
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No outputs in graph"))?;
        let output_shape_exprs = inner.output_shapes[0].clone();

        inner.runtime.set_data(input_id, input_data);
        inner.runtime.execute(&dyn_map);
        let result = inner.runtime.get_f32(output_id);

        let shape = resolve_shape(&output_shape_exprs, &dyn_map);
        let total_elements: usize = shape.iter().product();
        let mut result = result;
        if result.len() != total_elements {
            eprintln!(
                "[luminal] Warning: output buffer size ({}) differs from expected shape size ({}). \
                 Resizing to match.",
                result.len(),
                total_elements
            );
            result.resize(total_elements, 0.0);
        }

        Ok((result, shape))
    }
}

fn set_dyn_dims_from_input(
    graph: &mut LuminalGraph,
    shape_exprs: &[Expression],
    input_shape: &[usize],
) {
    for (i, dim_expr) in shape_exprs.iter().enumerate() {
        if i < input_shape.len() {
            if let Some(c) = expr_single_var(dim_expr) {
                graph.set_dim(c, input_shape[i]);
            }
        }
    }
}

/// Detect if an expression is a bare single symbolic variable (e.g., just 'a').
/// Uses brute-force probing: evaluates the expression with each variable set to 42 and checks
/// if the result equals 42. This only detects bare symbol variables, not compound expressions
/// like `a + 1` or `a * b`.
fn expr_single_var(expr: &Expression) -> Option<char> {
    if expr.to_usize().is_some() {
        return None;
    }
    for c in 'a'..='z' {
        let mut map = FxHashMap::default();
        map.insert(c, 42);
        if let Some(result) = expr.exec(&map) {
            if result == 42 {
                return Some(c);
            }
        }
    }
    None
}

fn resolve_dim_sizes(
    sizes: &[pt2_schema::DimSize],
    sym_to_char: &std::collections::HashMap<String, char>,
) -> Vec<Expression> {
    sizes
        .iter()
        .map(|s| match s {
            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int as usize),
            pt2_schema::DimSize::Expr(e) => {
                if let Some(sym) = pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str) {
                    if let Some(c) = sym_to_char.get(&sym) {
                        Expression::from(*c)
                    } else {
                        Expression::from(1usize)
                    }
                } else {
                    Expression::from(1usize)
                }
            }
        })
        .collect()
}

fn resolve_shape(exprs: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Vec<usize> {
    exprs
        .iter()
        .map(|e| {
            e.exec(dyn_map)
                .unwrap_or_else(|| panic!("Cannot resolve shape expression: {e:?}"))
        })
        .collect()
}

#[pyfunction]
pub fn compile_pt2(
    pt2_path: &str,
    weights_path: &str,
    backend: &str,
    search_iters: usize,
) -> PyResult<Pt2CompiledModel> {
    compile_pt2_inner(pt2_path, weights_path, backend, search_iters)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

fn compile_pt2_inner(
    pt2_path: &str,
    weights_path: &str,
    backend: &str,
    search_iters: usize,
) -> anyhow::Result<Pt2CompiledModel> {
    let parsed = pt2_parser::parse_pt2(pt2_path)?;
    let translated = translator::translate(&parsed)?;
    let mut graph = translated.graph;

    for (sym_name, c) in &translated.sym_map.sym_to_char {
        if let Some(rc) = translated.sym_map.ranges.get(sym_name) {
            graph.set_dim(*c, rc.min_val as usize);
        }
    }

    let output_shapes: Vec<Vec<Expression>> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
                .unwrap_or_default()
        })
        .collect();

    let user_input_ids: Vec<NodeIndex> = translated
        .user_input_ids
        .iter()
        .map(|(_, id)| *id)
        .collect();
    let output_ids: Vec<NodeIndex> = translated.output_ids.iter().map(|(_, id)| *id).collect();

    let user_input_shapes: Vec<Vec<Expression>> = translated
        .user_input_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
                .unwrap_or_default()
        })
        .collect();

    let user_input_sizes: Vec<(NodeIndex, usize)> = translated
        .user_input_ids
        .iter()
        .map(|(name, id)| {
            let meta = parsed.tensor_meta(name);
            let n_elements = meta
                .map(|m| {
                    m.sizes
                        .iter()
                        .map(|s| s.hint().unwrap_or(1) as usize)
                        .product()
                })
                .unwrap_or(1);
            (*id, n_elements)
        })
        .collect();

    let runtime = match backend {
        "cpu" | "native" => {
            graph.build_search_space::<NativeRuntime>();
            let mut rt = graph.search(NativeRuntime::default(), search_iters);
            if !weights_path.is_empty() {
                load_safetensors_native(&mut rt, &graph, weights_path)?;
            }
            load_constants_native(&mut rt, &graph, &parsed)?;
            RuntimeBackend::Native(rt)
        }
        "cuda" | "gpu" => init_cuda_runtime(
            &mut graph,
            weights_path,
            &parsed,
            &user_input_sizes,
            search_iters,
        )?,
        other => {
            anyhow::bail!("Unknown backend: {other}. Use 'cpu' or 'cuda'.");
        }
    };

    Ok(Pt2CompiledModel {
        inner: Mutex::new(Pt2CompiledModelInner {
            runtime,
            graph,
            user_input_ids,
            user_input_shapes,
            output_ids,
            output_shapes,
        }),
    })
}

#[cfg(feature = "cuda")]
fn init_cuda_runtime(
    graph: &mut LuminalGraph,
    weights_path: &str,
    parsed: &pt2_parser::ParsedPT2,
    user_input_sizes: &[(NodeIndex, usize)],
    search_iters: usize,
) -> anyhow::Result<RuntimeBackend> {
    let cuda_ctx =
        CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA context init failed: {e}"))?;
    let stream = cuda_ctx.default_stream();

    graph.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    // Phase 1: Set ALL input nodes to safe dummy data (1.0) for search profiling.
    // Real weights/constants may contain -inf (e.g. causal attention mask) which
    // produce NaN in intermediate computations (e.g. -inf - (-inf) = NaN in softmax
    // decomposition), causing the search's has_nan_outputs check to reject ALL
    // candidates. We load real data only AFTER the search completes.
    set_all_inputs_dummy_cuda(&mut rt, graph, weights_path, parsed, user_input_sizes)?;

    let mut rt = graph.search(rt, search_iters);

    if !weights_path.is_empty() {
        load_safetensors_cuda(&mut rt, graph, weights_path)?;
    }
    load_constants_cuda(&mut rt, graph, parsed)?;

    Ok(RuntimeBackend::Cuda(Box::new(rt)))
}

#[cfg(not(feature = "cuda"))]
fn init_cuda_runtime(
    _graph: &mut LuminalGraph,
    _weights_path: &str,
    _parsed: &pt2_parser::ParsedPT2,
    _user_input_sizes: &[(NodeIndex, usize)],
    _search_iters: usize,
) -> anyhow::Result<RuntimeBackend> {
    anyhow::bail!("CUDA support not compiled. Rebuild with --features cuda")
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn load_safetensors_impl(
    cx: &LuminalGraph,
    file_path: &str,
    mut set_data: impl FnMut(NodeIndex, Vec<f32>),
) -> anyhow::Result<()> {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;
    use std::fs::File;

    let f = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("SafeTensors deserialize error: {e}"))?;

    for node in cx.graph.node_indices() {
        if let Some(input) = (*cx.graph[node])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
        {
            if let Ok(tensor) = st.tensor(&input.label) {
                let f32s = safetensor_bytes_to_f32(tensor.data(), tensor.dtype());
                set_data(node, f32s);
            }
        }
    }

    Ok(())
}

fn load_safetensors_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    file_path: &str,
) -> anyhow::Result<()> {
    load_safetensors_impl(cx, file_path, |node, data| rt.set_data(node, data))
}

#[cfg(feature = "cuda")]
fn load_safetensors_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    file_path: &str,
) -> anyhow::Result<()> {
    load_safetensors_impl(cx, file_path, |node, data| rt.set_data(node, data))
}

/// Set ALL input nodes to dummy 1.0 data for safe CUDA search profiling.
///
/// Real weights/constants may contain -inf (e.g. causal attention masks) which
/// produce NaN in intermediate computations during profiling. We compute the
/// element count for each Input node from the safetensors metadata and constants
/// config, then fill with 1.0.
#[cfg(feature = "cuda")]
fn set_all_inputs_dummy_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    weights_path: &str,
    parsed: &pt2_parser::ParsedPT2,
    user_input_sizes: &[(NodeIndex, usize)],
) -> anyhow::Result<()> {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;
    use std::collections::HashMap;
    use std::fs::File;

    // Build a map: label -> n_elements from safetensors metadata (shapes only, no data copy)
    let mut label_sizes: HashMap<String, usize> = HashMap::new();

    if !weights_path.is_empty() {
        let f = File::open(weights_path)?;
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("SafeTensors deserialize error: {e}"))?;
        for (name, info) in st.tensors() {
            let n: usize = info.shape().iter().product();
            label_sizes.insert(name.to_string(), n);
        }
    }

    // Add constants from the .pt2 archive
    if let Some(cc) = &parsed.constants_config {
        for (name, entry) in &cc.config {
            let n: usize = entry
                .tensor_meta
                .sizes
                .iter()
                .map(|s| s.hint().unwrap_or(1) as usize)
                .product();
            label_sizes.insert(name.clone(), n);
        }
    }

    // Set all Input nodes whose label matches a known weight/constant to 1.0
    for node_id in cx.graph.node_indices() {
        if let Some(input) = (*cx.graph[node_id])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
        {
            if let Some(&n) = label_sizes.get(&input.label) {
                if n > 0 {
                    rt.set_data(node_id, vec![1.0f32; n]);
                }
            }
        }
    }

    // Set user input nodes
    for &(id, n_elements) in user_input_sizes {
        rt.set_data(id, vec![1.0f32; n_elements]);
    }

    Ok(())
}

fn safetensor_bytes_to_f32(bytes: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    }
}

/// Convert raw bytes to f32 using PT2 dtype numbering.
/// 1=uint8, 2=int8, 3=int16, 4=int32, 5=int64, 6=float16, 7=float32, 8=float64, 12=bool, 13=bfloat16
fn constant_bytes_to_f32(bytes: &[u8], dtype: u32) -> Vec<f32> {
    match dtype {
        7 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        6 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        13 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        8 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        5 => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        4 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        3 => bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
            .collect(),
        2 => bytes.iter().map(|&b| (b as i8) as f32).collect(),
        1 => bytes.iter().map(|&b| b as f32).collect(),
        12 => bytes
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    }
}

fn load_constants_impl(
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
    mut set_data: impl FnMut(NodeIndex, Vec<f32>),
) -> anyhow::Result<()> {
    let constants_config = match &parsed.constants_config {
        Some(c) => c,
        None => return Ok(()),
    };

    for (name, entry) in &constants_config.config {
        let raw_bytes = match pt2_parser::read_constant_bytes(
            &parsed.pt2_path,
            &parsed.archive_prefix,
            entry,
        ) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let f32_data = constant_bytes_to_f32(&raw_bytes, entry.tensor_meta.dtype);

        for node_id in cx.graph.node_indices() {
            if let Some(input) = (*cx.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
            {
                if input.label == *name {
                    set_data(node_id, f32_data.clone());
                }
            }
        }
    }
    Ok(())
}

fn load_constants_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
) -> anyhow::Result<()> {
    load_constants_impl(cx, parsed, |node, data| rt.set_data(node, data))
}

#[cfg(feature = "cuda")]
fn load_constants_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
) -> anyhow::Result<()> {
    load_constants_impl(cx, parsed, |node, data| rt.set_data(node, data))
}
