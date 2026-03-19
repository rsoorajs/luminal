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

// Safety: All access is guarded by Mutex.
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

        let input_id = *inner
            .user_input_ids
            .first()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No user inputs"))?;

        if let Some(input_shape_exprs) = inner.user_input_shapes.first().cloned() {
            set_dyn_dims_from_input(&mut inner.graph, &input_shape_exprs, &input_shape);
        }

        let dyn_map = inner.graph.dyn_map.clone();
        let output_id = *inner.output_ids.first().unwrap();
        let output_shape_exprs = inner.output_shapes[0].clone();

        inner.runtime.set_data(input_id, input_data);
        inner.runtime.execute(&dyn_map);
        let result = inner.runtime.get_f32(output_id);

        let shape = resolve_shape(&output_shape_exprs, &dyn_map);
        let total_elements: usize = shape.iter().product();
        let mut result = result;
        result.resize(total_elements, 0.0);

        Ok((result, shape))
    }
}

fn set_dyn_dims_from_input(graph: &mut LuminalGraph, shape_exprs: &[Expression], input_shape: &[usize]) {
    for (i, dim_expr) in shape_exprs.iter().enumerate() {
        if i < input_shape.len() {
            if let Some(c) = expr_single_var(dim_expr) {
                graph.set_dim(c, input_shape[i]);
            }
        }
    }
}

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
                .map(|meta| {
                    meta.sizes
                        .iter()
                        .map(|s| match s {
                            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int as usize),
                            pt2_schema::DimSize::Expr(e) => {
                                if let Some(sym) =
                                    pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str)
                                {
                                    if let Some(c) = translated.sym_map.sym_to_char.get(&sym) {
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
                })
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
                .map(|meta| {
                    meta.sizes
                        .iter()
                        .map(|s| match s {
                            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int as usize),
                            pt2_schema::DimSize::Expr(e) => {
                                if let Some(sym) =
                                    pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str)
                                {
                                    if let Some(c) = translated.sym_map.sym_to_char.get(&sym) {
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
                })
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
        "cuda" | "gpu" => {
            init_cuda_runtime(
                &mut graph,
                weights_path,
                &parsed,
                &user_input_sizes,
                search_iters,
            )?
        }
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

    if !weights_path.is_empty() {
        load_safetensors_cuda(&mut rt, graph, weights_path)?;
    }
    load_constants_cuda(&mut rt, graph, parsed)?;
    for &(id, n_elements) in user_input_sizes {
        rt.set_data(id, vec![0.0f32; n_elements]);
    }

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

fn load_safetensors_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    file_path: &str,
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
                rt.set_data(node, f32s);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn load_safetensors_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    file_path: &str,
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
                rt.set_data(node, f32s);
            }
        }
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
            .map(|b| {
                i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            })
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
        12 => bytes.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    }
}

fn load_constants_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
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
                    rt.set_data(node_id, f32_data.clone());
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn load_constants_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
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
                    rt.set_data(node_id, f32_data.clone());
                }
            }
        }
    }
    Ok(())
}
