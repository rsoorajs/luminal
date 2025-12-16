use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use itertools::Itertools;
use luminal::{
    graph::{extract_expr_list, SerializedEGraph},
    shape::Expression,
    utils::{
        flatten_strides, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::kernel::KernelOp;

pub type Ops = (KernelAdd, KernelGather);

#[derive(Default, Debug, Clone)]
pub struct KernelAdd {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

impl EgglogOp for KernelAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Add ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
    )
    (
        (union ?a (KernelAdd ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
    )
    :name \"kernel add\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a egraph_serialize::NodeId>,
        list_cache: &mut rustc_hash::FxHashMap<&'a egraph_serialize::NodeId, Vec<Expression>>,
        expr_cache: &mut rustc_hash::FxHashMap<&'a egraph_serialize::NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a egraph_serialize::NodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelAdd {
    fn compile(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void add_k(float *C, const float *A, const float *B) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] + B[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let ptx = compile_ptx_with_opts(
            &kernel,
            CompileOptions {
                arch: Some("sm_90a"),
                options: vec!["--std=c++17".to_string(), "-default-device".to_string()],
                include_paths: vec![
                    "/usr/local/cuda/include".to_string(),
                    "/usr/include".to_string(),
                ],
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("add_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            (
                self.out_shape.iter().copied().product::<Expression>(),
                1.into(),
                1.into(),
            ),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

impl EgglogOp for KernelGather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Gather ?out_shape ?indexes ?index_strides ?data ?data_strides))
    )
    (
        (let ?out_strides (RowMajor ?out_shape))
        (union ?a (KernelGather ?out_shape ?indexes ?index_strides ?data ?data_strides ?out_strides))
    )
    :name \"kernel gather\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &Vec<&'a egraph_serialize::NodeId>,
        list_cache: &mut rustc_hash::FxHashMap<&'a egraph_serialize::NodeId, Vec<Expression>>,
        expr_cache: &mut rustc_hash::FxHashMap<&'a egraph_serialize::NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a egraph_serialize::NodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelGather {
    fn compile(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void gather(float *C, const int *indexes, const float *data) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        float* out = C + {};
        const_z = indexes[{}];
        *out = data[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_strides(&self.out_shape, &self.index_stride).to_kernel(),
            flatten_strides(&self.out_shape, &self.data_stride).to_kernel()
        );
        let ptx = compile_ptx_with_opts(
            &kernel,
            CompileOptions {
                arch: Some("sm_90a"),
                options: vec!["--std=c++17".to_string(), "-default-device".to_string()],
                include_paths: vec![
                    "/usr/local/cuda/include".to_string(),
                    "/usr/include".to_string(),
                ],
                ..Default::default()
            },
        )
        .unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("gather").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            (self.out_shape.iter().copied().product(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }
}
