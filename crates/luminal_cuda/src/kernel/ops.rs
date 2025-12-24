use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream},
    nvrtc::{compile_ptx, CompileOptions},
};
use itertools::Itertools;
use luminal::{
    graph::{extract_dtype, extract_expr, extract_expr_list, SerializedEGraph},
    op::DType,
    prelude::ENodeId,
    shape::Expression,
    utils::{
        flatten_mul_strides, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{cuda_dtype, kernel::KernelOp};

pub type Ops = (KernelAdd, KernelMul, KernelIota, KernelGather);

#[derive(Default, Debug, Clone)]
pub struct KernelAdd {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Add ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (union ?a (KernelAdd ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty))
    )
    :name \"kernel add\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
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
        String,
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
        let dtype = cuda_dtype(self.dtype);
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void add_k({dtype} *C, const {dtype} *A, const {dtype} *B) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] + B[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("add_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            kernel,
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
pub struct KernelMul {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Mul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= (dtype ?inp_a) (Int))
    )
    (
        (union ?a (KernelMul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides (Int)))
    )
    :name \"kernel mul\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelMul {
    fn compile(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        String,
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
        let dtype = cuda_dtype(self.dtype);
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void mul_k({dtype} *C, const {dtype} *A, const {dtype} *B) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] * B[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("mul_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            kernel,
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
    dtype: DType,
}

impl EgglogOp for KernelGather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Gather ?indexes ?out_shape ?index_strides ?data ?data_shape ?data_strides))
        (= ?dty (dtype ?data))
    )
    (
        (let ?out_strides (RowMajor ?out_shape))
        (union ?a (KernelGather ?out_shape ?indexes ?index_strides ?data ?data_strides ?out_strides ?dty))
    )
    :name \"kernel gather\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
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
        String,
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
        let dtype = cuda_dtype(self.dtype);
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        {dtype}* out = C + {};
        const_z = indexes[{}];
        *out = data[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.index_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.data_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("gather").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            kernel,
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

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for KernelIota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelIota".to_string(), vec![Expr, Expr])
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Iota ?expr ?range))
    )
    (
        (let ?kernel_iota (KernelIota ?expr ?range))
        (union ?a ?kernel_iota)
        (set (dtype ?kernel_iota) (Int))
    )
    :name \"kernel iota\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                expr: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelIota {
    fn compile(
        &self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self.expr.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void iota_k(int *C) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[const_z] = {};
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            self.expr.to_kernel(),
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let func = module.load_function("iota_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            kernel,
            (self.range, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.range
    }
}
