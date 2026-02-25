use std::sync::Arc;

use crate::{
    cuda_dtype,
    kernel::{CudaFunctionExt, KernelOp},
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::{CompileOptions, compile_ptx, compile_ptx_with_opts},
};
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, app, eq, rule, set, sort, union, v},
        base::{DTYPE, ELIST, EXPRESSION, F64, IR, OP_SORTS, SORTS, dtype},
        extract_dtype, extract_expr, extract_expr_list,
    },
    hlir::{Add, Exp2, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Scatter, Sin, Sqrt, SumReduce},
    op::*,
    prelude::*,
};

/// Generates CUDA include directives based on the dtypes used in a kernel
pub fn dtype_includes(dtypes: &[DType]) -> String {
    let needs_fp16 = dtypes.iter().any(|d| matches!(d, DType::F16));
    let needs_bf16 = dtypes.iter().any(|d| matches!(d, DType::Bf16));
    let needs_fp8 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0));
    let needs_fp6 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F6E2M3 | DType::F6E3M2));
    let needs_fp4 = dtypes.iter().any(|d| matches!(d, DType::F4E2M1));
    let mut s = String::new();
    if needs_fp16 {
        s.push_str("#include <cuda_fp16.h>\n");
    }
    if needs_bf16 {
        s.push_str("#include <cuda_bf16.h>\n");
    }
    if needs_fp8 {
        s.push_str("#include <cuda_fp8.h>\n");
    }
    if needs_fp6 {
        s.push_str("#include <cuda_fp6.h>\n");
    }
    if needs_fp4 {
        s.push_str("#include <cuda_fp4.h>\n");
    }
    s
}

/// Compiles a CUDA kernel with proper include paths for special types
pub fn compile_kernel(kernel: &str, dtypes: &[DType]) -> cudarc::nvrtc::Ptx {
    let needs_special_types = dtypes.iter().any(|d| {
        matches!(
            d,
            DType::F16
                | DType::Bf16
                | DType::F8E4M3
                | DType::F8E5M2
                | DType::F8UE8M0
                | DType::F6E2M3
                | DType::F6E3M2
                | DType::F4E2M1
        )
    });

    if needs_special_types {
        compile_ptx_with_opts(
            kernel,
            CompileOptions {
                include_paths: vec![
                    "/usr/local/cuda/include".to_string(),
                    "/usr/include".to_string(),
                ],
                ..Default::default()
            },
        )
        .unwrap()
    } else {
        compile_ptx(kernel).unwrap()
    }
}

pub type Ops = (
    KernelAdd,
    KernelMul,
    KernelMod,
    KernelLessThan,
    KernelIota,
    KernelGather,
    KernelScatter,
    KernelSumReduce,
    KernelMaxReduce,
    KernelExp2,
    KernelLog2,
    KernelSin,
    KernelRecip,
    KernelSqrt,
    KernelConstant,
    KernelCast,
    KernelEmbed,
);

/// Build a rewrite that matches an HLIR op, reads dtype(s) from the given source fields,
/// and unions with a kernel op that has the same fields plus the dtype(s) appended.
fn kernel_rewrite<H: Default + EgglogOp, L: Default + EgglogOp>() -> Rule {
    let hlir = H::default().sort();
    let llir = L::default().sort();
    let (mut args, hlir_match) = hlir.new_call();
    let dt = v("?__dt");
    args.add("dtype", dt.clone());
    rule(union(hlir_match.clone(), llir.call(&args))).fact(eq(dt, dtype(hlir_match)))
}

#[derive(Default, Debug, Clone)]

pub struct KernelMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMaxReduce {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelMax",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("inp", IR),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<MaxReduce, Self>()]
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
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            }) as Box<dyn KernelOp>),
            vec![children[2]],
        )
    }
}

impl KernelOp for KernelMaxReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        __shared__ {dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};
        long long iter_stride = {iter_stride};

        {dtype} max_value = NEG_INF_F;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            max_value = fmaxf(max_value, in[in_start + i * iter_stride]);
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            max_value = fmaxf(max_value, __shfl_down_sync(FULL_MASK, max_value, s));
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = max_value;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {dtype} block_max = tid < cnt ? warp_sums[tid] : NEG_INF_F;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_max = fmaxf(block_max, __shfl_down_sync(FULL_MASK, block_max, s));
            }}

            if (tid == 0) {{
                out[{out_index}] = block_max;
            }}
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self
                .iter_stride
                .substitute('z', Expression::from(1))
                .simplify()
                .to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_max_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn kernel_name(&self) -> &'static str {
        "MaxReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelSumReduce {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelSum",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("inp", IR),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<SumReduce, Self>()]
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
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            }) as Box<dyn KernelOp>),
            vec![children[2]],
        )
    }
}

impl KernelOp for KernelSumReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();

        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = blockIdx.x;

        long long in_start = {in_index};
        long long iters = {iters};
        long long iter_stride = {iter_stride};

        {dtype} sum = 0;
        for (long long i = 0; i < iters; i++) {{
            sum += in[in_start + i * iter_stride];
        }}

        out[{out_index}] = sum;
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self
                .iter_stride
                .substitute('z', Expression::from(1))
                .simplify()
                .to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()), // grid
            (1.into(), 1.into(), 1.into()),  // blocks (single-threaded)
            0.into(),                        // shmem size
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn kernel_name(&self) -> &'static str {
        "SumReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelAdd {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelAdd {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelAdd",
            &[
                ("shape", ELIST),
                ("inp_a", IR),
                ("a_strides", ELIST),
                ("inp_b", IR),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Add, Self>()]
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
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        // Add dyn_dims parameter if we have dynamic dimensions
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void add_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] + B[{b_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype, self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("add_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        // Return empty constants map - we now use shared dyn_dims buffer
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(), // No per-module constants needed
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Add"
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
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelMul",
            &[
                ("shape", ELIST),
                ("inp_a", IR),
                ("a_strides", ELIST),
                ("inp_b", IR),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Mul, Self>()]
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
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mul_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] * B[{b_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype, self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mul_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Mul"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGather {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelGather",
            &[
                ("out_shape", ELIST),
                ("indexes", IR),
                ("index_strides", ELIST),
                ("data", IR),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (gather_args, gather_match) = luminal::hlir::Gather::default().sort().new_call();
        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), gather_args["index_shape"].clone()));
        let dt = v("?__dt");
        let kernel_args = [
            ("out_shape".to_string(), gather_args["index_shape"].clone()),
            ("indexes".to_string(), gather_args["indexes"].clone()),
            (
                "index_strides".to_string(),
                gather_args["index_strides"].clone(),
            ),
            ("data".to_string(), gather_args["data"].clone()),
            ("data_shape".to_string(), gather_args["data_shape"].clone()),
            (
                "data_strides".to_string(),
                gather_args["data_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        vec![
            rule(union(gather_match, self.sort().call(kernel_args)))
                .fact(eq(dt, dtype(gather_args["data"].clone()))),
        ]
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
                data_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                data_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[6], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[7]),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelGather {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let idx_idx = flatten_strides(&self.out_shape, &self.index_stride).to_kernel();
        let data_idx = flatten_strides(&self.data_shape, &self.data_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        {dtype}* out = C + {out_idx};
        const_z = indexes[{idx_idx}];
        *out = data[{data_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.out_shape.iter().copied().product(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Data + indices (indices are always int32)
        (self.output_size() * self.dtype.bits()).ceil_div(8) + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Gather"
    }
}

// KernelScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Two-phase: copy kernel runs via pre_launch (stream-level), scatter runs in CUDA graph.
#[derive(Debug, Clone)]
pub struct KernelScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
    copy_func: std::sync::OnceLock<(CudaFunction, Arc<CudaModule>, String)>,
}

impl Default for KernelScatter {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            dtype: DType::F32,
            copy_func: std::sync::OnceLock::new(),
        }
    }
}

impl EgglogOp for KernelScatter {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelScatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("dest", IR),
                ("indexes", IR),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src", IR),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (scatter_args, scatter_match) = luminal::hlir::Scatter::default().sort().new_call();
        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), scatter_args["dest_shape"].clone()));
        let dt = v("?__dt");
        let kernel_args = [
            ("dest_shape".to_string(), scatter_args["dest_shape"].clone()),
            (
                "dest_strides".to_string(),
                scatter_args["dest_strides"].clone(),
            ),
            ("dest".to_string(), scatter_args["dest"].clone()),
            ("indexes".to_string(), scatter_args["indexes"].clone()),
            (
                "index_shape".to_string(),
                scatter_args["index_shape"].clone(),
            ),
            (
                "index_strides".to_string(),
                scatter_args["index_strides"].clone(),
            ),
            ("src".to_string(), scatter_args["src"].clone()),
            (
                "src_strides".to_string(),
                scatter_args["src_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        vec![
            rule(union(scatter_match, self.sort().call(kernel_args)))
                .fact(eq(dt, dtype(scatter_args["src"].clone()))),
        ]
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
                dest_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                dest_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, children[8], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[9]),
                copy_func: std::sync::OnceLock::new(),
            })),
            vec![children[2], children[3], children[6]], // dest, indexes, src
        )
    }
}

impl KernelOp for KernelScatter {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let all_vars: FxHashSet<char> = self
            .dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
        let dyn_dims_param = if all_vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        // Compile copy kernel: copies dest → output (one thread per dest element)
        let n_dest = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let copy_out_idx = flatten_strides(&self.dest_shape, &self.out_strides).to_kernel();
        let copy_in_idx = flatten_strides(&self.dest_shape, &self.dest_strides).to_kernel();
        let copy_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter_copy({dtype} *out, const {dtype} *dest{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_dest}) return;
        out[{copy_out_idx}] = dest[{copy_in_idx}];
    }}
}}"
        );
        // Compile and store the copy function
        let (copy_module, copy_func) = if let Some((module, func)) = compile_cache.get(&copy_kernel)
        {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&copy_kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("scatter_copy").unwrap();
            compile_cache.insert(copy_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let _ = self.copy_func.set((copy_func, copy_module, copy_kernel));

        // Compile scatter kernel: writes src[i] → output[indexes[i]] (one thread per src element)
        let n_src_elements = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let n_dest_elements = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let scatter_idx_idx = flatten_strides(&self.index_shape, &self.index_strides).to_kernel();
        let scatter_src_idx = flatten_strides(&self.index_shape, &self.src_strides).to_kernel();
        let scatter_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter({dtype} *out, const int *indexes, const {dtype} *src{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_src_elements}) return;
        int idx = indexes[{scatter_idx_idx}];
        if (idx >= 0 && idx < {n_dest_elements}) {{
            out[idx] = src[{scatter_src_idx}];
        }}
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&scatter_kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&scatter_kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("scatter").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let n_src: Expression = self.index_shape.iter().copied().product();
        (
            func,
            module,
            scatter_kernel,
            (n_src, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.dest_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter output_bytes: {other:?}"),
        }
        .into();
        self.output_size() * elem_size
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // scatter kernel: (out, indexes, src [, dyn_dims])
        // input_ptrs: [dest, indexes, src]
        let mut params = vec![output_ptr, input_ptrs[1], input_ptrs[2]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn pre_launch(
        &self,
        stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        dyn_dims_ptr: u64,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let (copy_func, _copy_module, _) = self.copy_func.get().expect("copy kernel not compiled");
        let n_dest = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;
        let dest_ptr = input_ptrs[0]; // dest is input 0

        let has_dyn = dyn_dims_ptr != 0;
        let mut param_values: Vec<u64> = vec![output_ptr, dest_ptr];
        if has_dyn {
            param_values.push(dyn_dims_ptr);
        }
        let mut param_ptrs: Vec<*mut std::ffi::c_void> = param_values
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();

        let cu_func = unsafe { copy_func.raw_function() };
        unsafe {
            cudarc::driver::sys::cuLaunchKernel(
                cu_func,
                n_dest,
                1,
                1,
                1,
                1,
                1,
                0,
                stream.cu_stream() as *mut _,
                param_ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
            .result()?;
        }
        Ok(())
    }

    fn bytes_loaded(&self) -> Expression {
        let data_elem_size: Expression = match self.dtype {
            DType::F64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter bytes_loaded: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // dest (copy) + indices + src
        self.output_size() * data_elem_size + n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Scatter"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for KernelIota {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_iota) = luminal::hlir::Iota::default().sort().new_call();
        let kernel_iota = self.sort().call(&args);
        vec![
            rule(union(hlir_iota, kernel_iota.clone()))
                .set(dtype(kernel_iota), app(&SORTS.int_dt, vec![])),
        ]
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
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self.expr.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void iota_k(int *C{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[const_z] = {};
    }}
}}",
            self.expr.to_kernel(),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[DType::Int]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("iota_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.range, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn output_bytes(&self) -> Expression {
        // Iota always outputs int32 (4 bytes)
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Iota"
    }
}

// =============================================================================
// Unary Operations: Exp2, Log2, Sin, Recip, Sqrt
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelExp2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelExp2 {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelExp2",
            &[
                ("shape", ELIST),
                ("inp", IR),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Exp2, Self>()]
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
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
        )
    }
}

impl KernelOp for KernelExp2 {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void exp2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = exp2f(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("exp2_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Exp2"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLog2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelLog2 {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelLog2",
            &[
                ("shape", ELIST),
                ("inp", IR),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Log2, Self>()]
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
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
        )
    }
}

impl KernelOp for KernelLog2 {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void log2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = log2f(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("log2_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Log2"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSin {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelSin {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelSin",
            &[
                ("shape", ELIST),
                ("inp", IR),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Sin, Self>()]
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
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
        )
    }
}

impl KernelOp for KernelSin {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void sin_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = sinf(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("sin_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Sin"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelRecip {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelRecip {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelRecip",
            &[
                ("shape", ELIST),
                ("inp", IR),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Recip, Self>()]
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
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
        )
    }
}

impl KernelOp for KernelRecip {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void recip_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = 1.0f / in[{in_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("recip_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Recip"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSqrt {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelSqrt {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelSqrt",
            &[
                ("shape", ELIST),
                ("inp", IR),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Sqrt, Self>()]
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
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
        )
    }
}

impl KernelOp for KernelSqrt {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let vars = self
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void sqrt_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = sqrtf(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("sqrt_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Sqrt"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelMod {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMod {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelMod",
            &[
                ("shape", ELIST),
                ("inp_a", IR),
                ("a_strides", ELIST),
                ("inp_b", IR),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Mod, Self>()]
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

impl KernelOp for KernelMod {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mod_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = fmodf(A[{a_idx}], B[{b_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mod_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Both inputs have same dtype
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Mod"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLessThan {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelLessThan {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelLessThan",
            &[
                ("shape", ELIST),
                ("inp_a", IR),
                ("a_strides", ELIST),
                ("inp_b", IR),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let hlir = LessThan::default().sort();
        let (mut args, hlir_match) = hlir.new_call();
        // LessThan's dtype is Bool (output type), but the kernel needs the INPUT dtype
        let dt = v("?__dt");
        args.add("dtype", dt.clone());
        vec![
            rule(union(hlir_match, self.sort().call(&args)))
                .fact(eq(dt, dtype(args["inp_a"].clone()))),
        ]
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

impl KernelOp for KernelLessThan {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
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

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void less_than_k(unsigned char *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] < B[{b_idx}] ? 1 : 0;
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype, self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("less_than_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        // LessThan outputs Bool (unsigned char, 1 byte per element)
        self.output_size()
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "LessThan"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelConstant {
    value: f32,
}

impl EgglogOp for KernelConstant {
    fn sort(&self) -> SortDef {
        sort(IR, "KernelConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, const_match) = luminal::hlir::Constant::default().sort().new_call();
        let kernel_const = self.sort().call(&args);
        vec![
            rule(union(const_match, kernel_const.clone()))
                .set(dtype(kernel_const), app(&SORTS.f32_dt, vec![])),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelConstant {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let kernel = format!(
            "
extern \"C\" {{
    __global__ void constant_k(float *out) {{
        out[0] = {:.10}f;
    }}
}}",
            self.value
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[DType::F32]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("constant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (1.into(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        1.into()
    }

    fn output_bytes(&self) -> Expression {
        // Constant always outputs F32
        4.into()
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Constant"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelCast {
    size: Expression,
    in_dtype: DType,
    out_dtype: DType,
}

impl EgglogOp for KernelCast {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelCast",
            &[
                ("inp", IR),
                ("size", EXPRESSION),
                ("dtype", DTYPE),
                ("src_dtype", DTYPE),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (mut args, cast_match) = luminal::hlir::Cast::default().sort().new_call();
        let out_dty = args.remove("dtype");
        let in_dty = v("?__in_dt");
        args.add("dtype", in_dty.clone());
        args.add("src_dtype", out_dty);
        vec![
            rule(union(cast_match, self.sort().call(&args)))
                .fact(eq(in_dty, dtype(args["inp"].clone()))),
        ]
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
                size: extract_expr(egraph, children[1], expr_cache).unwrap_or_default(),
                in_dtype: extract_dtype(egraph, children[2]),
                out_dtype: extract_dtype(egraph, children[3]),
            })),
            vec![children[0]],
        )
    }
}

impl KernelOp for KernelCast {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let out_dtype = cuda_dtype(self.out_dtype);
        let includes = dtype_includes(&[self.in_dtype, self.out_dtype]);

        let kernel = if self.in_dtype.bits() < 8 {
            // Sub-byte packed types: multiple values packed per byte.
            // Extract the correct bits using bit-level addressing.
            let bits = self.in_dtype.bits();
            let in_cuda_type = cuda_dtype(self.in_dtype);
            let mask = (1u32 << bits) - 1;
            format!(
                "{includes}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const unsigned char *in_raw) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long bit_offset = idx * {bits};
        long long byte_idx = bit_offset >> 3;
        int bit_pos = (int)(bit_offset & 7);
        unsigned short raw = (unsigned short)in_raw[byte_idx];
        if (bit_pos + {bits} > 8) raw |= ((unsigned short)in_raw[byte_idx + 1]) << 8;
        {in_cuda_type} val;
        val.__x = (unsigned char)((raw >> bit_pos) & {mask}u);
        out[idx] = ({out_dtype})val;
    }}
}}"
            )
        } else {
            let in_dtype = cuda_dtype(self.in_dtype);
            format!(
                "{includes}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[const_z] = ({out_dtype})in[const_z];
    }}
}}"
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.in_dtype, self.out_dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.size, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn output_bytes(&self) -> Expression {
        (self.size * self.out_dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.size * self.in_dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Cast"
    }
}

/// Generate #define macros for dynamic dimensions that read from a shared dyn_dims buffer.
/// The buffer layout is alphabetically sorted by dim char for consistency.
/// Returns (defines_string, sorted_dims) where sorted_dims gives the order of dims in the buffer.
pub fn generate_dyn_dims_defines(vars: &FxHashSet<char>) -> (String, Vec<char>) {
    if vars.is_empty() {
        return (String::new(), Vec::new());
    }
    let mut sorted_dims: Vec<char> = vars.iter().copied().collect();
    sorted_dims.sort();
    let defines = sorted_dims
        .iter()
        .enumerate()
        .map(|(idx, dim)| format!("#define const_{dim} (dyn_dims + {idx})"))
        .collect::<Vec<_>>()
        .join("\n");
    (defines, sorted_dims)
}

/// Get the offset for a dynamic dimension in the shared dyn_dims buffer.
/// Returns None if the dim is not in the set.
pub fn get_dyn_dim_offset(dim: char, sorted_dims: &[char]) -> Option<usize> {
    sorted_dims.iter().position(|&d| d == dim)
}

#[derive(Default, Debug, Clone)]
pub struct KernelEmbed {
    batch_shape: Vec<Expression>,  // batch dimensions (e.g., [seq_len])
    token_stride: Vec<Expression>, // stride for token_ids input
    out_stride: Vec<Expression>,   // stride for output
    embed_dim: Expression,         // embedding dimension
}

impl EgglogOp for KernelEmbed {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelEmbed",
            &[
                ("batch_shape", ELIST),
                ("token_ids", IR),
                ("token_stride", ELIST),
                ("embed_table", IR),
                ("out_stride", ELIST),
                ("embed_dim", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // Match Gather with Add(Mul(Cast(token_ids), const), Iota) indices
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?iota_result (Iota ?iota_expr ?iota_range))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (KernelEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with cast mul\"
            )"),
            // Match Gather with Add(Iota, Mul(Cast(token_ids), const)) indices (reversed order)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?iota_result (Iota ?iota_expr ?iota_range))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (KernelEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with cast mul reversed\"
            )"),
            // Match Gather with Add(Mul(token_ids, const), Iota) indices (no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?iota_result (Iota ?iota_expr ?iota_range))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (KernelEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with mul\"
            )"),
            // Match Gather with Add(Iota, Mul(token_ids, const)) indices (reversed order, no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?iota_result (Iota ?iota_expr ?iota_range))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (KernelEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with mul reversed\"
            )"),
        ]
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
                batch_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                    .unwrap(),
                token_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                embed_dim: extract_expr(egraph, children[5], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]], // token_ids, embedding_table
        )
    }
}

impl KernelOp for KernelEmbed {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        let vars = self
            .batch_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.token_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.embed_dim.dyn_vars())
            .collect::<FxHashSet<_>>();
        let token_offset_expr = flatten_strides(&self.batch_shape, &self.token_stride).to_kernel();
        let out_offset_expr = flatten_strides(&self.batch_shape, &self.out_stride).to_kernel();
        let embed_dim_expr = self.embed_dim.to_kernel();
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void embed(float *out, const int *token_ids, const float *embed_table) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long embed_dim = {embed_dim_expr};
        long long batch_idx = idx / embed_dim;
        long long embed_idx = idx % embed_dim;
        long long const_z = batch_idx;
        long long token_offset = {token_offset_expr};
        long long out_offset = {out_offset_expr};
        int token_id = token_ids[token_offset];
        out[out_offset * embed_dim + embed_idx] = embed_table[(long long)token_id * embed_dim + embed_idx];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("embed").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        let total_threads = batch_size * self.embed_dim;
        (
            func,
            module,
            kernel,
            (total_threads, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
            * self.embed_dim
    }

    fn output_bytes(&self) -> Expression {
        // Embed outputs F32
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        // Load: 1 token ID (4 bytes) per batch + 1 embedding row (embed_dim * 4 bytes) per batch
        batch_size * (4 + self.embed_dim * 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store: 1 embedding row per batch element
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // No FLOPs - just memory copy
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Embed"
    }
}
