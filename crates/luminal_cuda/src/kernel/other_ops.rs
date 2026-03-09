use std::sync::Arc;

use crate::{
    cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{compile_kernel, dtype_includes, generate_dyn_dims_defines},
};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::CompileOptions,
};
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, IR},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

pub type Ops = (KernelMeanReduce,);

#[derive(Default, Debug, Clone)]

pub struct KernelMeanReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMeanReduce {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "KernelMean",
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
        // Disabled: the e-graph union introduced by this rule can cause the search
        // to select genomes with accumulated FP precision issues over many layers.
        // The unfused Sum + Mul(Recip(Cast(Iota))) path produces equivalent results.
        vec![]
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
            {
                let out_shape =
                    extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap();
                let iters = extract_expr(egraph, children[1], expr_cache).unwrap();
                let in_stride =
                    extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap();
                let iter_stride = extract_expr(egraph, children[4], expr_cache).unwrap();
                let out_stride =
                    extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap();
                let dtype = extract_dtype(egraph, children[6]);
                LLIROp::new::<dyn KernelOp>(Box::new(Self {
                    out_shape,
                    iters,
                    in_stride,
                    iter_stride,
                    out_stride,
                    dtype,
                }) as Box<dyn KernelOp>)
            },
            vec![children[2]],
        )
    }
}

impl KernelOp for KernelMeanReduce {
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

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_mean_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = blockIdx.x;
        long long n_elements = {n_outputs};
        if (const_z >= n_elements) return;

        long long in_start = {in_index};
        long long iters = {iters};

        {dtype} sum = 0;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            sum += in[in_start + {iter_stride_of_i}];
        }}

        out[{out_index}] = ({dtype})(sum / ({dtype})iters);
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            n_outputs = n_outputs.to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride_of_i = iter_stride_of_i,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_kernel(&kernel, &[self.dtype]);
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_mean_k").unwrap();
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
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        n_outputs * self.iters + n_outputs
    }

    fn kernel_name(&self) -> &'static str {
        "MeanReduce"
    }
}
