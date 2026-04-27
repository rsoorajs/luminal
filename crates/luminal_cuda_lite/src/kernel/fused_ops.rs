// =========================================================================
// Fused elementwise op variants used inside FusionStart/FusionEnd regions.
//
// Each `FusedX` struct mirrors its un-fused `KernelX` sibling field-for-field
// and serves a single purpose during e-graph rewriting: give the egglog rules
// a distinct sort to rewrite into, so a pair-fuse rule's RHS can never
// re-match its own LHS pattern. Cascade prevention by typing.
//
// In PR1 (the rule-design PR) these structs survive into kernel_to_host as
// independent nodes — `compile()` emits the same kernel its un-fused sibling
// would, so each FusedX in a region launches its own kernel. The post-
// extraction collapse pass in PR2 will rewrite each FusionEnd-rooted region
// into a single `FusedRegion` op, after which the FusedX::compile bodies
// become unreachable and these helpers can be deleted.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, OP_KIND},
        extract_dtype, extract_expr_list,
    },
    op::*,
    prelude::*,
};

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
};

pub type Ops = (
    FusedSin,
    FusedSqrt,
    FusedExp,
    FusedExp2,
    FusedLog2,
    FusedRecip,
    FusedAdd,
    FusedMul,
);

// =========================================================================
// Codegen helpers — same kernel templates as KernelSin / KernelAdd, but
// parameterised by the per-op body so every FusedX::compile shrinks to a
// single helper call.
// =========================================================================

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compile_unary_kernel(
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    kernel_name: &str,
    body_expr: &str,
    shape: &[Expression],
    in_strides: &[Expression],
    out_strides: &[Expression],
    dtype: DType,
) -> (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
) {
    let vars = shape
        .iter()
        .flat_map(|e| e.dyn_vars())
        .chain(in_strides.iter().flat_map(|e| e.dyn_vars()))
        .chain(out_strides.iter().flat_map(|e| e.dyn_vars()))
        .collect::<FxHashSet<_>>();
    let cuda_ty = cuda_dtype(dtype);
    let includes = dtype_includes(&[dtype]);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
    let dyn_dims_param = if vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };
    let n_elements = shape.iter().copied().product::<Expression>().to_kernel();
    let out_idx = flatten_strides(shape, out_strides).to_kernel();
    let in_idx = flatten_strides(shape, in_strides).to_kernel();
    let body = body_expr.replace("{in_idx}", &in_idx);
    let kernel = format!(
        "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void {kernel_name}({cuda_ty} *out, const {cuda_ty} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = {body};
    }}
}}"
    );
    let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
        (module.clone(), func.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function(kernel_name).unwrap();
        compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
        (module, func)
    };
    let out_size = shape.iter().copied().product::<Expression>();
    (
        func,
        module,
        kernel,
        (out_size.ceil_div(256), 1.into(), 1.into()),
        (out_size.min(256), 1.into(), 1.into()),
        0.into(),
        FxHashMap::default(),
    )
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compile_binary_kernel(
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    kernel_name: &str,
    op_str: &str,
    out_shape: &[Expression],
    a_stride: &[Expression],
    b_stride: &[Expression],
    out_stride: &[Expression],
    dtype: DType,
) -> (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
) {
    let vars = out_shape
        .iter()
        .flat_map(|e| e.dyn_vars())
        .chain(a_stride.iter().flat_map(|e| e.dyn_vars()))
        .chain(b_stride.iter().flat_map(|e| e.dyn_vars()))
        .chain(out_stride.iter().flat_map(|e| e.dyn_vars()))
        .collect::<FxHashSet<_>>();
    let cuda_ty = cuda_dtype(dtype);
    let includes = dtype_includes(&[dtype, dtype]);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
    let dyn_dims_param = if vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };
    let n_elements = out_shape
        .iter()
        .copied()
        .product::<Expression>()
        .to_kernel();
    let out_idx = flatten_strides(out_shape, out_stride).to_kernel();
    let a_idx = flatten_strides(out_shape, a_stride).to_kernel();
    let b_idx = flatten_strides(out_shape, b_stride).to_kernel();
    let kernel = format!(
        "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void {kernel_name}({cuda_ty} *C, const {cuda_ty} *A, const {cuda_ty} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] {op_str} B[{b_idx}];
    }}
}}"
    );
    let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
        (module.clone(), func.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function(kernel_name).unwrap();
        compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
        (module, func)
    };
    let out_size = out_shape.iter().copied().product::<Expression>();
    (
        func,
        module,
        kernel,
        (out_size.ceil_div(256), 1.into(), 1.into()),
        (out_size.min(256), 1.into(), 1.into()),
        0.into(),
        FxHashMap::default(),
    )
}

// =========================================================================
// FusedSin
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedSin {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedSin {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedSin",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedSin {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_sin_k",
            "sinf(in[{in_idx}])",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "FusedSin"
    }
}

// =========================================================================
// FusedSqrt
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedSqrt {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedSqrt {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedSqrt",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedSqrt {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_sqrt_k",
            "sqrtf(in[{in_idx}])",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusedSqrt"
    }
}

// =========================================================================
// FusedExp
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedExp {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedExp {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedExp",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedExp {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_exp_k",
            "expf(in[{in_idx}])",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusedExp"
    }
}

// =========================================================================
// FusedExp2
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedExp2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedExp2 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedExp2",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedExp2 {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_exp2_k",
            "exp2f(in[{in_idx}])",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusedExp2"
    }
}

// =========================================================================
// FusedLog2
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedLog2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedLog2 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedLog2",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedLog2 {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_log2_k",
            "log2f(in[{in_idx}])",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusedLog2"
    }
}

// =========================================================================
// FusedRecip
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedRecip {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedRecip {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedRecip",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedRecip {
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
        compile_unary_kernel(
            stream,
            compile_cache,
            "fused_recip_k",
            "1.0f / in[{in_idx}]",
            &self.shape,
            &self.in_strides,
            &self.out_strides,
            self.dtype,
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
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusedRecip"
    }
}

// =========================================================================
// FusedAdd
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedAdd {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedAdd {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedAdd",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedAdd {
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
        compile_binary_kernel(
            stream,
            compile_cache,
            "fused_add_k",
            "+",
            &self.out_shape,
            &self.a_stride,
            &self.b_stride,
            &self.out_stride,
            self.dtype,
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

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "FusedAdd"
    }
}

// =========================================================================
// FusedMul
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusedMul {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for FusedMul {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedMul",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        Vec::new()
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusedMul {
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
        compile_binary_kernel(
            stream,
            compile_cache,
            "fused_mul_k",
            "*",
            &self.out_shape,
            &self.a_stride,
            &self.b_stride,
            &self.out_stride,
            self.dtype,
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

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "FusedMul"
    }
}
