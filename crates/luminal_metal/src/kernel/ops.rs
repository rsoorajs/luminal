//! Metal kernel operation implementations
//!
//! This module contains the concrete implementations of Metal kernels
//! for each HLIR operation.

use super::MetalKernelOp;
use luminal::{
    prelude::{ENodeId, *},
    serialized_egraph::SerializedEGraph,
    utils::{
        flatten_mul_strides, EgglogOp, LLIROp,
        OpParam::{self, *},
    },
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

/// All Metal kernel operations
pub type MetalOps = (MetalExp2, MetalAdd, MetalMul);

// Utility function to compile a Metal shader
fn compile_shader(device: &Device, source: &str, function_name: &str) -> ComputePipelineState {
    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .expect("Failed to compile Metal shader");
    let function = library
        .get_function(function_name, None)
        .expect("Failed to get function from library");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create compute pipeline state")
}

// Helper macro for creating simple unary operations
macro_rules! metal_unary_op {
    ($name:ident, $op_name:expr, $metal_op:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn term(&self) -> (String, Vec<OpParam>) {
                ($op_name.to_string(), vec![EList, Input, EList, EList])
            }

            fn rewrites(&self) -> Vec<String> {
                vec![format!(
                    r#"(rule
                        ((= ?e ({} ?shape ?x ?x_stride ?out_stride))
                         (= ?dt (dtype ?x)))
                        ((let ?me ({} ?shape ?x ?x_stride ?out_stride))
                         (union ?e ?me)
                         (set (dtype ?me) ?dt))
                    )"#,
                    $op_name.replace("Metal", ""),
                    $op_name
                )]
            }

            fn cleanup(&self) -> bool {
                false
            }

            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                children: &[&'a ENodeId],
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::graph::extract_expr_list;
                (
                    LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        input_strides: extract_expr_list(
                            egraph,
                            children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    vec![children[1]],
                )
            }
        }

        impl MetalKernelOp for $name {
            fn compile(&self, device: &Device) -> ComputePipelineState {
                // Generate strided index expressions
                let inp_index = flatten_mul_strides(&self.shape, &self.input_strides);
                let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

                // Convert expressions to Metal code
                let inp_idx = inp_index.to_kernel().replace("const_z", "idx");
                let out_idx = out_index.to_kernel().replace("const_z", "idx");

                let source = format!(
                    r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mkernel(
                        device float *inp [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        device uint &n_elements [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]
                    ) {{
                        if (idx < n_elements) {{
                            out[{out_idx}] = {metal_op}(inp[{inp_idx}]);
                        }}
                    }}
                    "#,
                    metal_op = $metal_op,
                    inp_idx = inp_idx,
                    out_idx = out_idx
                );
                compile_shader(device, &source, "mkernel")
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn encode(
                &self,
                encoder: &ComputeCommandEncoderRef,
                pipeline: &ComputePipelineState,
                inputs: &[&Buffer],
                output: &Buffer,
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(inputs[0]), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &n_elements as *const u32 as *const _,
                );

                let thread_group_size = MTLSize::new(256, 1, 1);
                let thread_groups = MTLSize::new((n_elements as u64 + 255) / 256, 1, 1);
                encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            }
        }
    };
}

// Define unary operations
metal_unary_op!(MetalExp2, "MetalExp2", "exp2");

// Binary operations need a different structure
#[derive(Debug, Default, Clone)]
pub struct MetalAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalAdd ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalAdd {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        // Generate strided index expressions using 'z' = thread index
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] + b[{b_idx}];
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetalMul {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Mul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalMul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalMul {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        // Generate strided index expressions using 'z' = thread index
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] * b[{b_idx}];
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}
