use std::sync::{Arc, OnceLock};

use luminal::{
    graph::extract_expr,
    op::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    host::{HostOp, cublas::parse_cublas_op},
    cudarc::{
        driver::{CudaSlice, CudaStream, DevicePtr},
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasLtMatmulDesc_t,
                cublasLtMatrixLayout_t,
                cublasLtMatmulPreference_t,
                cublasLtMatmulHeuristicResult_t,
                cublasLtMatmulDescCreate,
                cublasLtMatmulDescSetAttribute,
                cublasLtMatrixLayoutCreate,
                cublasLtMatmulPreferenceCreate,
                cublasLtMatmulPreferenceSetAttribute,
                cublasLtMatmulPreferenceAttributes_t,
                cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmul,
                cublasLtMatmulPreferenceDestroy,
                cublasLtMatrixLayoutDestroy,
                cublasLtMatmulDescDestroy,
            }
        }
    }
};


#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
    
}

// Useless default for IntoEgglogOp
impl Default for CuBlasLt {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: cublasOperation_t::CUBLAS_OP_N, // IGNORE NOT REAL
            b_layout: cublasOperation_t::CUBLAS_OP_T, // IGNORE NOT REAL
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            cublaslt: OnceLock::new(),
        }
    }
}

impl EgglogOp for CuBlasLt {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublaslt".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout,
            vec![Input, Input, Expr, Expr, Expr, Str, Str, Expr, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["cublaslt_RmRm_rewrite.egg"].to_string(), // row row
            include_str!["cublaslt_RmCm_rewrite.egg"].to_string(), // row col
            include_str!["cublaslt_CmRm_rewrite.egg"].to_string(), // col row
            include_str!["cublaslt_CmCm_rewrite.egg"].to_string(), // col col
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k = extract_expr(egraph, children[4], expr_cache).unwrap();

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[children[5]].0;
        let b_layout_str = &egraph.enodes[children[6]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, children[7], expr_cache).unwrap();
        let ldb = extract_expr(egraph, children[8], expr_cache).unwrap();
        let ldc = extract_expr(egraph, children[9], expr_cache).unwrap();

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            cublaslt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, vec![children[0], children[1]])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for CuBlasLt {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[&CudaSlice<u8>],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as u64;
        let n = self.n.exec(dyn_map).unwrap() as u64;
        let k = self.k.exec(dyn_map).unwrap() as u64;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i64;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i64;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i64;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Get device pointers
        let (a_ptr, _a_guard) = inputs[1].device_ptr(stream);
        let (b_ptr, _b_guard) = inputs[2].device_ptr(stream);
        let (c_ptr, _c_guard) = inputs[0].device_ptr(stream);

        // Debug tracing
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            inputs[1].len(),
            m * k * 4,
            inputs[2].len(),
            k * n * 4,
            inputs[0].len(),
            m * n * 4
        );
        let _span = span!(
            Level::TRACE,
            "cuBLASLT",
            m, n, k, alpha, beta, lda, ldb, ldc, ?a_layout, ?b_layout,
        )
        .entered();

        let cublaslt = self
            .cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));

        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;

        // Allocate workspace (32 MiB)
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE)? };
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

        unsafe {
            // Create matmul descriptor
            cublasLtMatmulDescCreate(
                &mut matmul_desc,
                cudarc::cublaslt::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cudarc::cublaslt::sys::cudaDataType::CUDA_R_32F,
            ).result()?;

            // Set transpose attributes
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &a_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            ).result()?;
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &b_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            ).result()?;

            // Create matrix layout descriptors
            let (a_rows, a_cols) = if a_layout == cublasOperation_t::CUBLAS_OP_N {
                (m, k)
            } else {
                (k, m)
            };
            let (b_rows, b_cols) = if b_layout == cublasOperation_t::CUBLAS_OP_N {
                (k, n)
            } else {
                (n, k)
            };

            cublasLtMatrixLayoutCreate(
                &mut a_desc,
                cudarc::cublaslt::sys::cudaDataType::CUDA_R_32F,
                a_rows, a_cols, lda,
            ).result()?;
            cublasLtMatrixLayoutCreate(
                &mut b_desc,
                cudarc::cublaslt::sys::cudaDataType::CUDA_R_32F,
                b_rows, b_cols, ldb,
            ).result()?;
            cublasLtMatrixLayoutCreate(
                &mut c_desc,
                cudarc::cublaslt::sys::cudaDataType::CUDA_R_32F,
                m, n, ldc,
            ).result()?;

            // Create preference and set workspace size
            cublasLtMatmulPreferenceCreate(&mut preference).result()?;
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            ).result()?;

            // Get heuristic (best algorithm)
            cublasLtMatmulAlgoGetHeuristic(
                *cublaslt.handle(),
                matmul_desc,
                a_desc,
                b_desc,
                c_desc,
                c_desc, // D layout same as C
                preference,
                1, // Request 1 result
                &mut heuristic,
                &mut algo_count,
            ).result()?;

            if algo_count == 0 {
                // Cleanup before returning error
                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(c_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatmulDescDestroy(matmul_desc);
                return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
            }

            // Execute matmul
            cublasLtMatmul(
                *cublaslt.handle(),
                matmul_desc,
                &alpha as *const _ as *const std::ffi::c_void,
                a_ptr as *const std::ffi::c_void,
                a_desc,
                b_ptr as *const std::ffi::c_void,
                b_desc,
                &beta as *const _ as *const std::ffi::c_void,
                c_ptr as *const std::ffi::c_void,
                c_desc,
                c_ptr as *mut std::ffi::c_void,
                c_desc, // D layout same as C
                &heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                WORKSPACE_SIZE,
                stream.cu_stream() as *mut _,
            ).result()?;

            // Cleanup
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        }

        stream.synchronize()?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }
}