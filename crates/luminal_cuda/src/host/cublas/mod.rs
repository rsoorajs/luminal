use cudarc::cublas::{
    sys::{cublasOperation_t, cublasSgemm_v2, cublasStatus_t},
    CudaBlas,
};
use cudarc::driver::{CudaStream, DevicePtr};
use luminal::{
    graph::extract_expr,
    op::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::*,
};
use std::sync::Arc;
use tracing::{span, Level};

use crate::{cudarc::driver::CudaSlice, host::HostOp};

pub type Ops = (
    CuBlasSgemmV2,
);


#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CuBlasSgemmV2 {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
}

impl EgglogOp for CuBlasSgemmV2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublasSgemm_V2".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout, 
            vec![Input, Input, Expr, Expr, Expr, Str, Str],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![include_str!["sgemm_v2_rewrites.egg"].to_string()]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let m_extract = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n_extract = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k_extract = extract_expr(egraph, children[4], expr_cache).unwrap();
        let a_layout_extract = cublasOperation_t::CUBLAS_OP_N; 
        let b_layout_extract = cublasOperation_t::CUBLAS_OP_N; 

        let _span = span!(
            Level::TRACE,
            "cuBLAS_SGEMM_V2_TN_extract",
            m = ?m_extract,
            n = ?n_extract,
            k = ?k_extract,
            input_a = ?children[0],
            input_b = ?children[1],
            a_layout = ?a_layout_extract,
            b_layout = ?b_layout_extract,
        )
        .entered();

        (
            LLIROp::new::<dyn HostOp>(Box::new(Self {
                m: m_extract,
                n: n_extract,
                k: k_extract,
            }) as Box<dyn HostOp>),
            vec![children[0], children[1]],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for CuBlasSgemmV2 {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[&CudaSlice<u8>],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let blas = CudaBlas::new(stream.clone())?;

        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as i32;
        let n = self.n.exec(dyn_map).unwrap() as i32;
        let k = self.k.exec(dyn_map).unwrap() as i32;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Get device pointers
        let (a_ptr, _a_guard) = inputs[1].device_ptr(stream);
        let (b_ptr, _b_guard) = inputs[2].device_ptr(stream);
        let (c_ptr, _c_guard) = inputs[0].device_ptr(stream);

        let z = cublasOperation_t::CUBLAS_OP_T; 
        // Debug: Check buffer sizes
        let _buffer_span = span!(
            Level::TRACE,
            "buffer_validation",
            a_size_bytes = inputs[1].len(),
            a_expected_bytes = m * k * 4,
            b_size_bytes = inputs[2].len(),
            b_expected_bytes = k * n * 4,
            c_size_bytes = inputs[0].len(),
            c_expected_bytes = m * n * 4
        )
        .entered();

        // Execute GEMM using raw cuBLAS API
        // A is row-major (m x k), B is column-major (k x n), C is row-major (m x n)
        // cuBLAS expects column-major storage. To compute C = A * B with:
        //   - A row-major -> treat as A^T column-major
        //   - B column-major -> use as-is
        //   - C row-major -> treat as C^T column-major
        // We compute: C^T = B^T * A^T
        // Since B is already column-major, B^T in column-major = B row-major (transpose)
        // Since A is row-major, A^T in column-major = A column-major (no transpose needed)
        // So we call: C^T = B^T * A (swap matrices, transpose B, don't transpose A)
        stream.synchronize()?;
        let _sgemm_span = span!(
            Level::INFO,
            "cuBLAS_SGEMM_V2_TN",
            m = m,
            n = n,
            k = k,
            alpha = alpha,
            beta = beta,
            lda = k,
            ldb = k,
            ldc = n,
            transpose_a = "T",
            transpose_b = "N"
        )
        .entered();

        let status = unsafe {
            cublasSgemm_v2(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_T, // Transpose B (column-major -> treat as B^T)
                cublasOperation_t::CUBLAS_OP_N, // No transpose for A (row-major -> use as-is)
                n,                              // m: rows of output C^T = cols of C
                m,                              // n: cols of output C^T = rows of C
                k,                              // k: common dimension
                &alpha as *const f32,
                b_ptr as *const f32, // B (column-major k x n, transposed to n x k)
                k,                   // lda: leading dimension of B in column-major = k
                a_ptr as *const f32, // A (row-major m x k, treated as column-major A^T)
                k,                   // ldb: leading dimension of A treated as column-major = k
                &beta as *const f32,
                c_ptr as *mut f32, // C (row-major m x n, written as column-major C^T n x m)
                n,                 // ldc: leading dimension of C^T in column-major = n
            )
        };

        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuBLAS SGEMM TN failed with status: {:?}",
                status
            ));
        }

        stream.synchronize()?;

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }
}
