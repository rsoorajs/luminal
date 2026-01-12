use cudarc::cublas::{
    sys::{cublasOperation_t, cublasSgemm_v2, cublasStatus_t},
    CudaBlas,
};
use cudarc::driver::{CudaStream, DevicePtr};
use luminal::{
    graph::extract_expr,
    prelude::*,
    utils::{
        LLIROp,
        OpParam::{self, *},
    },
};
use std::sync::Arc;
use tracing::{trace, span, Level};

use crate::{cudarc::driver::CudaSlice, host::HostOp};

#[derive(Debug, Clone, Default)]
pub struct HostMatmul {
    m: Expression,
    n: Expression,
    k: Expression,
}

impl EgglogOp for HostMatmul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "HostMatmul".to_string(),
            //    A      B      m     n      k
            vec![Input, Input, Expr, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![include_str!["rewrite.egg"].to_string()]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::serialized_egraph::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let m_extract = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n_extract = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k_extract = extract_expr(egraph, children[4], expr_cache).unwrap();

        trace!("{:?}", m_extract);
        trace!("{:?}", n_extract);
        trace!("{:?}", k_extract);
        trace!("{:?}", children[0]);
        trace!("{:?}", children[1]);

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

impl HostOp for HostMatmul {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[&CudaSlice<u8>],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let blas = CudaBlas::new(stream.clone())?;

        trace!("\n");
        trace!("{:?}", self.m);
        trace!("{:?}", self.n);
        trace!("{:?}", self.k);

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

        // Debug: Check buffer sizes
        trace!("Buffer sizes (in bytes):");
        trace!(
            "  A: {} (expected: {} floats = {} bytes)",
            inputs[1].len(),
            m * k,
            m * k * 4
        );
        trace!(
            "  B: {} (expected: {} floats = {} bytes)",
            inputs[2].len(),
            k * n,
            k * n * 4
        );
        trace!(
            "  C: {} (expected: {} floats = {} bytes)",
            inputs[0].len(),
            m * n,
            m * n * 4
        );

        // Execute GEMM using raw cuBLAS API
        // cuBLAS expects column-major, but our matrices are row-major.
        // Use C^T = B^T * A^T (swap operands and dimensions)
        trace!("\nCalling cuBLAS with (row-major conversion):");
        trace!("  cublasSgemm_v2(handle, OP_N, OP_N, m={}, n={}, k={}, alpha={}, B, lda={}, A, ldb={}, beta={}, C, ldc={})",
                  n, m, k, alpha, n, k, beta, n);
        stream.synchronize()?;
        let sgemm_span = span!(Level::INFO, "SGEMM_V2 Call");
        let _entered = sgemm_span.enter();
        unsafe {
            let status = cublasSgemm_v2(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N, // No transpose
                cublasOperation_t::CUBLAS_OP_N, // No transpose
                n,                              // m: rows of B^T
                m,                              // n: cols of A^T
                k,                              // k: common dimension
                &alpha as *const f32,
                b_ptr as *const f32, // B (interpreted as B^T)
                n,                   // lda: leading dimension = n
                a_ptr as *const f32, // A (interpreted as A^T)
                k,                   // ldb: leading dimension = k
                &beta as *const f32,
                c_ptr as *mut f32, // C (interpreted as C^T)
                n,                 // ldc: leading dimension = n
            );

            trace!("  cuBLAS status: {:?}", status);

            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!(
                    "cuBLAS SGEMM failed with status: {:?}",
                    status
                ));
            }
        }
        trace!("\nSynchronizing stream...");
        stream.synchronize()?;
        drop(_entered);
        drop(sgemm_span);

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }
}
