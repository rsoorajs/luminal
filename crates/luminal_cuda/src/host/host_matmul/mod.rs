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

        dbg!(m_extract);
        dbg!(n_extract);
        dbg!(k_extract);
        dbg!(children[0]);
        dbg!(children[1]);

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
        inputs: &mut Vec<CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        let blas = CudaBlas::new(stream.clone())?;

        eprintln!();
        dbg!(self.m);
        dbg!(self.n);
        dbg!(self.k);

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
        eprintln!("Buffer sizes (in bytes):");
        eprintln!(
            "  A: {} (expected: {} floats = {} bytes)",
            inputs[1].len(),
            m * k,
            m * k * 4
        );
        eprintln!(
            "  B: {} (expected: {} floats = {} bytes)",
            inputs[2].len(),
            k * n,
            k * n * 4
        );
        eprintln!(
            "  C: {} (expected: {} floats = {} bytes)",
            inputs[0].len(),
            m * n,
            m * n * 4
        );

        // Debug: Read data from GPU to verify buffers are valid
        eprintln!("\nReading data from GPU buffers:");

        // Copy from device to host and convert bytes to f32
        let a_bytes: Vec<u8> = stream.memcpy_dtov(&inputs[1])?;
        let b_bytes: Vec<u8> = stream.memcpy_dtov(&inputs[2])?;
        let c_bytes: Vec<u8> = stream.memcpy_dtov(&inputs[0])?;

        // Reinterpret bytes as f32
        let a_host: &[f32] =
            unsafe { std::slice::from_raw_parts(a_bytes.as_ptr() as *const f32, (m * k) as usize) };
        let b_host: &[f32] =
            unsafe { std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, (k * n) as usize) };
        let c_host: &[f32] =
            unsafe { std::slice::from_raw_parts(c_bytes.as_ptr() as *const f32, (m * n) as usize) };

        eprintln!("  A data: {:?}", a_host);
        eprintln!("  B data: {:?}", b_host);
        eprintln!("  C data (initial): {:?}", c_host);

        // Execute GEMM using raw cuBLAS API
        // cuBLAS expects column-major, but our matrices are row-major.
        // Use C^T = B^T * A^T (swap operands and dimensions)
        eprintln!("\nCalling cuBLAS with (row-major conversion):");
        eprintln!("  cublasSgemm_v2(handle, OP_N, OP_N, m={}, n={}, k={}, alpha={}, B, lda={}, A, ldb={}, beta={}, C, ldc={})",
                  n, m, k, alpha, n, k, beta, n);

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

            eprintln!("  cuBLAS status: {:?}", status);

            if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!(
                    "cuBLAS SGEMM failed with status: {:?}",
                    status
                ));
            }
        }

        eprintln!("\nSynchronizing stream...");
        stream.synchronize()?;

        // Read back result to verify
        let result_bytes: Vec<u8> = stream.memcpy_dtov(&inputs[0])?;
        let result: &[f32] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, (m * n) as usize)
        };
        eprintln!("Result C: {:?}", result);
        eprintln!("HostMatmul completed successfully!");

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }
}
