pub mod block;
pub mod kernel;
pub mod logical;
pub mod runtime;
pub use cudarc;

#[cfg(test)]
mod tests;

use luminal::op::DType;

fn cuda_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Bf16 => todo!(),
        DType::Int => "int",
    }
}
