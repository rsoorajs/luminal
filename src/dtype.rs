use std::fmt::Display;

/// Supported dtypes
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DType {
    /// 32-bit float (8e23m)
    #[default]
    F32,
    /// 64-bit float (11e52m)
    F64, 
    
    /// 16-bit float (5e10m)
    Fp16,
    /// 16-bit float (8e7m)
    Bf16,
    /// 32-bit signed integer
    Int,
    /// 4-bit signed integer 
    I4,
    /// 4-bit unsigned integer
    U4,  
    /// 8-bit signed integer 
    I8, 
    /// 8-bit unsigned integer
    U8, 
    /// 16-bit signed integer 
    I16,
    /// 16-bit unsigned integer
    U16, 

    /// Boolean (stored as u8, 0 or 1)
    Bool,
    /// NVIDIA FP4 (E2M1) with block-scaled quantization.
    /// Each element is 4 bits. Every 16 elements share an FP8 (E4M3) scale factor.
    /// Storage: n/2 bytes (packed FP4) + n/16 bytes (block scales) = 9n/16 bytes per n elements.
    NvFp4,
    /// OCP MXFP4 (E2M1) with E8M0 block-scaled quantization.
    /// Each element is 4 bits. Every 32 elements share an E8M0 (8-bit exponent) scale factor.
    /// Storage: n/2 bytes (packed FP4) + n/32 bytes (block scales) = 17n/32 bytes per n elements.
    Mxfp4,
    
    /// 8-bit unsigned float (e8m0)
    F8UE8M0, 
    /// 8-bit float (e4m3)
    F8E4M3, 
    /// 8-bit float (e5m2)
    F8E5M2, 

    F6E2M3, //  
    F6E3M2, // 

    F4E2M1, // Base FP4 
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl DType {
    /// Returns bytes per element for fixed-size dtypes.
    /// Panics for block-scaled types like NvFp4 — use `size_of_n` instead.
    pub fn sizeof(&self) -> usize {
        match self {
            DType::F32 | DType::Int => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::Bool => 1,
            DType::NvFp4 => panic!("NvFp4 has no fixed per-element size; use size_of_n(n) instead"),
            DType::Mxfp4 => panic!("Mxfp4 has no fixed per-element size; use size_of_n(n) instead"),
        }
    }

    /// Returns the total number of bytes needed to store `n` elements of this dtype.
    /// For NvFp4, `n` must be divisible by 16 (the block size).
    pub fn size_of_n(&self, n: usize) -> usize {
        match self {
            DType::F32 | DType::Int => n * 4,
            DType::Bf16 | DType::F16 => n * 2,
            DType::Bool => n,
            DType::NvFp4 => {
                assert!(
                    n % 16 == 0,
                    "NvFp4 requires element count divisible by 16 (block size), got {n}"
                );
                // n/2 bytes packed FP4 data (2 elements per byte)
                // + n/16 bytes FP8 block scales (1 scale per 16 elements)
                n / 2 + n / 16
            }
            DType::Mxfp4 => {
                assert!(
                    n % 32 == 0,
                    "Mxfp4 requires element count divisible by 32 (block size), got {n}"
                );
                // n/2 bytes packed FP4 data (2 elements per byte)
                // + n/32 bytes E8M0 block scales (1 scale per 32 elements)
                n / 2 + n / 32
            }
        }
    }
}