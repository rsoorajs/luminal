use std::{
    fmt::Debug,
    vec::Vec,
    sync::Arc,
};

use luminal::prelude::*;
use crate::cudarc::driver::{CudaSlice, CudaStream}; 
mod host_matmul;


pub type Ops = (
    host_matmul::HostMatmul,
);

pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// With the convention, and this is a bad way to do it, that the first is the output buffer.
    fn execute(&self, stream: &Arc<CudaStream>, node_inputs: &mut Vec<CudaSlice<u8>>, dyn_map: &FxHashMap<char, usize>) -> anyhow::Result<()>;

    fn output_size(&self) -> Expression;
}
