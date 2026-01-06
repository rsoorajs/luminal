use std::fmt::Debug; 
use luminal::prelude::*; 
mod host_matmul;

pub type Ops = (host_matmul::HostMatmul,);


pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    fn execute(&self) -> anyhow::Result<()>;
}
