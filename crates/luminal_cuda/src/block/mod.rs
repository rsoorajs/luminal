mod ops;
pub use ops::*;

use cudarc::driver::CudaStream;
use luminal::{prelude::FxHashMap, shape::Expression, utils::EgglogOp};
use std::fmt::Debug;

use crate::runtime::CustomState;

#[allow(unused_variables)]
pub trait BlockOp: Debug + as_any::AsAny + EgglogOp {
    fn launch_range(&self) -> Vec<Expression> {
        unimplemented!()
    }
    fn output_size(&self) -> Expression {
        unimplemented!()
    }
    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        unimplemented!()
    }
    fn cuda_op(&self) -> (String, String) {
        ("".to_string(), "".to_string())
    } // C dtype, C function
    fn schedule_op(
        &self,
        custom_state: &mut FxHashMap<String, CustomState>,
        stream: &CudaStream,
        expressions: &FxHashMap<Expression, i32>,
    ) -> Vec<u8> {
        unimplemented!()
    } // C struct
    fn expressions(&self) -> Vec<Expression> {
        vec![]
    }
}

luminal::impl_into_ops!(BlockOp);
