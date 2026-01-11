mod ops;
pub use ops::*;

use cudarc::driver::CudaStream;
use luminal::{prelude::FxHashMap, shape::Expression};
use std::fmt::Debug;

#[allow(unused_variables)]
pub trait BlockOp: Debug + as_any::AsAny {
    fn op_name(&self) -> &'static str;
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
    #[allow(clippy::mutable_key_type)]
    fn schedule_op(
        &self,
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
