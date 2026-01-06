use luminal::{
    prelude::*,
    utils::{
        LLIROp,
        OpParam::{self, *},
    },
};

use crate::host::HostOp;

#[derive(Debug, Clone)]
pub struct HostMatmul {
    state: i32,
}


impl EgglogOp for HostMatmul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "HostMatmul".to_string(),
            // God this is so illegible
            vec![Input, Input, Expr, Expr, Expr, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![include_str!["rewrite.egg"].to_string()]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::serialized_egraph::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn HostOp>(Box::new(Self { state: 42 }) as Box<dyn HostOp>),
            vec![],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for HostMatmul{
    fn execute(&self) -> anyhow::Result<()> {
        Ok(())
    }
}