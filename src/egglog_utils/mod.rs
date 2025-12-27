use crate::utils::EgglogOp;
use itertools::Itertools;
use std::{str, sync::Arc};

pub const BASE: &str = include_str!("base.egg");
pub const BASE_CLEANUP: &str = include_str!("base_cleanup.egg");
pub const RUN_SCHEDULE: &str = include_str!("run_schedule.egg");

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    format!(
        "
    (datatype IR
    (OutputJoin IR IR)
    {}
    )
    (function dtype (IR) DType :merge new)
    ",
        ops.iter()
            .map(|o| {
                let (name, body) = o.term();
                format!(
                    "({name} {})",
                    body.into_iter().map(|j| format!("{j:?}")).join(" ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn op_rewrites_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    ops.iter().flat_map(|o| o.rewrites()).join("\n")
}

fn op_cleanups_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    format!(
        "
    {}
    ",
        ops.iter()
            .filter(|op| op.cleanup())
            .map(|o| {
                let (name, body) = o.term();
                let body_terms = (0..body.len()).map(|i| (b'a' + i as u8) as char).join(" ");
                format!(
                    "(rule
                ((= ?m ({name} {body_terms})))
                ((delete ({name} {body_terms})))
                :ruleset cleanup
            )"
                )
            })
            .join("\n")
    )
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    let mut code = BASE.to_string();
    code.push_str(&op_defs_string(ops));
    code.push_str("\n");
    code.push_str(&op_rewrites_string(ops));
    code.push_str("\n");
    if cleanup {
        code.push_str(&op_cleanups_string(ops));
        code.push_str("\n");
    }
    code.push_str(BASE_CLEANUP);
    code.push_str("\n");
    code.push_str(program);
    code.push_str("\n");
    code.push_str(RUN_SCHEDULE);
    code
}
