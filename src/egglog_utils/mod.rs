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

pub fn early_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> String {
    vec![
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.early_rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        format!(
            "(run-schedule
                (saturate expr)
                (run)
                (saturate base_cleanup)
            )
            (extract {root})"
        ),
    ]
    .join("\n")
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    vec![
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        RUN_SCHEDULE.to_string(),
    ]
    .join("\n")
}
