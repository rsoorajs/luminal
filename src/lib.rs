pub mod graph;
pub mod graph_tensor;
pub mod hl_ops;
pub mod op;
pub mod shape;
pub mod utils;
pub mod visualization;
pub mod egglog_utils;
pub mod serialized_egraph;

#[cfg(test)]
pub mod tests;

pub mod prelude {
    pub use crate::graph::*;
    pub use crate::graph_tensor::*;
    pub use crate::hl_ops::binary::F32Pow;
    pub use crate::hl_ops::*;
    pub use crate::op::*;
    pub use crate::shape::*;
    pub use crate::utils::*;
    pub use egraph_serialize::NodeId as ENodeId;
    pub use half::{bf16, f16};
    pub use petgraph;
    pub use petgraph::stable_graph::NodeIndex;
    pub use rustc_hash::{FxHashMap, FxHashSet};
    pub use tinyvec;
}

pub use paste::paste;
