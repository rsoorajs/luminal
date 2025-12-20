use anyhow::Result;
use egglog::EGraph;
use std::string::String;

use crate::prelude::petgraph::dot::{Config, Dot};

pub trait ToHtml {
    fn to_html(&self) -> Result<String>;
}

pub trait ToDot {
    fn to_dot(&self) -> Result<String>;
}

impl ToHtml for EGraph {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json = serde_json::to_string_pretty(
            &self.serialize(egglog::SerializeConfig::default()).egraph,
        )?;
        Ok(include_str!("egraph_viz_template.html").replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

impl ToDot for EGraph {
    fn to_dot(&self) -> Result<String> {
        let serialized: egglog::SerializeOutput =
            self.serialize(egglog::SerializeConfig::default());
        Ok(serialized.egraph.to_dot())
    }
}

/// Implements `ToDot` for [`LLIRGraph`] and [`HLIRGraph`]
impl<N, E> ToDot for petgraph::stable_graph::StableGraph<N, E>
where
    N: std::fmt::Debug,
    E: std::fmt::Debug,
{
    fn to_dot(&self) -> Result<String> {
        Ok(format!(
            "{:?}",
            Dot::with_config(self, &[Config::EdgeNoLabel])
        ))
    }
}
