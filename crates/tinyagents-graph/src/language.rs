//! Materialization of declarative language blueprints into executable graphs.
//!
//! This is the bridge from `tinyagents-language`'s parsed [`Blueprint`] (a
//! `.rag` program) to this crate's [`GraphBuilder`]/[`CompiledGraph`]: a host
//! supplies a [`NodeFactory`] that turns each [`NodeSpec`] into a
//! [`BoxedNode`] handler, and [`build_graph`] wires those handlers into a
//! whole-state (`GraphBuilder::overwrite`) graph and compiles it. This module
//! knows nothing about what a node handler actually does — that is entirely
//! the factory's responsibility — only how to assemble the compiled topology
//! around it.

use std::sync::Arc;

use tinyagents_harness::error::Result;
use tinyagents_language::{Blueprint, NodeSpec, Routing};

use crate::{CompiledGraph, GraphBuilder, NodeHandler};

/// A durable node handler materialized from a declarative node specification.
pub type BoxedNode<State> = Arc<NodeHandler<State, State>>;

/// Builds runtime node handlers from declarative node specifications.
pub trait NodeFactory<State> {
    /// Materializes one executable handler.
    ///
    /// # Errors
    ///
    /// Returns an error when the node kind is unsupported or a required
    /// capability binding is unavailable.
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<State>>;
}

/// Wires a blueprint into a durable whole-state graph.
///
/// # Errors
///
/// Propagates factory errors and graph topology validation failures.
pub fn build_graph<State, F>(
    blueprint: &Blueprint,
    factory: &F,
) -> Result<CompiledGraph<State, State>>
where
    State: Clone + Send + Sync + 'static,
    F: NodeFactory<State>,
{
    let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());

    for spec in &blueprint.nodes {
        let handler = factory.make(spec)?;
        builder = builder.add_node(spec.name.as_str(), move |state, ctx| {
            (handler.clone())(state, ctx)
        });
        builder = match &spec.routing {
            Routing::Next(target) => builder.add_edge(spec.name.as_str(), target.as_str()),
            // Conditional routing is not lowered into `add_conditional_edges`
            // here: the node is marked command-routing instead, so the
            // materialized handler itself must resolve `spec.routing`'s
            // labeled targets and return them via `Command::goto` at runtime.
            Routing::Conditional(_) => builder.mark_command_routing(spec.name.as_str()),
            Routing::Terminal => builder.set_finish(spec.name.as_str()),
        };
    }

    builder.compile()
}
