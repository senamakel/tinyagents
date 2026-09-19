//! The named capability registry: a higher-level catalog of capabilities
//! addressable by name — the data structure that lets a workflow reference
//! sub-capabilities it didn't hardcode.
//!
//! This layer is deliberately distinct from the harness'
//! [`tinyagents_harness::model_registry::ModelRegistry`] and
//! [`tinyagents_harness::tool::ToolRegistry`], which are per-run executable stores.
//! The [`CapabilityRegistry`] is a *capability catalog*: it owns named models,
//! tools, agents, graphs, routers, and reducers so host code can resolve
//! capabilities by name.

use std::collections::HashMap;
use std::sync::Arc;

use crate::component::{ComponentKind, ComponentMetadata};
use tinyagents_definition::AgentDefinition;
use tinyinference_llm::model::ChatModel;
use tinytools::Tool;

/// A name-addressable catalog of registered capabilities.
///
/// The registry is generic over the application `State` because models and
/// tools are generic over it. The default `State = ()` matches the common case
/// of stateless capabilities.
///
/// Storage is partitioned by [`ComponentKind`]:
///
/// - **Models and tools** keep executable values.
/// - **Graphs, routers, reducers**, and the reserved kinds are name-only
///   descriptors.
///
/// The [`metadata`](CapabilityRegistry::metadata) map is the source of truth for
/// *presence*: every successful registration records a
/// [`ComponentMetadata`] entry keyed by `(kind, name)`, so
/// [`has`](CapabilityRegistry::has) and [`names`](CapabilityRegistry::names)
/// work uniformly across kinds.
pub struct CapabilityRegistry<State = ()>
where
    State: Send + Sync,
{
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
    /// Declarative agent definitions keyed by their stable id. Execution is
    /// host-owned through graph's explicit `AgentInvoker` boundary.
    pub(crate) agents: HashMap<String, AgentDefinition>,
    /// Presence + discovery metadata, keyed by `(kind, canonical name)`.
    pub(crate) meta: HashMap<(ComponentKind, String), ComponentMetadata>,
    /// Alias map, keyed by `(kind, alias)` -> canonical name.
    pub(crate) aliases: HashMap<(ComponentKind, String), String>,
}
