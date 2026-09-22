//! Runtime-owned model registration and selection types.
//!
//! See `mod.rs` for [`ModelRegistry::resolve`], the algorithm that consumes
//! [`ModelSelection`] and returns a [`ResolvedModelBinding`].

use std::collections::HashMap;
use std::sync::Arc;

use tinyinference_llm::model::{CapabilitySet, ChatModel, ModelHint, ResolvedModel};

/// Input policy for resolving one registered model.
///
/// Field order below matches the precedence [`ModelRegistry::resolve`]
/// applies: `requested` beats `previous` (when `reuse_previous`) beats
/// `hints` (priority-sorted) beats `agent_default` beats the registry-wide
/// default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSelection {
    /// Explicit per-request model override; wins over every other source
    /// when the named model is registered and eligible.
    pub requested: Option<String>,
    /// A durable prior selection (name, and how it was originally resolved).
    /// Only consulted when [`Self::reuse_previous`] is set.
    pub previous: Option<ResolvedModel>,
    /// Whether [`Self::previous`] may be reused for this resolution.
    pub reuse_previous: bool,
    /// Ordered runtime hints, consulted in descending
    /// [`ModelHint::priority`] order (original order breaks ties).
    pub hints: Vec<ModelHint>,
    /// The agent definition's own default model name, consulted after every
    /// hint has been tried and rejected.
    pub agent_default: Option<String>,
    /// Capabilities a resolved model must satisfy, checked against every
    /// candidate regardless of which source it came from.
    pub required_capabilities: Option<CapabilitySet>,
    /// Whether a retired model may still be selected (e.g. to finish an
    /// in-flight conversation already pinned to it) rather than being
    /// treated as ineligible.
    pub allow_retired: bool,
}

/// Name-keyed runtime registry of executable models.
pub struct ModelRegistry<State: Send + Sync> {
    pub(crate) models: HashMap<String, Arc<dyn ChatModel<State>>>,
    pub(crate) default: Option<String>,
}

impl<State: Send + Sync> std::fmt::Debug for ModelRegistry<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<&str> = self.models.keys().map(String::as_str).collect();
        names.sort_unstable();
        formatter
            .debug_struct("ModelRegistry")
            .field("models", &names)
            .field("default", &self.default)
            .finish()
    }
}

/// Executable binding plus durable selection metadata.
pub struct ResolvedModelBinding<State: Send + Sync> {
    /// Selected-model metadata.
    pub resolved: ResolvedModel,
    /// Executable model handle.
    pub model: Arc<dyn ChatModel<State>>,
}

impl<State: Send + Sync> std::fmt::Debug for ResolvedModelBinding<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedModelBinding")
            .field("resolved", &self.resolved)
            .finish_non_exhaustive()
    }
}
