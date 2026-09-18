//! Harness-side registration and execution support for canonical tools.
//!
//! `tinytools` owns the public tool vocabulary. This module owns only the host
//! concerns: name lookup, provider-schema projection, timeout settings, error
//! routing, and the explicit recursive-dispatch handoff.

mod error_policy;
mod schema;
mod schema_prepare;
pub mod select;
mod timeout;
mod types;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

pub use error_policy::{ToolErrorPolicy, is_control_flow_error};
pub use schema::*;
pub use schema_prepare::*;
pub use select::*;
pub use timeout::*;
pub use types::ToolExecutionContext;

/// A host-owned dispatch hook for the rare canonical tool that must execute
/// against the *typed* parent run (currently recursive sub-agents).
///
/// Normal registrations use [`ToolRegistry::register`] and dispatch through
/// `tinytools::Tool::execute_with_context`. A recursive registration must be
/// explicit: no downcast, global registry, or hidden argument is involved.
#[async_trait]
pub trait ToolDispatch<State: Send + Sync, Ctx: Send + Sync>: Send + Sync {
    /// Canonical declaration exposed to the model and policy layer.
    fn tool(&self) -> Arc<dyn tinytools::Tool>;

    /// Supplies authoritative values for `ToolInjectedArgumentSource::Host`.
    ///
    /// This is deliberately an explicit registration-time dispatch concern;
    /// model arguments never carry host authority. The default is right for
    /// tools that only declare call-id injection (or no injected values).
    fn injected_arguments(
        &self,
        _call: &tinytools::ToolCall,
    ) -> anyhow::Result<tinytools::InjectedToolArguments> {
        Ok(tinytools::InjectedToolArguments::new())
    }

    /// Executes with the full typed parent run when the dispatch needs it.
    async fn execute(
        &self,
        state: &State,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &crate::context::RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult>;
}

struct CanonicalDispatch {
    tool: Arc<dyn tinytools::Tool>,
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for CanonicalDispatch {
    fn tool(&self) -> Arc<dyn tinytools::Tool> {
        self.tool.clone()
    }

    async fn execute(
        &self,
        _state: &State,
        arguments: Value,
        options: tinytools::ToolCallOptions,
        parent: &crate::context::RunContext<Ctx>,
    ) -> anyhow::Result<tinytools::ToolResult> {
        let context = ToolExecutionContext::from_run_context(parent);
        self.tool
            .execute_with_context(arguments, options, Some(&context))
            .await
    }
}

/// A name-keyed canonical tool registry.
pub struct ToolRegistry<State: Send + Sync, Ctx: Send + Sync> {
    tools: HashMap<String, Arc<dyn ToolDispatch<State, Ctx>>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> ToolRegistry<State, Ctx> {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a canonical tool under its declared name.
    pub fn register(&mut self, tool: Arc<dyn tinytools::Tool>) -> &mut Self {
        let name = tool.name().to_owned();
        self.tools
            .insert(name, Arc::new(CanonicalDispatch { tool }));
        self
    }

    /// Registers an explicit typed-parent dispatcher for a canonical tool.
    pub fn register_dispatch(&mut self, dispatch: Arc<dyn ToolDispatch<State, Ctx>>) -> &mut Self {
        let name = dispatch.tool().name().to_owned();
        self.tools.insert(name, dispatch);
        self
    }

    /// Looks up the complete host dispatch entry.
    pub(crate) fn dispatch(&self, name: &str) -> Option<Arc<dyn ToolDispatch<State, Ctx>>> {
        self.tools.get(name).cloned()
    }

    /// Looks up a canonical tool declaration.
    pub fn get(&self, name: &str) -> Option<Arc<dyn tinytools::Tool>> {
        self.dispatch(name).map(|dispatch| dispatch.tool())
    }

    /// Returns registered names in sorted order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Returns provider request schemas projected from canonical declarations.
    #[must_use]
    pub fn schemas(&self) -> Vec<tinyinference_llm::tool::ToolSchema> {
        let mut schemas: Vec<_> = self
            .tools
            .values()
            .map(|dispatch| provider_schema(dispatch.tool().as_ref()))
            .collect();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    /// Returns canonical specs including host-injected fields for introspection.
    #[must_use]
    pub fn declared_specs(&self) -> Vec<tinytools::ToolSpec> {
        let mut specs: Vec<_> = self
            .tools
            .values()
            .map(|dispatch| dispatch.tool().spec())
            .collect();
        specs.sort_by(|left, right| left.name.cmp(&right.name));
        specs
    }

    /// Returns declared policies keyed by tool name.
    #[must_use]
    pub fn policies(&self) -> HashMap<String, tinytools::ToolPolicy> {
        self.tools
            .iter()
            .map(|(name, dispatch)| (name.clone(), dispatch.tool().policy()))
            .collect()
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Default for ToolRegistry<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a canonical spec into the inference provider's request schema.
/// Host-injected values are removed before a model sees the schema.
#[must_use]
pub(crate) fn provider_schema(tool: &dyn tinytools::Tool) -> tinyinference_llm::tool::ToolSchema {
    let spec = tool.spec();
    tinyinference_llm::tool::ToolSchema {
        name: spec.name,
        description: spec.description,
        parameters: tinytools::project_injected_arguments(
            &spec.parameters,
            &tool.injected_arguments(),
        ),
        format: tinyinference_llm::tool::ToolFormat::Json,
    }
}

#[cfg(test)]
mod canonical_test;
#[cfg(test)]
mod timeout_test;
