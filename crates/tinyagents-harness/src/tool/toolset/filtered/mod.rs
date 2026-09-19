//! [`FilteredToolSet`]: keep only the tools a predicate accepts.

mod types;
#[cfg(test)]
mod test;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::{FilteredToolSet, ToolFilterPredicate};

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::ToolSet;

impl<State: Send + Sync, Ctx: Send + Sync> FilteredToolSet<State, Ctx> {
    /// Wraps `inner`, keeping only tools for which `predicate` returns
    /// `true`.
    pub fn new(inner: Arc<dyn ToolSet<State, Ctx>>, predicate: ToolFilterPredicate) -> Self {
        Self { inner, predicate }
    }

    /// Convenience constructor keeping only the named tools.
    pub fn allowing(
        inner: Arc<dyn ToolSet<State, Ctx>>,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let allowed: std::collections::HashSet<String> =
            names.into_iter().map(Into::into).collect();
        Self::new(inner, Arc::new(move |tool: &dyn Tool| tool_name_allowed(&allowed, tool.name())))
    }
}

/// Shared allowlist membership test: `true` when `name` is in `allowed`.
///
/// Pulled out of [`FilteredToolSet::allowing`]'s predicate so
/// [`crate::middleware::library::ToolAllowlistMiddleware`] (whose own
/// `HashSet<String>`-based check historically duplicated this exact test —
/// `docs/runtime-comparison/pydantic-ai.md` §4 calls this duplication out
/// directly) can share the one implementation instead of a second copy that
/// could drift from it.
pub(crate) fn tool_name_allowed(allowed: &std::collections::HashSet<String>, name: &str) -> bool {
    allowed.contains(name)
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for FilteredToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .inner
            .tools(ctx)
            .await?
            .into_iter()
            .filter(|tool| (self.predicate)(tool.as_ref()))
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        let exposed = self.tools(ctx).await?;
        if !exposed.iter().any(|tool| tool.name() == name) {
            return Err(TinyAgentsError::ToolNotFound(name.to_string()));
        }
        self.inner.call(name, args, ctx).await
    }

    fn instructions(&self) -> Option<String> {
        self.inner.instructions()
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        self.inner.for_run(ctx).await
    }
}
