//! [`ApprovalRequiredToolSet`]: flag matching tools as requiring approval.

#[cfg(test)]
mod test;
mod types;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::{ApprovalPredicate, ApprovalRequiredToolSet};

use crate::context::RunContext;
use crate::error::Result;
use crate::tool::toolset::{OverrideTool, ToolSet};

impl<State: Send + Sync, Ctx: Send + Sync> ApprovalRequiredToolSet<State, Ctx> {
    /// Wraps `inner`, marking every tool for which `predicate` returns
    /// `true` as requiring approval.
    pub fn new(inner: Arc<dyn ToolSet<State, Ctx>>, predicate: ApprovalPredicate) -> Self {
        Self { inner, predicate }
    }

    /// Convenience constructor requiring approval for the named tools.
    pub fn for_names(
        inner: Arc<dyn ToolSet<State, Ctx>>,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let flagged: std::collections::HashSet<String> =
            names.into_iter().map(Into::into).collect();
        Self::new(
            inner,
            Arc::new(move |tool: &dyn Tool| flagged.contains(tool.name())),
        )
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx>
    for ApprovalRequiredToolSet<State, Ctx>
{
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .inner
            .tools(ctx)
            .await?
            .into_iter()
            .map(|tool| {
                if (self.predicate)(tool.as_ref()) {
                    Arc::new(OverrideTool::new(tool).with_policy_transform(Arc::new(
                        |policy: tinytools::ToolPolicy| policy.requiring_approval(),
                    ))) as Arc<dyn Tool>
                } else {
                    tool
                }
            })
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        // Approval is a declaration a host's gate reads before dispatch (see
        // the type doc comment); this adaptor never blocks the call itself,
        // so it delegates unconditionally.
        self.inner.call(name, args, ctx).await
    }

    fn instructions(&self) -> Option<String> {
        self.inner.instructions()
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        self.inner.for_run(ctx).await
    }
}
