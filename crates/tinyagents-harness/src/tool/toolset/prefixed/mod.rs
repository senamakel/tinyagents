//! [`PrefixedToolSet`]: prefix every advertised tool name.

mod types;
#[cfg(test)]
mod test;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::PrefixedToolSet;

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::{OverrideTool, ToolSet};

impl<State: Send + Sync, Ctx: Send + Sync> PrefixedToolSet<State, Ctx> {
    /// Wraps `inner`, prefixing every advertised name with `prefix`.
    pub fn new(inner: Arc<dyn ToolSet<State, Ctx>>, prefix: impl Into<String>) -> Self {
        Self {
            inner,
            prefix: prefix.into(),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for PrefixedToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .inner
            .tools(ctx)
            .await?
            .into_iter()
            .map(|tool| {
                let prefixed_name = format!("{}{}", self.prefix, tool.name());
                Arc::new(OverrideTool::new(tool).with_name(prefixed_name)) as Arc<dyn Tool>
            })
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        let stripped = name
            .strip_prefix(self.prefix.as_str())
            .ok_or_else(|| TinyAgentsError::ToolNotFound(name.to_string()))?;
        self.inner.call(stripped, args, ctx).await
    }

    fn instructions(&self) -> Option<String> {
        self.inner.instructions()
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        self.inner.for_run(ctx).await
    }
}
