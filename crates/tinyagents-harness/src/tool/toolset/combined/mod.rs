//! [`CombinedToolSet`]: merge multiple toolsets into one.

mod types;
#[cfg(test)]
mod test;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::CombinedToolSet;

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::ToolSet;

impl<State: Send + Sync, Ctx: Send + Sync> CombinedToolSet<State, Ctx> {
    /// Merges `members` in order; the first member (in this order) exposing
    /// a given name owns dispatch for it.
    pub fn new(members: Vec<Arc<dyn ToolSet<State, Ctx>>>) -> Self {
        Self { members }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for CombinedToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        let mut all = Vec::new();
        for member in &self.members {
            all.extend(member.tools(ctx).await?);
        }
        Ok(all)
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        for member in &self.members {
            let owns = member
                .tools(ctx)
                .await?
                .iter()
                .any(|tool| tool.name() == name);
            if owns {
                return member.call(name, args, ctx).await;
            }
        }
        Err(TinyAgentsError::ToolNotFound(name.to_string()))
    }

    fn instructions(&self) -> Option<String> {
        let combined: Vec<String> = self
            .members
            .iter()
            .filter_map(|member| member.instructions())
            .collect();
        if combined.is_empty() {
            None
        } else {
            Some(combined.join("\n\n"))
        }
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        for member in &self.members {
            member.for_run(ctx).await?;
        }
        Ok(())
    }
}
