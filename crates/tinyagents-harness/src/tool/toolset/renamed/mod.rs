//! [`RenamedToolSet`]: rename tools per an explicit map.

mod types;
#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::RenamedToolSet;

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::{OverrideTool, ToolSet};

impl<State: Send + Sync, Ctx: Send + Sync> RenamedToolSet<State, Ctx> {
    /// Wraps `inner`, renaming tools per `renames` (declared name -> advertised
    /// name). A tool whose name is not a key keeps its original name.
    pub fn new(inner: Arc<dyn ToolSet<State, Ctx>>, renames: HashMap<String, String>) -> Self {
        Self { inner, renames }
    }

    fn advertised_name(&self, declared: &str) -> String {
        self.renames
            .get(declared)
            .cloned()
            .unwrap_or_else(|| declared.to_string())
    }

    /// Resolves an advertised name back to the declared name the inner
    /// toolset owns, if `advertised` matches a rename target. Falls back to
    /// treating `advertised` as already-declared when it is not a rename
    /// target — so an un-renamed tool still resolves.
    fn declared_name(&self, advertised: &str) -> String {
        self.renames
            .iter()
            .find(|(_, renamed)| renamed.as_str() == advertised)
            .map(|(declared, _)| declared.clone())
            .unwrap_or_else(|| advertised.to_string())
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for RenamedToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .inner
            .tools(ctx)
            .await?
            .into_iter()
            .map(|tool| {
                let advertised = self.advertised_name(tool.name());
                if advertised == tool.name() {
                    tool
                } else {
                    Arc::new(OverrideTool::new(tool).with_name(advertised)) as Arc<dyn Tool>
                }
            })
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        let declared = self.declared_name(name);
        // Confirm `name` is one this wrapper currently advertises, so an
        // un-renamed original name (or a stale rename target) does not
        // silently reach the inner toolset.
        let exposed = self.tools(ctx).await?;
        if !exposed.iter().any(|tool| tool.name() == name) {
            return Err(TinyAgentsError::ToolNotFound(name.to_string()));
        }
        self.inner.call(&declared, args, ctx).await
    }

    fn instructions(&self) -> Option<String> {
        self.inner.instructions()
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        self.inner.for_run(ctx).await
    }
}
