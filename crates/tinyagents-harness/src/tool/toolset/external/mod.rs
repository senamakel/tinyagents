//! [`ExternalToolSet`]: schema-only tools the host executes.

mod types;
#[cfg(test)]
mod test;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult, ToolSpec};

pub use types::ExternalToolSet;

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::ToolSet;

impl ExternalToolSet {
    /// Builds an external toolset that advertises `schemas` but executes
    /// none of them locally. See the type doc comment for the host
    /// integration contract.
    pub fn new(schemas: Vec<ToolSpec>) -> Self {
        Self { schemas }
    }
}

/// A [`Tool`] declaration for one [`ExternalToolSet`] entry.
///
/// Its [`Tool::execute`] always fails: this tool has no local implementation
/// by design. A caller should reach [`ToolSet::call`] instead (which returns
/// [`TinyAgentsError::CallDeferred`]) rather than invoking this directly, but
/// `execute` must still answer safely for a caller that reaches it through
/// a generic `dyn Tool` path.
struct ExternalTool {
    spec: ToolSpec,
}

#[async_trait]
impl Tool for ExternalTool {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn description(&self) -> &str {
        &self.spec.description
    }

    fn parameters_schema(&self) -> Value {
        self.spec.parameters.clone()
    }

    fn exposure(&self) -> tinytools::ToolExposure {
        tinytools::ToolExposure::Direct
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Err(anyhow::anyhow!(
            "tool `{}` is deferred to the host and has no local implementation; \
             call it through `ToolSet::call`, which reports `TinyAgentsError::CallDeferred`",
            self.spec.name
        ))
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for ExternalToolSet {
    async fn tools(&self, _ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .schemas
            .iter()
            .cloned()
            .map(|spec| Arc::new(ExternalTool { spec }) as Arc<dyn Tool>)
            .collect())
    }

    async fn call(&self, name: &str, args: Value, _ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        if !self.schemas.iter().any(|spec| spec.name == name) {
            return Err(TinyAgentsError::ToolNotFound(name.to_string()));
        }
        Err(TinyAgentsError::CallDeferred {
            name: name.to_string(),
            arguments: args,
        })
    }

    fn instructions(&self) -> Option<String> {
        None
    }
}
