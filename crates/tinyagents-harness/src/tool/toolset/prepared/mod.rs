//! [`PreparedToolSet`]: a per-step schema transform over an inner toolset.

mod types;
#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinytools::{Tool, ToolResult};

pub use types::{PreparedToolSet, SchemaTransform};

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::tool::toolset::{OverrideTool, ToolSet};

impl<State: Send + Sync, Ctx: Send + Sync> PreparedToolSet<State, Ctx> {
    /// Wraps `inner`, re-applying `transform` to its declared schemas every
    /// time [`ToolSet::tools`] is called.
    pub fn new(inner: Arc<dyn ToolSet<State, Ctx>>, transform: SchemaTransform<Ctx>) -> Self {
        Self { inner, transform }
    }

    /// Computes the effective (post-transform) tool list for `ctx`, paired
    /// with the original inner tool each still-present entry came from.
    async fn effective(
        &self,
        ctx: &RunContext<Ctx>,
    ) -> Result<Vec<(tinyinference_llm::tool::ToolSchema, Arc<dyn Tool>)>> {
        let inner_tools = self.inner.tools(ctx).await?;
        let by_name: HashMap<&str, &Arc<dyn Tool>> = inner_tools
            .iter()
            .map(|tool| (tool.name(), tool))
            .collect();
        let declared_schemas: Vec<_> = inner_tools
            .iter()
            .map(|tool| crate::tool::provider_schema(tool.as_ref()))
            .collect();
        let transformed = (self.transform)(ctx, declared_schemas);
        Ok(transformed
            .into_iter()
            .filter_map(|schema| {
                by_name
                    .get(schema.name.as_str())
                    .map(|tool| (schema, Arc::clone(*tool)))
            })
            .collect())
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for PreparedToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(self
            .effective(ctx)
            .await?
            .into_iter()
            .map(|(schema, tool)| {
                if schema.description == tool.description()
                    && schema.parameters == tool.parameters_schema()
                {
                    tool
                } else {
                    Arc::new(
                        OverrideTool::new(tool)
                            .with_name(schema.name)
                            .with_description(schema.description)
                            .with_parameters(schema.parameters),
                    ) as Arc<dyn Tool>
                }
            })
            .collect())
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        let effective = self.effective(ctx).await?;
        if !effective.iter().any(|(schema, _)| schema.name == name) {
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
