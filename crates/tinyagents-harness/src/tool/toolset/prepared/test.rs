//! Tests for [`PreparedToolSet`], including a per-step transform that
//! produces a different schema depending on the run context.

use std::sync::Arc;

use serde_json::json;

use super::PreparedToolSet;
use crate::context::{RunConfig, RunContext};
use crate::tool::ToolRegistry;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::EchoTool;

/// `Ctx` here is the "step" the caller is on, so the transform can read it
/// and prove the effective schema differs between two different
/// ctx/step invocations — as the task requires.
fn ctx_for_step(step: u32) -> RunContext<u32> {
    RunContext::new(RunConfig::new("run-prepared"), step)
}

fn registry() -> Arc<dyn ToolSet<(), u32>> {
    let mut registry: ToolRegistry<(), u32> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("search")));
    Arc::new(registry)
}

#[tokio::test]
async fn per_step_transform_hides_the_tool_on_an_early_step() {
    let prepared = PreparedToolSet::new(
        registry(),
        Arc::new(
            |ctx: &RunContext<u32>, schemas| {
                if ctx.data < 2 { Vec::new() } else { schemas }
            },
        ),
    );

    let early = prepared.tools(&ctx_for_step(0)).await.expect("tools");
    assert!(early.is_empty());

    let later = prepared.tools(&ctx_for_step(2)).await.expect("tools");
    assert_eq!(later.len(), 1);
    assert_eq!(later[0].name(), "search");
}

#[tokio::test]
async fn per_step_transform_rewrites_the_description() {
    let prepared = PreparedToolSet::new(
        registry(),
        Arc::new(|ctx: &RunContext<u32>, mut schemas| {
            for schema in &mut schemas {
                schema.description = format!("step {} description", ctx.data);
            }
            schemas
        }),
    );

    let step_one = prepared.tools(&ctx_for_step(1)).await.expect("tools");
    let step_two = prepared.tools(&ctx_for_step(2)).await.expect("tools");
    assert_eq!(step_one[0].description(), "step 1 description");
    assert_eq!(step_two[0].description(), "step 2 description");
    assert_ne!(step_one[0].description(), step_two[0].description());
}

#[tokio::test]
async fn hidden_tool_cannot_be_called() {
    let prepared = PreparedToolSet::new(registry(), Arc::new(|_ctx, _schemas| Vec::new()));
    let ctx = ctx_for_step(0);
    let err = prepared
        .call("search", json!({"text": "hi"}), &ctx)
        .await
        .expect_err("the transform hid every tool");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::ToolNotFound(_)
    ));
}

#[tokio::test]
async fn visible_tool_still_calls_through() {
    let prepared = PreparedToolSet::new(registry(), Arc::new(|_ctx, schemas| schemas));
    let ctx = ctx_for_step(0);
    let result = prepared
        .call("search", json!({"text": "hi"}), &ctx)
        .await
        .expect("the transform kept `search`");
    assert!(!result.is_error);
}
