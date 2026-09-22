//! Tests for [`FilteredToolSet`].

use std::sync::Arc;

use serde_json::json;

use super::FilteredToolSet;
use crate::context::{RunConfig, RunContext};
use crate::tool::ToolRegistry;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::EchoTool;

fn ctx() -> RunContext<()> {
    RunContext::new(RunConfig::new("run-filtered"), ())
}

fn registry_with(names: &[&str]) -> Arc<dyn ToolSet<(), ()>> {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    for name in names {
        registry.register(Arc::new(EchoTool::new(*name)));
    }
    Arc::new(registry)
}

#[tokio::test]
async fn keeps_only_tools_the_predicate_accepts() {
    let inner = registry_with(&["alpha", "beta"]);
    let filtered = FilteredToolSet::new(inner, Arc::new(|tool| tool.name() == "alpha"));

    let ctx = ctx();
    let names: Vec<_> = filtered
        .tools(&ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    assert_eq!(names, vec!["alpha".to_string()]);
}

#[tokio::test]
async fn filtered_out_tool_cannot_be_called() {
    let inner = registry_with(&["alpha", "beta"]);
    let filtered = FilteredToolSet::new(inner, Arc::new(|tool| tool.name() == "alpha"));

    let ctx = ctx();
    let err = filtered
        .call("beta", json!({"text": "hi"}), &ctx)
        .await
        .expect_err("beta was filtered out");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::ToolNotFound(name) if name == "beta"
    ));
}

#[tokio::test]
async fn allowed_tool_still_calls_through() {
    let inner = registry_with(&["alpha", "beta"]);
    let filtered = FilteredToolSet::allowing(inner, ["alpha"]);

    let ctx = ctx();
    let result = filtered
        .call("alpha", json!({"text": "hi"}), &ctx)
        .await
        .expect("alpha is allowed");
    assert!(!result.is_error);
}
