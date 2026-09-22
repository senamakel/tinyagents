//! Tests for [`ExternalToolSet`].

use serde_json::json;
use tinytools::ToolSpec;

use super::ExternalToolSet;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::ctx;

fn spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        description: "Executed by the host.".to_string(),
        parameters: json!({"type": "object"}),
    }
}

#[tokio::test]
async fn advertises_its_schemas() {
    let external = ExternalToolSet::new(vec![spec("host_only")]);
    let ctx: crate::context::RunContext<()> = ctx();
    let tools = <ExternalToolSet as ToolSet<(), ()>>::tools(&external, &ctx)
        .await
        .expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "host_only");
}

#[tokio::test]
async fn call_is_always_deferred_to_the_host() {
    let external = ExternalToolSet::new(vec![spec("host_only")]);
    let ctx: crate::context::RunContext<()> = ctx();
    let err =
        <ExternalToolSet as ToolSet<(), ()>>::call(&external, "host_only", json!({"a": 1}), &ctx)
            .await
            .expect_err("execution is never local");
    match err {
        crate::error::TinyAgentsError::CallDeferred { metadata } => {
            assert_eq!(metadata, serde_json::Value::Null);
        }
        other => panic!("expected CallDeferred, got {other:?}"),
    }
}

#[tokio::test]
async fn unknown_name_is_tool_not_found_not_deferred() {
    let external = ExternalToolSet::new(vec![spec("host_only")]);
    let ctx: crate::context::RunContext<()> = ctx();
    let err = <ExternalToolSet as ToolSet<(), ()>>::call(&external, "missing", json!({}), &ctx)
        .await
        .expect_err("missing was never advertised");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::ToolNotFound(_)
    ));
}

#[tokio::test]
async fn advertised_tools_carry_the_external_tool_marker() {
    // Same admission-time recognition as `ToolRegistry::register_external`
    // (`crate::tool::is_external_tool`), so a tool bridged into a harness's
    // registry via `ToolSetDispatchBridge` is deferred proactively rather
    // than reaching `ToolSet::call`.
    let external = ExternalToolSet::new(vec![spec("host_only")]);
    let ctx: crate::context::RunContext<()> = ctx();
    let tools = <ExternalToolSet as ToolSet<(), ()>>::tools(&external, &ctx)
        .await
        .expect("tools");
    assert!(crate::tool::is_external_tool(tools[0].as_ref()));
}

#[tokio::test]
async fn direct_execute_also_fails_safely() {
    let external = ExternalToolSet::new(vec![spec("host_only")]);
    let ctx: crate::context::RunContext<()> = ctx();
    let tools = <ExternalToolSet as ToolSet<(), ()>>::tools(&external, &ctx)
        .await
        .expect("tools");
    let result = tools[0].execute(json!({})).await;
    assert!(result.is_err());
}
