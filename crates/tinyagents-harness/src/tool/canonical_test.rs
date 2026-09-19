use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::{RegisterOutcome, ToolRegistry};

struct Echo;

#[async_trait]
impl tinytools::Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Returns text."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"text": {"type": "string"}, "call_id": {"type": "string"}},
            "required": ["text", "call_id"]
        })
    }

    fn injected_arguments(&self) -> Vec<tinytools::ToolInjectedArgument> {
        vec![tinytools::ToolInjectedArgument::tool_call_id("call_id")]
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> {
        Ok(tinytools::ToolResult::success(
            arguments["text"].as_str().unwrap_or_default(),
        ))
    }
}

#[test]
fn registry_accepts_the_canonical_trait_and_hides_injected_values() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(Echo));

    let schema = registry.schemas().pop().expect("registered schema");
    assert!(schema.parameters["properties"].get("call_id").is_none());
    assert_eq!(schema.parameters["required"], json!(["text"]));
}

/// M-5 regression: a second `Echo` registered under the same name used to
/// silently overwrite the first with no signal at all. `register` keeps its
/// `&mut Self`-chaining, non-breaking signature, but `try_register` now
/// reports the collision so a caller that wants to detect it can.
#[test]
fn try_register_reports_a_duplicate_name() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    assert_eq!(
        registry.try_register(Arc::new(Echo)),
        RegisterOutcome::Registered
    );
    assert_eq!(
        registry.try_register(Arc::new(Echo)),
        RegisterOutcome::Replaced("echo".to_string())
    );
    // Still only one entry under the name; the second registration replaced
    // the first rather than being rejected outright.
    assert_eq!(registry.names(), vec!["echo".to_string()]);
}

#[test]
fn register_still_replaces_silently_for_the_non_breaking_api() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    // `register` keeps its existing `&mut Self` chaining contract even on a
    // duplicate name; the collision is only surfaced through `try_register`
    // or the `tracing::warn!` diagnostic.
    registry.register(Arc::new(Echo)).register(Arc::new(Echo));
    assert_eq!(registry.names(), vec!["echo".to_string()]);
}

#[test]
fn preparation_discards_a_forged_call_id_before_validation() {
    let call = tinytools::ToolCall::new(
        tinytools::ToolCallId::new("real-call"),
        "echo",
        json!({"text": "hi", "call_id": "forged"}),
    );
    let prepared = tinytools::prepare_tool_arguments(
        &call,
        &[tinytools::ToolInjectedArgument::tool_call_id("call_id")],
        &tinytools::InjectedToolArguments::new(),
    )
    .expect("canonical preparation succeeds");

    assert_eq!(prepared["call_id"], "real-call");
}
