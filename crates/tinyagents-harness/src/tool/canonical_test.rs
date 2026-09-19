//! Tests for [`ToolRegistry`]'s canonical (non-recursive) dispatch path:
//! registration, lookup, name listing, schema/spec/policy projection, and
//! execution through [`CanonicalDispatch`](super::CanonicalDispatch).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::ToolRegistry;

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
