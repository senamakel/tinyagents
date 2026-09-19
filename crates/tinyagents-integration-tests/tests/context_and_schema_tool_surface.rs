//! Integration coverage for canonical TinyTools declarations at the harness
//! registration boundary.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::tool::ToolRegistry;
use tinytools::{
    InjectedToolArguments, Tool, ToolCall, ToolCallId, ToolInjectedArgument, ToolResult,
    prepare_tool_arguments,
};

struct ContextualTool;

#[async_trait]
impl Tool for ContextualTool {
    fn name(&self) -> &str {
        "contextual"
    }

    fn description(&self) -> &str {
        "Acts within the caller's thread"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "note": {"type": "string"},
                "thread_id": {"type": "string"},
            },
            "required": ["note", "thread_id"],
        })
    }

    fn injected_arguments(&self) -> Vec<ToolInjectedArgument> {
        vec![ToolInjectedArgument::host("thread_id")]
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(
            arguments["note"].as_str().unwrap_or_default(),
        ))
    }
}

#[test]
fn tool_result_keeps_structured_content_without_a_harness_shim() {
    let result = ToolResult::json(json!({"rows": [{"id": 1}, {"id": 2}, {"id": 3}]}));

    assert_eq!(result.text(), "");
    assert!(result.output().contains("\"rows\""));
    assert!(!result.is_error);
}

#[test]
fn injected_arguments_are_hidden_from_the_model_and_host_owned_at_execution() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(ContextualTool));

    let wire = registry.schemas();
    assert!(wire[0].parameters["properties"].get("thread_id").is_none());
    assert_eq!(wire[0].parameters["required"], json!(["note"]));

    let call = ToolCall::new(
        ToolCallId::new("c1"),
        "contextual",
        json!({"note": "hi", "thread_id": "forged"}),
    );
    let mut host_values = InjectedToolArguments::new();
    host_values.insert("thread_id", json!("trusted-thread"));
    let prepared =
        prepare_tool_arguments(&call, &ContextualTool.injected_arguments(), &host_values)
            .expect("host values prepare canonical arguments");

    assert_eq!(
        prepared,
        json!({"note": "hi", "thread_id": "trusted-thread"})
    );
}
