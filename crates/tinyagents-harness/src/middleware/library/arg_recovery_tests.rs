//! Tests for [`ArgRecoveryMiddleware`].

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::Middleware;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

struct Stub {
    name: &'static str,
    required: bool,
}

#[async_trait::async_trait]
impl Tool for Stub {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "stub"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        if self.required {
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
        } else {
            json!({"type": "object", "properties": {}})
        }
    }

    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

fn mw() -> ArgRecoveryMiddleware {
    let set: Vec<Box<dyn Tool>> = vec![
        Box::new(Stub {
            name: "loose",
            required: false,
        }),
        Box::new(Stub {
            name: "strict",
            required: true,
        }),
    ];
    ArgRecoveryMiddleware::new(vec![Arc::new(set)])
}

async fn recover(name: &str, arguments: serde_json::Value) -> serde_json::Value {
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    let mut call = ToolCall::new("c1", name, arguments);
    mw().before_tool(&mut ctx, &(), &mut call).await.unwrap();
    call.arguments
}

#[tokio::test]
async fn an_object_is_left_alone() {
    assert_eq!(
        recover("strict", json!({"q": "x"})).await,
        json!({"q": "x"})
    );
}

#[tokio::test]
async fn a_json_encoded_string_is_decoded() {
    assert_eq!(
        recover("strict", json!("{\"q\": \"x\"}")).await,
        json!({"q": "x"})
    );
}

#[tokio::test]
async fn a_fenced_json_string_is_decoded() {
    assert_eq!(
        recover("strict", json!("```json\n{\"q\": \"x\"}\n```")).await,
        json!({"q": "x"})
    );
}

#[tokio::test]
async fn a_non_object_for_a_schema_without_required_fields_becomes_an_empty_object() {
    assert_eq!(recover("loose", json!(null)).await, json!({}));
}

#[tokio::test]
async fn a_non_object_for_a_schema_with_required_fields_is_left_for_the_crate_gate() {
    assert_eq!(recover("strict", json!(null)).await, json!(null));
}

#[tokio::test]
async fn an_unknown_tool_is_treated_as_permissive() {
    assert_eq!(recover("nope", json!(3)).await, json!({}));
}

#[tokio::test]
async fn provider_invalid_arguments_keep_their_raw_string_for_admission_repair() {
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    let raw = "{query:\"weather\"}";
    let mut call = ToolCall::invalid("c1", "loose", raw, "key must be a string");
    mw().before_tool(&mut ctx, &(), &mut call).await.unwrap();
    assert_eq!(call.arguments, json!(raw));
    assert!(call.invalid.is_some());
}

#[tokio::test]
async fn a_provider_invalid_fenced_object_is_decoded_and_cleared() {
    let mut ctx: RunContext = RunContext::new(RunConfig::new("mw-test"), ());
    let raw = "```json\n{\"q\": \"x\"}\n```";
    let mut call = ToolCall::invalid("c1", "strict", raw, "expected value");
    mw().before_tool(&mut ctx, &(), &mut call).await.unwrap();
    assert_eq!(call.arguments, json!({"q": "x"}));
    assert!(call.invalid.is_none());
}
