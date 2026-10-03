use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::runtime::{AgentHarness, RunPolicy};
use tinyinference_llm::message::{ContentBlock, Message, ToolMessage};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinytools::{Tool, ToolResult};

#[test]
fn duration_suffix_renders_seconds_with_one_decimal() {
    assert_eq!(duration_suffix(12_345), "[took 12.3s]");
    assert_eq!(duration_suffix(0), "[took 0.0s]");
    assert_eq!(duration_suffix(49), "[took 0.0s]");
    assert_eq!(duration_suffix(950), "[took 0.9s]");
    assert_eq!(duration_suffix(900_300), "[took 900.3s]");
}

fn tool_message(content: Vec<ContentBlock>) -> ToolMessage {
    ToolMessage {
        tool_call_id: "call-1".to_string(),
        content,
        trusted_verbatim: false,
        artifact: None,
    }
}

#[test]
fn append_duration_adds_a_trailing_line_after_the_output() {
    let mut message = tool_message(vec![ContentBlock::Text("ok".to_string())]);
    append_duration(&mut message, 1_500);
    assert_eq!(
        message.content,
        vec![
            ContentBlock::Text("ok".to_string()),
            ContentBlock::Text("\n[took 1.5s]".to_string()),
        ]
    );
}

#[test]
fn append_duration_leaves_verbatim_results_untouched() {
    let mut message = tool_message(vec![ContentBlock::Text("exact".to_string())]);
    message.trusted_verbatim = true;
    append_duration(&mut message, 1_500);
    assert_eq!(
        message.content,
        vec![ContentBlock::Text("exact".to_string())]
    );
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "test tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("tool-output"))
    }
}

fn tool_call_then_text() -> MockModel {
    let mut call = ModelResponse::assistant("");
    call.message.content.clear();
    call.message.tool_calls = vec![ToolCall::new("call-1", "lookup", json!({}))];
    call.finish_reason = Some("tool_calls".to_string());
    MockModel::with_responses(vec![call, ModelResponse::assistant("done")])
}

async fn tool_row_text(policy: RunPolicy) -> String {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(tool_call_then_text()));
    harness.register_tool(Arc::new(EchoTool));
    harness.with_policy(policy);
    let run = harness
        .invoke_default(&(), vec![Message::user("look it up")])
        .await
        .expect("run succeeds");
    let row = run
        .messages
        .iter()
        .find(|message| matches!(message, Message::Tool(_)))
        .expect("the transcript has a tool row");
    row.text()
}

#[tokio::test]
async fn tool_rows_carry_no_duration_by_default() {
    assert_eq!(tool_row_text(RunPolicy::default()).await, "tool-output");
}

#[tokio::test]
async fn tool_rows_carry_their_duration_when_the_policy_asks() {
    let text = tool_row_text(RunPolicy {
        tool_result_durations: true,
        ..RunPolicy::default()
    })
    .await;
    assert!(
        text.starts_with("tool-output\n[took ") && text.ends_with("s]"),
        "unexpected tool row: {text:?}"
    );
}

/// A row that is one JSON document is read by machines as well as the model
/// (hosts parse workflow proposals and sub-agent payloads out of it), so a
/// trailing line would make it unparseable.
#[test]
fn append_duration_leaves_json_documents_parseable() {
    let mut text_json = tool_message(vec![ContentBlock::Text(
        "  {\"type\": \"workflow_proposal\"}\n".to_string(),
    )]);
    append_duration(&mut text_json, 1_500);
    assert_eq!(text_json.content.len(), 1);

    let mut block_json = tool_message(vec![ContentBlock::Json(json!({"ok": true}))]);
    append_duration(&mut block_json, 1_500);
    assert_eq!(block_json.content.len(), 1);

    let mut array = tool_message(vec![ContentBlock::Text("[1, 2]".to_string())]);
    append_duration(&mut array, 1_500);
    assert_eq!(array.content.len(), 1);

    // Text that merely starts like JSON is still prose.
    let mut prose = tool_message(vec![ContentBlock::Text("{not json} done".to_string())]);
    append_duration(&mut prose, 1_500);
    assert_eq!(prose.content.len(), 2);
}
