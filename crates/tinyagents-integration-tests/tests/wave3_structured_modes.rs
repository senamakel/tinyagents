//! Coverage for A6 — prompted / union structured-output modes and
//! `EndStrategy`.
//!
//! `docs/runtime-comparison/plan.md` Phase 2 item A6 adds
//! `StructuredStrategy::Prompted`/`ToolCallUnion` (driven by
//! `RunPolicy::structured_strategy_override`) and `RunPolicy::end_strategy`,
//! which replaces the old always-continue behavior for a turn that mixes a
//! structured-output call with real tool calls. This file exercises each
//! `EndStrategy` variant and both new structured modes end to end through
//! `AgentHarness`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::runtime::{AgentHarness, EndStrategy, RunPolicy, StructuredStrategyOverride};
use tinyagents_harness::testkit::FakeTool;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelProfile, ModelRequest, ModelResponse, ResponseFormat};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;

/// A model that records every request it receives, for the Prompted-mode
/// test's assertion that the schema landed in a system message.
struct RecordingModel {
    script: Mutex<Vec<ModelResponse>>,
    seen: Mutex<Vec<ModelRequest>>,
}

impl RecordingModel {
    fn new(script: Vec<ModelResponse>) -> Self {
        Self {
            script: Mutex::new(script),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.seen.lock().expect("poisoned").clone()
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    fn profile(&self) -> Option<&ModelProfile> {
        None
    }

    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.seen.lock().expect("poisoned").push(request);
        let mut script = self.script.lock().expect("poisoned");
        if script.len() > 1 {
            Ok(script.remove(0))
        } else {
            Ok(script[0].clone())
        }
    }
}

fn schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"]
    })
}

fn mixed_turn_then_final(second_answer: &str) -> Vec<ModelResponse> {
    vec![
        ModelResponse {
            message: tinyinference_llm::message::AssistantMessage {
                id: Some("m1".to_string()),
                content: Vec::new(),
                tool_calls: vec![
                    ToolCall::new("c1", "search", json!({})),
                    ToolCall::new("c2", "result", json!({"answer": "first"})),
                ],
                usage: None,
            },
            usage: None,
            finish_reason: None,
            raw: None,
            resolved_model: None,
            continue_turn: None,
            served_from_cache: false,
            correlation: None,
            resolved_route: None,
        },
        ModelResponse::assistant(format!(r#"{{"answer":"{second_answer}"}}"#)),
    ]
}

// ── EndStrategy::Graceful (default) ─────────────────────────────────────────

#[tokio::test]
async fn graceful_runs_the_tool_then_finishes_with_the_first_answer() {
    let model = Arc::new(MockModel::with_responses(mixed_turn_then_final("second")));
    let search = Arc::new(FakeTool::returning("search", "hits"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .register_tool(search.clone())
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::auto("result", schema())),
            end_strategy: EndStrategy::Graceful,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(search.calls().len(), 1, "the accompanying tool call still runs");
    assert_eq!(run.structured, Some(json!({"answer": "first"})));
    assert_eq!(
        model.call_count(),
        1,
        "Graceful finishes after the mixed turn; the model is never asked again"
    );
}

// ── EndStrategy::Early ───────────────────────────────────────────────────────

#[tokio::test]
async fn early_finishes_immediately_and_never_runs_the_accompanying_tool() {
    let model = Arc::new(MockModel::with_responses(mixed_turn_then_final("second")));
    let search = Arc::new(FakeTool::returning("search", "hits"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .register_tool(search.clone())
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::auto("result", schema())),
            end_strategy: EndStrategy::Early,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        search.calls().len(),
        0,
        "Early must not run the accompanying tool call"
    );
    assert_eq!(run.structured, Some(json!({"answer": "first"})));
    assert_eq!(model.call_count(), 1);
}

// ── EndStrategy::Exhaustive ──────────────────────────────────────────────────

#[tokio::test]
async fn exhaustive_ignores_the_first_output_tool_and_waits_for_a_clean_turn() {
    let model = Arc::new(MockModel::with_responses(mixed_turn_then_final("second")));
    let search = Arc::new(FakeTool::returning("search", "hits"));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .register_tool(search.clone())
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::auto("result", schema())),
            end_strategy: EndStrategy::Exhaustive,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        search.calls().len(),
        1,
        "the tool call from the mixed turn still runs"
    );
    // The mixed turn's output tool is ignored outright; the *second* turn's
    // clean answer (no accompanying tool calls) is the one that is recorded.
    assert_eq!(run.structured, Some(json!({"answer": "second"})));
    assert_eq!(
        model.call_count(),
        2,
        "Exhaustive keeps going past the mixed turn"
    );
}

// ── Prompted mode ────────────────────────────────────────────────────────────

#[tokio::test]
async fn prompted_mode_injects_the_schema_into_the_system_prompt_and_extracts_from_text() {
    let model = Arc::new(RecordingModel::new(vec![ModelResponse::assistant(
        r#"{"answer":"prompted"}"#,
    )]));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone()).with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto("result", schema())),
        structured_strategy_override: Some(StructuredStrategyOverride::Prompted { template: None }),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.structured, Some(json!({"answer": "prompted"})));
    let sent = model
        .requests()
        .first()
        .expect("one request was sent")
        .messages
        .clone();
    let has_schema_instructions = sent.iter().any(|message| {
        matches!(message, Message::System(_)) && message.text().contains("JSON Schema")
    });
    assert!(
        has_schema_instructions,
        "Prompted mode must inject the schema into a system message, got: {sent:?}"
    );
}

// ── ToolCallUnion mode ───────────────────────────────────────────────────────

#[tokio::test]
async fn tool_call_union_records_which_variant_matched() {
    let model = Arc::new(MockModel::with_tool_call(
        "failure",
        json!({"reason": "not found"}),
    ));

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone()).with_policy(RunPolicy {
        default_response_format: Some(ResponseFormat::auto(
            "result",
            json!({ "type": "object" }),
        )),
        structured_strategy_override: Some(StructuredStrategyOverride::ToolCallUnion {
            variants: vec![
                (
                    "success".to_string(),
                    json!({
                        "type": "object",
                        "properties": { "value": { "type": "string" } },
                        "required": ["value"]
                    }),
                ),
                (
                    "failure".to_string(),
                    json!({
                        "type": "object",
                        "properties": { "reason": { "type": "string" } },
                        "required": ["reason"]
                    }),
                ),
            ],
        }),
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.structured, Some(json!({"reason": "not found"})));
    assert_eq!(run.structured_variant.as_deref(), Some("failure"));
}
