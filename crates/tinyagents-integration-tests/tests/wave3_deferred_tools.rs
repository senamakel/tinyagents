//! Coverage for A2 — deferred tool calls as a typed, resumable output.
//!
//! `docs/runtime-comparison/plan.md` Phase 2 item A2 adds
//! `AgentRun::deferred` / `DeferredToolRequests`, `DeferredToolResults`,
//! `AgentHarness::resume_deferred`, `ToolRegistry::register_external`, and
//! the `DeferredToolHandler` trait. This file exercises the public surface
//! end to end across a simulated process restart: the first leg's
//! `run.messages` + `run.deferred` round-trip through JSON before the
//! second leg resumes from them.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::testkit::EventRecorder;
use tinyagents_harness::tool::{DeferredToolRequests, DeferredToolResults};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::{ToolCall, ToolFormat, ToolSchema};
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolPolicy, ToolResult};

struct DeleteTool {
    seen: Mutex<Vec<serde_json::Value>>,
}

#[async_trait]
impl Tool for DeleteTool {
    fn name(&self) -> &str {
        "delete"
    }
    fn description(&self) -> &str {
        "delete a path"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        ToolPolicy::classified().requiring_approval()
    }
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.seen.lock().unwrap().push(arguments);
        Ok(ToolResult::success("deleted"))
    }
}

fn assistant(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(text.to_string())]
    };
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage: Some(Usage::new(1, 1)),
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn harness(delete: Arc<DeleteTool>) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            assistant(
                vec![
                    ToolCall::new("call-delete", "delete", json!({"path": "/tmp/x"})),
                    ToolCall::new("call-click", "browser_click", json!({"x": 1})),
                ],
                "",
            ),
            assistant(Vec::new(), "done"),
        ])),
    );
    harness.register_tool(delete);
    harness.register_external_tool(ToolSchema {
        name: "browser_click".into(),
        description: "client-side click".into(),
        parameters: json!({"type": "object"}),
        format: ToolFormat::Json,
    });
    harness
}

#[tokio::test]
async fn deferred_run_survives_a_json_round_trip_and_resumes_in_a_fresh_harness() {
    let delete = Arc::new(DeleteTool {
        seen: Mutex::new(Vec::new()),
    });
    let recorder = EventRecorder::new();

    // Leg 1: both calls defer (one approval, one external); the run pauses.
    let first = harness(delete.clone())
        .invoke_in_context(
            &(),
            RunContext::new(RunConfig::new("leg-1"), ()).with_events(recorder.sink()),
            vec![Message::user("clean up")],
        )
        .await
        .expect("a deferral is not an error");
    let pending = first.deferred.clone().expect("deferred");
    assert_eq!(pending.approvals.len(), 1);
    assert_eq!(pending.calls.len(), 1);
    assert_eq!(
        recorder
            .events()
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolDeferred { .. }))
            .count(),
        2
    );

    // "Process restart": only the transcript and the requests survive.
    let messages_json = serde_json::to_string(&first.messages).unwrap();
    let pending_json = serde_json::to_string(&pending).unwrap();
    let messages: Vec<Message> = serde_json::from_str(&messages_json).unwrap();
    let pending: DeferredToolRequests = serde_json::from_str(&pending_json).unwrap();

    // Leg 2: a brand-new harness (same registrations) resumes.
    let results = DeferredToolResults::new()
        .approve_with_args("call-delete", json!({"path": "/tmp/y"}))
        .respond("call-click", ToolResult::success("clicked"));
    assert!(pending.remaining(&results).is_empty());
    let results_json = serde_json::to_string(&results).unwrap();
    let results: DeferredToolResults = serde_json::from_str(&results_json).unwrap();

    let delete2 = Arc::new(DeleteTool {
        seen: Mutex::new(Vec::new()),
    });
    let mut second_harness = harness(delete2.clone());
    // The resumed leg only needs the final answer from the model.
    second_harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![assistant(Vec::new(), "done")])),
    );
    let run = second_harness
        .resume_deferred(
            &(),
            RunContext::new(RunConfig::new("leg-2"), ()),
            messages,
            results,
        )
        .await
        .expect("resume completes");

    assert!(delete.seen.lock().unwrap().is_empty());
    assert_eq!(delete2.seen.lock().unwrap().clone(), vec![json!({"path": "/tmp/y"})]);
    assert!(run.deferred.is_none());
    assert_eq!(run.text().as_deref(), Some("done"));
    let tool_rows: Vec<(String, String)> = run
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::Tool(t) => Some((t.tool_call_id.clone(), m.text())),
            _ => None,
        })
        .collect();
    assert_eq!(
        tool_rows,
        vec![
            ("call-delete".to_string(), "deleted".to_string()),
            ("call-click".to_string(), "clicked".to_string()),
        ]
    );
}
