//! Tests for deferred tool calls (A2): the loop exiting with
//! `AgentRun::deferred`, resuming with `DeferredToolResults`, external tools,
//! the inline `DeferredToolHandler` path, and `HumanApprovalMiddleware`'s
//! `ApprovalOutcome::Defer`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::runtime::AgentHarness;
use crate::testkit::EventRecorder;
use crate::tool::{DeferredToolRequests, DeferredToolResults};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolPolicy, ToolResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A tool that records the arguments it ran with and returns a fixed reply.
/// `policy` lets a test declare `approval_required`.
struct RecordingTool {
    name: &'static str,
    reply: &'static str,
    policy: ToolPolicy,
    seen: Mutex<Vec<serde_json::Value>>,
}

impl RecordingTool {
    fn plain(name: &'static str, reply: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            reply,
            policy: ToolPolicy::read_only(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn approval_gated(name: &'static str, reply: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            reply,
            policy: ToolPolicy::classified().requiring_approval(),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<serde_json::Value> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "recording tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    fn policy(&self) -> ToolPolicy {
        self.policy.clone()
    }
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.seen.lock().unwrap().push(arguments);
        Ok(ToolResult::success(self.reply))
    }
}

fn response(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
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

/// The two-call batch every deferral test starts from: `delete` needs
/// approval, `lookup` does not.
fn mixed_batch() -> ModelResponse {
    response(
        vec![
            ToolCall::new("call-delete", "delete", json!({"path": "/tmp/x"})),
            ToolCall::new("call-lookup", "lookup", json!({"q": "x"})),
        ],
        "",
    )
}

fn tool_result_text(messages: &[Message], call_id: &str) -> Option<String> {
    messages.iter().find_map(|message| match message {
        Message::Tool(tool) if tool.tool_call_id == call_id => Some(message.text()),
        _ => None,
    })
}

// ── Deferral ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn approval_required_call_defers_the_run_after_its_siblings_execute() {
    let recorder = EventRecorder::new();
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            mixed_batch(),
            response(Vec::new(), "never reached"),
        ])),
    );
    let delete = RecordingTool::approval_gated("delete", "deleted");
    let lookup = RecordingTool::plain("lookup", "found");
    harness.register_tool(delete.clone());
    harness.register_tool(lookup.clone());

    let ctx = RunContext::new(RunConfig::new("defer"), ()).with_events(recorder.sink());
    let result = harness
        .invoke_in_context_with_status(&(), ctx, vec![Message::user("go")])
        .await
        .expect("a deferral is not an error");
    let run = result.run;

    // The non-deferred sibling ran and its result is on the transcript; the
    // assistant's tool-call row is intact and the deferred call is unanswered.
    assert_eq!(lookup.calls().len(), 1);
    assert!(delete.calls().is_empty(), "an approval-gated tool must not run");
    assert!(matches!(&run.messages[1], Message::Assistant(a) if a.tool_calls.len() == 2));
    assert_eq!(
        tool_result_text(&run.messages, "call-lookup").as_deref(),
        Some("found")
    );
    assert!(tool_result_text(&run.messages, "call-delete").is_none());
    assert_eq!(run.model_calls, 1, "the loop must not call the model again");
    assert!(run.final_response.is_none());

    let deferred = run.deferred.expect("run reports the pending approval");
    assert_eq!(deferred.approvals.len(), 1);
    assert_eq!(deferred.approvals[0].id, "call-delete");
    assert!(deferred.calls.is_empty());
    assert_eq!(deferred.remaining(&DeferredToolResults::default()).len(), 1);
    assert_eq!(
        result.status.status,
        crate::ids::ExecutionStatus::Interrupted,
        "a deferred run is interrupted, not completed"
    );
    assert!(recorder.events().iter().any(|event| matches!(
        event,
        AgentEvent::ToolDeferred { call_id, reason }
            if call_id == &CallId::new("call-delete") && reason == "approval_required"
    )));
}
