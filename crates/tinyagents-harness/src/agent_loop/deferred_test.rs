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

// ── Resume ──────────────────────────────────────────────────────────────────

/// Runs the mixed batch to its deferral and returns the harness, the tools,
/// and the deferred run, ready to resume.
async fn deferred_run(
    recorder: &EventRecorder,
) -> (
    AgentHarness<()>,
    Arc<RecordingTool>,
    Arc<RecordingTool>,
    crate::middleware::AgentRun,
) {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            mixed_batch(),
            response(Vec::new(), "all done"),
        ])),
    );
    let delete = RecordingTool::approval_gated("delete", "deleted");
    let lookup = RecordingTool::plain("lookup", "found");
    harness.register_tool(delete.clone());
    harness.register_tool(lookup.clone());
    let ctx = RunContext::new(RunConfig::new("first"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("first leg defers");
    assert!(run.deferred.is_some());
    (harness, delete, lookup, run)
}

#[tokio::test]
async fn resume_with_approve_runs_the_tool_and_continues_to_the_model() {
    let recorder = EventRecorder::new();
    let (harness, delete, lookup, first) = deferred_run(&recorder).await;
    let pending: DeferredToolRequests = first.deferred.clone().unwrap();

    let results = DeferredToolResults::new().approve("call-delete");
    assert!(pending.remaining(&results).is_empty());
    let ctx = RunContext::new(RunConfig::new("second"), ()).with_events(recorder.sink());
    let run = harness
        .resume_deferred(&(), ctx, first.messages.clone(), results)
        .await
        .expect("resume completes the run");

    assert_eq!(delete.calls(), vec![json!({"path": "/tmp/x"})]);
    assert_eq!(lookup.calls().len(), 1, "the sibling is not re-run on resume");
    assert_eq!(
        tool_result_text(&run.messages, "call-delete").as_deref(),
        Some("deleted")
    );
    assert_eq!(run.text().as_deref(), Some("all done"));
    assert!(run.deferred.is_none());
    assert_eq!(run.model_calls, 1, "resume spends exactly one new model call");
    assert!(recorder.events().iter().any(|event| matches!(
        event,
        AgentEvent::ToolApproved { call_id } if call_id == &CallId::new("call-delete")
    )));
}

#[tokio::test]
async fn resume_with_approve_with_args_runs_the_tool_with_the_edited_arguments() {
    let recorder = EventRecorder::new();
    let (harness, delete, _lookup, first) = deferred_run(&recorder).await;

    let results = DeferredToolResults::new()
        .approve_with_args("call-delete", json!({"path": "/tmp/safer"}));
    let ctx = RunContext::new(RunConfig::new("second"), ()).with_events(recorder.sink());
    let run = harness
        .resume_deferred(&(), ctx, first.messages.clone(), results)
        .await
        .expect("resume completes the run");

    assert_eq!(delete.calls(), vec![json!({"path": "/tmp/safer"})]);
    assert_eq!(
        tool_result_text(&run.messages, "call-delete").as_deref(),
        Some("deleted")
    );
    assert_eq!(run.text().as_deref(), Some("all done"));
}

#[tokio::test]
async fn resume_with_deny_answers_the_call_with_the_message_and_never_runs_it() {
    let recorder = EventRecorder::new();
    let (harness, delete, _lookup, first) = deferred_run(&recorder).await;

    let results = DeferredToolResults::new().deny("call-delete", "operator refused the delete");
    let ctx = RunContext::new(RunConfig::new("second"), ()).with_events(recorder.sink());
    let run = harness
        .resume_deferred(&(), ctx, first.messages.clone(), results)
        .await
        .expect("a denial is not a failure");

    assert!(delete.calls().is_empty());
    let denial = run
        .messages
        .iter()
        .find_map(|message| match message {
            Message::Tool(tool) if tool.tool_call_id == "call-delete" => Some(tool.clone()),
            _ => None,
        })
        .expect("the denial is a tool-result row");
    assert_eq!(denial.content, vec![ContentBlock::Text("operator refused the delete".into())]);
    assert_eq!(denial.artifact.as_ref().unwrap()["is_error"], true);
    assert_eq!(run.text().as_deref(), Some("all done"));
    assert!(!run.executed_tools.iter().any(|name| name == "delete"));
    assert!(recorder.events().iter().any(|event| matches!(
        event,
        AgentEvent::ToolDenied { call_id, message }
            if call_id == &CallId::new("call-delete") && message == "operator refused the delete"
    )));
}

#[tokio::test]
async fn resume_refuses_an_incomplete_resolution_and_names_the_missing_ids() {
    let recorder = EventRecorder::new();
    let (harness, delete, _lookup, first) = deferred_run(&recorder).await;
    let pending = first.deferred.clone().unwrap();

    let results = DeferredToolResults::new();
    assert_eq!(pending.remaining(&results), vec![CallId::new("call-delete")]);
    let ctx = RunContext::new(RunConfig::new("second"), ());
    let error = harness
        .resume_deferred(&(), ctx, first.messages.clone(), results)
        .await
        .expect_err("nothing was resolved");
    assert!(
        matches!(&error, TinyAgentsError::Validation(message) if message.contains("call-delete")),
        "{error}"
    );
    assert!(delete.calls().is_empty());
}

// ── External tools ──────────────────────────────────────────────────────────

#[tokio::test]
async fn external_tool_call_is_deferred_and_its_host_result_is_injected_on_resume() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            response(
                vec![ToolCall::new("call-ext", "browser_click", json!({"x": 1, "y": 2}))],
                "",
            ),
            response(Vec::new(), "clicked"),
        ])),
    );
    harness.register_external_tool(tinyinference_llm::tool::ToolSchema {
        name: "browser_click".into(),
        description: "Click at a screen coordinate (runs in the client).".into(),
        parameters: json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}),
        format: tinyinference_llm::tool::ToolFormat::Json,
    });

    let first = harness
        .invoke_default(&(), vec![Message::user("click it")])
        .await
        .expect("first leg defers");
    let pending = first.deferred.clone().expect("external call is pending");
    assert!(pending.approvals.is_empty());
    assert_eq!(pending.calls.len(), 1);
    assert_eq!(pending.calls[0].name, "browser_click");
    assert_eq!(pending.calls[0].arguments, json!({"x": 1, "y": 2}));
    assert!(tool_result_text(&first.messages, "call-ext").is_none());

    let results = DeferredToolResults::new().respond("call-ext", ToolResult::success("ok: clicked (1,2)"));
    let ctx = RunContext::new(RunConfig::new("second"), ());
    let run = harness
        .resume_deferred(&(), ctx, first.messages.clone(), results)
        .await
        .expect("resume completes the run");
    assert_eq!(
        tool_result_text(&run.messages, "call-ext").as_deref(),
        Some("ok: clicked (1,2)")
    );
    assert_eq!(run.text().as_deref(), Some("clicked"));
    assert!(run.executed_tools.is_empty(), "the harness never ran the external tool");
}

// ── Inline handler ──────────────────────────────────────────────────────────

/// A handler that approves everything and records what it was asked.
struct ApproveAllHandler {
    asked: Mutex<Vec<DeferredToolRequests>>,
}

#[async_trait]
impl crate::tool::DeferredToolHandler for ApproveAllHandler {
    async fn handle(
        &self,
        requests: &DeferredToolRequests,
    ) -> crate::error::Result<DeferredToolResults> {
        self.asked.lock().unwrap().push(requests.clone());
        Ok(requests.approve_all())
    }
}

#[tokio::test]
async fn inline_handler_resolves_deferrals_without_surfacing_them() {
    let recorder = EventRecorder::new();
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            mixed_batch(),
            response(Vec::new(), "all done"),
        ])),
    );
    let delete = RecordingTool::approval_gated("delete", "deleted");
    harness.register_tool(delete.clone());
    harness.register_tool(RecordingTool::plain("lookup", "found"));
    let handler = Arc::new(ApproveAllHandler {
        asked: Mutex::new(Vec::new()),
    });
    harness.with_deferred_tool_handler(handler.clone());

    let ctx = RunContext::new(RunConfig::new("inline"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("the handler settles the batch");

    assert!(run.deferred.is_none(), "the caller never sees the deferral");
    assert_eq!(run.text().as_deref(), Some("all done"));
    assert_eq!(delete.calls(), vec![json!({"path": "/tmp/x"})]);
    assert_eq!(
        tool_result_text(&run.messages, "call-delete").as_deref(),
        Some("deleted")
    );
    let asked = handler.asked.lock().unwrap();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].approvals[0].id, "call-delete");
    let events = recorder.events();
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolDeferred { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolApproved { .. })));
}

/// A handler that leaves the request unresolved.
struct SilentHandler;

#[async_trait]
impl crate::tool::DeferredToolHandler for SilentHandler {
    async fn handle(
        &self,
        _requests: &DeferredToolRequests,
    ) -> crate::error::Result<DeferredToolResults> {
        Ok(DeferredToolResults::new())
    }
}

#[tokio::test]
async fn inline_handler_that_leaves_calls_unresolved_fails_the_run() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::with_responses(vec![mixed_batch()])));
    harness.register_tool(RecordingTool::approval_gated("delete", "deleted"));
    harness.register_tool(RecordingTool::plain("lookup", "found"));
    harness.with_deferred_tool_handler(Arc::new(SilentHandler));

    let error = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect_err("an incomplete resolution is a validation failure");
    assert!(
        matches!(&error, TinyAgentsError::Validation(message) if message.contains("call-delete")),
        "{error}"
    );
}

// ── Execution-time deferral (`Err(ApprovalRequired)` from the tool) ─────────

struct SelfDeferringTool;

#[async_trait]
impl Tool for SelfDeferringTool {
    fn name(&self) -> &str {
        "wire_money"
    }
    fn description(&self) -> &str {
        "asks for approval from inside execute"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Err(TinyAgentsError::ApprovalRequired {
            metadata: json!({"amount": arguments["amount"]}),
        }
        .into())
    }
}

#[tokio::test]
async fn tool_raising_approval_required_defers_with_its_metadata() {
    let recorder = EventRecorder::new();
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![response(
            vec![ToolCall::new("call-wire", "wire_money", json!({"amount": 500}))],
            "",
        )])),
    );
    harness.register_tool(Arc::new(SelfDeferringTool));

    let ctx = RunContext::new(RunConfig::new("exec-defer"), ()).with_events(recorder.sink());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("pay")])
        .await
        .expect("a deferral is not an error");
    let pending = run.deferred.expect("pending approval");
    assert_eq!(pending.approvals[0].id, "call-wire");
    assert_eq!(
        pending.metadata.get(&CallId::new("call-wire")),
        Some(&json!({"amount": 500}))
    );
    // The `ToolStarted` emitted before execution has exactly one terminal
    // partner, the `ToolDeferred`, and no `ToolFailed`.
    let events = recorder.events();
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolStarted { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolDeferred { .. })));
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::ToolFailed { .. })));
    assert_eq!(run.tool_calls, 0);
}

// ── HumanApprovalMiddleware ─────────────────────────────────────────────────

#[tokio::test]
async fn human_approval_middleware_defer_outcome_produces_the_deferred_exit() {
    use crate::middleware::library::{ApprovalOutcome, HumanApprovalMiddleware};

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            mixed_batch(),
            response(Vec::new(), "all done"),
        ])),
    );
    // Neither tool declares approval in its policy; the middleware decides.
    let delete = RecordingTool::plain("delete", "deleted");
    let lookup = RecordingTool::plain("lookup", "found");
    harness.register_tool(delete.clone());
    harness.register_tool(lookup.clone());
    harness.push_middleware(Arc::new(
        HumanApprovalMiddleware::new(["delete"]).with_approval_outcome(Arc::new(
            |call: &ToolCall| {
                if call.arguments["path"] == "/tmp/x" {
                    ApprovalOutcome::Defer
                } else {
                    ApprovalOutcome::Allow
                }
            },
        )),
    ));

    let first = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("defer is not an error");
    let pending = first.deferred.clone().expect("the flagged call is pending");
    assert_eq!(pending.approvals[0].id, "call-delete");
    assert!(delete.calls().is_empty());
    assert_eq!(lookup.calls().len(), 1);

    // On resume the same middleware sees the approval and lets it through.
    let run = harness
        .resume_deferred(
            &(),
            RunContext::new(RunConfig::new("second"), ()),
            first.messages.clone(),
            DeferredToolResults::new().approve("call-delete"),
        )
        .await
        .expect("resume completes");
    assert_eq!(delete.calls().len(), 1);
    assert_eq!(run.text().as_deref(), Some("all done"));
}

#[tokio::test]
async fn human_approval_middleware_deny_outcome_answers_the_model_without_running() {
    use crate::middleware::library::{ApprovalOutcome, HumanApprovalMiddleware};

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(MockModel::with_responses(vec![
            mixed_batch(),
            response(Vec::new(), "understood"),
        ])),
    );
    let delete = RecordingTool::plain("delete", "deleted");
    harness.register_tool(delete.clone());
    harness.register_tool(RecordingTool::plain("lookup", "found"));
    harness.push_middleware(Arc::new(
        HumanApprovalMiddleware::new(["delete"]).with_approval_outcome(Arc::new(
            |_call: &ToolCall| ApprovalOutcome::Deny("policy forbids deletes".into()),
        )),
    ));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("a denial is answered, not raised");
    assert!(delete.calls().is_empty());
    assert!(run.deferred.is_none());
    assert_eq!(
        tool_result_text(&run.messages, "call-delete").as_deref(),
        Some("policy forbids deletes")
    );
    assert_eq!(run.text().as_deref(), Some("understood"));
}
