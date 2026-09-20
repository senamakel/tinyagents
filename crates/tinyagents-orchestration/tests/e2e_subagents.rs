//! TRUE end-to-end: agent-calling-agent composition (sub-agents).
//!
//! A parent [`AgentHarness`] whose scripted [`MockModel`] calls a
//! [`SubAgentTool`] drives a child [`AgentHarness`] (also a `MockModel`) and
//! composes the child's answer into a final assistant reply. The parent run's
//! [`EventSink`] is wired to a testkit [`EventRecorder`] so we can reconstruct
//! a [`Trajectory`] and assert *structurally* that the sub-agent really ran —
//! never on model prose.
//!
//! A second test exercises the deterministic recursion-depth guard: nesting a
//! sub-agent past the harness's `max_depth` fails fast with
//! [`TinyAgentsError::SubAgentDepth`] *before* any model call, both through the
//! direct invoke path and through the tool path.

use std::sync::Arc;

use serde_json::json;

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::limits::RunLimits;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::{EventRecorder, Trajectory};
use tinyagents_orchestration::subagent::{
    ChildDataPolicy, SubAgent, SubAgentJobStatus, SubAgentTool,
};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// A tool-call assistant turn: no text, a single tool call.
fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(9, 4)),
            origin: None,
        },
        usage: Some(Usage::new(9, 4)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// A plain-text assistant turn.
fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.to_string())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(5, 3)),
            origin: None,
        },
        usage: Some(Usage::new(5, 3)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// A child harness whose model always answers with `answer`.
fn child_harness(answer: &str) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("child-model", Arc::new(MockModel::constant(answer)));
    harness
}

/// A child harness capped at `max_depth`.
fn child_harness_with_max_depth(answer: &str, max_depth: usize) -> AgentHarness<()> {
    let mut harness = child_harness(answer);
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(max_depth),
        ..RunPolicy::default()
    });
    harness
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn parent_drives_subagent_and_composes_answer() {
    // Child agent that "researches" and returns a fixed answer.
    let child = Arc::new(SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("RUST_IS_A_SYSTEMS_LANGUAGE")),
    ));
    let tool = Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(|parent: &()| *parent),
    ));
    let jobs = tool.job_registry().clone();

    // Parent: first turn delegates to the sub-agent tool, second turn composes
    // the final answer.
    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool_dispatch(tool);
    parent.register_model(
        "parent-model",
        Arc::new(MockModel::with_responses(vec![
            tool_call_response("c1", "researcher", json!({ "input": "what is rust?" })),
            text_response("Based on the researcher: a systems language."),
        ])),
    );

    // Wire the parent run's events into a shared recorder so we can assert on
    // the trajectory (the sub-agent tool really ran) afterward.
    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("parent-run"), ()).with_events(recorder.sink());

    let run = parent
        .invoke_in_context(&(), ctx, vec![Message::user("delegate this")])
        .await
        .expect("parent run succeeds");

    // Behavior / structure assertions — never on the exact prose.
    assert_eq!(run.tool_calls, 1, "parent invoked the sub-agent tool once");
    assert_eq!(run.model_calls, 2, "parent made two model calls");
    assert_eq!(
        run.text(),
        Some("Based on the researcher: a systems language.".to_string())
    );

    // The tool result contains a stable job id rather than blocking for output.
    let job_id = jobs
        .list()
        .into_iter()
        .next()
        .expect("the subagent tool registers one job")
        .id
        .to_string();
    let job = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            let job = jobs.get(&job_id).expect("job is registered");
            if job.status.is_terminal() {
                break job;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("job completes");
    assert_eq!(job.status, SubAgentJobStatus::Completed);
    assert_eq!(job.output.as_deref(), Some("RUST_IS_A_SYSTEMS_LANGUAGE"));

    // Trajectory assertion: the sub-agent (exposed as the `researcher` tool)
    // really ran, and the run completed cleanly.
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_tool_called("researcher");
    assert_eq!(traj.tool_call_count("researcher"), 1);
    traj.assert_model_called_times(3);
    traj.assert_completed();
    traj.assert_order(&["run.started", "researcher", "run.completed"])
        .expect("sub-agent tool runs between run start and completion");
}

#[tokio::test]
async fn child_subagents_derive_unique_thread_ids_from_parent_thread() {
    let child = SubAgent::new(
        "researcher",
        "answers research questions",
        Arc::new(child_harness("RUST_IS_A_SYSTEMS_LANGUAGE")),
    );
    let recorder = EventRecorder::new();
    let parent = RunContext::new(
        RunConfig::new("parent-run").with_thread("parent-thread"),
        (),
    )
    .with_events(recorder.sink());

    child
        .invoke_in_parent(&(), (), &parent, "first question")
        .await
        .expect("first child run succeeds");
    child
        .invoke_in_parent(&(), (), &parent, "second question")
        .await
        .expect("second child run succeeds");

    let child_threads: Vec<String> = recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::RunStarted {
                thread_id: Some(thread_id),
                ..
            } => Some(thread_id.to_string()),
            _ => None,
        })
        .collect();

    assert_eq!(child_threads.len(), 2);
    assert_ne!(
        child_threads[0], child_threads[1],
        "each child run should get an isolated thread"
    );
    for thread in child_threads {
        assert!(
            thread.starts_with("parent-thread-subagent-researcher-d1-"),
            "child thread should inherit the parent thread as a hyphenated prefix: {thread}"
        );
        assert!(
            !thread.contains('/'),
            "child thread id should not use slash separators: {thread}"
        );
    }
}

#[tokio::test]
async fn nesting_past_max_depth_is_a_deterministic_error() {
    // Cap the child harness at depth 1: a child run is allowed at depth 1
    // (parent_depth 0) but not at depth 2 (parent_depth 1).
    let subagent = Arc::new(SubAgent::new(
        "deep",
        "a deep agent",
        Arc::new(child_harness_with_max_depth("ok", 1)),
    ));

    // Within the cap: parent_depth 0 -> child depth 1.
    let ok_run = subagent
        .invoke(&(), (), 0, "ok")
        .await
        .expect("child run at depth 1 is within the cap");
    assert_eq!(ok_run.text(), Some("ok".to_string()));

    // Direct invoke past the cap: parent_depth 1 -> child depth 2 > cap of 1.
    let err = subagent
        .invoke(&(), (), 1, "too deep")
        .await
        .expect_err("child depth 2 exceeds the cap");
    assert!(
        matches!(err, TinyAgentsError::SubAgentDepth(1)),
        "expected SubAgentDepth(1), got {err:?}"
    );

    // Tool path past the cap: the typed parent context carries depth 1, so the
    // child run would exceed its maximum depth of 1.
    let tool = SubAgentTool::new(subagent, ChildDataPolicy::new(|parent: &()| *parent));
    let parent = RunContext::new(
        RunConfig::new("deep-parent")
            .with_depth(1)
            .with_max_depth(1),
        (),
    );
    let tool_result = tool
        .invoke_in_parent_context(
            &(),
            json!({ "input": "x" }),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .expect("the tool returns a failed tool result");
    let tool_error = tool_result.output();
    assert!(
        tool_error.contains("recursion depth limit") && tool_error.contains("maximum depth of 1"),
        "expected SubAgentDepth(1) from the tool path, got {tool_error:?}"
    );
}
