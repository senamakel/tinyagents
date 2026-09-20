//! TRUE end-to-end (offline): sub-agent ERROR propagation.
//!
//! A [`SubAgent`] is built over a child [`AgentHarness`] whose model always
//! requests a tool that fails ([`FakeTool::failing`]). The tool failure
//! propagates out of the child agent loop, so:
//!
//! - [`SubAgent::invoke`] returns `Err(TinyAgentsError::Tool(..))`,
//! - [`SubAgentTool`] returns a job id immediately, and the job registry later
//!   records the sanitized failure, and
//! - an orchestrator remains live after spawning a failing child and can query
//!   that failure through host tools.
//!
//! All assertions are structural / on the error variant — never on model prose.

use std::sync::Arc;

use serde_json::json;

use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::testkit::{EventRecorder, FakeTool, ScriptedModel, Trajectory};
use tinyagents_orchestration::subagent::{
    ChildDataPolicy, SubAgent, SubAgentJobRegistry, SubAgentJobStatus, SubAgentTool,
};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;

async fn wait_for_failed_job(jobs: &SubAgentJobRegistry) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if let Some(job) = jobs.list().into_iter().next()
                && job.status.is_terminal()
            {
                assert_eq!(job.status, SubAgentJobStatus::Failed);
                assert_eq!(
                    job.error.as_deref(),
                    Some("tool error: tool dispatch failed")
                );
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("failing child job reaches a terminal state");
}

/// Builds a child harness whose model always asks for the `broken` tool, which
/// fails with a foreign `anyhow` error. The harness maps that to its stable,
/// non-sensitive tool-failure surface before a sub-agent can propagate it.
fn failing_child_harness() -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "child-model",
        Arc::new(MockModel::with_tool_call("broken", json!({}))),
    );
    harness.register_tool(Arc::new(FakeTool::failing("broken", "boom")));
    harness
}

#[tokio::test]
async fn subagent_invoke_propagates_tool_failure() {
    let subagent = SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    );

    let err = subagent
        .invoke(&(), (), 0, "do the thing")
        .await
        .expect_err("the child tool failure must propagate out of SubAgent::invoke");

    match err {
        TinyAgentsError::Tool(msg) => assert_eq!(msg, "tool dispatch failed"),
        other => panic!("expected a sanitized TinyAgentsError::Tool, got {other:?}"),
    }
}

#[tokio::test]
async fn subagent_tool_call_returns_job_id_and_records_failure() {
    let subagent = Arc::new(SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    ));
    let tool = SubAgentTool::new(subagent, ChildDataPolicy::new(|parent: &()| *parent));
    let jobs = tool.job_registry().clone();
    let parent = RunContext::new(RunConfig::new("parent"), ());

    let result = tool
        .invoke_in_parent_context(
            &(),
            json!({ "input": "x" }),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .expect("spawning a child is independent from its eventual result");
    assert!(result.output().contains("job_id"));
    wait_for_failed_job(&jobs).await;
}

#[tokio::test]
async fn orchestrator_survives_a_failing_subagent_job() {
    let subagent = Arc::new(SubAgent::new(
        "broken_worker",
        "a worker whose tool always fails",
        Arc::new(failing_child_harness()),
    ));
    let tool = Arc::new(SubAgentTool::new(
        subagent,
        ChildDataPolicy::new(|parent: &()| *parent),
    ));
    let jobs = tool.job_registry().clone();

    let mut orchestrator: AgentHarness<()> = AgentHarness::new();
    orchestrator.register_tool_dispatch(tool);
    let mut delegation = ModelResponse::assistant("");
    delegation.message.tool_calls.push(ToolCall::new(
        "delegate",
        "broken_worker",
        json!({ "input": "delegate" }),
    ));
    orchestrator.register_model(
        "parent-model",
        Arc::new(ScriptedModel::new(vec![
            delegation,
            ModelResponse::assistant("job spawned"),
        ])),
    );

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("orchestrator-run"), ()).with_events(recorder.sink());

    let run = orchestrator
        .invoke_in_context(&(), ctx, vec![Message::user("delegate this")])
        .await
        .expect("child failure does not abort the spawning orchestrator");
    assert_eq!(run.text().as_deref(), Some("job spawned"));
    wait_for_failed_job(&jobs).await;

    // The orchestrator run emitted a RunFailed event (on_error fan-out path).
    let traj = Trajectory::from_events(recorder.events());
    traj.assert_completed();
}
