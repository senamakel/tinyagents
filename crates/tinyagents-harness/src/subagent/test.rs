//! Typed-parent sub-agent contracts.

use std::sync::Arc;

use serde_json::json;

use super::{ChildDataPolicy, SubAgent, SubAgentSession, SubAgentTool};
use crate::cancel::CancellationToken;
use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::{AgentEvent, EventSink, RecordingListener};
use crate::limits::RunLimits;
use crate::runtime::{AgentHarness, RunPolicy};
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::MockModel;

#[derive(Clone, Debug, PartialEq, Eq)]
struct NonDefaultContext {
    value: String,
}

fn child_harness<Ctx: Send + Sync>(answer: &str) -> AgentHarness<(), Ctx> {
    let mut harness = AgentHarness::new();
    harness.register_model("child", Arc::new(MockModel::constant(answer)));
    harness
}

#[test]
fn child_data_policy_is_an_explicit_typed_transform() {
    let policy = ChildDataPolicy::new(|parent: &NonDefaultContext| NonDefaultContext {
        value: format!("{}/child", parent.value),
    });
    assert_eq!(
        policy.child_data(&NonDefaultContext { value: "root".into() }),
        NonDefaultContext { value: "root/child".into() }
    );
}

#[tokio::test]
async fn typed_tool_dispatch_runs_child_with_non_default_parent_data() {
    let child = Arc::new(SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<NonDefaultContext>("child answer")),
    ));
    let tool = SubAgentTool::new(
        child,
        ChildDataPolicy::new(|parent: &NonDefaultContext| NonDefaultContext {
            value: format!("{}:delegated", parent.value),
        }),
    );
    let parent = RunContext::new(
        RunConfig::new("parent").with_max_model_calls(1),
        NonDefaultContext { value: "root".into() },
    );
    let result = tool
        .invoke_in_parent_context(&(), json!({"input": "work"}), tinytools::ToolCallOptions::default(), &parent)
        .await
        .unwrap();
    assert!(!result.is_error);
    assert_eq!(result.output(), "child answer");
}

#[tokio::test]
async fn stricter_parent_depth_cap_is_a_recoverable_typed_tool_result() {
    let child = Arc::new(SubAgent::new("worker", "works", Arc::new(child_harness::<NonDefaultContext>("unused"))));
    let tool = SubAgentTool::new(child, ChildDataPolicy::new(Clone::clone));
    let parent = RunContext::new(RunConfig::new("parent").with_max_depth(0), NonDefaultContext { value: "root".into() });
    let result = tool
        .invoke_in_parent_context(&(), json!({"input": "work"}), tinytools::ToolCallOptions::default(), &parent)
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.output().contains("delegated-agent limit signal"));
}

#[tokio::test]
async fn invoke_in_parent_shares_events_cancellation_and_child_lifecycle() {
    let child = SubAgent::new("worker", "works", Arc::new(child_harness::<NonDefaultContext>("done")));
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let cancellation = CancellationToken::new();
    let parent = RunContext::new(RunConfig::new("parent").with_thread("thread"), NonDefaultContext { value: "root".into() })
        .with_events(events)
        .with_cancellation(cancellation.clone());
    assert_eq!(child.invoke_in_parent(&(), NonDefaultContext { value: "one".into() }, &parent, "one").await.unwrap().text().as_deref(), Some("done"));
    assert_eq!(child.invoke_in_parent(&(), NonDefaultContext { value: "two".into() }, &parent, "two").await.unwrap().text().as_deref(), Some("done"));
    assert_eq!(recorder.events().iter().filter(|event| matches!(event.event, AgentEvent::SubAgentStarted { depth: 1, .. })).count(), 2);
    cancellation.cancel();
    assert!(matches!(child.invoke_in_parent(&(), NonDefaultContext { value: "three".into() }, &parent, "three").await, Err(TinyAgentsError::Cancelled)));
}

#[tokio::test]
async fn child_harness_depth_cap_is_enforced_before_model_work() {
    let mut harness = child_harness::<()>("unused");
    harness.with_policy(RunPolicy { limits: RunLimits::default().with_max_depth(0), ..RunPolicy::default() });
    let child = SubAgent::new("worker", "works", Arc::new(harness));
    assert!(matches!(child.invoke(&(), (), 0, "work").await, Err(TinyAgentsError::SubAgentDepth(0))));
}

#[tokio::test]
async fn session_reuses_harness_and_retains_transcript() {
    let child = Arc::new(SubAgent::new("worker", "works", Arc::new(child_harness::<()>("done"))));
    let mut session = SubAgentSession::new(child);
    session.send(&(), (), vec![Message::user("first")]).await.unwrap();
    session.send(&(), (), vec![Message::user("second")]).await.unwrap();
    assert_eq!(session.turns(), 2);
    assert!(session.transcript().len() >= 4);
    session.reset();
    assert!(session.transcript().is_empty());
}
