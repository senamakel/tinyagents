//! Typed-parent sub-agent contracts.
//!
//! Covers `ChildDataPolicy` as an explicit transform, `SubAgentTool` dispatch
//! (child-data inheritance, a stricter child depth cap surfacing as a
//! recoverable tool result rather than an error, cancellation inheritance),
//! `SubAgent::invoke_in_parent` event/cancellation/lifecycle propagation, the
//! depth cap being enforced before any model call, and `SubAgentSession`
//! reusing its harness while retaining the transcript across sends.

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{ChildDataPolicy, SubAgent, SubAgentSession, SubAgentTool};
use crate::cancel::CancellationToken;
use crate::context::{RunConfig, RunContext};
use crate::error::TinyAgentsError;
use crate::events::{AgentEvent, EventSink, RecordingListener};
use crate::limits::RunLimits;
use crate::runtime::{AgentHarness, RunPolicy};
use crate::tool::ToolRegistry;
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
        policy.child_data(&NonDefaultContext {
            value: "root".into()
        }),
        NonDefaultContext {
            value: "root/child".into()
        }
    );
}

#[tokio::test]
async fn typed_tool_dispatch_runs_child_with_non_default_parent_data() {
    let observed_parent_data = Arc::new(Mutex::new(Vec::new()));
    let policy_observation = observed_parent_data.clone();
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let child = Arc::new(SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<NonDefaultContext>("child answer")),
    ));
    let tool = Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(move |parent: &NonDefaultContext| {
            policy_observation
                .lock()
                .expect("data policy observation lock")
                .push(parent.value.clone());
            NonDefaultContext {
                value: format!("{}:delegated", parent.value),
            }
        }),
    ));
    let mut registry: ToolRegistry<(), NonDefaultContext> = ToolRegistry::new();
    registry.register_dispatch(tool);
    let parent = RunContext::new(
        RunConfig::new("parent")
            .with_thread("typed-parent-thread")
            .with_max_model_calls(1),
        NonDefaultContext {
            value: "root".into(),
        },
    )
    .with_events(events);
    let result = registry
        .dispatch("worker")
        .expect("typed parent dispatcher is registered")
        .execute(
            &(),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .unwrap();
    assert!(!result.is_error);
    assert_eq!(result.output(), "child answer");
    assert_eq!(
        *observed_parent_data
            .lock()
            .expect("data policy observation lock"),
        vec!["root"]
    );
    assert!(recorder.events().iter().any(|record| matches!(
        &record.event,
        AgentEvent::RunStarted { run_id, thread_id: Some(thread_id) }
            if run_id.as_str().starts_with("worker-d1-")
                && thread_id.as_str().starts_with("typed-parent-thread-subagent-worker-d1-")
    )));
}

#[tokio::test]
async fn stricter_parent_depth_cap_is_a_recoverable_typed_tool_result() {
    let child = Arc::new(SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<NonDefaultContext>("unused")),
    ));
    let tool = Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(|parent: &NonDefaultContext| parent.clone()),
    ));
    let mut registry: ToolRegistry<(), NonDefaultContext> = ToolRegistry::new();
    registry.register_dispatch(tool);
    let parent = RunContext::new(
        RunConfig::new("parent").with_max_depth(0),
        NonDefaultContext {
            value: "root".into(),
        },
    );
    let result = registry
        .dispatch("worker")
        .expect("typed parent dispatcher is registered")
        .execute(
            &(),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.output().contains("delegated-agent limit signal"));
}

#[tokio::test]
async fn typed_tool_dispatch_inherits_parent_cancellation() {
    let child = Arc::new(SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<NonDefaultContext>("unused")),
    ));
    let tool = Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(|parent: &NonDefaultContext| parent.clone()),
    ));
    let mut registry: ToolRegistry<(), NonDefaultContext> = ToolRegistry::new();
    registry.register_dispatch(tool);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let parent = RunContext::new(
        RunConfig::new("parent"),
        NonDefaultContext {
            value: "root".into(),
        },
    )
    .with_cancellation(cancellation);

    let error = registry
        .dispatch("worker")
        .expect("typed parent dispatcher is registered")
        .execute(
            &(),
            json!({"input": "work"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .expect_err("cancelled parent stops the typed child invocation");
    assert!(matches!(
        error.downcast_ref::<TinyAgentsError>(),
        Some(TinyAgentsError::Cancelled)
    ));
}

#[tokio::test]
async fn invoke_in_parent_shares_events_cancellation_and_child_lifecycle() {
    let child = SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<NonDefaultContext>("done")),
    );
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let cancellation = CancellationToken::new();
    let parent = RunContext::new(
        RunConfig::new("parent").with_thread("thread"),
        NonDefaultContext {
            value: "root".into(),
        },
    )
    .with_events(events)
    .with_cancellation(cancellation.clone());
    assert_eq!(
        child
            .invoke_in_parent(
                &(),
                NonDefaultContext {
                    value: "one".into()
                },
                &parent,
                "one"
            )
            .await
            .unwrap()
            .text()
            .as_deref(),
        Some("done")
    );
    assert_eq!(
        child
            .invoke_in_parent(
                &(),
                NonDefaultContext {
                    value: "two".into()
                },
                &parent,
                "two"
            )
            .await
            .unwrap()
            .text()
            .as_deref(),
        Some("done")
    );
    let records = recorder.events();
    assert_eq!(
        records
            .iter()
            .filter(|event| matches!(event.event, AgentEvent::SubAgentStarted { depth: 1, .. }))
            .count(),
        2
    );
    assert_eq!(
        records
            .iter()
            .filter(|event| matches!(event.event, AgentEvent::SubAgentCompleted { depth: 1, .. }))
            .count(),
        2
    );
    let child_runs: Vec<(String, String)> = records
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::RunStarted {
                run_id,
                thread_id: Some(thread_id),
            } => Some((run_id.to_string(), thread_id.to_string())),
            _ => None,
        })
        .collect();
    assert_eq!(child_runs.len(), 2);
    assert_ne!(child_runs[0], child_runs[1]);
    for (run_id, thread_id) in child_runs {
        assert!(run_id.starts_with("worker-d1-"), "run id: {run_id}");
        assert!(
            thread_id.starts_with("thread-subagent-worker-d1-"),
            "thread id: {thread_id}"
        );
    }
    cancellation.cancel();
    assert!(matches!(
        child
            .invoke_in_parent(
                &(),
                NonDefaultContext {
                    value: "three".into()
                },
                &parent,
                "three"
            )
            .await,
        Err(TinyAgentsError::Cancelled)
    ));
}

#[tokio::test]
async fn child_harness_depth_cap_is_enforced_before_model_work() {
    let mut harness = child_harness::<()>("unused");
    harness.with_policy(RunPolicy {
        limits: RunLimits::default().with_max_depth(0),
        ..RunPolicy::default()
    });
    let child = SubAgent::new("worker", "works", Arc::new(harness));
    assert!(matches!(
        child.invoke(&(), (), 0, "work").await,
        Err(TinyAgentsError::SubAgentDepth(0))
    ));
}

#[tokio::test]
async fn session_reuses_harness_and_retains_transcript() {
    let child = Arc::new(SubAgent::new(
        "worker",
        "works",
        Arc::new(child_harness::<()>("done")),
    ));
    let events = EventSink::new();
    let recorder = Arc::new(RecordingListener::new());
    events.subscribe(recorder.clone());
    let mut session = SubAgentSession::new(child).with_events(events);
    session
        .send(&(), (), vec![Message::user("first")])
        .await
        .unwrap();
    session
        .send(&(), (), vec![Message::user("second")])
        .await
        .unwrap();
    assert_eq!(session.turns(), 2);
    assert!(session.transcript().len() >= 4);
    assert!(
        recorder
            .events()
            .iter()
            .any(|record| matches!(record.event, AgentEvent::SubAgentReused { turn: 1, .. }))
    );
    session.reset();
    assert!(session.transcript().is_empty());
}
