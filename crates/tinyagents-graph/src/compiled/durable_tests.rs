//! Durable executor tests: interrupt→resume and failure→retry across dropped
//! and reopened driver handles on an explicitly owned testkit backend.
//! Physical persistence is covered by the storage engine's own tests.

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
use crate::checkpoint::Checkpointer;
use crate::command::{Command, Interrupt, NodeResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tinyagents_harness::ids::ExecutionStatus;
use tinystoragedrivers_testkit::MemoryBackend;

/// Same "human approval" graph shape as `interrupt_then_resume_reruns_node`:
/// pauses on first run, applies a resume-supplied bump on the second.
fn approve_graph() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                Some(value) => {
                    let bump = value.get("bump").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                    Ok(NodeResult::Update(s + bump))
                }
                None => Ok(NodeResult::Interrupt(Interrupt::new(
                    "approve",
                    json!({ "ask": "approve?" }),
                ))),
            }
        })
        .add_node("done", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("approve")
        .add_edge("approve", "done")
        .set_finish("done")
        .compile()
        .unwrap()
}

/// Same "flaky node" shape as `flaky_graph` in `test.rs`: fails the first
/// `fail_times` invocations, then succeeds with `+1`.
fn flaky_graph(fail_times: usize, attempts: Arc<AtomicUsize>) -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("flaky", move |s, _c: NodeContext| {
            let attempts = attempts.clone();
            async move {
                let n = attempts.fetch_add(1, AtomicOrdering::SeqCst);
                if n < fail_times {
                    Err(TinyAgentsError::Model(format!("transient blip {n}")))
                } else {
                    Ok(NodeResult::Update(s + 1))
                }
            }
        })
        .set_entry("flaky")
        .set_finish("flaky")
        .compile()
        .unwrap()
}

// ── DriverCheckpointer over testkit ports ────────────────────────────────

#[tokio::test]
async fn driver_backend_interrupt_then_reopen_then_resume() {
    let checkpoint_backend = MemoryBackend::new();

    // Write phase runs the graph to its interrupt point.
    {
        let cp: Arc<dyn Checkpointer<i32>> = Arc::new(
            crate::test_support::dir_checkpointer::<i32>(&checkpoint_backend),
        );
        let graph = approve_graph().with_checkpointer(cp);
        let paused = graph.run_with_thread("hitl", 10).await.unwrap();
        assert!(paused.is_interrupted());
        assert_eq!(paused.status.status, ExecutionStatus::Interrupted);
        assert_eq!(paused.interrupts.len(), 1);
        // Drop the graph and its checkpointer.
    }

    // Read/resume phase uses a new driver over the same backend.
    let cp2: Arc<dyn Checkpointer<i32>> = Arc::new(crate::test_support::dir_checkpointer::<i32>(
        &checkpoint_backend,
    ));
    let graph2 = approve_graph().with_checkpointer(cp2);
    let resumed = graph2
        .resume("hitl", Command::resume(json!({ "bump": 5 })))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.state, 15);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn driver_backend_failure_then_reopen_then_retry() {
    let checkpoint_backend = MemoryBackend::new();
    let attempts = Arc::new(AtomicUsize::new(0));

    // Write phase: the node fails once and the run aborts, leaving a
    // resumable failure checkpoint.
    {
        let cp: Arc<dyn Checkpointer<i32>> = Arc::new(
            crate::test_support::dir_checkpointer::<i32>(&checkpoint_backend),
        );
        let graph = flaky_graph(1, attempts.clone()).with_checkpointer(cp);
        let err = graph.run_with_thread("net", 100).await.unwrap_err();
        assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        // Drop the graph and its checkpointer.
    }

    // Read/retry phase uses a new driver over the same backend. The transient
    // condition has cleared by the time this attempt runs.
    let cp2: Arc<dyn Checkpointer<i32>> = Arc::new(crate::test_support::dir_checkpointer::<i32>(
        &checkpoint_backend,
    ));
    let graph2 = flaky_graph(1, attempts.clone()).with_checkpointer(cp2);
    let resumed = graph2.retry("net").await.unwrap();
    assert_eq!(resumed.state, 101);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
}
