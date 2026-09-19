//! `NodeContext::durable_task`: per-task memoisation of a side-effecting
//! sub-step, keyed by `(task_id, key)` in the checkpoint write ledger.
//!
//! The contract under test: a handler is re-run from its start after an
//! interrupt/resume, a failure/retry, or an in-process node retry, but a
//! side effect wrapped in `durable_task` happens **once** — the re-run gets
//! the stored output back without polling the future — and distinct keys
//! are memoised independently of each other.

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
use crate::checkpoint::{Checkpointer, FileCheckpointer, InMemoryCheckpointer};
use crate::command::{Command, Interrupt, NodeResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tinyagents_harness::ids::ExecutionStatus;
use tinyagents_harness::retry::RetryPolicy;

fn memory() -> Arc<dyn Checkpointer<i32>> {
    Arc::new(InMemoryCheckpointer::<i32>::new())
}

/// A single node that performs a counted side effect inside
/// `durable_task("side-effect")`, then either pauses (first pass, no resume
/// value) or commits `state + <memoised value>`.
fn interrupting_graph(effects: Arc<AtomicUsize>) -> GraphBuilder<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("work", move |s, ctx: NodeContext| {
            let effects = effects.clone();
            async move {
                let n: usize = ctx
                    .durable_task("side-effect", async {
                        effects.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(effects.load(AtomicOrdering::SeqCst))
                    })
                    .await?;
                if ctx.resume.is_none() {
                    return Ok(NodeResult::Interrupt(Interrupt::new(
                        "work",
                        json!({ "ask": "continue?" }),
                    )));
                }
                Ok(NodeResult::Update(s + n as i32))
            }
        })
        .set_entry("work")
        .set_finish("work")
}

#[tokio::test]
async fn durable_task_is_memoised_across_an_interrupt_and_resume() {
    let effects = Arc::new(AtomicUsize::new(0));
    let graph = interrupting_graph(effects.clone())
        .compile()
        .unwrap()
        .with_checkpointer(memory());

    let paused = graph.run_with_thread("memo", 100).await.unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);

    // The memo travelled with the pending task into the checkpoint ledger.
    let snapshot = graph.get_state("memo", None).await.unwrap().unwrap();
    let tuple = graph
        .checkpointer
        .as_ref()
        .unwrap()
        .get_tuple(snapshot.config)
        .await
        .unwrap()
        .unwrap();
    let memos: Vec<_> = tuple
        .pending_writes
        .iter()
        .filter(|w| w.is_durable_task())
        .collect();
    assert_eq!(memos.len(), 1);
    assert_eq!(memos[0].durable_task_key(), Some("side-effect"));
    assert_eq!(memos[0].payload, json!(1));
    assert!(memos[0].idx >= 1, "durable-task writes never reuse idx 0");

    // The handler re-runs from the top on resume, but the side effect does
    // not: the memoised `1` is what it sees.
    let resumed = graph
        .resume("memo", Command::resume(json!({ "go": true })))
        .await
        .unwrap();
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 101);
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);
}

/// `work` performs a counted side effect, then fails hard on its first
/// attempt (per `fails`) and succeeds afterwards.
fn failing_graph(effects: Arc<AtomicUsize>, fails: Arc<AtomicUsize>) -> GraphBuilder<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("work", move |s, ctx: NodeContext| {
            let effects = effects.clone();
            let fails = fails.clone();
            async move {
                let n: usize = ctx
                    .durable_task("side-effect", async {
                        effects.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(effects.load(AtomicOrdering::SeqCst))
                    })
                    .await?;
                if fails.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                    return Err(TinyAgentsError::Graph("crash after the side effect".into()));
                }
                Ok(NodeResult::Update(s + n as i32))
            }
        })
        .set_entry("work")
        .set_finish("work")
}

#[tokio::test]
async fn durable_task_is_memoised_across_a_failure_and_retry() {
    let effects = Arc::new(AtomicUsize::new(0));
    let fails = Arc::new(AtomicUsize::new(0));
    let graph = failing_graph(effects.clone(), fails.clone())
        .compile()
        .unwrap()
        .with_checkpointer(memory());

    let err = graph.run_with_thread("retry", 100).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Graph(_)), "got {err:?}");
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);

    let resumed = graph.retry("retry").await.unwrap();
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 101);
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(
        fails.load(AtomicOrdering::SeqCst),
        2,
        "handler itself ran twice"
    );
}

#[tokio::test]
async fn durable_task_keys_are_memoised_independently() {
    // `a` runs, the handler crashes, then on retry `a` is a hit and `b` a
    // fresh miss — proving memoisation is per key, not per node.
    let a_effects = Arc::new(AtomicUsize::new(0));
    let b_effects = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    let (a, b, att) = (a_effects.clone(), b_effects.clone(), attempts.clone());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("work", move |s, ctx: NodeContext| {
            let (a, b, att) = (a.clone(), b.clone(), att.clone());
            async move {
                let x: i32 = ctx
                    .durable_task("a", async {
                        a.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(10)
                    })
                    .await?;
                if att.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                    return Err(TinyAgentsError::Graph("crash between a and b".into()));
                }
                let y: i32 = ctx
                    .durable_task("b", async {
                        b.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(20)
                    })
                    .await?;
                Ok(NodeResult::Update(s + x + y))
            }
        })
        .set_entry("work")
        .set_finish("work")
        .compile()
        .unwrap()
        .with_checkpointer(memory());

    graph.run_with_thread("keys", 0).await.unwrap_err();
    assert_eq!(a_effects.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(
        b_effects.load(AtomicOrdering::SeqCst),
        0,
        "`b` never reached"
    );

    let resumed = graph.retry("keys").await.unwrap();
    assert_eq!(resumed.state, 30);
    assert_eq!(
        a_effects.load(AtomicOrdering::SeqCst),
        1,
        "`a` replayed from memo"
    );
    assert_eq!(
        b_effects.load(AtomicOrdering::SeqCst),
        1,
        "`b` ran fresh once"
    );
}

#[tokio::test]
async fn durable_task_memo_is_shared_by_in_process_node_retries() {
    // Under a node retry policy the handler is re-invoked in-process with a
    // cloned context; the clone shares the memo buffer, so the retry hits.
    let effects = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(AtomicUsize::new(0));
    let (eff, att) = (effects.clone(), attempts.clone());
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("work", move |s, ctx: NodeContext| {
            let (eff, att) = (eff.clone(), att.clone());
            async move {
                let n: i32 = ctx
                    .durable_task("side-effect", async {
                        eff.fetch_add(1, AtomicOrdering::SeqCst);
                        Ok(7)
                    })
                    .await?;
                if att.fetch_add(1, AtomicOrdering::SeqCst) < 2 {
                    return Err(TinyAgentsError::Model("transient".into()));
                }
                Ok(NodeResult::Update(s + n))
            }
        })
        .set_entry("work")
        .set_finish("work")
        .compile()
        .unwrap()
        .with_node_retry(RetryPolicy::default().with_max_attempts(4));

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 7);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 3);
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn durable_task_memo_survives_a_file_checkpointer_restart() {
    let dir = tempfile::tempdir().unwrap();
    let effects = Arc::new(AtomicUsize::new(0));
    {
        let cp: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
        let graph = interrupting_graph(effects.clone())
            .compile()
            .unwrap()
            .with_checkpointer(cp);
        let paused = graph.run_with_thread("disk", 0).await.unwrap();
        assert!(paused.is_interrupted());
        assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);
    }
    let cp: Arc<dyn Checkpointer<i32>> = Arc::new(FileCheckpointer::<i32>::new(dir.path()));
    let graph = interrupting_graph(effects.clone())
        .compile()
        .unwrap()
        .with_checkpointer(cp);
    let resumed = graph
        .resume("disk", Command::resume(json!({})))
        .await
        .unwrap();
    assert_eq!(resumed.state, 1);
    assert_eq!(effects.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn durable_task_does_not_memoise_a_failed_future() {
    // An `Err` from the wrapped future is returned as-is and leaves no memo,
    // so the next attempt re-runs it.
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("work", move |s, ctx: NodeContext| {
            let c = c.clone();
            async move {
                let n: i32 = ctx
                    .durable_task("flaky", async {
                        if c.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                            Err(TinyAgentsError::Model("first call fails".into()))
                        } else {
                            Ok(3)
                        }
                    })
                    .await?;
                Ok(NodeResult::Update(s + n))
            }
        })
        .set_entry("work")
        .set_finish("work")
        .compile()
        .unwrap()
        .with_node_retry(RetryPolicy::default().with_max_attempts(3));

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 3);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
}
