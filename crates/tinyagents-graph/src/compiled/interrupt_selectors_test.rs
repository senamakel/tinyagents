//! Executor-level interrupt injection (`GraphBuilder::interrupt_before` /
//! `interrupt_after`, with `mark_interrupt` as the `before` alias) and
//! fail-closed `Interrupt::response_schema` validation on resume.
//!
//! Every test counts handler invocations through a shared `AtomicUsize`:
//! the load-bearing contract of both selectors is that the paused node's
//! handler runs **exactly once** across the pause and the resume — never
//! zero times (a lost write) and never twice (a repeated side effect).

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
use crate::checkpoint::{Checkpointer, InMemoryCheckpointer};
use crate::command::{Command, Interrupt, NodeResult};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tinyagents_harness::ids::ExecutionStatus;

fn memory() -> Arc<dyn Checkpointer<i32>> {
    Arc::new(InMemoryCheckpointer::<i32>::new())
}

/// `a (+1) -> b (+10, counted, adds any resume `bump`) -> c (+100)`, with
/// the interrupt selectors applied by `configure`.
fn chain(
    b_runs: Arc<AtomicUsize>,
    configure: impl FnOnce(GraphBuilder<i32, i32>) -> GraphBuilder<i32, i32>,
) -> CompiledGraph<i32, i32> {
    let builder = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", move |s, ctx: NodeContext| {
            let b_runs = b_runs.clone();
            async move {
                b_runs.fetch_add(1, AtomicOrdering::SeqCst);
                let bump = ctx
                    .resume
                    .as_ref()
                    .and_then(|v| v.get("bump"))
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                Ok(NodeResult::Update(s + 10 + bump))
            }
        })
        .add_node("c", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 100))
        })
        .add_sequence(["a", "b", "c"])
        .set_entry("a")
        .set_finish("c");
    configure(builder)
        .compile()
        .unwrap()
        .with_checkpointer(memory())
}

fn phase(interrupt: &Interrupt) -> &str {
    interrupt.payload["phase"].as_str().unwrap_or("")
}

#[tokio::test]
async fn interrupt_before_pauses_without_running_the_handler_and_resume_runs_it_once() {
    let b_runs = Arc::new(AtomicUsize::new(0));
    let graph = chain(b_runs.clone(), |b| b.interrupt_before(["b"]));

    let paused = graph.run_with_thread("before", 0).await.unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(paused.status.status, ExecutionStatus::Interrupted);
    assert_eq!(paused.interrupts.len(), 1);
    assert_eq!(paused.interrupts[0].node.as_str(), "b");
    assert_eq!(phase(&paused.interrupts[0]), "before");
    assert!(
        paused.interrupts[0].task_id.is_some(),
        "stamped with its task id"
    );
    // `a` committed; `b` never ran.
    assert_eq!(paused.state, 1);
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 0);

    // The pause is a checkpoint carrying the interrupt.
    let history = graph.get_state_history("before", None).await.unwrap();
    assert!(history[0].metadata.has_interrupts);
    assert_eq!(history[0].pending_interrupts.len(), 1);
    assert_eq!(
        history[0]
            .next_nodes
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    assert_eq!(history[0].values, 1);

    // Resume runs `b` normally (with the resume value) and finishes.
    let resumed = graph
        .resume("before", Command::resume(json!({ "bump": 5 })))
        .await
        .unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 1 + 10 + 5 + 100);
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn mark_interrupt_is_an_alias_for_interrupt_before() {
    let b_runs = Arc::new(AtomicUsize::new(0));
    let graph = chain(b_runs.clone(), |b| b.mark_interrupt("b"));

    // Export marker still set...
    let topology = graph.topology();
    let b = topology.nodes.iter().find(|n| n.id == "b").unwrap();
    assert!(b.interrupt);

    // ...and the runtime pause is real.
    let paused = graph.run_with_thread("alias", 0).await.unwrap();
    assert_eq!(phase(&paused.interrupts[0]), "before");
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 0);
    let resumed = graph.retry("alias").await.unwrap();
    assert_eq!(resumed.state, 111);
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn interrupt_after_runs_the_handler_once_and_holds_its_update_until_resume() {
    let b_runs = Arc::new(AtomicUsize::new(0));
    let graph = chain(b_runs.clone(), |b| b.interrupt_after(["b"]));

    let paused = graph.run_with_thread("after", 0).await.unwrap();
    assert!(paused.is_interrupted());
    assert_eq!(paused.interrupts.len(), 1);
    assert_eq!(paused.interrupts[0].node.as_str(), "b");
    assert_eq!(phase(&paused.interrupts[0]), "after");
    // The handler ran exactly once, but its `+10` is not yet committed.
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(paused.state, 1);

    // The checkpoint also holds the pre-update state, the interrupt, and
    // `b` as the pending task with its deferred result in the ledger.
    let snapshot = graph.get_state("after", None).await.unwrap().unwrap();
    assert_eq!(snapshot.values, 1);
    assert!(snapshot.metadata.has_interrupts);
    assert_eq!(
        snapshot
            .next_nodes
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    let tuple = graph
        .checkpointer
        .as_ref()
        .unwrap()
        .get_tuple(snapshot.config.clone())
        .await
        .unwrap()
        .unwrap();
    let deferred: Vec<_> = tuple
        .pending_writes
        .iter()
        .filter(|w| w.is_interrupt_after())
        .collect();
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].payload["update"], json!(11));
    let history = graph.get_state_history("after", None).await.unwrap();
    assert!(history[0].metadata.has_interrupts);

    // Resume replays the stored result: no second handler run, update applied.
    let resumed = graph.retry("after").await.unwrap();
    assert!(!resumed.is_interrupted());
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 111);
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn interrupt_after_replays_the_deferred_command_goto_on_resume() {
    let b_runs = Arc::new(AtomicUsize::new(0));
    let runs = b_runs.clone();
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1))
        })
        .add_node("b", move |s, _c: NodeContext| {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Command(
                    Command::update(s + 10).with_goto(["d"]),
                ))
            }
        })
        .add_node("c", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 100))
        })
        .add_node("d", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s + 1000))
        })
        .set_entry("a")
        .add_edge("a", "b")
        .mark_command_routing("b")
        .set_finish("c")
        .set_finish("d")
        .interrupt_after(["b"])
        .compile()
        .unwrap()
        .with_checkpointer(memory());

    let paused = graph.run_with_thread("goto", 0).await.unwrap();
    assert_eq!(phase(&paused.interrupts[0]), "after");
    assert_eq!(paused.state, 1);

    let resumed = graph.retry("goto").await.unwrap();
    // `b`'s +10 applied, and its explicit `goto d` honoured (not `c`).
    assert_eq!(resumed.state, 1 + 10 + 1000);
    assert_eq!(
        resumed
            .visited
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b", "d"]
    );
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn interrupt_before_and_after_on_one_node_pause_twice_and_run_it_once() {
    let b_runs = Arc::new(AtomicUsize::new(0));
    let graph = chain(b_runs.clone(), |b| {
        b.interrupt_before(["b"]).interrupt_after(["b"])
    });

    let first = graph.run_with_thread("both", 0).await.unwrap();
    assert_eq!(phase(&first.interrupts[0]), "before");
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 0);

    let second = graph.retry("both").await.unwrap();
    assert!(second.is_interrupted());
    assert_eq!(phase(&second.interrupts[0]), "after");
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(second.state, 1);

    let done = graph.retry("both").await.unwrap();
    assert_eq!(done.status.status, ExecutionStatus::Completed);
    assert_eq!(done.state, 111);
    assert_eq!(b_runs.load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn interrupt_after_in_a_parallel_step_defers_every_branch() {
    // Two fan-out branches, both `interrupt_after`: both handlers run once,
    // both updates are held, both are replayed on resume through the
    // additive reducer.
    let runs = Arc::new(AtomicUsize::new(0));
    let r1 = runs.clone();
    let r2 = runs.clone();
    let graph = GraphBuilder::<i32, i32>::new()
        .set_reducer(crate::reducer::ClosureStateReducer::new(
            |s: i32, u: i32| Ok(s + u),
        ))
        .with_parallel(true)
        .add_node("fan", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(0))
        })
        .add_node("x", move |_s, _c: NodeContext| {
            let r1 = r1.clone();
            async move {
                r1.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(10))
            }
        })
        .add_node("y", move |_s, _c: NodeContext| {
            let r2 = r2.clone();
            async move {
                r2.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(20))
            }
        })
        .set_entry("fan")
        .add_edge("fan", "x")
        .add_edge("fan", "y")
        .set_finish("x")
        .set_finish("y")
        .interrupt_after(["x", "y"])
        .compile()
        .unwrap()
        .with_checkpointer(memory());

    let paused = graph.run_with_thread("par", 0).await.unwrap();
    assert_eq!(paused.interrupts.len(), 2);
    assert!(paused.interrupts.iter().all(|i| phase(i) == "after"));
    assert_eq!(paused.state, 0);
    assert_eq!(runs.load(AtomicOrdering::SeqCst), 2);

    let resumed = graph.retry("par").await.unwrap();
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 30);
    assert_eq!(runs.load(AtomicOrdering::SeqCst), 2);
}

#[test]
fn interrupt_selectors_must_name_real_nodes() {
    let err = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move {
            Ok(NodeResult::Update(s))
        })
        .set_entry("a")
        .set_finish("a")
        .interrupt_before(["ghost"])
        .compile()
        .unwrap_err();
    assert!(
        matches!(&err, TinyAgentsError::MissingNode(n) if n == "ghost"),
        "got {err:?}"
    );
}

// ── Interrupt::response_schema ───────────────────────────────────────────

/// The schema every test below resumes against.
fn approval_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["approved"],
        "properties": { "approved": { "type": "boolean" } }
    })
}

/// `approve` pauses with a schema-bearing interrupt; once resumed it commits
/// `+1` when approved, `-1` otherwise.
fn approval_graph() -> CompiledGraph<i32, i32> {
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("approve", |s, ctx: NodeContext| async move {
            match ctx.resume {
                Some(value) => {
                    let approved = value["approved"].as_bool().unwrap_or(false);
                    Ok(NodeResult::Update(if approved { s + 1 } else { s - 1 }))
                }
                None => Ok(NodeResult::Interrupt(
                    Interrupt::new("approve", json!({ "ask": "approve?" }))
                        .with_response_schema(approval_schema()),
                )),
            }
        })
        .set_entry("approve")
        .set_finish("approve")
        .compile()
        .unwrap()
        .with_checkpointer(memory())
}

#[tokio::test]
async fn response_schema_accepts_a_conforming_resume_value() {
    let graph = approval_graph();
    let paused = graph.run_with_thread("ok", 10).await.unwrap();
    assert_eq!(
        paused.interrupts[0].response_schema,
        Some(approval_schema())
    );
    let resumed = graph
        .resume("ok", Command::resume(json!({ "approved": true })))
        .await
        .unwrap();
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, 11);
}

#[tokio::test]
async fn response_schema_rejects_bad_values_before_touching_the_checkpoint() {
    let graph = approval_graph();
    let paused = graph.run_with_thread("bad", 10).await.unwrap();
    let before = graph.get_state_history("bad", None).await.unwrap();
    let latest_id = before[0].config.checkpoint_id.clone();

    for bad in [json!({ "approved": "yes" }), json!({}), json!("approve")] {
        let err = graph
            .resume("bad", Command::resume(bad.clone()))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, TinyAgentsError::Validation(msg) if msg.contains("response_schema")),
            "{bad}: got {err:?}"
        );
    }

    // Fail-closed: no new checkpoint, same latest id, same pending interrupt.
    let after = graph.get_state_history("bad", None).await.unwrap();
    assert_eq!(after.len(), before.len());
    assert_eq!(after[0].config.checkpoint_id, latest_id);
    assert_eq!(after[0].values, 10);
    assert_eq!(after[0].pending_interrupts, paused.interrupts);
    assert!(after[0].metadata.has_interrupts);

    // A conforming value still resumes the untouched checkpoint.
    let resumed = graph
        .resume("bad", Command::resume(json!({ "approved": false })))
        .await
        .unwrap();
    assert_eq!(resumed.state, 9);
    assert_eq!(
        graph.get_state_history("bad", None).await.unwrap().len(),
        before.len() + 1
    );
}

#[tokio::test]
async fn response_schema_validates_per_task_resume_values() {
    let graph = approval_graph();
    let paused = graph.run_with_thread("by-task", 0).await.unwrap();
    let task = paused.interrupts[0].task_id.clone().unwrap();

    let err = graph
        .resume(
            "by-task",
            Command::resume_tasks([(task.clone(), json!({ "approved": 1 }))]),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::Validation(_)), "got {err:?}");

    let resumed = graph
        .resume(
            "by-task",
            Command::resume_tasks([(task, json!({ "approved": true }))]),
        )
        .await
        .unwrap();
    assert_eq!(resumed.state, 1);
}

#[tokio::test]
async fn retry_without_a_resume_value_skips_schema_validation() {
    // A bare `retry` delivers no value, so there is nothing to validate;
    // the node simply pauses again.
    let graph = approval_graph();
    graph.run_with_thread("retry", 0).await.unwrap();
    let again = graph.retry("retry").await.unwrap();
    assert!(again.is_interrupted());
}
