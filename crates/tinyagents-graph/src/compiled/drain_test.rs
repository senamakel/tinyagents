//! Graceful drain (`RunOptions::drain`, `DrainSignal`/`DrainHandle`): the
//! superstep in flight finishes and commits, the next step's activations are
//! checkpointed instead of run, and the run reports `Drained` — resumable
//! to the same final state an undrained run reaches.

use super::*;
use crate::builder::{GraphBuilder, NodeContext};
use crate::checkpoint::InMemoryCheckpointer;
use crate::command::NodeResult;
use crate::stream::{CollectingSink, GraphEvent};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use tinyagents_harness::ids::ExecutionStatus;

/// Three sequential supersteps `a (+1) -> b (+10) -> c (+100)`, each node
/// counted; `a` raises `drain` (when given) from inside its handler, i.e.
/// while step 1 is in flight.
fn chain(counts: [Arc<AtomicUsize>; 3], drain: Option<DrainHandle>) -> CompiledGraph<i32, i32> {
    let [a, b, c] = counts;
    GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", move |s, _c: NodeContext| {
            let a = a.clone();
            let drain = drain.clone();
            async move {
                a.fetch_add(1, AtomicOrdering::SeqCst);
                if let Some(drain) = drain {
                    drain.drain();
                }
                Ok(NodeResult::Update(s + 1))
            }
        })
        .add_node("b", move |s, _c: NodeContext| {
            let b = b.clone();
            async move {
                b.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(s + 10))
            }
        })
        .add_node("c", move |s, _c: NodeContext| {
            let c = c.clone();
            async move {
                c.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(s + 100))
            }
        })
        .add_sequence(["a", "b", "c"])
        .set_entry("a")
        .set_finish("c")
        .compile()
        .unwrap()
}

fn counters() -> [Arc<AtomicUsize>; 3] {
    std::array::from_fn(|_| Arc::new(AtomicUsize::new(0)))
}

#[tokio::test]
async fn drain_finishes_the_in_flight_step_and_stops_before_the_next() {
    let counts = counters();
    let (handle, signal) = DrainSignal::new();
    let sink = Arc::new(CollectingSink::new());
    let graph = chain(counts.clone(), Some(handle.clone()))
        .with_checkpointer(Arc::new(InMemoryCheckpointer::<i32>::new()))
        .with_event_sink(sink.clone());

    let run = graph
        .run_with_thread_options("drain", 0, RunOptions::with_drain(signal))
        .await
        .unwrap();

    assert!(run.drained);
    assert_eq!(run.status.status, ExecutionStatus::Drained);
    assert!(run.status.is_terminal());
    assert!(!run.is_interrupted());
    assert!(handle.is_requested());
    // Step 1 (`a`) completed and committed; `b`/`c` never started.
    assert_eq!(run.state, 1);
    assert_eq!(run.steps, 1);
    assert_eq!(counts[0].load(AtomicOrdering::SeqCst), 1);
    assert_eq!(counts[1].load(AtomicOrdering::SeqCst), 0);
    assert_eq!(counts[2].load(AtomicOrdering::SeqCst), 0);
    assert_eq!(
        run.status
            .active_nodes
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    assert!(run.checkpoint_id.is_some());

    // The terminal event is `RunDrained` (and only that).
    let events = sink.events();
    assert!(events.iter().any(|e| matches!(
        e,
        GraphEvent::RunDrained { run_id, steps: 1 } if *run_id == run.run_id
    )));
    assert!(!events.iter().any(|e| matches!(
        e,
        GraphEvent::RunCompleted { .. }
            | GraphEvent::RunCancelled { .. }
            | GraphEvent::RunFailed { .. }
    )));

    // The checkpoint names `b` as the pending work and is marked drained.
    let snapshot = graph.get_state("drain", None).await.unwrap().unwrap();
    assert_eq!(snapshot.values, 1);
    assert_eq!(
        snapshot
            .next_nodes
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b"]
    );
    assert!(!snapshot.metadata.has_interrupts);
    let tuple = graph
        .checkpointer
        .as_ref()
        .unwrap()
        .get_tuple(snapshot.config)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        tuple.checkpoint.metadata["drained"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn resume_after_drain_reaches_the_same_state_as_an_undrained_run() {
    // Reference: the same graph, no drain.
    let reference = chain(counters(), None).run(0).await.unwrap();
    assert_eq!(reference.status.status, ExecutionStatus::Completed);

    let counts = counters();
    let (handle, signal) = DrainSignal::new();
    let graph = chain(counts.clone(), Some(handle))
        .with_checkpointer(Arc::new(InMemoryCheckpointer::<i32>::new()));
    let drained = graph
        .run_with_thread_options("resume", 0, RunOptions::with_drain(signal))
        .await
        .unwrap();
    assert!(drained.drained);

    // `retry`/`resume` continue from the drained checkpoint like any other
    // pending-tasks checkpoint: `b` then `c`, each once.
    let resumed = graph.retry("resume").await.unwrap();
    assert!(!resumed.drained);
    assert_eq!(resumed.status.status, ExecutionStatus::Completed);
    assert_eq!(resumed.state, reference.state);
    assert_eq!(resumed.state, 111);
    assert_eq!(
        resumed
            .visited
            .iter()
            .map(|n| n.as_str())
            .collect::<Vec<_>>(),
        vec!["b", "c"]
    );
    assert_eq!(counts[0].load(AtomicOrdering::SeqCst), 1);
    assert_eq!(counts[1].load(AtomicOrdering::SeqCst), 1);
    assert_eq!(counts[2].load(AtomicOrdering::SeqCst), 1);
}

#[tokio::test]
async fn drain_requested_before_the_run_starts_runs_nothing() {
    let counts = counters();
    let (handle, signal) = DrainSignal::new();
    handle.drain();
    let graph =
        chain(counts.clone(), None).with_checkpointer(Arc::new(InMemoryCheckpointer::<i32>::new()));
    let run = graph
        .run_with_thread_options("early", 5, RunOptions::with_drain(signal))
        .await
        .unwrap();
    assert!(run.drained);
    assert_eq!(run.steps, 0);
    assert_eq!(run.state, 5);
    assert!(counts.iter().all(|c| c.load(AtomicOrdering::SeqCst) == 0));
    // Still resumable from the entry node.
    let resumed = graph.retry("early").await.unwrap();
    assert_eq!(resumed.state, 116);
}

#[tokio::test]
async fn drain_without_a_thread_reports_drained_without_a_checkpoint() {
    let counts = counters();
    let (handle, signal) = DrainSignal::new();
    let graph = chain(counts.clone(), Some(handle));
    let run = graph
        .run_with_options(0, RunOptions::with_drain(signal))
        .await
        .unwrap();
    assert!(run.drained);
    assert_eq!(run.status.status, ExecutionStatus::Drained);
    assert_eq!(run.state, 1);
    assert!(run.checkpoint_id.is_none());
    assert_eq!(counts[1].load(AtomicOrdering::SeqCst), 0);
}

#[tokio::test]
async fn undrained_runs_report_drained_false_on_every_outcome() {
    let completed = chain(counters(), None).run(0).await.unwrap();
    assert!(!completed.drained);
    let (_handle, signal) = DrainSignal::new();
    let unsignalled = chain(counters(), None)
        .run_with_options(0, RunOptions::with_drain(signal))
        .await
        .unwrap();
    assert!(!unsignalled.drained);
    assert_eq!(unsignalled.status.status, ExecutionStatus::Completed);
}

#[tokio::test]
async fn probe_sequential_stall_keeps_unstarted_siblings_pending() {
    use crate::command::Interrupt;
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("a", |s, _c: NodeContext| async move { Ok(NodeResult::Update(s)) })
        .add_node("b", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Interrupt(Interrupt::new("b", serde_json::json!({}))))
        })
        .add_node("c", |s, _c: NodeContext| async move { Ok(NodeResult::Update(s + 1)) })
        .set_entry("a")
        .add_edge("a", "b")
        .add_edge("a", "c")
        .set_finish("b")
        .set_finish("c")
        .compile()
        .unwrap()
        .with_checkpointer(Arc::new(InMemoryCheckpointer::<i32>::new()));
    let paused = graph.run_with_thread("probe", 0).await.unwrap();
    let snapshot = graph.get_state("probe", None).await.unwrap().unwrap();
    panic!("pending after sequential stall: {:?} (visited {:?})", snapshot.next_nodes, paused.visited);
}
