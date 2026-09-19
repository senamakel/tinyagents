//! Unit tests for per-node execution policy (Unit A): per-node
//! retry/timeout overrides, idle timeouts with heartbeats, task caching,
//! `on_error` recovery, and real `defer` scheduling.

use super::*;
use crate::builder::{GraphBuilder, NodeContext, NodePolicy};
use crate::command::NodeResult;
use tinyagents_harness::retry::RetryPolicy;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

/// A single-node graph whose handler fails (with a retryable model error)
/// the first `fail_times` invocations, then succeeds with `+1`.
fn flaky_builder(fail_times: usize, attempts: Arc<AtomicUsize>) -> GraphBuilder<i32, i32> {
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
}

// ── A.1: per-node retry / timeout overrides ──────────────────────────────────

/// A node's own `NodePolicy::retry` wins over the graph-wide
/// `with_node_retry` policy: the node retries per its own attempt cap even
/// though the graph-wide policy would have given up earlier.
#[tokio::test]
async fn per_node_retry_policy_overrides_graph_wide_retry() {
    let attempts = Arc::new(AtomicUsize::new(0));
    // Fails 3 times; graph-wide budget is 2 attempts (would fail), per-node
    // budget is 5 attempts (recovers on the 4th).
    let graph = flaky_builder(3, attempts.clone())
        .with_node_policy(
            "flaky",
            NodePolicy {
                retry: Some(
                    RetryPolicy::default()
                        .with_max_attempts(5)
                        .with_backoff_sleep(false),
                ),
                ..NodePolicy::default()
            },
        )
        .compile()
        .unwrap()
        .with_node_retry(
            RetryPolicy::default()
                .with_max_attempts(2)
                .with_backoff_sleep(false),
        );

    let run = graph.run(10).await.unwrap();
    assert_eq!(run.state, 11);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 4, "1 try + 3 retries");
}

/// With no graph-wide retry at all, a per-node retry policy still applies.
#[tokio::test]
async fn per_node_retry_policy_applies_without_graph_wide_retry() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = flaky_builder(1, attempts.clone())
        .with_node_policy(
            "flaky",
            NodePolicy {
                retry: Some(
                    RetryPolicy::default()
                        .with_max_attempts(2)
                        .with_backoff_sleep(false),
                ),
                ..NodePolicy::default()
            },
        )
        .compile()
        .unwrap();

    let run = graph.run(10).await.unwrap();
    assert_eq!(run.state, 11);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
}

/// A node's own `NodePolicy::timeout` (shorter than the graph-wide
/// `with_node_timeout`) is what bounds it: the handler times out at the
/// per-node value even though the graph default would have let it finish.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_node_timeout_overrides_graph_wide_timeout() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .with_node_timeout(Duration::from_secs(5))
        .add_node("slow", |s: i32, _c: NodeContext| async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            Ok(NodeResult::Update(s))
        })
        .with_node_policy(
            "slow",
            NodePolicy::default().with_timeout(Duration::from_millis(20)),
        )
        .set_entry("slow")
        .set_finish("slow")
        .compile()
        .unwrap();

    let started = std::time::Instant::now();
    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "the per-node 20ms timeout fired, not the 5s graph-wide one"
    );
}

/// `set_node_defaults` is the middle precedence layer: a node with no
/// per-node timeout uses the defaults' timeout over the legacy graph-wide
/// `with_node_timeout`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_defaults_timeout_beats_legacy_graph_wide_timeout() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .with_node_timeout(Duration::from_millis(20))
        .set_node_defaults(NodePolicy::default().with_timeout(Duration::from_secs(5)))
        .add_node("slow", |s: i32, _c: NodeContext| async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            Ok(NodeResult::Update(s + 1))
        })
        .set_entry("slow")
        .set_finish("slow")
        .compile()
        .unwrap();

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 1, "the 5s default timeout let the 60ms node finish");
}
