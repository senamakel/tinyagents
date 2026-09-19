//! Unit tests for per-node execution policy (Unit A): per-node
//! retry/timeout overrides, idle timeouts with heartbeats, task caching,
//! `on_error` recovery, and real `defer` scheduling.

use super::*;
use crate::builder::{GraphBuilder, NodeContext, NodePolicy};
use crate::command::NodeResult;
use tinyagents_harness::retry::RetryPolicy;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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
