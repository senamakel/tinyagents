//! Unit tests for per-node execution policy (Unit A): per-node
//! retry/timeout overrides, idle timeouts with heartbeats, task caching,
//! `on_error` recovery, and real `defer` scheduling.

use super::*;
use crate::builder::{GraphBuilder, NodeCachePolicy, NodeContext, NodePolicy};
use crate::cache::InMemoryTaskCache;
use crate::command::{Command, NodeResult, RouteTarget, Send};
use crate::reducer::ClosureStateReducer;
use crate::stream::{CollectingSink, GraphEvent};
use serde_json::json;
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
    assert_eq!(
        attempts.load(AtomicOrdering::SeqCst),
        4,
        "1 try + 3 retries"
    );
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
    assert_eq!(
        run.state, 1,
        "the 5s default timeout let the 60ms node finish"
    );
}

// ── A.1: idle timeout + heartbeat ────────────────────────────────────────────

/// A handler that heartbeats more often than its `idle_timeout` survives
/// well past what a flat timeout of that same duration would have allowed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heartbeat_keeps_idle_timeout_from_firing() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("worker", |s: i32, ctx: NodeContext| async move {
            // Runs for 200ms total, heartbeating every 20ms — far inside the
            // 60ms idle window, but 3x longer than a flat 60ms timeout.
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(20)).await;
                ctx.heartbeat();
            }
            Ok(NodeResult::Update(s + 1))
        })
        .with_node_policy(
            "worker",
            NodePolicy::default().with_idle_timeout(Duration::from_millis(60)),
        )
        .set_entry("worker")
        .set_finish("worker")
        .compile()
        .unwrap();

    let run = graph.run(0).await.unwrap();
    assert_eq!(run.state, 1);
}

/// With only `idle_timeout` set and no heartbeat ever sent, the node times
/// out at (approximately) exactly the idle duration after it starts — the
/// idle timeout degrades to a flat timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_timeout_without_heartbeats_is_a_flat_timeout() {
    let idle = Duration::from_millis(80);
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("silent", |s: i32, _c: NodeContext| async move {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(NodeResult::Update(s))
        })
        .with_node_policy("silent", NodePolicy::default().with_idle_timeout(idle))
        .set_entry("silent")
        .set_finish("silent")
        .compile()
        .unwrap();

    let started = std::time::Instant::now();
    let err = graph.run(0).await.unwrap_err();
    let elapsed = started.elapsed();
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert!(
        elapsed >= idle && elapsed < idle + Duration::from_millis(150),
        "expected the idle timeout to fire in [{idle:?}, {idle:?} + slop), got {elapsed:?}"
    );
}

/// A flat `timeout` and an `idle_timeout` on the same node are independent
/// ceilings: a handler that heartbeats forever is still cut off by the
/// flat timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flat_timeout_still_bounds_a_heartbeating_handler() {
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("chatty", |s: i32, ctx: NodeContext| async move {
            loop {
                tokio::time::sleep(Duration::from_millis(10)).await;
                ctx.heartbeat();
                if false {
                    break;
                }
            }
            #[allow(unreachable_code)]
            Ok(NodeResult::Update(s))
        })
        .with_node_policy(
            "chatty",
            NodePolicy::default()
                .with_idle_timeout(Duration::from_millis(100))
                .with_timeout(Duration::from_millis(60)),
        )
        .set_entry("chatty")
        .set_finish("chatty")
        .compile()
        .unwrap();

    let started = std::time::Instant::now();
    let err = graph.run(0).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Timeout(_)), "got {err:?}");
    assert!(started.elapsed() < Duration::from_millis(300));
}

// ── A.2: task cache ──────────────────────────────────────────────────────────

/// A single-node graph counting handler invocations, cached on the input
/// state's value.
fn counting_cached_graph(
    calls: Arc<AtomicUsize>,
    ttl: Option<Duration>,
) -> (CompiledGraph<i32, i32>, Arc<InMemoryTaskCache>) {
    let cache = Arc::new(InMemoryTaskCache::new());
    let mut policy = NodeCachePolicy::new(|s: &i32, _arg| format!("state={s}"));
    if let Some(ttl) = ttl {
        policy = policy.with_ttl(ttl);
    }
    let graph = GraphBuilder::<i32, i32>::overwrite()
        .add_node("compute", move |s: i32, _c: NodeContext| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(NodeResult::Update(s * 10))
            }
        })
        .set_entry("compute")
        .set_finish("compute")
        .compile()
        .unwrap()
        .with_task_cache(cache.clone())
        .with_cached_node("compute", policy);
    (graph, cache)
}

/// A second run with the same cache key skips the handler entirely, replays
/// the cached update, and reports the hit as `TaskCompleted { cached: true }`.
#[tokio::test]
async fn cache_hit_skips_handler_and_emits_cached_task_completed() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (graph, _cache) = counting_cached_graph(calls.clone(), None);
    let sink = Arc::new(CollectingSink::new());
    let graph = graph.with_event_sink(sink.clone());

    let first = graph.run(4).await.unwrap();
    assert_eq!(first.state, 40);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    assert!(
        !sink
            .events()
            .iter()
            .any(|e| matches!(e, GraphEvent::TaskCompleted { cached: true, .. })),
        "the first run was a miss"
    );

    let second = graph.run(4).await.unwrap();
    assert_eq!(second.state, 40, "the cached update was replayed");
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        1,
        "the handler was not invoked on the cache hit"
    );
    assert_eq!(
        second.visited,
        vec![NodeId::from("compute")],
        "a cached node still counts as visited"
    );
    let hit = sink.events().into_iter().find(|e| {
        matches!(
            e,
            GraphEvent::TaskCompleted {
                cached: true,
                step: 1,
                ..
            }
        )
    });
    assert!(hit.is_some(), "expected a cached TaskCompleted event");

    // A different key is a miss again.
    let third = graph.run(5).await.unwrap();
    assert_eq!(third.state, 50);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
}

/// Once an entry's TTL elapses the handler runs again (and repopulates).
#[tokio::test]
async fn cache_ttl_expiry_reruns_the_handler() {
    let calls = Arc::new(AtomicUsize::new(0));
    let (graph, _cache) = counting_cached_graph(calls.clone(), Some(Duration::from_millis(40)));

    graph.run(1).await.unwrap();
    graph.run(1).await.unwrap();
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        1,
        "second run was a hit"
    );

    tokio::time::sleep(Duration::from_millis(80)).await;
    let run = graph.run(1).await.unwrap();
    assert_eq!(run.state, 10);
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        2,
        "the expired entry forced a re-run"
    );
}

/// The cache key function receives each activation's `send_arg`, so a
/// `Send` fan-out of one node keys (and hits) per argument.
#[tokio::test]
async fn cache_key_receives_send_arg_per_fanout_activation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen_args = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let cache = Arc::new(InMemoryTaskCache::new());
    let key_args = seen_args.clone();
    let graph = GraphBuilder::<Vec<String>, Vec<String>>::new()
        .set_reducer(ClosureStateReducer::new(
            |mut s: Vec<String>, u: Vec<String>| {
                s.extend(u);
                Ok(s)
            },
        ))
        .add_node("fan", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command {
                update: None,
                goto: vec![
                    RouteTarget::Send(Send::new("work", json!("a"))),
                    RouteTarget::Send(Send::new("work", json!("b"))),
                ],
                resume: None,
                resume_by_task: Default::default(),
            }))
        })
        .add_node("work", {
            let calls = calls.clone();
            move |_s, ctx: NodeContext| {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, AtomicOrdering::SeqCst);
                    let arg = ctx.send_arg.and_then(|v| v.as_str().map(String::from));
                    Ok(NodeResult::Update(vec![format!(
                        "work:{}",
                        arg.unwrap_or_default()
                    )]))
                }
            }
        })
        .mark_command_routing("fan")
        .set_entry("fan")
        .set_finish("work")
        .compile()
        .unwrap()
        .with_task_cache(cache.clone())
        .with_cached_node(
            "work",
            NodeCachePolicy::new(move |_s: &Vec<String>, arg: Option<&serde_json::Value>| {
                let arg = arg.map(|v| v.to_string()).unwrap_or_default();
                key_args.lock().unwrap().push(arg.clone());
                format!("arg={arg}")
            }),
        );

    let first = graph.run(vec![]).await.unwrap();
    let mut got = first.state.clone();
    got.sort();
    assert_eq!(got, vec!["work:a", "work:b"]);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    {
        let mut args = seen_args.lock().unwrap();
        args.sort();
        assert_eq!(*args, vec!["\"a\"", "\"b\""], "key fn saw each send_arg");
    }

    // Second run: both fan-out activations are cache hits, the fan node
    // itself (uncached) still runs.
    let second = graph.run(vec![]).await.unwrap();
    let mut got = second.state.clone();
    got.sort();
    assert_eq!(
        got,
        vec!["work:a", "work:b"],
        "cached updates were replayed"
    );
    assert_eq!(
        calls.load(AtomicOrdering::SeqCst),
        2,
        "neither fan-out activation invoked the handler"
    );
}

// ── A.3: on_error recovery ────────────────────────────────────────────────

/// `on_error` is consulted only after the retry budget is exhausted (or the
/// error is non-retryable); returning `Some(command)` recovers the node with
/// that command's update instead of failing the run.
#[tokio::test]
async fn on_error_recovers_after_retries_are_exhausted() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = flaky_builder(usize::MAX, attempts.clone())
        .with_node_policy(
            "flaky",
            NodePolicy {
                retry: Some(
                    RetryPolicy::default()
                        .with_max_attempts(2)
                        .with_backoff_sleep(false),
                ),
                on_error: Some(Arc::new(|state: &i32, _err: &TinyAgentsError| {
                    Some(Command {
                        update: Some(state + 100),
                        goto: vec![],
                        resume: None,
                        resume_by_task: Default::default(),
                    })
                })),
                ..NodePolicy::default()
            },
        )
        .compile()
        .unwrap();

    let run = graph.run(10).await.unwrap();
    assert_eq!(run.state, 110, "on_error's command update was applied");
    assert_eq!(
        attempts.load(AtomicOrdering::SeqCst),
        2,
        "1 try + 1 retry, then on_error recovered instead of a 3rd attempt"
    );
}

/// `on_error` returning `None` falls through to the ordinary escalation —
/// the run still fails with the underlying error.
#[tokio::test]
async fn on_error_none_falls_through_to_the_original_error() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let graph = flaky_builder(usize::MAX, attempts.clone())
        .with_node_policy(
            "flaky",
            NodePolicy {
                on_error: Some(Arc::new(|_state: &i32, _err: &TinyAgentsError| None)),
                ..NodePolicy::default()
            },
        )
        .compile()
        .unwrap();

    let err = graph.run(10).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::Model(_)), "got {err:?}");
    assert_eq!(
        attempts.load(AtomicOrdering::SeqCst),
        1,
        "no retry policy set"
    );
}

// ── A.4: real defer ──────────────────────────────────────────────────────

/// A deferred node fanned out to alongside a non-deferred sibling is held
/// back: it runs in a later superstep than its sibling, once the sibling's
/// own successor leaves nothing non-deferred in the frontier — not
/// concurrently with it, which is what a purely cosmetic `mark_deferred`
/// marker (metadata-only, no scheduling effect) would have produced.
#[tokio::test]
async fn deferred_node_runs_only_once_the_frontier_has_no_other_work() {
    let graph = GraphBuilder::<Vec<String>, Vec<String>>::new()
        .set_reducer(ClosureStateReducer::new(
            |mut s: Vec<String>, u: Vec<String>| {
                s.extend(u);
                Ok(s)
            },
        ))
        .add_node("start", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command {
                update: None,
                goto: vec![
                    RouteTarget::Node(NodeId::from("worker")),
                    RouteTarget::Node(NodeId::from("synth")),
                ],
                resume: None,
                resume_by_task: Default::default(),
            }))
        })
        .add_node("worker", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(vec!["worker".to_string()]))
        })
        .add_node("synth", |_s, _c: NodeContext| async move {
            Ok(NodeResult::Update(vec!["synth".to_string()]))
        })
        .mark_command_routing("start")
        .mark_deferred("synth")
        .set_entry("start")
        .set_finish("worker")
        .set_finish("synth")
        .compile()
        .unwrap();

    let sink = Arc::new(CollectingSink::new());
    let graph = graph.with_event_sink(sink.clone());
    let run = graph.run(vec![]).await.unwrap();
    let mut got = run.state.clone();
    got.sort();
    assert_eq!(got, vec!["synth", "worker"], "both branches still ran");

    let started_step = |name: &str| {
        sink.events().into_iter().find_map(|e| match e {
            GraphEvent::NodeStarted { node, step } if node.as_str() == name => Some(step),
            _ => None,
        })
    };
    let worker_step = started_step("worker").expect("worker started");
    let synth_step = started_step("synth").expect("synth started");
    assert!(
        synth_step > worker_step,
        "the deferred node must run in a later superstep than its \
         non-deferred sibling, got worker={worker_step} synth={synth_step}"
    );
}
