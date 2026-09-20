//! Tests for member worker graph execution.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tinyagents_graph::stream::CollectingSink;

use super::*;

#[tokio::test]
async fn member_graph_routes_completed_and_failed_workers() {
    let complete = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let complete_flag = complete.clone();
    let failed_flag = failed.clone();
    run_member_graph(
        None,
        || async {
            Ok(MemberOutcome::Completed {
                output: "done".into(),
            })
        },
        move |output| {
            let complete = complete_flag.clone();
            async move {
                assert_eq!(output, "done");
                complete.store(true, Ordering::SeqCst);
                Ok(())
            }
        },
        move |_| {
            let failed = failed_flag.clone();
            async move {
                failed.store(true, Ordering::SeqCst);
                Ok(())
            }
        },
    )
    .await
    .unwrap();
    assert!(complete.load(Ordering::SeqCst));
    assert!(!failed.load(Ordering::SeqCst));

    let failed = Arc::new(AtomicBool::new(false));
    let failed_flag = failed.clone();
    run_member_graph(
        None,
        || async {
            Ok(MemberOutcome::Failed {
                reason: "boom".into(),
            })
        },
        |_| async { Ok(()) },
        move |reason| {
            let failed = failed_flag.clone();
            async move {
                assert_eq!(reason, "boom");
                failed.store(true, Ordering::SeqCst);
                Ok(())
            }
        },
    )
    .await
    .unwrap();
    assert!(failed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn worker_engine_errors_propagate() {
    let result = run_member_graph(
        None,
        || async { Err(anyhow::anyhow!("worker unavailable")) },
        |_| async { Ok(()) },
        |_| async { Ok(()) },
    )
    .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn injected_event_sink_observes_member_graph_lifecycle() {
    let sink = Arc::new(CollectingSink::new());
    run_member_graph(
        Some(sink.clone()),
        || async {
            Ok(MemberOutcome::Completed {
                output: "done".into(),
            })
        },
        |_| async { Ok(()) },
        |_| async { Ok(()) },
    )
    .await
    .unwrap();

    assert!(
        !sink.is_empty(),
        "an injected sink must receive the member graph lifecycle"
    );
}
