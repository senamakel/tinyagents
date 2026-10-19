use super::*;
use crate::context::RunConfig;
use crate::runtime::AgentHarness;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{future::Future, pin::Pin, task::Poll};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};

struct Lease {
    renewals: AtomicUsize,
    lose_after: usize,
}

struct BlockingModel {
    entered: Arc<tokio::sync::Notify>,
    released: Arc<tokio::sync::Notify>,
    dropped: Arc<AtomicBool>,
}

struct ModelCallDrop(Arc<AtomicBool>);

async fn poll_pending<F: Future>(mut future: Pin<&mut F>) {
    std::future::poll_fn(|context| {
        assert!(future.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
}

impl Drop for ModelCallDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl ChatModel<()> for BlockingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let _drop = ModelCallDrop(Arc::clone(&self.dropped));
        self.entered.notify_one();
        self.released.notified().await;
        Ok(ModelResponse::assistant("done"))
    }
}

#[async_trait]
impl RunLeaseHeartbeat for Lease {
    fn lease_duration(&self) -> Duration {
        Duration::from_secs(120)
    }

    async fn renew(&self) -> Result<bool> {
        let count = self.renewals.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(count < self.lose_after)
    }
}

#[tokio::test]
async fn pending_work_heartbeats_and_stops_when_lease_is_lost() {
    tokio::time::pause();
    let lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: 2,
    };
    let dropped = Arc::new(AtomicBool::new(false));
    struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let flag = DropFlag(Arc::clone(&dropped));
    let pending = async move {
        let _flag = flag;
        std::future::pending::<()>().await;
    };
    let run = run_with_lease(pending, &lease, Duration::from_secs(10));
    tokio::pin!(run);
    tokio::select! {
        _ = tokio::task::yield_now() => {},
        result = &mut run => panic!("pending work ended unexpectedly: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(10)).await;
    poll_pending(run.as_mut()).await;
    tokio::time::advance(Duration::from_secs(10)).await;
    poll_pending(run.as_mut()).await;
    let result = run.await;
    assert!(result.is_err());
    assert_eq!(lease.renewals.load(Ordering::SeqCst), 2);
    assert!(
        dropped.load(Ordering::SeqCst),
        "lease loss drops in-flight work"
    );
}

#[tokio::test]
async fn completed_or_cancelled_wrapper_does_not_leave_a_heartbeat_behind() {
    tokio::time::pause();
    let lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: usize::MAX,
    };
    assert_eq!(
        run_with_lease(async { 7 }, &lease, Duration::from_secs(5))
            .await
            .unwrap(),
        7
    );
    let before_cancel = lease.renewals.load(Ordering::SeqCst);
    let entered = Arc::new(tokio::sync::Notify::new());
    let run_entered = Arc::clone(&entered);
    let pending = async move {
        run_entered.notify_one();
        std::future::pending::<()>().await;
    };
    let mut cancelled = Box::pin(run_with_lease(pending, &lease, Duration::from_secs(5)));
    tokio::select! {
        _ = entered.notified() => {},
        result = &mut cancelled => panic!("pending work ended unexpectedly: {result:?}"),
    }
    drop(cancelled);
    tokio::time::advance(Duration::from_secs(20)).await;
    tokio::task::yield_now().await;
    assert_eq!(lease.renewals.load(Ordering::SeqCst), before_cancel + 1);
}

struct StalledLease(AtomicUsize);

struct NearExpiryLease(AtomicUsize);

struct NeverLease;

#[async_trait]
impl RunLeaseHeartbeat for NeverLease {
    fn lease_duration(&self) -> Duration {
        Duration::from_secs(60)
    }

    async fn renew(&self) -> Result<bool> {
        std::future::pending().await
    }
}

#[async_trait]
impl RunLeaseHeartbeat for StalledLease {
    fn lease_duration(&self) -> Duration {
        Duration::from_secs(60)
    }

    async fn renew(&self) -> Result<bool> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(true)
        } else {
            std::future::pending().await
        }
    }
}

#[async_trait]
impl RunLeaseHeartbeat for NearExpiryLease {
    fn lease_duration(&self) -> Duration {
        Duration::from_secs(31)
    }

    async fn renew(&self) -> Result<bool> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_secs(9)).await;
            Ok(true)
        } else {
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn stalled_renewal_stops_near_expiry_before_the_granted_lease_expires() {
    tokio::time::pause();
    let lease = NearExpiryLease(AtomicUsize::new(0));
    let started_at = tokio::time::Instant::now();
    let entered = Arc::new(tokio::sync::Notify::new());
    let work_entered = Arc::clone(&entered);
    let dropped = Arc::new(AtomicBool::new(false));
    struct DropOnCancel(Arc<AtomicBool>);
    impl Drop for DropOnCancel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let work_dropped = Arc::clone(&dropped);
    let work = async move {
        let _guard = DropOnCancel(work_dropped);
        work_entered.notify_one();
        std::future::pending::<()>().await;
    };
    let mut run = Box::pin(run_with_lease(work, &lease, Duration::from_secs(10)));
    tokio::select! {
        _ = entered.notified() => {},
        result = &mut run => panic!("pending work ended unexpectedly: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(10)).await;
    poll_pending(run.as_mut()).await;
    tokio::time::advance(Duration::from_secs(10)).await;
    poll_pending(run.as_mut()).await;

    assert!(run.await.is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert!(tokio::time::Instant::now().duration_since(started_at) < lease.lease_duration());
}

#[tokio::test]
async fn stalled_renewal_is_bounded_and_cancels_in_flight_work() {
    tokio::time::pause();
    let entered = Arc::new(tokio::sync::Notify::new());
    let task_entered = Arc::clone(&entered);
    let dropped = Arc::new(AtomicBool::new(false));
    struct DropOnCancel(Arc<AtomicBool>);
    impl Drop for DropOnCancel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped_by_task = Arc::clone(&dropped);
    let task = async move {
        let _guard = DropOnCancel(dropped_by_task);
        task_entered.notify_one();
        std::future::pending::<()>().await;
    };
    let lease = StalledLease(AtomicUsize::new(0));
    let mut run = Box::pin(run_with_lease(task, &lease, Duration::from_secs(5)));
    tokio::select! {
        _ = entered.notified() => {},
        result = &mut run => panic!("pending work ended unexpectedly: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(5)).await;
    poll_pending(run.as_mut()).await;
    tokio::time::advance(Duration::from_secs(5)).await;
    poll_pending(run.as_mut()).await;
    assert!(run.await.is_err());
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn invalid_timing_or_initial_lease_loss_never_starts_work() {
    tokio::time::pause();
    let invalid_lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: usize::MAX,
    };
    let started = Arc::new(AtomicBool::new(false));
    let work_started = Arc::clone(&started);
    let invalid = run_with_lease(
        async move {
            work_started.store(true, Ordering::SeqCst);
        },
        &invalid_lease,
        Duration::from_secs(40),
    )
    .await;
    assert!(invalid.is_err());
    assert_eq!(invalid_lease.renewals.load(Ordering::SeqCst), 0);
    assert!(!started.load(Ordering::SeqCst));

    let work_started = Arc::clone(&started);
    let stalled = async move {
        work_started.store(true, Ordering::SeqCst);
    };
    let mut initial_stall = Box::pin(run_with_lease(stalled, &NeverLease, Duration::from_secs(5)));
    tokio::select! {
        _ = tokio::task::yield_now() => {},
        result = &mut initial_stall => panic!("stalled initial heartbeat unexpectedly ended: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(5)).await;
    tokio::task::yield_now().await;
    assert!(initial_stall.await.is_err());
    assert!(!started.load(Ordering::SeqCst));

    let expired_lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: 0,
    };
    let work_started = Arc::clone(&started);
    let lost = run_with_lease(
        async move {
            work_started.store(true, Ordering::SeqCst);
        },
        &expired_lease,
        Duration::from_secs(5),
    )
    .await;
    assert!(lost.is_err());
    assert_eq!(expired_lease.renewals.load(Ordering::SeqCst), 1);
    assert!(!started.load(Ordering::SeqCst));
}

#[tokio::test]
async fn agent_harness_renews_during_pending_model_call_and_cancels_on_lease_loss() {
    tokio::time::pause();
    let entered = Arc::new(tokio::sync::Notify::new());
    let released = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(AtomicBool::new(false));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(BlockingModel {
            entered: Arc::clone(&entered),
            released,
            dropped: Arc::clone(&dropped),
        }),
    );
    let lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: 2,
    };
    let mut run = Box::pin(harness.invoke_with_lease(
        &(),
        (),
        RunConfig::new("run"),
        vec![Message::user("continue")],
        &lease,
        Duration::from_secs(10),
    ));
    tokio::select! {
        _ = entered.notified() => {},
        result = &mut run => panic!("model run ended before its await: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_secs(10)).await;
    assert!(run.await.is_err());
    assert_eq!(lease.renewals.load(Ordering::SeqCst), 2);
    assert!(
        dropped.load(Ordering::SeqCst),
        "lease loss cancels model work"
    );
}

#[tokio::test]
async fn successful_agent_run_stops_renewing_after_model_completion() {
    tokio::time::pause();
    let entered = Arc::new(tokio::sync::Notify::new());
    let released = Arc::new(tokio::sync::Notify::new());
    let release_model = Arc::clone(&released);
    let dropped = Arc::new(AtomicBool::new(false));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model(
        "mock",
        Arc::new(BlockingModel {
            entered: Arc::clone(&entered),
            released,
            dropped: Arc::clone(&dropped),
        }),
    );
    let lease = Lease {
        renewals: AtomicUsize::new(0),
        lose_after: usize::MAX,
    };
    let mut run = Box::pin(harness.invoke_with_lease(
        &(),
        (),
        RunConfig::new("run"),
        vec![Message::user("continue")],
        &lease,
        Duration::from_secs(10),
    ));
    tokio::select! {
        _ = entered.notified() => {},
        result = &mut run => panic!("model run ended before its await: {result:?}"),
    }
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(lease.renewals.load(Ordering::SeqCst), 2);
    release_model.notify_one();
    assert!(run.await.is_ok());
    tokio::time::advance(Duration::from_secs(30)).await;
    tokio::task::yield_now().await;
    assert_eq!(lease.renewals.load(Ordering::SeqCst), 2);
    assert!(dropped.load(Ordering::SeqCst));
}
