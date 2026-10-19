use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Lease {
    renewals: AtomicUsize,
    lose_after: usize,
}

#[async_trait]
impl RunLeaseHeartbeat for Lease {
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
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_secs(10)).await;
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
    let cancelled = run_with_lease(std::future::pending::<()>(), &lease, Duration::from_secs(5));
    tokio::pin!(cancelled);
    drop(cancelled);
    tokio::time::advance(Duration::from_secs(20)).await;
    assert_eq!(lease.renewals.load(Ordering::SeqCst), 0);
}
