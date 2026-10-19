//! Lease lifecycle helpers for long-running harness work.

use std::future::Future;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::{Result, TinyAgentsError};

/// A host-provided lease renewal operation.
///
/// The harness does not own persistence or lease policy. A session or host
/// adapter implements this port using an owner/epoch conditional update.
#[async_trait]
pub trait RunLeaseHeartbeat: Send + Sync {
    /// Extends the current lease if this owner and fencing epoch still own it.
    /// Returns `false` when ownership was lost or the lease already expired.
    async fn renew(&self) -> Result<bool>;
}

/// Runs a future while periodically renewing its lease.
///
/// The future is dropped immediately when renewal fails, and the renewal loop
/// is dropped when the future completes or this function is cancelled. This
/// covers pending model and tool calls; status/progress writes are not needed
/// to keep ownership alive. A renewal operation is bounded by one heartbeat
/// interval; callers must choose an interval whose two-interval worst case is
/// shorter than the lease remaining at the start of a renewal.
pub async fn run_with_lease<F, T>(
    future: F,
    lease: &dyn RunLeaseHeartbeat,
    heartbeat_every: Duration,
) -> Result<T>
where
    F: Future<Output = T>,
{
    if heartbeat_every.is_zero() {
        return Err(TinyAgentsError::Validation(
            "run lease heartbeat interval must be positive".into(),
        ));
    }
    tokio::pin!(future);
    let heartbeat = async {
        loop {
            tokio::time::sleep(heartbeat_every).await;
            let renewed = tokio::time::timeout(heartbeat_every, lease.renew())
                .await
                .map_err(|_| {
                    TinyAgentsError::Storage("run lease heartbeat timed out".into())
                })??;
            if !renewed {
                return Err(TinyAgentsError::Storage(
                    "run lease ownership was lost".into(),
                ));
            }
        }
        #[allow(unreachable_code)]
        Ok::<(), TinyAgentsError>(())
    };
    tokio::pin!(heartbeat);
    tokio::select! {
        result = &mut future => Ok(result),
        result = &mut heartbeat => result.map(|()| unreachable!("heartbeat loop is infinite")),
    }
}

#[cfg(test)]
#[path = "lease_tests.rs"]
mod tests;
