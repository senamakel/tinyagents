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
    /// Duration granted by each successful renewal.
    fn lease_duration(&self) -> Duration;

    /// Extends the current lease if this owner and fencing epoch still own it.
    /// Returns `false` when ownership was lost or the lease already expired.
    async fn renew(&self) -> Result<bool>;
}

/// Runs a future while periodically renewing its lease.
///
/// A successful renewal runs before the work future is first polled. The work
/// future is dropped immediately when a later renewal fails, and the renewal
/// loop is dropped when the future completes or this function is cancelled.
/// Each renewal is bounded by one heartbeat interval. Since a granted expiry
/// is calculated before the async persistence write, three intervals must fit
/// strictly inside the adapter-reported lease duration: initial renewal,
/// cadence wait, and the next bounded renewal.
pub async fn run_with_lease<F, T>(
    future: F,
    lease: &dyn RunLeaseHeartbeat,
    heartbeat_every: Duration,
) -> Result<T>
where
    F: Future<Output = T>,
{
    if heartbeat_every.is_zero()
        || lease.lease_duration().is_zero()
        || heartbeat_every
            .checked_mul(3)
            .is_none_or(|budget| budget >= lease.lease_duration())
    {
        return Err(TinyAgentsError::Validation(
            "run lease duration must exceed three positive heartbeat intervals".into(),
        ));
    }
    renew_once(lease, heartbeat_every).await?;
    tokio::pin!(future);
    let heartbeat = async {
        loop {
            tokio::time::sleep(heartbeat_every).await;
            renew_once(lease, heartbeat_every).await?;
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

async fn renew_once(lease: &dyn RunLeaseHeartbeat, budget: Duration) -> Result<()> {
    let renewed = tokio::time::timeout(budget, lease.renew())
        .await
        .map_err(|_| TinyAgentsError::Storage("run lease heartbeat timed out".into()))??;
    if renewed {
        Ok(())
    } else {
        Err(TinyAgentsError::Storage(
            "run lease ownership was lost".into(),
        ))
    }
}

#[cfg(test)]
#[path = "lease_tests.rs"]
mod tests;
