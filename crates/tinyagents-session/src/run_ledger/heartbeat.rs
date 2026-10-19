//! Harness heartbeat adapter for the session run-ledger lease.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use tinyagents_harness::{RunLeaseHeartbeat, TinyAgentsError};

use super::{AgentRunLeaseToken, RunLedgerStore};

/// Renews one agent-run lease through the session ledger.
///
/// Pass this to `AgentHarness::invoke_with_lease` so model and tool awaits are
/// covered by the same owner/epoch fence as persisted run state.
pub struct AgentRunLeaseHeartbeat {
    store: Arc<dyn RunLedgerStore>,
    token: AgentRunLeaseToken,
    lease_for: Duration,
    clock: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
}

impl AgentRunLeaseHeartbeat {
    /// Creates a heartbeat that reads the current UTC clock on each renewal.
    pub fn new(
        store: Arc<dyn RunLedgerStore>,
        token: AgentRunLeaseToken,
        lease_for: Duration,
    ) -> Self {
        Self::with_clock(store, token, lease_for, Utc::now)
    }

    /// Creates a heartbeat with an injected clock for deterministic callers
    /// and tests.
    pub fn with_clock<F>(
        store: Arc<dyn RunLedgerStore>,
        token: AgentRunLeaseToken,
        lease_for: Duration,
        clock: F,
    ) -> Self
    where
        F: Fn() -> DateTime<Utc> + Send + Sync + 'static,
    {
        Self {
            store,
            token,
            lease_for,
            clock: Arc::new(clock),
        }
    }
}

#[async_trait::async_trait]
impl RunLeaseHeartbeat for AgentRunLeaseHeartbeat {
    async fn renew(&self) -> tinyagents_harness::Result<bool> {
        self.store
            .renew_agent_run_lease(&self.token, (self.clock)(), self.lease_for)
            .await
            .map_err(|error| TinyAgentsError::Storage(error.to_string()))
    }
}
