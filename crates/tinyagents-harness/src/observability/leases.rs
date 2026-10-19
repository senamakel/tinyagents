//! In-memory lease lifecycle operations for harness run statuses.

use std::time::SystemTime;

use super::{HarnessRunStatus, InMemoryStatusStore, is_active_status};
use crate::error::{Result, TinyAgentsError};

impl InMemoryStatusStore {
    pub(super) async fn claim_status(
        &self,
        mut status: HarnessRunStatus,
        owner: &str,
        now: SystemTime,
        expires_at: SystemTime,
    ) -> Result<Option<HarnessRunStatus>> {
        if owner.trim().is_empty() || expires_at <= now || !is_active_status(&status) {
            return Err(TinyAgentsError::Validation(
                "status lease requires an owner, future expiry, and active status".into(),
            ));
        }
        let mut state = self.state.lock().map_err(|error| {
            TinyAgentsError::Storage(format!("InMemoryStatusStore lock poisoned: {error}"))
        })?;
        let run_id = status.run_id.as_str().to_string();
        let epoch = match state.statuses.get(&run_id) {
            None => 1,
            Some(current)
                if is_active_status(current)
                    && current.lease_owner.is_some()
                    && current
                        .lease_expires_at
                        .is_some_and(|deadline| deadline <= now) =>
            {
                current.lease_epoch.checked_add(1).ok_or_else(|| {
                    TinyAgentsError::Storage("status lease epoch exhausted".into())
                })?
            }
            Some(_) => return Ok(None),
        };
        status.lease_owner = Some(owner.to_owned());
        status.lease_expires_at = Some(expires_at);
        status.lease_epoch = epoch;
        status.updated_at = now;
        if !state.statuses.contains_key(&run_id) {
            state.order.push_back(run_id.clone());
        }
        state.statuses.insert(run_id, status.clone());
        Ok(Some(status))
    }

    pub(super) async fn renew_status_lease(
        &self,
        run_id: &str,
        owner: &str,
        epoch: u64,
        now: SystemTime,
        expires_at: SystemTime,
    ) -> Result<bool> {
        if expires_at <= now {
            return Err(TinyAgentsError::Validation(
                "status lease expiry must be in the future".into(),
            ));
        }
        let mut state = self.state.lock().map_err(|error| {
            TinyAgentsError::Storage(format!("InMemoryStatusStore lock poisoned: {error}"))
        })?;
        let Some(status) = state.statuses.get_mut(run_id) else {
            return Ok(false);
        };
        if !is_active_status(status)
            || status.lease_owner.as_deref() != Some(owner)
            || status.lease_epoch != epoch
            || status
                .lease_expires_at
                .is_none_or(|deadline| deadline <= now)
        {
            return Ok(false);
        }
        status.lease_expires_at = Some(expires_at);
        status.updated_at = now;
        Ok(true)
    }

    pub(super) async fn finish_status_lease(
        &self,
        mut replacement: HarnessRunStatus,
        owner: &str,
        epoch: u64,
        now: SystemTime,
    ) -> Result<bool> {
        let mut state = self.state.lock().map_err(|error| {
            TinyAgentsError::Storage(format!("InMemoryStatusStore lock poisoned: {error}"))
        })?;
        let run_id = replacement.run_id.as_str().to_string();
        let Some(current) = state.statuses.get(&run_id) else {
            return Ok(false);
        };
        if is_active_status(&replacement)
            || !is_active_status(current)
            || current.lease_owner.as_deref() != Some(owner)
            || current.lease_epoch != epoch
            || current
                .lease_expires_at
                .is_none_or(|deadline| deadline <= now)
        {
            return Ok(false);
        }
        replacement.lease_owner = None;
        replacement.lease_expires_at = None;
        replacement.lease_epoch = epoch;
        replacement.updated_at = now;
        state.statuses.insert(run_id, replacement);
        Ok(true)
    }
}
