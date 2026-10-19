//! Native agent records. Merge rules match the classic ledger without workspace lookup.
use super::storage::AGENTS;
use super::*;

impl DriverRunLedger {
    pub(super) async fn agent(&self, id: &str) -> Result<Option<AgentRun>> {
        use super::storage::{CAS_ATTEMPTS, key};
        use tinystoragedrivers_core::{ErrorKind, ExpectedDocument, ReadCheck};
        for _ in 0..CAS_ATTEMPTS {
            let Some(previous) = self.read::<AgentRun>(AGENTS, id).await? else {
                return Ok(None);
            };
            let (telemetry, check) = self.telemetry_dependency(id).await?;
            let checks = vec![
                check,
                ReadCheck {
                    collection: AGENTS.into(),
                    id: key(&[id])?,
                    expected: ExpectedDocument::Version(previous.version),
                },
            ];
            match self
                .scoped
                .documents()
                .atomic_batch_checked(checks, vec![])
                .await
            {
                Ok(_) => {
                    let mut row = previous.doc;
                    row.telemetry = telemetry;
                    return Ok(Some(row));
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent read contention budget exhausted").into())
    }

    pub(super) async fn telemetry_dependency(
        &self,
        id: &str,
    ) -> Result<(Option<RunTelemetry>, tinystoragedrivers_core::ReadCheck)> {
        use tinystoragedrivers_core::{ExpectedDocument, ReadCheck};
        let telemetry = self.read::<RunTelemetry>(TELEMETRY, id).await?;
        let check = ReadCheck {
            collection: TELEMETRY.into(),
            id: super::storage::key(&[id])?,
            expected: telemetry.as_ref().map_or(ExpectedDocument::Absent, |row| {
                ExpectedDocument::Version(row.version)
            }),
        };
        Ok((telemetry.map(|row| row.doc), check))
    }

    pub(super) async fn agent_telemetry(&self, mut row: AgentRun) -> Result<AgentRun> {
        row.telemetry = self
            .read::<RunTelemetry>(TELEMETRY, &row.id)
            .await?
            .map(|t| t.doc);
        Ok(row)
    }

    pub(super) async fn upsert_agent(&self, upsert: AgentRunUpsert) -> Result<AgentRun> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{ErrorKind, Precondition};
        let now = now();
        for _ in 0..CAS_ATTEMPTS {
            let previous = self.read::<AgentRun>(AGENTS, &upsert.id).await?;
            if previous.as_ref().is_some_and(|row| row.doc.lease_epoch > 0) {
                return Err(StorageError::conflict(
                    "agent run has a lease history; use its owner-fenced operation",
                )
                .into());
            }
            let condition = previous.as_ref().map_or(Precondition::Absent, |row| {
                Precondition::Version(row.version)
            });
            let mut row = merge_agent(previous.map(|row| row.doc), &upsert, now);
            let (telemetry, check) = self.telemetry_dependency(&upsert.id).await?;
            row.telemetry = None;
            let writes = vec![put_op(AGENTS, key(&[&row.id])?, &row, condition)?];
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![check], writes)
                .await
            {
                Ok(_) => {
                    row.telemetry = telemetry;
                    return Ok(row);
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent upsert contention budget exhausted").into())
    }

    pub(super) async fn claim_agent(
        &self,
        upsert: AgentRunUpsert,
        owner: &str,
        now: DateTime<Utc>,
        lease_for: chrono::Duration,
    ) -> Result<AgentRunLeaseClaim> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{ErrorKind, Precondition};
        let ttl = lease_for.num_milliseconds();
        if owner.trim().is_empty() || ttl <= 0 || upsert.status != AgentRunStatus::Running {
            return Err(StorageError::conflict(
                "starting an agent run requires an owner, positive lease, and running status",
            )
            .into());
        }
        self.declare().await?;
        for _ in 0..CAS_ATTEMPTS {
            let previous = self.read::<AgentRun>(AGENTS, &upsert.id).await?;
            let mut row = match previous.as_ref() {
                None => merge_agent(None, &upsert, now),
                Some(found) => {
                    let mut row = found.doc.clone();
                    let reclaimable = matches!(
                        row.status,
                        AgentRunStatus::Running | AgentRunStatus::Pending
                    ) && row.lease_owner.is_some()
                        && row.lease_expires_at.is_some_and(|expires| expires <= now);
                    if !reclaimable {
                        row.telemetry = self
                            .read::<RunTelemetry>(TELEMETRY, &row.id)
                            .await?
                            .map(|telemetry| telemetry.doc);
                        return Ok(AgentRunLeaseClaim::Busy(row));
                    }
                    row
                }
            };
            let epoch = if previous.is_some() {
                row.lease_epoch
                    .checked_add(1)
                    .ok_or_else(|| StorageError::conflict("agent lease epoch exhausted"))?
            } else {
                1
            };
            row.status = AgentRunStatus::Running;
            row.lease_owner = Some(owner.to_owned());
            row.lease_expires_at = Some(now + lease_for);
            row.lease_epoch = epoch;
            row.updated_at = now;
            let condition = previous.as_ref().map_or(Precondition::Absent, |found| {
                Precondition::Version(found.version)
            });
            let write = put_op(AGENTS, key(&[&row.id])?, &row, condition)?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![], vec![write])
                .await
            {
                Ok(_) => {
                    let row = self.agent_telemetry(row).await?;
                    return Ok(AgentRunLeaseClaim::Acquired {
                        token: AgentRunLeaseToken {
                            run_id: upsert.id,
                            owner: owner.to_owned(),
                            epoch,
                        },
                        run: row,
                    });
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent claim contention budget exhausted").into())
    }

    pub(super) async fn renew_agent(
        &self,
        token: &AgentRunLeaseToken,
        now: DateTime<Utc>,
        lease_for: chrono::Duration,
    ) -> Result<bool> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{ErrorKind, Precondition};
        let ttl = lease_for.num_milliseconds();
        if ttl <= 0 {
            return Err(StorageError::conflict("agent lease duration must be positive").into());
        }
        self.declare().await?;
        for _ in 0..CAS_ATTEMPTS {
            let Some(previous) = self.read::<AgentRun>(AGENTS, &token.run_id).await? else {
                return Ok(false);
            };
            let mut row = previous.doc;
            if !agent_lease_matches(&row, token, now) {
                return Ok(false);
            }
            row.lease_expires_at = Some(now + lease_for);
            row.updated_at = now;
            let write = put_op(
                AGENTS,
                key(&[&row.id])?,
                &row,
                Precondition::Version(previous.version),
            )?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![], vec![write])
                .await
            {
                Ok(_) => return Ok(true),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent heartbeat contention budget exhausted").into())
    }

    pub(super) async fn finish_agent(
        &self,
        upsert: AgentRunUpsert,
        token: &AgentRunLeaseToken,
        now: DateTime<Utc>,
    ) -> Result<Option<AgentRun>> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{ErrorKind, Precondition};
        if upsert.id != token.run_id || !upsert.status.is_terminal() {
            return Err(StorageError::conflict(
                "finishing an agent run requires its lease token and a terminal status",
            )
            .into());
        }
        self.declare().await?;
        for _ in 0..CAS_ATTEMPTS {
            let Some(previous) = self.read::<AgentRun>(AGENTS, &token.run_id).await? else {
                return Ok(None);
            };
            if !agent_lease_matches(&previous.doc, token, now) {
                return Ok(None);
            }
            let mut row = merge_agent(Some(previous.doc), &upsert, now);
            row.lease_owner = None;
            row.lease_expires_at = None;
            row.lease_epoch = token.epoch;
            row.telemetry = None;
            let (telemetry, check) = self.telemetry_dependency(&token.run_id).await?;
            let write = put_op(
                AGENTS,
                key(&[&row.id])?,
                &row,
                Precondition::Version(previous.version),
            )?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![check], vec![write])
                .await
            {
                Ok(_) => {
                    row.telemetry = telemetry;
                    return Ok(Some(row));
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("agent completion contention budget exhausted").into())
    }
}

impl DriverRunLedger {
    pub(super) async fn agents(
        &self,
        request: &AgentRunListRequest,
    ) -> Result<AgentRunListResponse> {
        let filter = eq_nonblank(Filter::All, "status", request.status.as_deref());
        let filter = eq_nonblank(filter, "kind", request.kind.as_deref());
        let filter = eq_nonblank(filter, "parent_run_id", request.parent_run_id.as_deref());
        let filter = eq_nonblank(
            filter,
            "parent_thread_id",
            request.parent_thread_id.as_deref(),
        );
        let (rows, count) = self
            .page::<AgentRun>(
                AGENTS,
                filter,
                request.limit.unwrap_or(50).min(500) as u64,
                request.offset.unwrap_or(0) as u64,
            )
            .await?;
        let mut runs = Vec::with_capacity(rows.len());
        for row in rows {
            runs.push(self.agent_telemetry(row).await?);
        }
        Ok(AgentRunListResponse { runs, count })
    }

    // The port admits this sweep only under exclusive process ownership.
    pub(super) async fn interrupt_agents(&self) -> Result<usize> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{DocumentStoreExt, ErrorKind, Precondition};
        self.declare().await?;
        let filter = Filter::eq("status", "running").or(Filter::eq("status", "pending"));
        for _ in 0..CAS_ATTEMPTS {
            let rows = self
                .scoped
                .documents()
                .query_all(AGENTS, &Query::filter(filter.clone()))
                .await?;
            let count = rows.len();
            if count == 0 {
                return Ok(0);
            }
            let now = now();
            let writes = rows
                .into_iter()
                .map(|previous| {
                    let mut row: AgentRun = super::storage::decode(previous.doc)?;
                    row.status = AgentRunStatus::Interrupted;
                    row.completed_at = row.completed_at.or(Some(now));
                    row.updated_at = now;
                    put_op(
                        AGENTS,
                        key(&[&row.id])?,
                        &row,
                        Precondition::Version(previous.version),
                    )
                })
                .collect::<std::result::Result<Vec<_>, StorageError>>()?;
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![], writes)
                .await
            {
                Ok(_) => return Ok(count),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("exclusive agent sweep contention budget exhausted").into())
    }

    /// Shared-backend reaping only settles rows whose recorded lease expired.
    pub(super) async fn interrupt_expired_agents(&self, now: DateTime<Utc>) -> Result<usize> {
        use super::storage::{CAS_ATTEMPTS, key, put_op};
        use tinystoragedrivers_core::{DocumentStoreExt, ErrorKind, Precondition};
        self.declare().await?;
        let filter = Filter::eq("status", "running").or(Filter::eq("status", "pending"));
        for _ in 0..CAS_ATTEMPTS {
            let rows = self
                .scoped
                .documents()
                .query_all(AGENTS, &Query::filter(filter.clone()))
                .await?;
            let mut writes = Vec::new();
            for previous in rows {
                let mut row: AgentRun = super::storage::decode(previous.doc)?;
                let expired = row.lease_owner.is_some()
                    && row.lease_expires_at.is_some_and(|expires| expires <= now);
                if !expired {
                    continue;
                }
                row.status = AgentRunStatus::Interrupted;
                row.completed_at = row.completed_at.or(Some(now));
                row.updated_at = now;
                row.lease_owner = None;
                row.lease_expires_at = None;
                writes.push(put_op(
                    AGENTS,
                    key(&[&row.id])?,
                    &row,
                    Precondition::Version(previous.version),
                )?);
            }
            let count = writes.len();
            if count == 0 {
                return Ok(0);
            }
            match self
                .scoped
                .documents()
                .atomic_batch_checked(vec![], writes)
                .await
            {
                Ok(_) => return Ok(count),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(StorageError::conflict("expired agent reaper contention budget exhausted").into())
    }
}

fn agent_lease_matches(row: &AgentRun, token: &AgentRunLeaseToken, now: DateTime<Utc>) -> bool {
    row.id == token.run_id
        && row.lease_owner.as_deref() == Some(token.owner.as_str())
        && row.lease_epoch == token.epoch
        && row.lease_expires_at.is_some_and(|expires| expires > now)
        && row.status == AgentRunStatus::Running
}

fn merge_agent(
    previous: Option<AgentRun>,
    upsert: &AgentRunUpsert,
    now: DateTime<Utc>,
) -> AgentRun {
    let first = previous.is_none();
    let mut row = previous.unwrap_or(AgentRun {
        id: upsert.id.clone(),
        kind: upsert.kind,
        parent_run_id: None,
        parent_thread_id: None,
        agent_id: None,
        status: upsert.status,
        prompt_ref: None,
        worker_thread_id: None,
        checkpoint_path: None,
        checkpoint: None,
        summary: None,
        error: None,
        metadata: upsert.metadata.clone(),
        telemetry: None,
        started_at: upsert.started_at.map(trunc).unwrap_or(now),
        updated_at: now,
        completed_at: None,
        lease_owner: None,
        lease_expires_at: None,
        lease_epoch: 0,
    });
    if !(row.kind == AgentRunKind::WorkerThread && upsert.kind == AgentRunKind::Subagent) {
        row.kind = upsert.kind;
    }
    row.parent_run_id = upsert.parent_run_id.clone().or(row.parent_run_id);
    row.parent_thread_id = upsert.parent_thread_id.clone().or(row.parent_thread_id);
    row.agent_id = upsert.agent_id.clone().or(row.agent_id);
    row.status = upsert.status;
    row.prompt_ref = upsert.prompt_ref.clone().or(row.prompt_ref);
    row.worker_thread_id = upsert.worker_thread_id.clone().or(row.worker_thread_id);
    row.checkpoint_path = upsert.checkpoint_path.clone().or(row.checkpoint_path);
    row.checkpoint = upsert.checkpoint.clone().or(row.checkpoint);
    row.summary = upsert.summary.clone().or(row.summary);
    row.error = upsert.error.clone().or(row.error);
    if first || upsert.metadata != serde_json::json!({}) {
        row.metadata = upsert.metadata.clone();
    }
    row.completed_at = upsert.completed_at.map(trunc).or(row.completed_at);
    row.updated_at = now;
    row
}
