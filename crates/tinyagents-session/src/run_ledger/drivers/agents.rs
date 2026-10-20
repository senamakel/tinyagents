//! Native agent records. Merge rules match the classic ledger without workspace lookup.
use super::storage::AGENTS;
use super::*;

impl DriverRunLedger {
    pub(super) async fn agent(&self, id: &str) -> Result<Option<AgentRun>> {
        use super::storage::{CAS_ATTEMPTS, key};
        use tinyagents_storage_kit::{ErrorKind, ExpectedDocument, ReadCheck};
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
    ) -> Result<(Option<RunTelemetry>, tinystoragedrivers_types::ReadCheck)> {
        use tinyagents_storage_kit::{ExpectedDocument, ReadCheck};
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
        use tinyagents_storage_kit::{ErrorKind, Precondition};
        let now = now();
        for _ in 0..CAS_ATTEMPTS {
            let previous = self.read::<AgentRun>(AGENTS, &upsert.id).await?;
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
        use tinyagents_storage_kit::{DocumentStoreExt, ErrorKind, Precondition};
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
