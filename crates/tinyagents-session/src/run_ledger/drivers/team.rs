//! Team-task claim, completion and member shutdown over validated batches.
//!
//! Each operation reads the documents it depends on, decides the outcome, and
//! commits the writes together with a [`ReadCheck`] for every dependency it
//! observed. A concurrent writer that changed a dependency, the task or the
//! member makes the batch conflict, and the operation re-reads and decides
//! again. The decision rules are the classic SQLite ones.
use super::storage::{CAS_ATTEMPTS, MEMBERS, TASKS, encode, key, now};
use super::*;
use tinyagents_storage_kit::{ErrorKind, ExpectedDocument, Precondition, ReadCheck, WriteOp};

const LOG_PREFIX: &str = "[run_ledger:team]";

/// A dependency task observed during a decision, with its read check.
struct Dependency {
    id: String,
    done: bool,
    check: ReadCheck,
}

impl DriverRunLedger {
    /// Read every dependency of `task`, scoped to `team_id` like the SQL gate.
    ///
    /// A dependency that belongs to another team counts as not done, exactly
    /// as the classic `WHERE id = ? AND team_id = ?` lookup did.
    async fn dependencies(&self, team_id: &str, task: &AgentTeamTask) -> Result<Vec<Dependency>> {
        let mut found = Vec::with_capacity(task.depends_on.len());
        for dep_id in &task.depends_on {
            let row = self.read::<AgentTeamTask>(TASKS, dep_id).await?;
            let done = row.as_ref().is_some_and(|row| {
                row.doc.team_id == team_id && row.doc.status == AgentTeamTaskStatus::Done
            });
            found.push(Dependency {
                id: dep_id.clone(),
                done,
                check: ReadCheck {
                    collection: TASKS.into(),
                    id: key(&[dep_id])?,
                    expected: row.as_ref().map_or(ExpectedDocument::Absent, |row| {
                        ExpectedDocument::Version(row.version)
                    }),
                },
            });
        }
        Ok(found)
    }

    /// Commit `writes` guarded by `checks`; `Ok(false)` means a native conflict.
    async fn commit_checked(&self, checks: Vec<ReadCheck>, writes: Vec<WriteOp>) -> Result<bool> {
        match self
            .scoped
            .documents()
            .atomic_batch_checked(checks, writes)
            .await
        {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Atomically claim `task_id` for `member_id` once its dependencies are done.
    pub(super) async fn claim_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome> {
        for _ in 0..CAS_ATTEMPTS {
            let Some(task) = self
                .read::<AgentTeamTask>(TASKS, task_id)
                .await?
                .filter(|row| row.doc.team_id == team_id)
            else {
                tracing::debug!("{LOG_PREFIX} claim.unknown team={team_id} task={task_id}");
                return Ok(ClaimOutcome::UnknownTask);
            };
            let deps = self.dependencies(team_id, &task.doc).await?;
            let mut checks: Vec<ReadCheck> = deps.iter().map(|dep| dep.check.clone()).collect();
            checks.push(ReadCheck {
                collection: TASKS.into(),
                id: key(&[task_id])?,
                expected: ExpectedDocument::Version(task.version),
            });
            let unmet: Vec<String> = deps
                .iter()
                .filter(|dep| !dep.done)
                .map(|dep| dep.id.clone())
                .collect();
            // A terminal task is not claimable whatever its claim column says:
            // `upsert` clears the claim on every non-`in_progress` status, so a
            // `done` task also has no claimant.
            let claimable = task.doc.claimed_by_member_id.is_none()
                && matches!(
                    task.doc.status,
                    AgentTeamTaskStatus::Todo
                        | AgentTeamTaskStatus::Ready
                        | AgentTeamTaskStatus::Blocked
                );
            if !unmet.is_empty() || !claimable {
                // The verdict rests on what was read; validate it still holds.
                if !self.commit_checked(checks, Vec::new()).await? {
                    continue;
                }
                if !unmet.is_empty() {
                    tracing::debug!(
                        "{LOG_PREFIX} claim.blocked team={team_id} task={task_id} unmet={}",
                        unmet.len()
                    );
                    return Ok(ClaimOutcome::Blocked { unmet });
                }
                tracing::debug!("{LOG_PREFIX} claim.already_claimed team={team_id} task={task_id}");
                return Ok(ClaimOutcome::AlreadyClaimed);
            }
            let mut claimed = task.doc;
            claimed.claimed_by_member_id = Some(member_id.into());
            claimed.claim_token = Some(claim_token.into());
            claimed.status = AgentTeamTaskStatus::InProgress;
            claimed.updated_at = now();
            // The task itself is guarded by the write precondition, so its read
            // check would be redundant (and a batch refuses contradictory ones).
            checks.pop();
            let write = WriteOp::Put {
                collection: TASKS.into(),
                id: key(&[task_id])?,
                doc: encode(&claimed)?,
                precondition: Precondition::Version(task.version),
            };
            if self.commit_checked(checks, vec![write]).await? {
                tracing::debug!("{LOG_PREFIX} claim.claimed team={team_id} task={task_id}");
                return Ok(ClaimOutcome::Claimed(Box::new(claimed)));
            }
        }
        Err(StorageError::conflict("team task claim contention budget exhausted").into())
    }

    /// Gate a task's completion and flip it to `done` when every invariant holds.
    pub(super) async fn complete_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        self.complete_team_task_if_claimed(
            team_id,
            task_id,
            member_id,
            None,
            evidence,
            require_evidence,
        )
        .await
    }

    /// Complete a task with an optional claim-token fence.
    pub(super) async fn complete_team_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: Option<&str>,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        for _ in 0..CAS_ATTEMPTS {
            let Some(task) = self
                .read::<AgentTeamTask>(TASKS, task_id)
                .await?
                .filter(|row| row.doc.team_id == team_id)
            else {
                tracing::debug!("{LOG_PREFIX} complete.unknown team={team_id} task={task_id}");
                return Ok(CompletionOutcome::UnknownTask);
            };
            let is_claimant = task.doc.claimed_by_member_id.as_deref() == Some(member_id)
                && claim_token.is_none_or(|token| task.doc.claim_token.as_deref() == Some(token));
            let in_progress = task.doc.status == AgentTeamTaskStatus::InProgress;
            let task_check = ReadCheck {
                collection: TASKS.into(),
                id: key(&[task_id])?,
                expected: ExpectedDocument::Version(task.version),
            };
            if !is_claimant || !in_progress {
                if !self.commit_checked(vec![task_check], Vec::new()).await? {
                    continue;
                }
                tracing::debug!(
                    "{LOG_PREFIX} complete.not_claimed team={team_id} task={task_id} \
                     claimant={is_claimant} in_progress={in_progress}"
                );
                return Ok(CompletionOutcome::NotClaimed);
            }

            // Evidence accumulates across attempts (order-preserving, deduplicated).
            let mut merged = task.doc.evidence.clone();
            for link in evidence {
                if !merged.contains(link) {
                    merged.push(link.clone());
                }
            }
            let deps = self.dependencies(team_id, &task.doc).await?;
            let mut reasons: Vec<String> = deps
                .iter()
                .filter(|dep| !dep.done)
                .map(|dep| format!("dependency {} is not done", dep.id))
                .collect();
            if let Some(owner) =
                task.doc.owner_member_id.as_ref().filter(|owner| {
                    Some(owner.as_str()) != task.doc.claimed_by_member_id.as_deref()
                })
            {
                reasons.push(format!(
                    "task is owned by {owner} but claimed by {}",
                    task.doc.claimed_by_member_id.as_deref().unwrap_or("nobody")
                ));
            }
            if require_evidence && merged.is_empty() {
                reasons.push("completion requires at least one evidence link".to_string());
            }

            let checks: Vec<ReadCheck> = deps.into_iter().map(|dep| dep.check).collect();
            let mut next = task.doc;
            next.evidence = merged;
            next.updated_at = now();
            if reasons.is_empty() {
                next.status = AgentTeamTaskStatus::Done;
                next.gate_status = "passed".into();
                next.gate_reason = None;
            } else {
                // The submission is real even when the gate fails: keep the
                // merged evidence, record only the verdict.
                next.gate_status = "failed".into();
                next.gate_reason = Some(reasons.join("; "));
            }
            let write = WriteOp::Put {
                collection: TASKS.into(),
                id: key(&[task_id])?,
                doc: encode(&next)?,
                precondition: Precondition::Version(task.version),
            };
            if !self.commit_checked(checks, vec![write]).await? {
                continue;
            }
            if reasons.is_empty() {
                tracing::debug!("{LOG_PREFIX} complete.completed team={team_id} task={task_id}");
                return Ok(CompletionOutcome::Completed(Box::new(next)));
            }
            tracing::debug!(
                "{LOG_PREFIX} complete.gate_failed team={team_id} task={task_id} reasons={}",
                reasons.len()
            );
            return Ok(CompletionOutcome::GateFailed { reasons });
        }
        Err(StorageError::conflict("team task completion contention budget exhausted").into())
    }

    /// Stop a member and release its in-progress tasks in one atomic batch.
    pub(super) async fn shutdown_team_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
        use tinyagents_storage_kit::{DocumentStoreExt, Filter, Query, Sort};
        for _ in 0..CAS_ATTEMPTS {
            let Some(member) = self
                .read::<AgentTeamMember>(MEMBERS, member_id)
                .await?
                .filter(|row| row.doc.team_id == team_id)
            else {
                tracing::debug!("{LOG_PREFIX} shutdown.unknown team={team_id} member={member_id}");
                return Ok(None);
            };
            let filter = Filter::eq("team_id", team_id)
                .and(Filter::eq("claimed_by_member_id", member_id))
                .and(Filter::eq("status", "in_progress"));
            let held = self
                .scoped
                .documents()
                .query_all(
                    TASKS,
                    &Query::filter(filter)
                        .sort(Sort::asc("order_index"))
                        .sort(Sort::asc("created_ms")),
                )
                .await?;
            let now = now();
            let mut released = Vec::with_capacity(held.len());
            let mut writes = Vec::with_capacity(held.len() + 1);
            for row in held {
                let mut task: AgentTeamTask = super::storage::decode(row.doc)?;
                released.push(task.id.clone());
                task.claimed_by_member_id = None;
                task.claim_token = None;
                task.status = AgentTeamTaskStatus::Todo;
                task.updated_at = now;
                writes.push(WriteOp::Put {
                    collection: TASKS.into(),
                    id: row.id,
                    doc: encode(&task)?,
                    precondition: Precondition::Version(row.version),
                });
            }
            let mut stopped = member.doc;
            stopped.member_status = AgentTeamMemberStatus::Stopped;
            stopped.current_task_id = None;
            stopped.updated_at = now;
            writes.push(WriteOp::Put {
                collection: MEMBERS.into(),
                id: key(&[member_id])?,
                doc: encode(&stopped)?,
                precondition: Precondition::Version(member.version),
            });
            if self.commit_checked(Vec::new(), writes).await? {
                tracing::debug!(
                    "{LOG_PREFIX} shutdown.stopped team={team_id} member={member_id} released={}",
                    released.len()
                );
                return Ok(Some((stopped, released)));
            }
        }
        Err(StorageError::conflict("team member shutdown contention budget exhausted").into())
    }
}
