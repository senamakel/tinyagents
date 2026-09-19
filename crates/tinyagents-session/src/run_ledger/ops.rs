//! CRUD, listing, and coordination primitives for the run ledger tables:
//! agent runs, workflow runs (with a compare-and-swap driver lease), run
//! events, run telemetry, and agent-team coordination (teams, members,
//! tasks with claim/completion).
//!
//! Every entry point opens its own connection or transaction via
//! `crate::store::with_connection` / `crate::store::with_transaction`
//! (aliased here through `init_run_ledger_schema`, now a no-op kept as the
//! conventional call site — see `super::store`). Anything that reads state
//! and then acts on it (an upsert reading its own write back, a claim, a
//! compare-and-swap) uses `with_transaction`; plain single-statement reads
//! and writes use `with_connection`. The `*_inner` helpers take an open
//! [`Connection`] directly so an upsert can read back the row it just wrote
//! inside the same transaction rather than reopening a connection and
//! possibly observing a concurrent writer's state instead of its own.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};

use tinyagents_harness::error::Result;

use super::super::context::StorageContext;
use super::store::init_run_ledger_schema;
use super::types::{
    AgentRun, AgentRunKind, AgentRunListRequest, AgentRunListResponse, AgentRunStatus,
    AgentRunUpsert, AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember,
    AgentTeamMemberStatus, AgentTeamMemberUpsert, AgentTeamStatus, AgentTeamTask,
    AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome, CompletionOutcome,
    RunEvent, RunEventAppend, RunEventListRequest, RunEventListResponse, RunTelemetry,
    RunTelemetryUpsert, WorkflowLeaseClaim, WorkflowRun, WorkflowRunListRequest,
    WorkflowRunListResponse, WorkflowRunStatus, WorkflowRunUpsert,
};

mod rows;
mod team;

use rows::{
    get_agent_run_inner, get_agent_team_inner, get_agent_team_member_inner,
    get_agent_team_task_inner, get_run_telemetry_inner, map_agent_run_row,
    map_agent_team_member_row, map_agent_team_row, map_agent_team_task_row, map_run_event_row,
    map_workflow_run_row,
};

pub use team::{
    claim_agent_team_task, complete_agent_team_task, get_agent_team, get_agent_team_member,
    get_agent_team_task, list_agent_team_members, list_agent_team_tasks, list_agent_teams,
    mark_agent_team_member_idle, mark_agent_team_member_running, release_agent_team_task,
    shutdown_agent_team_member, upsert_agent_team, upsert_agent_team_member,
    upsert_agent_team_task,
};

/// Grep prefix for run-ledger logging.
const LOG_PREFIX: &str = "[session_db:run_ledger]";

/// Inserts a new [`AgentRun`] or merges fields into an existing one with the
/// same id, returning the row as stored.
///
/// Most fields are `COALESCE`d against the existing row on conflict, so
/// passing `None` leaves them unchanged rather than clearing them — except
/// `status` and `updated_at`, which are always overwritten, and `metadata`,
/// which only replaces the stored value when the incoming JSON is non-empty
/// (`{}` is treated as "no metadata supplied"). `kind` additionally refuses
/// to downgrade a `worker_thread` run back to `subagent`, since a run that
/// has already been promoted to a worker thread should not silently revert.
/// Use [`transition_agent_run_status`] instead when a caller needs to
/// *clear* `error` or `completed_at`.
pub fn upsert_agent_run(workspace_dir: &Path, upsert: AgentRunUpsert) -> Result<AgentRun> {
    let now = Utc::now();
    let started_at = upsert.started_at.unwrap_or(now);
    let updated_at = now;
    let metadata_json =
        serde_json::to_string(&upsert.metadata).storage_context("serialize agent run metadata")?;
    let checkpoint_json = upsert
        .checkpoint
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .storage_context("serialize agent run checkpoint")?;

    tinyagents_tracing::debug!(
        "{LOG_PREFIX} upsert_agent_run id={} kind={} status={} parent={} thread={}",
        upsert.id,
        upsert.kind.as_str(),
        upsert.status.as_str(),
        upsert.parent_run_id.as_deref().unwrap_or("-"),
        upsert.parent_thread_id.as_deref().unwrap_or("-")
    );

    // One transaction for the write *and* the read-back. Committing the insert
    // on an autocommit connection, closing it, then re-opening to `get_*` hands
    // the caller whatever a concurrent writer left behind rather than what this
    // call wrote — an upsert that reports someone else's row.
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_runs (
                id, kind, parent_run_id, parent_thread_id, agent_id, status,
                prompt_ref, worker_thread_id, task_board_id, task_card_id,
                checkpoint_path, checkpoint_json, summary, error, metadata_json,
                started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
             ON CONFLICT(id) DO UPDATE SET
                kind = CASE
                    WHEN agent_runs.kind = 'worker_thread' AND excluded.kind = 'subagent' THEN agent_runs.kind
                    ELSE excluded.kind
                END,
                parent_run_id = COALESCE(excluded.parent_run_id, agent_runs.parent_run_id),
                parent_thread_id = COALESCE(excluded.parent_thread_id, agent_runs.parent_thread_id),
                agent_id = COALESCE(excluded.agent_id, agent_runs.agent_id),
                status = excluded.status,
                prompt_ref = COALESCE(excluded.prompt_ref, agent_runs.prompt_ref),
                worker_thread_id = COALESCE(excluded.worker_thread_id, agent_runs.worker_thread_id),
                task_board_id = COALESCE(excluded.task_board_id, agent_runs.task_board_id),
                task_card_id = COALESCE(excluded.task_card_id, agent_runs.task_card_id),
                checkpoint_path = COALESCE(excluded.checkpoint_path, agent_runs.checkpoint_path),
                checkpoint_json = COALESCE(excluded.checkpoint_json, agent_runs.checkpoint_json),
                summary = COALESCE(excluded.summary, agent_runs.summary),
                error = COALESCE(excluded.error, agent_runs.error),
                metadata_json = CASE
                    WHEN excluded.metadata_json = '{}' THEN agent_runs.metadata_json
                    ELSE excluded.metadata_json
                END,
                updated_at = excluded.updated_at,
                completed_at = COALESCE(excluded.completed_at, agent_runs.completed_at)",
            params![
                upsert.id,
                upsert.kind.as_str(),
                upsert.parent_run_id,
                upsert.parent_thread_id,
                upsert.agent_id,
                upsert.status.as_str(),
                upsert.prompt_ref,
                upsert.worker_thread_id,
                upsert.task_board_id,
                upsert.task_card_id,
                upsert.checkpoint_path,
                checkpoint_json,
                upsert.summary,
                upsert.error,
                metadata_json,
                started_at.to_rfc3339(),
                updated_at.to_rfc3339(),
                upsert.completed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert agent run")?;
        get_agent_run_inner(conn, &upsert.id)?.storage_context("agent run missing after upsert")
    })
}

/// Inserts a new [`WorkflowRun`] or merges fields into an existing one with
/// the same id, returning the row as stored.
///
/// Bumps `revision` on every call, including the first insert, so
/// [`compare_and_swap_workflow_run`] callers always have a fresh fencing
/// token. When `status` transitions to a terminal value the driver lease
/// (`lease_owner` / `lease_expires_at`) is cleared, since a finished run has
/// nothing left to drive. Prefer [`compare_and_swap_workflow_run`] or
/// [`compare_and_swap_workflow_run_lifecycle`] for a driver actively holding
/// a lease — this plain upsert has no revision fencing of its own and can
/// clobber a concurrent driver's write.
pub fn upsert_workflow_run(workspace_dir: &Path, upsert: WorkflowRunUpsert) -> Result<WorkflowRun> {
    let now = Utc::now();
    let started_at = upsert.started_at.unwrap_or(now);
    let input_json =
        serde_json::to_string(&upsert.input).storage_context("serialize workflow input")?;
    let phase_states_json = serde_json::to_string(&upsert.phase_states)
        .storage_context("serialize workflow phase states")?;
    let child_run_ids_json =
        serde_json::to_string(&upsert.child_run_ids).storage_context("serialize child run ids")?;

    // One transaction for the write *and* the read-back. Committing the insert
    // on an autocommit connection, closing it, then re-opening to `get_*` hands
    // the caller whatever a concurrent writer left behind rather than what this
    // call wrote — an upsert that reports someone else's row.
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO workflow_runs (
                id, definition_id, parent_thread_id, input_json, phase_states_json,
                child_run_ids_json, status, summary, started_at, updated_at, completed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
                definition_id = excluded.definition_id,
                parent_thread_id = COALESCE(excluded.parent_thread_id, workflow_runs.parent_thread_id),
                input_json = excluded.input_json,
                phase_states_json = excluded.phase_states_json,
                child_run_ids_json = excluded.child_run_ids_json,
                status = excluded.status,
                summary = COALESCE(excluded.summary, workflow_runs.summary),
                updated_at = excluded.updated_at,
                completed_at = COALESCE(excluded.completed_at, workflow_runs.completed_at),
                lease_owner = CASE
                    WHEN excluded.status IN ('completed', 'failed', 'cancelled', 'interrupted') THEN NULL
                    ELSE workflow_runs.lease_owner
                END,
                lease_expires_at = CASE
                    WHEN excluded.status IN ('completed', 'failed', 'cancelled', 'interrupted') THEN NULL
                    ELSE workflow_runs.lease_expires_at
                END,
                revision = workflow_runs.revision + 1",
            params![
                upsert.id,
                upsert.definition_id,
                upsert.parent_thread_id,
                input_json,
                phase_states_json,
                child_run_ids_json,
                upsert.status.as_str(),
                upsert.summary,
                started_at.to_rfc3339(),
                now.to_rfc3339(),
                upsert.completed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert workflow run")?;
        get_workflow_run_inner(conn, &upsert.id)?
            .storage_context("workflow run missing after upsert")
    })
}

/// Atomically lease a workflow run to one driver.  A second live driver gets
/// the authoritative row as `Busy`; it must not schedule a duplicate phase.
pub fn try_claim_workflow_run(
    workspace_dir: &Path,
    id: &str,
    owner: &str,
    lease_for: chrono::Duration,
) -> Result<WorkflowLeaseClaim> {
    let now = Utc::now();
    let expires = now + lease_for;
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let changed = conn.execute(
            "UPDATE workflow_runs
             SET lease_owner = ?1, lease_expires_at = ?2,
                 revision = revision + 1, updated_at = ?3
             WHERE id = ?4
               AND (lease_owner IS NULL OR lease_owner = ?1 OR lease_expires_at IS NULL OR lease_expires_at <= ?3)",
            params![owner, expires.to_rfc3339(), now.to_rfc3339(), id],
        )?;
        let Some(run) = get_workflow_run_inner(conn, id)? else {
            return Ok(WorkflowLeaseClaim::Missing);
        };
        Ok(if changed == 1 {
            WorkflowLeaseClaim::Acquired(run)
        } else {
            WorkflowLeaseClaim::Busy(run)
        })
    })
}

/// Compare-and-swap a workflow transition while renewing its driver's lease.
/// `None` means another driver or an out-of-band lifecycle operation changed
/// the row; callers must reload rather than overwrite that state.
pub fn compare_and_swap_workflow_run(
    workspace_dir: &Path,
    upsert: WorkflowRunUpsert,
    expected_revision: u64,
    owner: &str,
    lease_for: chrono::Duration,
) -> Result<Option<WorkflowRun>> {
    let now = Utc::now();
    let expires = now + lease_for;
    let input_json =
        serde_json::to_string(&upsert.input).storage_context("serialize workflow input")?;
    let phase_states_json = serde_json::to_string(&upsert.phase_states)
        .storage_context("serialize workflow phase states")?;
    let child_run_ids_json =
        serde_json::to_string(&upsert.child_run_ids).storage_context("serialize child run ids")?;
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let changed = conn.execute(
            "UPDATE workflow_runs SET
                definition_id = ?1, parent_thread_id = COALESCE(?2, parent_thread_id),
                input_json = ?3, phase_states_json = ?4, child_run_ids_json = ?5,
                status = ?6, summary = COALESCE(?7, summary), updated_at = ?8,
                completed_at = COALESCE(?9, completed_at),
                lease_owner = CASE WHEN ?6 IN ('completed', 'failed', 'cancelled', 'interrupted') THEN NULL ELSE lease_owner END,
                lease_expires_at = CASE WHEN ?6 IN ('completed', 'failed', 'cancelled', 'interrupted') THEN NULL ELSE ?10 END,
                revision = revision + 1
             WHERE id = ?11 AND revision = ?12 AND lease_owner = ?13
               AND lease_expires_at > ?8",
            params![
                upsert.definition_id, upsert.parent_thread_id, input_json, phase_states_json,
                child_run_ids_json, upsert.status.as_str(), upsert.summary,
                now.to_rfc3339(), upsert.completed_at.map(|dt| dt.to_rfc3339()),
                expires.to_rfc3339(), upsert.id, expected_revision as i64, owner,
            ],
        )?;
        if changed == 0 {
            Ok(None)
        } else {
            Ok(get_workflow_run_inner(conn, &upsert.id)?)
        }
    })
}

/// Renew a live workflow driver's lease without changing its durable revision.
///
/// A phase can legitimately run longer than the lease interval.  Renewing is
/// deliberately not a state transition: child registration and phase commits
/// retain their revision fencing, while this small heartbeat only proves that
/// the same owner is still alive. `false` means ownership was lost and the
/// caller must cancel its children and stop driving immediately.
pub fn renew_workflow_run_lease(
    workspace_dir: &Path,
    id: &str,
    owner: &str,
    lease_for: chrono::Duration,
) -> Result<bool> {
    let now = Utc::now();
    let expires = now + lease_for;
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        Ok(conn.execute(
            "UPDATE workflow_runs
             SET lease_expires_at = ?1, updated_at = ?2
             WHERE id = ?3 AND lease_owner = ?4 AND lease_expires_at > ?2",
            params![expires.to_rfc3339(), now.to_rfc3339(), id, owner],
        )? == 1)
    })
}

/// Compare-and-swap a host lifecycle transition (stop or resume).
///
/// Lifecycle commands are allowed to fence an in-flight driver, but only from
/// the exact revision their caller observed.  The write always clears the
/// driver lease.  That makes a stop→resume hand-off safe: the old driver can no
/// longer commit, and a resumer gets a fresh owner through `try_claim_*`.
pub fn compare_and_swap_workflow_run_lifecycle(
    workspace_dir: &Path,
    upsert: WorkflowRunUpsert,
    expected_revision: u64,
) -> Result<Option<WorkflowRun>> {
    let now = Utc::now();
    let id = upsert.id.clone();
    let input_json =
        serde_json::to_string(&upsert.input).storage_context("serialize workflow input")?;
    let phase_states_json = serde_json::to_string(&upsert.phase_states)
        .storage_context("serialize workflow phase states")?;
    let child_run_ids_json =
        serde_json::to_string(&upsert.child_run_ids).storage_context("serialize child run ids")?;
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let changed = conn.execute(
            "UPDATE workflow_runs SET
                definition_id = ?1, parent_thread_id = COALESCE(?2, parent_thread_id),
                input_json = ?3, phase_states_json = ?4, child_run_ids_json = ?5,
                status = ?6, summary = COALESCE(?7, summary), updated_at = ?8,
                completed_at = COALESCE(?9, completed_at),
                lease_owner = NULL, lease_expires_at = NULL,
                revision = revision + 1
             WHERE id = ?10 AND revision = ?11",
            params![
                upsert.definition_id,
                upsert.parent_thread_id,
                input_json,
                phase_states_json,
                child_run_ids_json,
                upsert.status.as_str(),
                upsert.summary,
                now.to_rfc3339(),
                upsert.completed_at.map(|dt| dt.to_rfc3339()),
                id,
                expected_revision as i64,
            ],
        )?;
        if changed == 0 {
            Ok(None)
        } else {
            Ok(get_workflow_run_inner(conn, &upsert.id)?)
        }
    })
}

/// Appends a new [`RunEvent`] to a run's event log, allocating its sequence
/// number atomically.
///
/// `sequence` is `MAX(sequence) + 1` for the run, computed by the same
/// `INSERT ... SELECT` statement that writes the row (see the inline comment
/// at the call site) so two connections appending concurrently cannot
/// compute the same next value and lose one event to a primary-key conflict.
pub fn append_run_event(workspace_dir: &Path, event: RunEventAppend) -> Result<RunEvent> {
    let now = Utc::now();
    let payload_json =
        serde_json::to_string(&event.payload).storage_context("serialize run event")?;
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        // Allocate and insert the sequence in ONE statement. Reading
        // `MAX(sequence) + 1` and then inserting is a read-modify-write race:
        // two connections appending for the same run can read the same next
        // value, and the loser fails the `(run_id, sequence)` primary key —
        // silently dropping a real run event unless every caller implements an
        // undocumented retry. The sub-select is evaluated inside the same
        // statement, so SQLite's write lock serializes the whole allocation.
        let next_sequence: i64 = conn
            .query_row(
                "INSERT INTO run_events (run_id, sequence, event_type, payload_json, timestamp)
             VALUES (
                ?1,
                (SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?1),
                ?2, ?3, ?4
             )
             RETURNING sequence",
                params![
                    event.run_id,
                    event.event_type,
                    payload_json,
                    now.to_rfc3339(),
                ],
                |row| row.get(0),
            )
            .storage_context("append run event")?;
        Ok(RunEvent {
            run_id: event.run_id,
            sequence: next_sequence as u64,
            event_type: event.event_type,
            payload: serde_json::from_str(&payload_json).unwrap_or_else(|_| json!({})),
            timestamp: now,
        })
    })
}

/// Inserts a new [`RunTelemetry`] row or merges partial fields into an
/// existing one, returning the row as stored.
///
/// Every counter field is `Option`, and `None` means "leave this field
/// unchanged" rather than "reset to zero" — see the inline comment at the
/// call site for why the insert and update sides need different `COALESCE`
/// targets to make that true on first insert as well as on later updates.
pub fn upsert_run_telemetry(
    workspace_dir: &Path,
    upsert: RunTelemetryUpsert,
) -> Result<RunTelemetry> {
    let now = Utc::now();
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            // The counters are `Option` so a caller can update one field without
            // clobbering the rest, but the columns are `NOT NULL DEFAULT`, and
            // SQLite does NOT apply a column default to an explicitly supplied
            // NULL. Binding the raw `None` therefore made every partial upsert
            // (say, recording only `model` or only `error`) fail a NOT NULL
            // constraint on first write. The insert side coalesces to the
            // column default; the update side re-reads the SAME parameter and
            // coalesces to the stored value, which keeps per-field optionality.
            // `excluded.*` cannot serve the update side here — it observes the
            // already-coalesced insert row, so a `None` would read as 0 and
            // overwrite the stored counter.
            "INSERT INTO run_telemetry (
                run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
             ) VALUES (
                ?1,
                COALESCE(?2, 0), COALESCE(?3, 0), COALESCE(?4, 0), COALESCE(?5, 0.0),
                ?6, COALESCE(?7, 0), ?8, ?9, ?10, ?11
             )
             ON CONFLICT(run_id) DO UPDATE SET
                input_tokens = COALESCE(?2, run_telemetry.input_tokens),
                output_tokens = COALESCE(?3, run_telemetry.output_tokens),
                cached_input_tokens = COALESCE(?4, run_telemetry.cached_input_tokens),
                cost_usd = COALESCE(?5, run_telemetry.cost_usd),
                elapsed_ms = COALESCE(?6, run_telemetry.elapsed_ms),
                tool_count = COALESCE(?7, run_telemetry.tool_count),
                model = COALESCE(?8, run_telemetry.model),
                provider = COALESCE(?9, run_telemetry.provider),
                error = COALESCE(?10, run_telemetry.error),
                updated_at = ?11",
            params![
                upsert.run_id,
                upsert.input_tokens.map(|v| v as i64),
                upsert.output_tokens.map(|v| v as i64),
                upsert.cached_input_tokens.map(|v| v as i64),
                upsert.cost_usd,
                upsert.elapsed_ms.map(|v| v as i64),
                upsert.tool_count.map(|v| v as i64),
                upsert.model,
                upsert.provider,
                upsert.error,
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert run telemetry")?;
        get_run_telemetry_inner(conn, &upsert.run_id)
    })
}

/// Fetches a single [`AgentRun`] by id, or `None` if no row matches.
pub fn get_agent_run(workspace_dir: &Path, id: &str) -> Result<Option<AgentRun>> {
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        get_agent_run_inner(conn, id)
    })
}

/// Apply a durable status transition to a single agent run.
///
/// Unlike [`upsert_agent_run`] — whose `ON CONFLICT` clause `COALESCE`s the
/// `error` and `completed_at` columns and can therefore only ever *set* them —
/// this is a direct `UPDATE` that can both set and *clear* both columns. That
/// is required by control verbs such as "retry", which moves a failed run back
/// to `pending` and must drop the stale failure reason and completion time.
///
/// `status` is always written. `error` and `completed_at` are written verbatim,
/// so passing `None` clears the column. `updated_at` is bumped to now. Returns
/// the freshly-read run, or `None` when no row matched `id` (e.g. it was
/// deleted between a prior read and this write).
pub fn transition_agent_run_status(
    workspace_dir: &Path,
    id: &str,
    status: AgentRunStatus,
    error: Option<&str>,
    completed_at: Option<DateTime<Utc>>,
) -> Result<Option<AgentRun>> {
    let now = Utc::now();
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} transition_agent_run_status id={id} status={} has_error={} has_completed_at={}",
        status.as_str(),
        error.is_some(),
        completed_at.is_some()
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_runs
                 SET status = ?1, error = ?2, completed_at = ?3, updated_at = ?4
                 WHERE id = ?5",
                params![
                    status.as_str(),
                    error,
                    completed_at.map(|dt| dt.to_rfc3339()),
                    now.to_rfc3339(),
                    id,
                ],
            )
            .storage_context("transition agent run status")?;
        if rows_affected == 0 {
            tinyagents_tracing::debug!("{LOG_PREFIX} transition_agent_run_status.miss id={id}");
            return Ok(None);
        }
        get_agent_run_inner(conn, id)
    })
}

/// Settle non-terminal `agent_runs` rows left behind by a previous process.
///
/// A freshly-booted core has no in-flight subagents — any detached run task
/// from a prior process is gone with that process. So a row still marked
/// `running` (or `pending`) at startup is, by definition, orphaned: its driver
/// died without firing the host's terminal completion notification, so the
/// host's run-ledger finalizer never settled it. Without
/// this sweep those rows render as perpetual "running" timeline entries on every
/// thread reopen.
///
/// We stamp them `interrupted` (outcome unknown — mirrors the turn-state
/// `mark_all_interrupted` recovery) and set `completed_at`. `awaiting_user` /
/// `paused` are intentionally left untouched: those are resumable states a user
/// may still continue.
///
pub fn interrupt_orphaned_agent_runs(workspace_dir: &Path) -> Result<usize> {
    let now = Utc::now();
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_runs
                 SET status = ?1, completed_at = COALESCE(completed_at, ?2), updated_at = ?2
                 WHERE status IN ('running', 'pending')",
                params![AgentRunStatus::Interrupted.as_str(), now.to_rfc3339()],
            )
            .storage_context("interrupt orphaned agent runs")?;
        if rows_affected > 0 {
            tinyagents_tracing::info!(
                "{LOG_PREFIX} interrupted {rows_affected} orphaned agent run(s) on startup"
            );
        }
        Ok(rows_affected)
    })
}

/// Lists agent runs, most-recently-updated first, with optional filters
/// (status, kind, parent run, parent thread) and pagination.
///
/// `limit` is capped at 500 regardless of the requested value.
pub fn list_agent_runs(
    workspace_dir: &Path,
    request: &AgentRunListRequest,
) -> Result<AgentRunListResponse> {
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }
        if let Some(kind) = request.kind.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(kind.to_string()));
            where_clauses.push(format!("kind = ?{}", values.len()));
        }
        if let Some(parent) = request
            .parent_run_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(parent.to_string()));
            where_clauses.push(format!("parent_run_id = ?{}", values.len()));
        }
        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM agent_runs {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        let offset = request.offset.unwrap_or(0) as i64;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, kind, parent_run_id, parent_thread_id, agent_id, status,
                    prompt_ref, worker_thread_id, task_board_id, task_card_id,
                    checkpoint_path, checkpoint_json, summary, error, metadata_json,
                    started_at, updated_at, completed_at
             FROM agent_runs {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| map_agent_run_row(conn, row))?;
        let mut runs = Vec::new();
        for row in rows {
            runs.push(row?);
        }
        Ok(AgentRunListResponse { runs, count })
    })
}

/// Lists a run's events in `sequence` order, optionally starting after a
/// given cursor (`after_sequence`), for polling "what's new" incrementally.
///
/// `limit` is capped at 1000 regardless of the requested value.
pub fn list_recent_run_events(
    workspace_dir: &Path,
    request: &RunEventListRequest,
) -> Result<RunEventListResponse> {
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let limit = request.limit.unwrap_or(100).min(1000) as i64;
        let after = request.after_sequence.unwrap_or(0) as i64;
        let mut stmt = conn.prepare(
            "SELECT run_id, sequence, event_type, payload_json, timestamp
             FROM run_events
             WHERE run_id = ?1 AND sequence > ?2
             ORDER BY sequence ASC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(params![request.run_id, after, limit], map_run_event_row)?;
        let mut events = Vec::new();
        for row in rows {
            events.push(row?);
        }
        Ok(RunEventListResponse {
            count: events.len(),
            events,
        })
    })
}

/// Connection-scoped workflow-run lookup, so an upsert can read its own write
/// back inside the same transaction.
fn get_workflow_run_inner(conn: &Connection, id: &str) -> Result<Option<WorkflowRun>> {
    let mut stmt = conn.prepare(
        "SELECT id, definition_id, parent_thread_id, input_json, phase_states_json,
                child_run_ids_json, status, summary, started_at, updated_at, completed_at,
                revision, lease_owner, lease_expires_at
         FROM workflow_runs WHERE id = ?1",
    )?;
    Ok(stmt
        .query_row(params![id], map_workflow_run_row)
        .optional()?)
}

/// Fetches a single [`WorkflowRun`] by id, or `None` if no row matches.
pub fn get_workflow_run(workspace_dir: &Path, id: &str) -> Result<Option<WorkflowRun>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} get_workflow_run.entry id={id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let run = get_workflow_run_inner(conn, id)?;
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} get_workflow_run.exit id={id} found={}",
            run.is_some()
        );
        Ok(run)
    })
}

/// List durable workflow runs, most-recently-updated first, with optional
/// filters (definition id, status, parent thread) and pagination. Mirrors
/// [`list_agent_runs`] for the workflow_runs table.
pub fn list_workflow_runs(
    workspace_dir: &Path,
    request: &WorkflowRunListRequest,
) -> Result<WorkflowRunListResponse> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} list_workflow_runs.entry definition={:?} status={:?} parent_thread={:?} limit={:?} offset={:?}",
        request.definition_id,
        request.status,
        request.parent_thread_id,
        request.limit,
        request.offset
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(definition) = request
            .definition_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(definition.to_string()));
            where_clauses.push(format!("definition_id = ?{}", values.len()));
        }
        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }
        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM workflow_runs {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        // `offset` is `u64`; convert checked so a value > i64::MAX surfaces a
        // clear error instead of wrapping negative and corrupting pagination.
        let offset = i64::try_from(request.offset.unwrap_or(0))
            .storage_context("workflow run list offset exceeds i64::MAX")?;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, definition_id, parent_thread_id, input_json, phase_states_json,
                    child_run_ids_json, status, summary, started_at, updated_at, completed_at,
                    revision, lease_owner, lease_expires_at
             FROM workflow_runs {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), map_workflow_run_row)?;
        let mut runs = Vec::new();
        for row in rows {
            runs.push(row?);
        }
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} list_workflow_runs.exit count={count} returned={}",
            runs.len()
        );
        Ok(WorkflowRunListResponse { runs, count })
    })
}
