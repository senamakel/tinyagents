//! Durable tool-effect ledger (B5): crash-safe bookkeeping of tool-call side
//! effects, backing `tinyagents_harness::tool::ToolEffectLedger`.
//!
//! A tool call that mutates external state is dangerous to blindly re-run
//! after a process restart. This module gives the agent loop a durable
//! `started` → `completed`/`failed`/`interrupted` row per call, written to
//! the `tool_effects` table added by migration 7 (see `crate::migrations`),
//! so a resumed run can tell an in-flight call apart from one that never
//! started or one that already settled.
//!
//! # Layout
//!
//! [`record_tool_started`], [`settle_tool_effect`],
//! [`list_unresolved_tool_effects`], and [`mark_interrupted`] are the plain,
//! synchronous, workspace-scoped operations, matching every other function
//! in [`super::ops`]. [`RunLedgerToolEffects`] wraps them behind the
//! `async` `tinyagents_harness::tool::ToolEffectLedger` trait so a harness
//! run can attach this ledger via
//! [`tinyagents_harness::context::RunContext::with_tool_effect_ledger`]
//! without either crate depending on the other's concrete storage type.
//!
//! # Why not reuse `agent_runs`
//!
//! `agent_runs` records one row per *run* (a whole harness invocation);
//! `tool_effects` records one row per *tool call within* a run — a much
//! finer grain, with its own terminal-status vocabulary
//! (`started`/`completed`/`failed`/`interrupted`) that only makes sense at
//! call scope. Folding tool-call effects into `agent_runs` would mean either
//! a variable-width JSON blob column (unqueryable) or overloading the run's
//! own status with per-call meaning it does not have.

use chrono::{DateTime, Utc};
use rusqlite::{OptionalExtension, params};

use async_trait::async_trait;
use tinyagents_harness::error::Result;
use tinyagents_harness::tool::{
    ToolEffect as HarnessToolEffect, ToolEffectLedger, ToolEffectSettle as HarnessToolEffectSettle,
    ToolEffectStart as HarnessToolEffectStart, ToolEffectStatus as HarnessToolEffectStatus,
};

use super::store::init_run_ledger_schema;
use super::super::context::StorageContext;

/// Grep prefix for tool-effect-ledger logging.
const LOG_PREFIX: &str = "[session_db:tool_effects]";

/// Lifecycle state of one recorded tool effect.
///
/// Mirrors `tinyagents_harness::tool::ToolEffectStatus` (see
/// [`ToolEffectStatus::from_harness`] / [`ToolEffectStatus::into_harness`])
/// rather than reusing it directly, matching how every other run-ledger
/// status enum (`AgentRunStatus`, `WorkflowRunStatus`, ...) stays
/// session-owned and serde-friendly independent of the harness crate's own
/// runtime types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffectStatus {
    /// Admitted and about to execute (or executing); no terminal outcome yet.
    Started,
    /// Returned a result (successful or a recoverable tool error).
    Completed,
    /// The call's execution future itself failed.
    Failed,
    /// Left `started` across a resume and settled as unsafe to re-execute.
    Interrupted,
}

impl ToolEffectStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }

    /// Parses a stored `status` string, defaulting to [`Self::Started`] for
    /// any unrecognized value — fail toward "still open" so a corrupt or
    /// forward-written value is never silently dropped from resume
    /// consideration.
    pub fn parse(raw: &str) -> Self {
        match raw {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "interrupted" => Self::Interrupted,
            _ => Self::Started,
        }
    }

    fn from_harness(status: HarnessToolEffectStatus) -> Self {
        match status {
            HarnessToolEffectStatus::Started => Self::Started,
            HarnessToolEffectStatus::Completed => Self::Completed,
            HarnessToolEffectStatus::Failed => Self::Failed,
            HarnessToolEffectStatus::Interrupted => Self::Interrupted,
        }
    }

    fn into_harness(self) -> HarnessToolEffectStatus {
        match self {
            Self::Started => HarnessToolEffectStatus::Started,
            Self::Completed => HarnessToolEffectStatus::Completed,
            Self::Failed => HarnessToolEffectStatus::Failed,
            Self::Interrupted => HarnessToolEffectStatus::Interrupted,
        }
    }
}

/// A persisted tool-effect row.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolEffectRow {
    pub run_id: String,
    pub call_id: String,
    pub tool: String,
    pub status: ToolEffectStatus,
    pub idempotency_key: Option<String>,
    pub effect_summary: Option<String>,
    pub started_at: DateTime<Utc>,
    pub settled_at: Option<DateTime<Utc>>,
}

/// Fields a caller supplies to [`record_tool_started`].
#[derive(Debug, Clone)]
pub struct ToolEffectStart {
    pub run_id: String,
    pub call_id: String,
    pub tool: String,
    pub idempotency_key: Option<String>,
    pub effect_summary: Option<String>,
}

/// Fields a caller supplies to [`settle_tool_effect`].
#[derive(Debug, Clone)]
pub struct ToolEffectSettle {
    pub run_id: String,
    pub call_id: String,
    /// Must not be [`ToolEffectStatus::Started`] — settling *to* `started`
    /// is not a meaningful transition; callers use [`record_tool_started`]
    /// for that.
    pub status: ToolEffectStatus,
    pub effect_summary: Option<String>,
}

/// Records that a tool call has been admitted and is about to execute.
///
/// Idempotent on `(run_id, call_id)`: a second `record_tool_started` for the
/// same call re-stamps `started_at` and overwrites the tool/idempotency/
/// summary fields rather than erroring, so a caller that retries the write
/// after an ambiguous failure (timeout, but the write actually landed) does
/// not have to special-case "already started".
pub fn record_tool_started(
    workspace_dir: &std::path::Path,
    start: ToolEffectStart,
) -> Result<ToolEffectRow> {
    let now = Utc::now();
    tracing::debug!(
        "{LOG_PREFIX} record_tool_started run_id={} call_id={} tool={}",
        start.run_id,
        start.call_id,
        start.tool
    );
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO tool_effects (
                run_id, call_id, tool, status, idempotency_key, effect_summary,
                started_at, settled_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL)
             ON CONFLICT(run_id, call_id) DO UPDATE SET
                tool = excluded.tool,
                status = excluded.status,
                idempotency_key = excluded.idempotency_key,
                effect_summary = excluded.effect_summary,
                started_at = excluded.started_at,
                settled_at = NULL",
            params![
                start.run_id,
                start.call_id,
                start.tool,
                ToolEffectStatus::Started.as_str(),
                start.idempotency_key,
                start.effect_summary,
                now.to_rfc3339(),
            ],
        )
        .storage_context("record tool effect started")?;
        get_tool_effect_inner(conn, &start.run_id, &start.call_id)?
            .storage_context("tool effect missing after record_tool_started")
    })
}

/// Records the terminal outcome of a previously-started tool call.
///
/// A settle for a `(run_id, call_id)` with no prior `started` row still
/// inserts one (with `started_at` backfilled to now) rather than erroring —
/// see [`tinyagents_harness::tool::ToolEffectLedger::settled`]'s contract:
/// a ledger attached only after some tools already began must not wedge the
/// loop on an unseen call id.
///
/// Returns `None` only if the write itself did not take effect, which should
/// not happen given the upsert above; kept `Option` to mirror the read-back
/// pattern used by the rest of this crate rather than unwrap in the caller.
pub fn settle_tool_effect(
    workspace_dir: &std::path::Path,
    settle: ToolEffectSettle,
) -> Result<Option<ToolEffectRow>> {
    let now = Utc::now();
    tracing::debug!(
        "{LOG_PREFIX} settle_tool_effect run_id={} call_id={} status={}",
        settle.run_id,
        settle.call_id,
        settle.status.as_str()
    );
    crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO tool_effects (
                run_id, call_id, tool, status, idempotency_key, effect_summary,
                started_at, settled_at
             ) VALUES (?1, ?2, '', ?3, NULL, ?4, ?5, ?5)
             ON CONFLICT(run_id, call_id) DO UPDATE SET
                status = excluded.status,
                effect_summary = COALESCE(excluded.effect_summary, tool_effects.effect_summary),
                settled_at = excluded.settled_at",
            params![
                settle.run_id,
                settle.call_id,
                settle.status.as_str(),
                settle.effect_summary,
                now.to_rfc3339(),
            ],
        )
        .storage_context("settle tool effect")?;
        get_tool_effect_inner(conn, &settle.run_id, &settle.call_id)
    })
}

/// Lists every effect for `run_id` still in [`ToolEffectStatus::Started`] —
/// admitted but never settled, the signature of a crash between the two.
pub fn list_unresolved_tool_effects(
    workspace_dir: &std::path::Path,
    run_id: &str,
) -> Result<Vec<ToolEffectRow>> {
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT run_id, call_id, tool, status, idempotency_key, effect_summary,
                    started_at, settled_at
             FROM tool_effects
             WHERE run_id = ?1 AND status = ?2
             ORDER BY started_at ASC",
        )?;
        let rows = stmt.query_map(
            params![run_id, ToolEffectStatus::Started.as_str()],
            map_tool_effect_row,
        )?;
        let mut effects = Vec::new();
        for row in rows {
            effects.push(row?);
        }
        Ok(effects)
    })
}

/// Marks one call's effect as [`ToolEffectStatus::Interrupted`] — the
/// resume-time verdict for a `started` row whose tool is not safe to
/// blindly re-execute (`tinytools::ToolReplay::Never`).
///
/// Returns `None` if no row exists for `(run_id, call_id)`.
pub fn mark_interrupted(
    workspace_dir: &std::path::Path,
    run_id: &str,
    call_id: &str,
) -> Result<Option<ToolEffectRow>> {
    settle_tool_effect(
        workspace_dir,
        ToolEffectSettle {
            run_id: run_id.to_string(),
            call_id: call_id.to_string(),
            status: ToolEffectStatus::Interrupted,
            effect_summary: None,
        },
    )
}

fn get_tool_effect_inner(
    conn: &rusqlite::Connection,
    run_id: &str,
    call_id: &str,
) -> Result<Option<ToolEffectRow>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, call_id, tool, status, idempotency_key, effect_summary,
                started_at, settled_at
         FROM tool_effects WHERE run_id = ?1 AND call_id = ?2",
    )?;
    Ok(stmt
        .query_row(params![run_id, call_id], map_tool_effect_row)
        .optional()?)
}

fn map_tool_effect_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ToolEffectRow> {
    Ok(ToolEffectRow {
        run_id: row.get(0)?,
        call_id: row.get(1)?,
        tool: row.get(2)?,
        status: ToolEffectStatus::parse(&row.get::<_, String>(3)?),
        idempotency_key: row.get(4)?,
        effect_summary: row.get(5)?,
        started_at: parse_rfc3339(&row.get::<_, String>(6)?)?,
        settled_at: parse_rfc3339_opt(row.get(7)?)?,
    })
}

fn parse_rfc3339(raw: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
        })
}

fn parse_rfc3339_opt(raw: Option<String>) -> rusqlite::Result<Option<DateTime<Utc>>> {
    match raw {
        Some(value) => parse_rfc3339(&value).map(Some),
        None => Ok(None),
    }
}

/// SQLite-backed `tinyagents_harness::tool::ToolEffectLedger`, so a harness
/// run can attach durable tool-effect bookkeeping with
/// `RunContext::with_tool_effect_ledger(Arc::new(RunLedgerToolEffects::new(workspace_dir)))`
/// without the harness crate depending on `tinyagents-session`.
///
/// Thin `async` wrapper: every method call is a bounded, synchronous SQLite
/// statement (matching every other write in this crate), so no `spawn_blocking`
/// is used — the same tradeoff `tinyagents_session::store` already makes
/// throughout.
#[derive(Clone)]
pub struct RunLedgerToolEffects {
    workspace_dir: std::path::PathBuf,
}

impl RunLedgerToolEffects {
    /// Builds a ledger backed by the session database under `workspace_dir`.
    pub fn new(workspace_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }
}

#[async_trait]
impl ToolEffectLedger for RunLedgerToolEffects {
    async fn started(&self, start: HarnessToolEffectStart) -> Result<()> {
        record_tool_started(
            &self.workspace_dir,
            ToolEffectStart {
                run_id: start.run_id.as_str().to_string(),
                call_id: start.call_id.as_str().to_string(),
                tool: start.tool,
                idempotency_key: Some(start.idempotency_key),
                effect_summary: start.effect_summary,
            },
        )?;
        Ok(())
    }

    async fn settled(&self, settle: HarnessToolEffectSettle) -> Result<()> {
        settle_tool_effect(
            &self.workspace_dir,
            ToolEffectSettle {
                run_id: settle.run_id.as_str().to_string(),
                call_id: settle.call_id.as_str().to_string(),
                status: ToolEffectStatus::from_harness(settle.status),
                effect_summary: settle.effect_summary,
            },
        )?;
        Ok(())
    }

    async fn unresolved(&self, run_id: &str) -> Result<Vec<HarnessToolEffect>> {
        let rows = list_unresolved_tool_effects(&self.workspace_dir, run_id)?;
        Ok(rows
            .into_iter()
            .map(|row| HarnessToolEffect {
                run_id: row.run_id,
                call_id: row.call_id,
                tool: row.tool,
                status: row.status.into_harness(),
                idempotency_key: row.idempotency_key,
                effect_summary: row.effect_summary,
                started_at: row.started_at,
                settled_at: row.settled_at,
            })
            .collect())
    }
}
