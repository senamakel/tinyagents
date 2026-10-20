//! Durable tool-effect ledger (B5): crash-safe bookkeeping of tool-call side
//! effects, backing `tinyagents_harness::tool::ToolEffectLedger`.
//!
//! A tool call that mutates external state is dangerous to blindly re-run
//! after a process restart. This module gives the agent loop a durable
//! `started` → `completed`/`failed`/`interrupted` row per call, stored by the
//! run ledger, so a resumed run can tell an in-flight call apart from one that
//! never started or one that already settled.
//!
//! # Layout
//!
//! The row types ([`ToolEffectRow`], [`ToolEffectStart`], [`ToolEffectSettle`],
//! [`ToolEffectStatus`]) are the vocabulary of the ledger's
//! `record_tool_started`, `settle_tool_effect`, `list_unresolved_tool_effects`
//! and `mark_tool_effect_interrupted` operations on [`RunLedgerStore`].
//! [`RunLedgerToolEffects`] wraps a ledger handle behind the `async`
//! `tinyagents_harness::tool::ToolEffectLedger` trait so a harness run can
//! attach it via
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

use std::sync::Arc;

use chrono::{DateTime, Utc};

use async_trait::async_trait;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::tool::{
    ToolEffect as HarnessToolEffect, ToolEffectLedger, ToolEffectSettle as HarnessToolEffectSettle,
    ToolEffectStart as HarnessToolEffectStart, ToolEffectStatus as HarnessToolEffectStatus,
};

use super::port::RunLedgerStore;

/// Lifecycle state of one recorded tool effect.
///
/// Mirrors `tinyagents_harness::tool::ToolEffectStatus` (see
/// `from_harness` / `into_harness`)
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
    /// Paused mid-execution by the tool itself (`ApprovalRequired`/
    /// `CallDeferred`) and waiting on a resume to answer it.
    Deferred,
}

impl ToolEffectStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
            Self::Deferred => "deferred",
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
            "deferred" => Self::Deferred,
            _ => Self::Started,
        }
    }

    fn from_harness(status: HarnessToolEffectStatus) -> Self {
        match status {
            HarnessToolEffectStatus::Started => Self::Started,
            HarnessToolEffectStatus::Completed => Self::Completed,
            HarnessToolEffectStatus::Failed => Self::Failed,
            HarnessToolEffectStatus::Interrupted => Self::Interrupted,
            HarnessToolEffectStatus::Deferred => Self::Deferred,
        }
    }

    fn into_harness(self) -> HarnessToolEffectStatus {
        match self {
            Self::Started => HarnessToolEffectStatus::Started,
            Self::Completed => HarnessToolEffectStatus::Completed,
            Self::Failed => HarnessToolEffectStatus::Failed,
            Self::Interrupted => HarnessToolEffectStatus::Interrupted,
            Self::Deferred => HarnessToolEffectStatus::Deferred,
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

/// Fields a caller supplies to [`RunLedgerStore::record_tool_started`].
#[derive(Debug, Clone)]
pub struct ToolEffectStart {
    pub run_id: String,
    pub call_id: String,
    pub tool: String,
    pub idempotency_key: Option<String>,
    pub effect_summary: Option<String>,
}

/// Fields a caller supplies to [`RunLedgerStore::settle_tool_effect`].
#[derive(Debug, Clone)]
pub struct ToolEffectSettle {
    pub run_id: String,
    pub call_id: String,
    /// Must not be [`ToolEffectStatus::Started`] — settling *to* `started`
    /// is not a meaningful transition; callers use [`RunLedgerStore::record_tool_started`]
    /// for that.
    pub status: ToolEffectStatus,
    pub effect_summary: Option<String>,
}

/// A `tinyagents_harness::tool::ToolEffectLedger` over a bound
/// [`RunLedgerStore`], so a harness run can attach durable tool-effect
/// bookkeeping with
/// `RunContext::with_tool_effect_ledger(Arc::new(RunLedgerToolEffects::new(ledger)))`
/// without the harness crate depending on `tinyagents-session`.
///
/// The ledger is whatever backend the host opened (SQLite, MongoDB, memory):
/// the adapter holds no path and opens no file. Calls are the ledger's own
/// atomic per-call operations, so a row is written `started` before the call
/// executes and settled in place afterwards.
#[derive(Clone)]
pub struct RunLedgerToolEffects {
    ledger: Arc<dyn RunLedgerStore>,
}

impl RunLedgerToolEffects {
    /// Builds a tool-effect ledger over `ledger`.
    pub fn new(ledger: Arc<dyn RunLedgerStore>) -> Self {
        Self { ledger }
    }
}

impl std::fmt::Debug for RunLedgerToolEffects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunLedgerToolEffects")
            .finish_non_exhaustive()
    }
}

fn storage(error: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Storage(format!("tool-effect ledger: {error}"))
}

#[async_trait]
impl ToolEffectLedger for RunLedgerToolEffects {
    async fn started(&self, start: HarnessToolEffectStart) -> Result<()> {
        self.ledger
            .record_tool_started(ToolEffectStart {
                run_id: start.run_id.as_str().to_string(),
                call_id: start.call_id.as_str().to_string(),
                tool: start.tool,
                idempotency_key: Some(start.idempotency_key),
                effect_summary: start.effect_summary,
            })
            .await
            .map(|_| ())
            .map_err(storage)
    }

    async fn settled(&self, settle: HarnessToolEffectSettle) -> Result<()> {
        self.ledger
            .settle_tool_effect(ToolEffectSettle {
                run_id: settle.run_id.as_str().to_string(),
                call_id: settle.call_id.as_str().to_string(),
                status: ToolEffectStatus::from_harness(settle.status),
                effect_summary: settle.effect_summary,
            })
            .await
            .map(|_| ())
            .map_err(storage)
    }

    async fn unresolved(&self, run_id: &str) -> Result<Vec<HarnessToolEffect>> {
        let rows = self
            .ledger
            .list_unresolved_tool_effects(run_id)
            .await
            .map_err(storage)?;
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
