//! Tool-effect ledger: a durable record of "this run started executing this
//! tool call" that survives a crash between the call starting and its result
//! landing.
//!
//! A tool call that mutates external state (sends an email, charges a card,
//! writes a file) is dangerous to blindly re-run after a process restart: the
//! harness itself has no way to tell "never started" from "started, effect
//! landed, but the process died before the result was folded back into the
//! transcript" from "started, effect landed, crash, and now we are about to
//! run it again". [`ToolEffectLedger`] closes that gap the same way a durable
//! job queue does: write a `started` row *before* the call executes, write a
//! `completed`/`failed` row after it returns, and treat any row still
//! `started` at resume time as evidence of an interrupted effect.
//!
//! This module owns only the harness-side vocabulary — the trait and its
//! payload types. `tinyagents-session`'s `run_ledger::tool_effects` module
//! supplies the durable, SQLite-backed implementation
//! (`RunLedgerToolEffects`), matching how `crate::store::Store` is declared
//! here and implemented by hosts.
//!
//! # Replay classification
//!
//! What a resumed run *does* with an unresolved `started` row is driven by
//! the tool's own [`tinytools::ToolPolicy::runtime`]
//! [`ToolReplay`][tinytools::ToolReplay] declaration, not by this trait:
//! `ToolReplay::Safe` means the call may be blindly re-executed, so the
//! resume path leaves the assistant tool-call unanswered and lets the loop
//! run it again; `ToolReplay::Never` (the default) means the effect might
//! already have landed, so the resume path must not re-run it and instead
//! synthesizes a tool-error result. See
//! [`crate::runtime::AgentHarness::reconcile_tool_effects`] for that logic.

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::ids::{CallId, RunId};

/// Lifecycle state of one recorded tool effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolEffectStatus {
    /// The call was admitted and is about to execute (or is executing); no
    /// terminal outcome has landed yet.
    Started,
    /// The call returned a result (successful or a recoverable tool error)
    /// and the effect is considered settled.
    Completed,
    /// The call's execution future itself failed (as opposed to returning a
    /// recoverable [`tinytools::ToolResult::error`]).
    Failed,
    /// The call was left `started` across a resume and, per the tool's
    /// [`tinytools::ToolReplay`] declaration, was not safe to re-execute —
    /// [`crate::runtime::AgentHarness::reconcile_tool_effects`] settled it
    /// this way instead of running it again.
    Interrupted,
    /// The call was paused mid-execution by the tool itself
    /// (`ApprovalRequired`/`CallDeferred`) and is waiting on
    /// [`crate::runtime::AgentHarness::resume_deferred`] to answer it. This
    /// is a deliberate pause, not a crash artifact: settling to this status
    /// (instead of leaving the row `started`) is what keeps
    /// [`crate::runtime::AgentHarness::reconcile_tool_effects`] — which only
    /// reconciles rows still `started` — from mistaking a live deferral for
    /// an interrupted run.
    Deferred,
}

impl ToolEffectStatus {
    /// Renders the status as the stable, lower-case string a durable backend
    /// stores and a host may log or filter on.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
            Self::Deferred => "deferred",
        }
    }

    /// Parses a stored status string, defaulting to [`Self::Started`] for any
    /// unrecognized value (mirrors the run-ledger status-enum convention:
    /// fail toward "still open" rather than silently dropping a row from
    /// resume consideration).
    pub fn parse(raw: &str) -> Self {
        match raw {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "interrupted" => Self::Interrupted,
            "deferred" => Self::Deferred,
            _ => Self::Started,
        }
    }
}

/// Fields recorded when a tool call is admitted, before it executes.
#[derive(Clone, Debug)]
pub struct ToolEffectStart {
    /// The run whose transcript this call belongs to.
    pub run_id: RunId,
    /// The call's correlation id (matches the transcript's `tool_call_id`).
    pub call_id: CallId,
    /// Name of the tool being invoked.
    pub tool: String,
    /// Deduplication key for this call, derived from the tool's declared
    /// idempotency key when it supplies one, otherwise a content hash of
    /// `(tool name, arguments)`. Lets a host detect "this exact call was
    /// already attempted" independent of the (per-attempt-unique) `call_id`.
    pub idempotency_key: String,
    /// Optional short, human-readable description of the effect about to be
    /// attempted (e.g. `"send email to a@example.com"`), for audit display.
    pub effect_summary: Option<String>,
}

/// Fields recorded when a previously-started tool effect settles.
#[derive(Clone, Debug)]
pub struct ToolEffectSettle {
    /// The run whose transcript this call belongs to.
    pub run_id: RunId,
    /// The call's correlation id; must match a prior [`ToolEffectStart`].
    pub call_id: CallId,
    /// Terminal status. [`ToolEffectStatus::Started`] is not a valid value
    /// here — an implementation may treat it as a programmer error.
    pub status: ToolEffectStatus,
    /// Optional short, human-readable description of the effect's outcome,
    /// overwriting the summary supplied at start (e.g. `"sent, message id
    /// abc123"`). `None` leaves the started summary as-is.
    pub effect_summary: Option<String>,
}

/// A durably-recorded tool effect, as returned by
/// [`ToolEffectLedger::unresolved`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolEffect {
    /// The run whose transcript this call belongs to.
    pub run_id: String,
    /// The call's correlation id.
    pub call_id: String,
    /// Name of the tool that was invoked.
    pub tool: String,
    /// Current lifecycle status.
    pub status: ToolEffectStatus,
    /// Deduplication key recorded at start, when the backend stores it.
    pub idempotency_key: Option<String>,
    /// Most recent human-readable effect summary.
    pub effect_summary: Option<String>,
    /// When the call was admitted.
    pub started_at: DateTime<Utc>,
    /// When the call settled, if it has.
    pub settled_at: Option<DateTime<Utc>>,
}

/// Durable record of tool-call side effects for crash recovery.
///
/// Implementations must be safe to call concurrently from multiple tool
/// executions within (and across) runs. `tinyagents-session` supplies a
/// SQLite-backed implementation (`run_ledger::tool_effects::RunLedgerToolEffects`);
/// a host may supply its own for a different durability substrate.
#[async_trait]
pub trait ToolEffectLedger: Send + Sync {
    /// Records that a tool call has been admitted and is about to execute.
    ///
    /// Must be durable *before* the tool actually runs — see
    /// [`LedgerFailure`] for how the harness reacts when this call itself
    /// fails.
    async fn started(&self, start: ToolEffectStart) -> Result<()>;

    /// Records the terminal outcome of a previously-started tool call.
    ///
    /// A backend that receives a settle for a `call_id` it never saw
    /// `started` should still record it rather than error, so a ledger
    /// attached only after some tools already began does not wedge the loop.
    async fn settled(&self, settle: ToolEffectSettle) -> Result<()>;

    /// Lists every effect for `run_id` still in [`ToolEffectStatus::Started`]
    /// — i.e. admitted but never settled, the signature of a crash between
    /// the two.
    async fn unresolved(&self, run_id: &str) -> Result<Vec<ToolEffect>>;
}

/// How the harness reacts when [`ToolEffectLedger::started`] itself returns
/// an error, before the tool call it was about to journal has executed.
///
/// The ledger exists to make crash recovery trustworthy; if writing to it is
/// itself unreliable, a host has to choose explicitly between two failure
/// modes rather than have one silently assumed:
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LedgerFailure {
    /// Fail the tool call (and, by the normal error-propagation path, the
    /// run) rather than execute a side-effecting call the ledger could not
    /// record. This is the default: a ledger that cannot be trusted to
    /// record "started" cannot be trusted to detect "interrupted" either, so
    /// failing closed is the safer default for any tool with real-world
    /// effects.
    #[default]
    Abort,
    /// Log the ledger failure and execute the tool call anyway, forgoing
    /// crash-recovery coverage for this one call. Appropriate for a host that
    /// would rather keep a run alive through a transient ledger outage than
    /// block on it, and that classifies its own tools as safe enough to run
    /// unrecorded.
    Continue,
}
