//! Typed records read by a host from the retired session database.
//!
//! The host owns source authorization and decoding. This module carries only
//! the rows that the session owner can validate and persist; it has no path or
//! database dependency.

use crate::run_ledger::tool_effects::ToolEffectRow;
use crate::run_ledger::types::{
    AgentRun, AgentTeam, AgentTeamMember, AgentTeamTask, RunEvent, RunTelemetry, WorkflowRun,
};
use crate::types::{SessionMessage, SessionRecord, SessionToolCall};

/// A host-decoded batch from the former `sessions.db`.
///
/// Message and tool-call IDs, run-event sequence numbers, row timestamps, and
/// source ordering are supplied by the reader and are never reallocated here.
/// Transcript JSONL and turn-state snapshots are separate import surfaces.
#[derive(Debug, Clone, Default)]
pub struct LegacySessionImport {
    /// Session history rows.
    pub sessions: Vec<SessionRecord>,
    /// Message rows with their original integer IDs.
    pub messages: Vec<SessionMessage>,
    /// Tool-call rows with their original integer IDs.
    pub tool_calls: Vec<SessionToolCall>,
    /// Agent runs.
    pub agent_runs: Vec<AgentRun>,
    /// Run telemetry rows.
    pub telemetry: Vec<RunTelemetry>,
    /// Run events with their original per-run sequence numbers.
    pub events: Vec<RunEvent>,
    /// Workflow runs, including any legacy live-lease fields.
    pub workflow_runs: Vec<WorkflowRun>,
    /// Agent teams.
    pub teams: Vec<AgentTeam>,
    /// Team members.
    pub members: Vec<AgentTeamMember>,
    /// Team tasks.
    pub tasks: Vec<AgentTeamTask>,
    /// Tool effects.
    pub tool_effects: Vec<ToolEffectRow>,
}

/// Counts for one family in one import call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ImportCounts {
    /// Source rows supplied by the host.
    pub read: usize,
    /// Rows inserted by this call.
    pub imported: usize,
    /// Identical rows already present and left unchanged.
    pub already_present: usize,
}

impl ImportCounts {
    pub(super) fn new(read: usize, imported: usize) -> Self {
        Self {
            read,
            imported,
            already_present: read.saturating_sub(imported),
        }
    }
}

/// Per-family result of an idempotent typed database import.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LegacyDatabaseReport {
    /// Session rows.
    pub sessions: ImportCounts,
    /// Message rows.
    pub messages: ImportCounts,
    /// Tool-call rows.
    pub tool_calls: ImportCounts,
    /// Agent-run rows.
    pub agent_runs: ImportCounts,
    /// Telemetry rows.
    pub telemetry: ImportCounts,
    /// Run-event rows.
    pub events: ImportCounts,
    /// Workflow-run rows.
    pub workflow_runs: ImportCounts,
    /// Team rows.
    pub teams: ImportCounts,
    /// Team-member rows.
    pub members: ImportCounts,
    /// Team-task rows.
    pub tasks: ImportCounts,
    /// Tool-effect rows.
    pub tool_effects: ImportCounts,
    /// Former entry-tree rows, unsupported by this owner and always zero.
    pub entries: ImportCounts,
    /// Derived full-text index rows inserted or repaired by this call.
    pub fts_rows: usize,
}
