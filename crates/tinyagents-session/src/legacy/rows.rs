//! Row readers for a classic `sessions.db`.
//!
//! One function per table, each returning domain records exactly as the
//! classic store would have read them. Reads are tolerant of older schemas:
//! a column added by a later migration (`reasoning_content`, the workflow
//! lease columns) reads as `NULL`/default when the source predates it, and a
//! table that does not exist yields no rows. Nothing here writes, migrates or
//! repairs the source.
//!
//! Timestamps are RFC 3339 strings in the source; a value that does not parse
//! fails the whole read rather than inventing a time, because the import is
//! id- and order-preserving and must not fabricate history.

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use tinystoragedrivers_core::{ErrorKind, StorageError};
use tinystoragedrivers_sqlite::rusqlite::{self, Row, types::Type};

use super::source::{LegacySourceSnapshot, source_sql_error};
use crate::run_ledger::tool_effects::ToolEffectRow;
use crate::run_ledger::{
    AgentRun, AgentRunKind, AgentRunStatus, AgentTeam, AgentTeamMember, AgentTeamMemberStatus,
    AgentTeamStatus, AgentTeamTask, AgentTeamTaskStatus, LedgerResult, RunEvent, RunTelemetry,
    ToolEffectStatus, WorkflowRun, WorkflowRunStatus,
};
use crate::types::{SessionMessage, SessionRecord, SessionStatus, SessionToolCall};

/// A `SELECT` list over `table` that substitutes a literal for each column the
/// source predates.
fn select_list(
    snapshot: &LegacySourceSnapshot,
    table: &str,
    wanted: &[(&str, &str)],
) -> LedgerResult<String> {
    let present = snapshot.columns(table)?;
    Ok(wanted
        .iter()
        .map(|(column, fallback)| {
            if present.iter().any(|name| name == column) {
                (*column).to_owned()
            } else {
                format!("{fallback} AS {column}")
            }
        })
        .collect::<Vec<_>>()
        .join(", "))
}

fn read_all<T>(
    snapshot: &LegacySourceSnapshot,
    table: &str,
    wanted: &[(&str, &str)],
    order_by: &str,
    map: impl FnMut(&Row<'_>) -> rusqlite::Result<T>,
) -> LedgerResult<Vec<T>> {
    if !snapshot.has_table(table) {
        return Ok(Vec::new());
    }
    let list = select_list(snapshot, table, wanted)?;
    let mut statement = snapshot
        .connection
        .prepare(&format!("SELECT {list} FROM {table} ORDER BY {order_by}"))
        .map_err(source_sql_error)?;
    let rows = statement
        .query_map([], map)
        .map_err(source_sql_error)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(source_sql_error)?;
    Ok(rows)
}

fn time(row: &Row<'_>, index: usize) -> rusqlite::Result<DateTime<Utc>> {
    let raw: String = row.get(index)?;
    DateTime::parse_from_rfc3339(&raw)
        .map(|time| time.with_timezone(&Utc))
        .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(error))
        })
}

fn time_opt(row: &Row<'_>, index: usize) -> rusqlite::Result<Option<DateTime<Utc>>> {
    match row.get::<_, Option<String>>(index)? {
        Some(raw) => DateTime::parse_from_rfc3339(&raw)
            .map(|time| Some(time.with_timezone(&Utc)))
            .map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(error))
            }),
        None => Ok(None),
    }
}

fn json_or_empty(raw: String) -> Value {
    serde_json::from_str(&raw).unwrap_or_else(|_| json!({}))
}

fn unsigned(row: &Row<'_>, index: usize) -> rusqlite::Result<u64> {
    let value: i64 = row.get(index)?;
    u64::try_from(value).map_err(|_| rusqlite::Error::InvalidQuery)
}

/// Every session, oldest first.
pub(crate) fn sessions(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<SessionRecord>> {
    read_all(
        snapshot,
        "sessions",
        &[
            ("id", "NULL"),
            ("agent_definition_id", "''"),
            ("agent_definition_name", "''"),
            ("session_key", "''"),
            ("parent_session_id", "NULL"),
            ("thread_id", "NULL"),
            ("source_channel", "NULL"),
            ("status", "'running'"),
            ("model", "NULL"),
            ("turn_count", "0"),
            ("input_tokens", "0"),
            ("output_tokens", "0"),
            ("cached_input_tokens", "0"),
            ("cost_usd", "0.0"),
            ("transcript_path", "NULL"),
            ("started_at", "NULL"),
            ("ended_at", "NULL"),
        ],
        "started_at, id",
        |row| {
            Ok(SessionRecord {
                id: row.get(0)?,
                agent_definition_id: row.get(1)?,
                agent_definition_name: row.get(2)?,
                session_key: row.get(3)?,
                parent_session_id: row.get(4)?,
                thread_id: row.get(5)?,
                source_channel: row.get(6)?,
                status: SessionStatus::parse(&row.get::<_, String>(7)?),
                model: row.get(8)?,
                turn_count: u32::try_from(row.get::<_, i64>(9)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
                input_tokens: unsigned(row, 10)?,
                output_tokens: unsigned(row, 11)?,
                cached_input_tokens: unsigned(row, 12)?,
                cost_usd: row.get(13)?,
                transcript_path: row.get(14)?,
                started_at: time(row, 15)?,
                ended_at: time_opt(row, 16)?,
            })
        },
    )
}

/// Every message, by original id.
pub(crate) fn messages(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<SessionMessage>> {
    read_all(
        snapshot,
        "session_messages",
        &[
            ("id", "NULL"),
            ("session_id", "NULL"),
            ("role", "''"),
            ("content", "''"),
            ("reasoning_content", "NULL"),
            ("model", "NULL"),
            ("input_tokens", "NULL"),
            ("output_tokens", "NULL"),
            ("cost_usd", "NULL"),
            ("created_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(SessionMessage {
                id: row.get(0)?,
                session_id: row.get(1)?,
                role: row.get(2)?,
                content: row.get(3)?,
                reasoning_content: row.get(4)?,
                model: row.get(5)?,
                input_tokens: row.get::<_, Option<i64>>(6)?.map(|v| v as u64),
                output_tokens: row.get::<_, Option<i64>>(7)?.map(|v| v as u64),
                cost_usd: row.get(8)?,
                created_at: time(row, 9)?,
            })
        },
    )
}

/// Every tool call, by original id.
pub(crate) fn tool_calls(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<SessionToolCall>> {
    read_all(
        snapshot,
        "session_tool_calls",
        &[
            ("id", "NULL"),
            ("session_id", "NULL"),
            ("message_id", "NULL"),
            ("tool_name", "''"),
            ("tool_input", "NULL"),
            ("tool_output", "NULL"),
            ("status", "'pending'"),
            ("duration_ms", "NULL"),
            ("created_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(SessionToolCall {
                id: row.get(0)?,
                session_id: row.get(1)?,
                message_id: row.get(2)?,
                tool_name: row.get(3)?,
                tool_input: row.get(4)?,
                tool_output: row.get(5)?,
                status: row.get(6)?,
                duration_ms: row.get(7)?,
                created_at: time(row, 8)?,
            })
        },
    )
}

/// Every agent run (without its telemetry; see [`telemetry`]).
///
/// An unknown `kind` or `status` string fails the read: the classic parser
/// would silently map it to a default, and an import must not rewrite the
/// meaning of history.
pub(crate) fn agent_runs(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<AgentRun>> {
    read_all(
        snapshot,
        "agent_runs",
        &[
            ("id", "NULL"),
            ("kind", "''"),
            ("parent_run_id", "NULL"),
            ("parent_thread_id", "NULL"),
            ("agent_id", "NULL"),
            ("status", "''"),
            ("prompt_ref", "NULL"),
            ("worker_thread_id", "NULL"),
            ("checkpoint_path", "NULL"),
            ("checkpoint_json", "NULL"),
            ("summary", "NULL"),
            ("error", "NULL"),
            ("metadata_json", "'{}'"),
            ("started_at", "NULL"),
            ("updated_at", "NULL"),
            ("completed_at", "NULL"),
        ],
        "id",
        |row| {
            let kind_raw: String = row.get(1)?;
            let status_raw: String = row.get(5)?;
            let kind = AgentRunKind::parse(&kind_raw);
            let status = AgentRunStatus::parse(&status_raw);
            if kind.as_str() != kind_raw || status.as_str() != status_raw {
                return Err(rusqlite::Error::InvalidQuery);
            }
            for index in [9, 12] {
                if let Some(raw) = row.get::<_, Option<String>>(index)? {
                    serde_json::from_str::<Value>(&raw).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            index,
                            Type::Text,
                            Box::new(error),
                        )
                    })?;
                }
            }
            let checkpoint: Option<String> = row.get(9)?;
            Ok(AgentRun {
                id: row.get(0)?,
                kind,
                parent_run_id: row.get(2)?,
                parent_thread_id: row.get(3)?,
                agent_id: row.get(4)?,
                status,
                prompt_ref: row.get(6)?,
                worker_thread_id: row.get(7)?,
                checkpoint_path: row.get(8)?,
                checkpoint: checkpoint.and_then(|raw| serde_json::from_str(&raw).ok()),
                summary: row.get(10)?,
                error: row.get(11)?,
                metadata: json_or_empty(row.get(12)?),
                telemetry: None,
                started_at: time(row, 13)?,
                updated_at: time(row, 14)?,
                completed_at: time_opt(row, 15)?,
            })
        },
    )
}

/// Every run-telemetry row. A negative counter fails the read.
pub(crate) fn telemetry(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<RunTelemetry>> {
    read_all(
        snapshot,
        "run_telemetry",
        &[
            ("run_id", "NULL"),
            ("input_tokens", "0"),
            ("output_tokens", "0"),
            ("cached_input_tokens", "0"),
            ("cost_usd", "0.0"),
            ("elapsed_ms", "NULL"),
            ("tool_count", "0"),
            ("model", "NULL"),
            ("provider", "NULL"),
            ("error", "NULL"),
            ("updated_at", "NULL"),
        ],
        "run_id",
        |row| {
            Ok(RunTelemetry {
                run_id: row.get(0)?,
                input_tokens: unsigned(row, 1)?,
                output_tokens: unsigned(row, 2)?,
                cached_input_tokens: unsigned(row, 3)?,
                cost_usd: row.get(4)?,
                elapsed_ms: row
                    .get::<_, Option<i64>>(5)?
                    .map(|v| u64::try_from(v).map_err(|_| rusqlite::Error::InvalidQuery))
                    .transpose()?,
                tool_count: unsigned(row, 6)?,
                model: row.get(7)?,
                provider: row.get(8)?,
                error: row.get(9)?,
                updated_at: Some(time(row, 10)?),
            })
        },
    )
}

/// Every run event in `(run, sequence)` order. A non-positive sequence fails the read.
pub(crate) fn run_events(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<RunEvent>> {
    read_all(
        snapshot,
        "run_events",
        &[
            ("run_id", "NULL"),
            ("sequence", "NULL"),
            ("event_type", "''"),
            ("payload_json", "'{}'"),
            ("timestamp", "NULL"),
        ],
        "run_id, sequence",
        |row| {
            let sequence: i64 = row.get(1)?;
            if sequence <= 0 {
                return Err(rusqlite::Error::InvalidQuery);
            }
            let payload: String = row.get(3)?;
            serde_json::from_str::<Value>(&payload).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(3, Type::Text, Box::new(error))
            })?;
            Ok(RunEvent {
                run_id: row.get(0)?,
                sequence: sequence as u64,
                event_type: row.get(2)?,
                payload: json_or_empty(payload),
                timestamp: time(row, 4)?,
            })
        },
    )
}

/// Every workflow run, including its lease when the source has one.
pub(crate) fn workflow_runs(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<WorkflowRun>> {
    read_all(
        snapshot,
        "workflow_runs",
        &[
            ("id", "NULL"),
            ("definition_id", "''"),
            ("parent_thread_id", "NULL"),
            ("input_json", "'{}'"),
            ("phase_states_json", "'{}'"),
            ("child_run_ids_json", "'[]'"),
            ("status", "''"),
            ("summary", "NULL"),
            ("started_at", "NULL"),
            ("updated_at", "NULL"),
            ("completed_at", "NULL"),
            ("revision", "0"),
            ("lease_owner", "NULL"),
            ("lease_expires_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(WorkflowRun {
                id: row.get(0)?,
                definition_id: row.get(1)?,
                parent_thread_id: row.get(2)?,
                input: json_or_empty(row.get(3)?),
                phase_states: json_or_empty(row.get(4)?),
                child_run_ids: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                status: WorkflowRunStatus::parse(&row.get::<_, String>(6)?),
                summary: row.get(7)?,
                started_at: time(row, 8)?,
                updated_at: time(row, 9)?,
                completed_at: time_opt(row, 10)?,
                revision: unsigned(row, 11)?,
                lease_owner: row.get(12)?,
                lease_expires_at: time_opt(row, 13)?,
            })
        },
    )
}

/// Every agent team.
pub(crate) fn teams(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<AgentTeam>> {
    read_all(
        snapshot,
        "agent_teams",
        &[
            ("id", "NULL"),
            ("parent_thread_id", "NULL"),
            ("lead_agent_id", "''"),
            ("status", "''"),
            ("summary", "NULL"),
            ("created_at", "NULL"),
            ("updated_at", "NULL"),
            ("closed_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(AgentTeam {
                id: row.get(0)?,
                parent_thread_id: row.get(1)?,
                lead_agent_id: row.get(2)?,
                status: AgentTeamStatus::parse(&row.get::<_, String>(3)?),
                summary: row.get(4)?,
                created_at: time(row, 5)?,
                updated_at: time(row, 6)?,
                closed_at: time_opt(row, 7)?,
            })
        },
    )
}

/// Every team member.
pub(crate) fn members(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<AgentTeamMember>> {
    read_all(
        snapshot,
        "agent_team_members",
        &[
            ("id", "NULL"),
            ("team_id", "NULL"),
            ("name", "''"),
            ("agent_id", "NULL"),
            ("member_status", "''"),
            ("current_task_id", "NULL"),
            ("worker_thread_id", "NULL"),
            ("run_id", "NULL"),
            ("created_at", "NULL"),
            ("updated_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(AgentTeamMember {
                id: row.get(0)?,
                team_id: row.get(1)?,
                name: row.get(2)?,
                agent_id: row.get(3)?,
                member_status: AgentTeamMemberStatus::parse(&row.get::<_, String>(4)?),
                current_task_id: row.get(5)?,
                worker_thread_id: row.get(6)?,
                run_id: row.get(7)?,
                created_at: time(row, 8)?,
                updated_at: time(row, 9)?,
            })
        },
    )
}

/// Every team task, with its claim and evidence.
pub(crate) fn team_tasks(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<AgentTeamTask>> {
    read_all(
        snapshot,
        "agent_team_tasks",
        &[
            ("id", "NULL"),
            ("team_id", "NULL"),
            ("title", "''"),
            ("objective", "NULL"),
            ("status", "''"),
            ("owner_member_id", "NULL"),
            ("claimed_by_member_id", "NULL"),
            ("claim_token", "NULL"),
            ("depends_on_json", "'[]'"),
            ("gate_status", "'pending'"),
            ("gate_reason", "NULL"),
            ("evidence_json", "'[]'"),
            ("source_run_id", "NULL"),
            ("order_index", "0"),
            ("created_at", "NULL"),
            ("updated_at", "NULL"),
        ],
        "id",
        |row| {
            Ok(AgentTeamTask {
                id: row.get(0)?,
                team_id: row.get(1)?,
                title: row.get(2)?,
                objective: row.get(3)?,
                status: AgentTeamTaskStatus::parse(&row.get::<_, String>(4)?),
                owner_member_id: row.get(5)?,
                claimed_by_member_id: row.get(6)?,
                claim_token: row.get(7)?,
                depends_on: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                gate_status: row.get(9)?,
                gate_reason: row.get(10)?,
                evidence: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                source_run_id: row.get(12)?,
                order_index: row.get(13)?,
                created_at: time(row, 14)?,
                updated_at: time(row, 15)?,
            })
        },
    )
}

/// Every tool-effect row.
pub(crate) fn tool_effects(snapshot: &LegacySourceSnapshot) -> LedgerResult<Vec<ToolEffectRow>> {
    read_all(
        snapshot,
        "tool_effects",
        &[
            ("run_id", "NULL"),
            ("call_id", "NULL"),
            ("tool", "''"),
            ("status", "'started'"),
            ("idempotency_key", "NULL"),
            ("effect_summary", "NULL"),
            ("started_at", "NULL"),
            ("settled_at", "NULL"),
        ],
        "run_id, call_id",
        |row| {
            Ok(ToolEffectRow {
                run_id: row.get(0)?,
                call_id: row.get(1)?,
                tool: row.get(2)?,
                status: ToolEffectStatus::parse(&row.get::<_, String>(3)?),
                idempotency_key: row.get(4)?,
                effect_summary: row.get(5)?,
                started_at: time(row, 6)?,
                settled_at: time_opt(row, 7)?,
            })
        },
    )
}

/// A source that is not a classic session database fails fast and typed.
pub(crate) fn require_classic(snapshot: &LegacySourceSnapshot) -> LedgerResult<()> {
    let known = [
        "sessions",
        "session_messages",
        "agent_runs",
        "workflow_runs",
        "agent_teams",
        "tool_effects",
    ];
    if known.iter().any(|table| snapshot.has_table(table)) {
        return Ok(());
    }
    Err(StorageError::new(
        ErrorKind::InvalidInput,
        "source has none of the classic session database tables",
    )
    .into())
}
