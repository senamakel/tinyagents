//! SQLite row decoding and transaction-scoped lookup helpers.
//!
//! Keeping this detail outside the public ledger operations makes the
//! operation module focused on state transitions and query semantics.

use super::*;

/// Connection-scoped team lookup, so an upsert can read its own write back
/// inside the same transaction.
pub(super) fn get_agent_team_inner(conn: &Connection, id: &str) -> Result<Option<AgentTeam>> {
    let mut stmt = conn.prepare(
        "SELECT id, parent_thread_id, lead_agent_id, status, summary,
                created_at, updated_at, closed_at
         FROM agent_teams WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_row)
        .optional()
        .map_err(Into::into)
}

/// Connection-scoped member lookup, so an upsert can read its own write back
/// inside the same transaction.
pub(super) fn get_agent_team_member_inner(
    conn: &Connection,
    id: &str,
) -> Result<Option<AgentTeamMember>> {
    let mut stmt = conn.prepare(
        "SELECT id, team_id, name, agent_id, member_status,
                current_task_id, worker_thread_id, run_id, created_at, updated_at
         FROM agent_team_members WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_member_row)
        .optional()
        .map_err(Into::into)
}

/// Connection-scoped task lookup, so a claim/completion transaction can read
/// its own write back inside the same transaction.
pub(super) fn get_agent_team_task_inner(
    conn: &Connection,
    id: &str,
) -> Result<Option<AgentTeamTask>> {
    let mut stmt = conn.prepare(
        "SELECT id, team_id, title, objective, status, owner_member_id,
                claimed_by_member_id, claim_token, depends_on_json, gate_status,
                gate_reason, evidence_json, source_run_id, order_index,
                created_at, updated_at
         FROM agent_team_tasks WHERE id = ?1",
    )?;
    stmt.query_row(params![id], map_agent_team_task_row)
        .optional()
        .map_err(Into::into)
}

pub(super) fn map_agent_team_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTeam> {
    Ok(AgentTeam {
        id: row.get(0)?,
        parent_thread_id: row.get(1)?,
        lead_agent_id: row.get(2)?,
        status: AgentTeamStatus::parse(&row.get::<_, String>(3)?),
        summary: row.get(4)?,
        created_at: parse_rfc3339(&row.get::<_, String>(5)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(6)?)?,
        closed_at: parse_rfc3339_opt(row.get(7)?)?,
    })
}

pub(super) fn map_agent_team_member_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<AgentTeamMember> {
    Ok(AgentTeamMember {
        id: row.get(0)?,
        team_id: row.get(1)?,
        name: row.get(2)?,
        agent_id: row.get(3)?,
        member_status: AgentTeamMemberStatus::parse(&row.get::<_, String>(4)?),
        current_task_id: row.get(5)?,
        worker_thread_id: row.get(6)?,
        run_id: row.get(7)?,
        created_at: parse_rfc3339(&row.get::<_, String>(8)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(9)?)?,
    })
}

pub(super) fn map_agent_team_task_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTeamTask> {
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
        created_at: parse_rfc3339(&row.get::<_, String>(14)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(15)?)?,
    })
}

/// Connection-scoped agent-run lookup, so an upsert can read its own write
/// back inside the same transaction.
pub(super) fn get_agent_run_inner(conn: &Connection, id: &str) -> Result<Option<AgentRun>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, parent_run_id, parent_thread_id, agent_id, status,
                prompt_ref, worker_thread_id, task_board_id, task_card_id,
                checkpoint_path, checkpoint_json, summary, error, metadata_json,
                started_at, updated_at, completed_at
         FROM agent_runs WHERE id = ?1",
    )?;
    stmt.query_row(params![id], |row| map_agent_run_row(conn, row))
        .optional()
        .map_err(Into::into)
}

/// Connection-scoped telemetry lookup that errors when the row is absent —
/// used right after [`upsert_run_telemetry`] writes it, where a miss means
/// the write silently failed.
pub(super) fn get_run_telemetry_inner(conn: &Connection, run_id: &str) -> Result<RunTelemetry> {
    let mut stmt = conn.prepare(
        "SELECT run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
         FROM run_telemetry WHERE run_id = ?1",
    )?;
    stmt.query_row(params![run_id], map_run_telemetry_row)
        .storage_context("run telemetry missing after upsert")
}

/// Connection-scoped telemetry lookup used when joining telemetry onto an
/// [`AgentRun`], where no telemetry row yet existing is a normal `None`
/// rather than an error.
pub(super) fn get_optional_run_telemetry(
    conn: &Connection,
    run_id: &str,
) -> rusqlite::Result<Option<RunTelemetry>> {
    let mut stmt = conn.prepare(
        "SELECT run_id, input_tokens, output_tokens, cached_input_tokens, cost_usd,
                elapsed_ms, tool_count, model, provider, error, updated_at
         FROM run_telemetry WHERE run_id = ?1",
    )?;
    stmt.query_row(params![run_id], map_run_telemetry_row)
        .optional()
}

pub(super) fn map_agent_run_row(
    conn: &Connection,
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<AgentRun> {
    let id: String = row.get(0)?;
    let checkpoint_json: Option<String> = row.get(11)?;
    let metadata_json: String = row.get(14)?;
    Ok(AgentRun {
        id: id.clone(),
        kind: AgentRunKind::parse(&row.get::<_, String>(1)?),
        parent_run_id: row.get(2)?,
        parent_thread_id: row.get(3)?,
        agent_id: row.get(4)?,
        status: AgentRunStatus::parse(&row.get::<_, String>(5)?),
        prompt_ref: row.get(6)?,
        worker_thread_id: row.get(7)?,
        task_board_id: row.get(8)?,
        task_card_id: row.get(9)?,
        checkpoint_path: row.get(10)?,
        checkpoint: parse_json_opt(checkpoint_json),
        summary: row.get(12)?,
        error: row.get(13)?,
        metadata: parse_json(metadata_json),
        telemetry: get_optional_run_telemetry(conn, &id)?,
        started_at: parse_rfc3339(&row.get::<_, String>(15)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(16)?)?,
        completed_at: parse_rfc3339_opt(row.get(17)?)?,
    })
}

pub(super) fn map_workflow_run_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkflowRun> {
    Ok(WorkflowRun {
        id: row.get(0)?,
        definition_id: row.get(1)?,
        parent_thread_id: row.get(2)?,
        input: parse_json(row.get(3)?),
        phase_states: parse_json(row.get(4)?),
        child_run_ids: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
        status: WorkflowRunStatus::parse(&row.get::<_, String>(6)?),
        summary: row.get(7)?,
        started_at: parse_rfc3339(&row.get::<_, String>(8)?)?,
        updated_at: parse_rfc3339(&row.get::<_, String>(9)?)?,
        completed_at: parse_rfc3339_opt(row.get(10)?)?,
        revision: row.get::<_, i64>(11)? as u64,
        lease_owner: row.get(12)?,
        lease_expires_at: parse_rfc3339_opt(row.get(13)?)?,
    })
}

pub(super) fn map_run_event_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunEvent> {
    Ok(RunEvent {
        run_id: row.get(0)?,
        sequence: row.get::<_, i64>(1)? as u64,
        event_type: row.get(2)?,
        payload: parse_json(row.get(3)?),
        timestamp: parse_rfc3339(&row.get::<_, String>(4)?)?,
    })
}

pub(super) fn map_run_telemetry_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunTelemetry> {
    Ok(RunTelemetry {
        run_id: row.get(0)?,
        input_tokens: row.get::<_, i64>(1)? as u64,
        output_tokens: row.get::<_, i64>(2)? as u64,
        cached_input_tokens: row.get::<_, i64>(3)? as u64,
        cost_usd: row.get(4)?,
        elapsed_ms: row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
        tool_count: row.get::<_, i64>(6)? as u64,
        model: row.get(7)?,
        provider: row.get(8)?,
        error: row.get(9)?,
        updated_at: Some(parse_rfc3339(&row.get::<_, String>(10)?)?),
    })
}

pub(super) fn parse_json(raw: String) -> Value {
    serde_json::from_str(&raw).unwrap_or_else(|_| json!({}))
}

pub(super) fn parse_json_opt(raw: Option<String>) -> Option<Value> {
    raw.and_then(|value| serde_json::from_str(&value).ok())
}

pub(super) fn parse_rfc3339(raw: &str) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(err))
        })
}

pub(super) fn parse_rfc3339_opt(raw: Option<String>) -> rusqlite::Result<Option<DateTime<Utc>>> {
    match raw {
        Some(value) => parse_rfc3339(&value).map(Some),
        None => Ok(None),
    }
}
