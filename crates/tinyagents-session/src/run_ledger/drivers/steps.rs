//! The v1 -> v2 steps of [`super::ledger_storage_schema`]. Each is a pure
//! function of its batch, so a resumed run produces the same output.
//!
//! A v1 document is `{schemaVersion: 1, updatedSql, createdSql, startedSql,
//! record: {camelCase wire body}}`. The wire types deserialize from `record`
//! unchanged, and the storage DTOs take it from there.

use super::dto::{EffectDto, from_ms};
use super::dto::{Stored, to_ms};
use super::storage::{
    AGENTS, EFFECTS, EVENT_HEADS, EVENT_STREAM, EVENTS, MEMBERS, SCHEMA_VERSION, TASKS, TEAMS,
    TELEMETRY, WORKFLOWS, encode, event_stream, key,
};
use crate::run_ledger::tool_effects::ToolEffectStatus;
use crate::run_ledger::types::*;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tinyagents_storage_kit::{SourceRecord, Step, TargetRecord};

/// The wire record inside a v1 envelope.
pub(super) fn v1_record<T: DeserializeOwned>(body: &Value) -> Option<T> {
    serde_json::from_value(body.get("record")?.clone()).ok()
}

fn target(
    item: &str,
    id: Option<String>,
    body: Option<Value>,
    source: &SourceRecord,
) -> Option<TargetRecord> {
    let record = TargetRecord::new(item, id?, body?);
    Some(if source.meta.created_ms > 0 {
        record.created_ms(source.meta.created_ms)
    } else {
        record
    })
}

/// A step that rewrites each v1 record of wire type `T` as its DTO under the
/// id `id_of` names; a record that does not decode is dropped.
fn convert<T: Stored + DeserializeOwned + 'static>(
    id: &'static str,
    item: &'static str,
    legacy: &'static str,
    id_of: fn(&T) -> Option<String>,
) -> Step {
    Step::new(id, item, 1, SCHEMA_VERSION, move |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let record: T = v1_record(&source.body)?;
                target(item, id_of(&record), encode(&record).ok(), source)
            })
            .collect())
    })
    .source(legacy)
}

fn key1(id: &str) -> Option<String> {
    key(&[id]).ok()
}

/// Rewrite a v1 record with `map` over the bare `record` object.
fn value_step(
    id: &'static str,
    item: &'static str,
    legacy: &'static str,
    map: fn(&Value) -> Option<(String, Value)>,
) -> Step {
    Step::new(id, item, 1, SCHEMA_VERSION, move |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let (doc_id, body) = map(source.body.get("record")?)?;
                target(item, Some(doc_id), Some(body), source)
            })
            .collect())
    })
    .source(legacy)
}

fn text<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    value.get(field)?.as_str()
}

/// `run_ledger_tool_effects`: the status string is kept verbatim.
fn tool_effect(record: &Value) -> Option<(String, Value)> {
    let run_id = text(record, "runId")?;
    let call_id = text(record, "callId")?;
    let time = |field: &str| {
        text(record, field)
            .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
            .map(|at| at.timestamp_millis())
    };
    let dto = EffectDto {
        run_id: run_id.into(),
        call_id: call_id.into(),
        tool: text(record, "tool")?.into(),
        status: text(record, "status")?.into(),
        idempotency_key: text(record, "idempotencyKey").map(Into::into),
        effect_summary: text(record, "effectSummary").map(Into::into),
        started_ms: time("startedAt")?,
        settled_ms: time("settledAt"),
    };
    let _ = ToolEffectStatus::parse(&dto.status);
    Some((
        key(&[run_id, call_id]).ok()?,
        serde_json::to_value(dto).ok()?,
    ))
}

/// `run_events:<sha>` -> `ledger.events/<run>`: the run id is in every entry
/// (the v1 stream name is a digest of it); entries drop it.
fn event_stream_step() -> Step {
    Step::stream(
        "run_events_v2",
        EVENT_STREAM,
        1,
        SCHEMA_VERSION,
        |_, entries| {
            let Some(run_id) = entries
                .first()
                .and_then(|entry| text(entry, "runId"))
                .map(str::to_owned)
            else {
                return Ok(None);
            };
            let mut mapped = Vec::with_capacity(entries.len());
            for entry in entries {
                let event: RunEvent = serde_json::from_value(entry)?;
                mapped.push(serde_json::to_value(super::dto::StreamEvent::of(&event))?);
            }
            Ok(Some((event_stream(&run_id)?, mapped)))
        },
    )
    .source_prefix("run_events:")
}

/// Every step of the `ledger` owner.
pub(super) fn steps() -> Vec<Step> {
    let _ = (from_ms, to_ms);
    vec![
        convert::<WorkflowRun>(
            "workflow_runs_v2",
            WORKFLOWS,
            "run_ledger_workflow_runs",
            |r| key1(&r.id),
        ),
        convert::<RunTelemetry>("telemetry_v2", TELEMETRY, "run_ledger_telemetry", |r| {
            key1(&r.run_id)
        }),
        convert::<AgentRun>("agent_runs_v2", AGENTS, "run_ledger_agent_runs", |r| {
            key1(&r.id)
        }),
        convert::<AgentTeam>("teams_v2", TEAMS, "run_ledger_teams", |r| key1(&r.id)),
        convert::<AgentTeamMember>("members_v2", MEMBERS, "run_ledger_members", |r| key1(&r.id)),
        convert::<AgentTeamTask>("tasks_v2", TASKS, "run_ledger_tasks", |r| key1(&r.id)),
        convert::<RunEvent>("event_outbox_v2", EVENTS, "run_ledger_event_outbox", |r| {
            key(&[&r.run_id, &r.sequence.to_string()]).ok()
        }),
        // The v1 head's id is a digest of the run id; the head is recomputed.
        Step::new("event_heads_v2", EVENT_HEADS, 1, SCHEMA_VERSION, |_| {
            Ok(Vec::new())
        })
        .source("run_ledger_event_heads"),
        value_step(
            "tool_effects_v2",
            EFFECTS,
            "run_ledger_tool_effects",
            tool_effect,
        ),
        event_stream_step(),
    ]
}
