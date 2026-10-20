//! The v1 -> v2 migration steps of [`super::storage_schema`]. Each is a pure
//! function of its input, so a resumed run produces the same output.

use serde_json::Value;
use tinyagents_storage_kit::{Key, SourceRecord, Step, TargetRecord};

use super::super::turn_states::storage;
use super::{ENTRIES_NAME, INDEX_NAME, TRANSCRIPT_VERSION, TURN_NAME, TURN_VERSION};
use crate::turn_state::TurnState;

/// A key for `parts`, or `None` when they make no valid id.
fn key<const N: usize>(parts: [&str; N]) -> Option<String> {
    Key::of(parts).ok().map(|key| key.as_str().to_owned())
}

/// The first-write time to keep: the v1 `created_at` (RFC 3339), else the
/// source record's own.
fn created_ms(source: &SourceRecord) -> Option<i64> {
    source
        .body
        .get("created_at")
        .and_then(Value::as_str)
        .and_then(|text| chrono::DateTime::parse_from_rfc3339(text).ok())
        .map(|at| at.timestamp_millis())
        .or_else(|| (source.meta.created_ms > 0).then_some(source.meta.created_ms))
}

fn with_created(record: TargetRecord, created: Option<i64>) -> TargetRecord {
    match created {
        Some(ms) => record.created_ms(ms),
        None => record,
    }
}

/// `session_transcripts` -> `session.transcripts`: id `Key::of([stem])`,
/// `created_at` becomes the application field `created_ms`, and rows that only
/// held a generation reservation (`written: false`) are dropped.
pub(super) fn transcripts() -> Step {
    Step::new(
        "transcripts_v2",
        INDEX_NAME,
        1,
        TRANSCRIPT_VERSION,
        |batch| {
            Ok(batch
                .iter()
                .filter_map(|source| {
                    if source.body.get("written") != Some(&Value::Bool(true)) {
                        return None;
                    }
                    let id = key([source.body.get("stem")?.as_str()?])?;
                    let mut body = source.body.clone();
                    let fields = body.as_object_mut()?;
                    let created = created_ms(source);
                    fields.remove("created_at");
                    fields.remove("reserved_at");
                    if let Some(created) = created {
                        fields.insert("created_ms".into(), Value::from(created));
                    }
                    Some(TargetRecord::new(INDEX_NAME, id, body))
                })
                .collect())
        },
    )
    .source("session_transcripts")
}

/// `session_transcript_entries` -> `session.transcript_entries`: id
/// `Key::of([stem, "<seq>"])`, the body unchanged.
pub(super) fn transcript_entries() -> Step {
    Step::new(
        "transcript_entries_v2",
        ENTRIES_NAME,
        1,
        TRANSCRIPT_VERSION,
        |batch| {
            Ok(batch
                .iter()
                .filter_map(|source| {
                    let stem = source.body.get("stem")?.as_str()?;
                    let seq = source.body.get("seq")?.as_u64()?;
                    let id = key([stem, format!("{seq:010}").as_str()])?;
                    Some(with_created(
                        TargetRecord::new(ENTRIES_NAME, id, source.body.clone()),
                        created_ms(source),
                    ))
                })
                .collect())
        },
    )
    .source("session_transcript_entries")
}

/// `session_turn_states` -> `session.turn_states`: unwrap `turn`, snake_case
/// body with `started_ms` / `updated_ms`, id `Key::of([thread_id, request_id])`.
/// A row that does not decode as a turn is dropped.
pub(super) fn turn_states() -> Step {
    Step::new("turn_states_v2", TURN_NAME, 1, TURN_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let turn: TurnState =
                    serde_json::from_value(source.body.get("turn")?.clone()).ok()?;
                let body = storage::encode(&turn).ok()?;
                let id = key([turn.thread_id.as_str(), turn.request_id.as_str()])?;
                Some(with_created(
                    TargetRecord::new(TURN_NAME, id, body),
                    created_ms(source),
                ))
            })
            .collect())
    })
    .source("session_turn_states")
}
