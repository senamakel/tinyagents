//! The v1 -> v2 migration steps of [`super::storage_schema`]. Each is a pure
//! function of its input, so a resumed run produces the same output.

use serde::Deserialize;
use serde_json::Value;
use tinyagents_storage_kit::{SourceRecord, Step, TargetRecord};

use super::dto::{CompletionDoc, TaskDoc, history_id_key, task_id_key, to_ms};
use super::{COMPLETIONS, HISTORY, RECORDS, SCHEMA_VERSION};
use crate::completions::CompletionRecord;
use crate::types::OrchestrationTaskRecord;

/// The v1 `<prefix>_tasks` and `<prefix>_task_history` body: only the revision
/// and the wrapped record are needed, the rest were indexed projections.
#[derive(Deserialize)]
struct V1Task {
    rev: u64,
    record: Value,
}

/// A target that keeps the time the record was first written.
fn target(collection: &str, id: String, body: Value, created_ms: i64) -> TargetRecord {
    let record = TargetRecord::new(collection, id, body);
    if created_ms > 0 {
        record.created_ms(created_ms)
    } else {
        record
    }
}

/// Decode a v1 task row, dropping (with a warning) one that no longer decodes.
fn v1_task(source: &SourceRecord) -> Option<(OrchestrationTaskRecord, u64)> {
    let decoded = serde_json::from_value::<V1Task>(source.body.clone())
        .map_err(|e| e.to_string())
        .and_then(|row| {
            serde_json::from_value::<OrchestrationTaskRecord>(row.record)
                .map(|record| (record, row.rev))
                .map_err(|e| e.to_string())
        });
    match decoded {
        Ok(found) => Some(found),
        Err(error) => {
            tracing::warn!(id = %source.id, %error, "[orchestration] dropped an unreadable v1 task row");
            None
        }
    }
}

/// `orchestration_tasks` -> `tasks.records`.
pub(super) fn records() -> Step {
    Step::new("records_v2", RECORDS, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let (record, rev) = v1_task(source)?;
                let id = task_id_key(record.spec.task_id.as_str()).ok()?;
                let doc = TaskDoc::from_record(&record, rev);
                let body = serde_json::to_value(&doc).ok()?;
                Some(target(RECORDS, id, body, doc.created_ms))
            })
            .collect())
    })
    .source("orchestration_tasks")
}

/// `orchestration_task_history` -> `tasks.history`: the id becomes
/// `Key::of([task_id, rev])`; a snapshot's first write is its last update.
pub(super) fn history() -> Step {
    Step::new("history_v2", HISTORY, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let (record, rev) = v1_task(source)?;
                let id = history_id_key(record.spec.task_id.as_str(), rev).ok()?;
                let written_ms = to_ms(record.updated_at);
                let body = serde_json::to_value(TaskDoc::from_record(&record, rev)).ok()?;
                Some(target(HISTORY, id, body, written_ms))
            })
            .collect())
    })
    .source("orchestration_task_history")
}

/// `orchestration_completions` -> `tasks.completions`: unwrap `record`;
/// `updatedAtMs` is the document's own `updated_ms` from now on.
pub(super) fn completions() -> Step {
    Step::new("completions_v2", COMPLETIONS, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let record = source
                    .body
                    .get("record")
                    .cloned()
                    .and_then(|record| serde_json::from_value::<CompletionRecord>(record).ok());
                let Some(record) = record else {
                    tracing::warn!(id = %source.id, "[completion-store] dropped an unreadable v1 row");
                    return None;
                };
                let id = task_id_key(&record.task_id).ok()?;
                let doc = CompletionDoc::from_record(&record);
                let body = serde_json::to_value(&doc).ok()?;
                Some(target(COMPLETIONS, id, body, doc.finished_ms))
            })
            .collect())
    })
    .source("orchestration_completions")
}
