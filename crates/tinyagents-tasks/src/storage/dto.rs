//! The stored forms of the task and completion records.
//!
//! [`OrchestrationTaskRecord`] and [`CompletionRecord`] are tool and RPC output
//! types that carry `SystemTime`. What the stores write is a separate body with
//! snake_case fields and `*_ms` times (spec schema-v2 §3), so the output types
//! do not change. A record's last-update time is the document's `updated_ms`
//! metadata, not a body field.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tinyagents_storage_kit::{Key, Result};

use crate::completions::{
    CompletionRecord, CompletionResult, CompletionState, CompletionStatus, NotifyMode,
};
use crate::types::{
    OrchestrationTaskRecord, OrchestrationTaskResult, OrchestrationTaskSpec,
    OrchestrationTaskStatus,
};

/// Whole milliseconds since the Unix epoch (`0` before it).
pub(crate) fn to_ms(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

/// The instant `ms` milliseconds after the Unix epoch (the epoch for a
/// negative count).
pub(crate) fn from_ms(ms: i64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(u64::try_from(ms).unwrap_or(0))
}

/// The id of the task record `task_id`: `Key::of([task_id])`.
pub(crate) fn task_id_key(task_id: &str) -> Result<String> {
    Ok(Key::of([task_id])?.as_str().to_owned())
}

/// The id of revision `rev` of `task_id` in the history:
/// `Key::of([task_id, rev])`.
pub(crate) fn history_id_key(task_id: &str, rev: u64) -> Result<String> {
    Ok(Key::of([task_id, &rev.to_string()])?.as_str().to_owned())
}

/// The body of a `tasks.records` or `tasks.history` document.
///
/// The spec is flattened to the top level, so its lineage fields
/// (`parent_run_id`, `root_run_id`, `thread_id`, `node_id`) are the indexed
/// columns [`TaskStore::list`](crate::TaskStore::list) filters on; `kind_label`
/// is the task kind's label for the same purpose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TaskDoc {
    #[serde(flatten)]
    pub(crate) spec: OrchestrationTaskSpec,
    pub(crate) kind_label: String,
    pub(crate) status: OrchestrationTaskStatus,
    /// Transitions so far, starting at 1.
    pub(crate) rev: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) result: Option<OrchestrationTaskResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    pub(crate) created_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) started_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ended_ms: Option<i64>,
}

impl TaskDoc {
    pub(crate) fn from_record(record: &OrchestrationTaskRecord, rev: u64) -> Self {
        Self {
            spec: record.spec.clone(),
            kind_label: record.spec.kind.as_str().to_owned(),
            status: record.status,
            rev,
            result: record.result.clone(),
            error: record.error.clone(),
            created_ms: to_ms(record.created_at),
            started_ms: record.started_at.map(to_ms),
            ended_ms: record.ended_at.map(to_ms),
        }
    }

    /// The record this body stores, last updated at `updated_ms` (the
    /// document's metadata).
    pub(crate) fn into_record(self, updated_ms: i64) -> OrchestrationTaskRecord {
        OrchestrationTaskRecord {
            spec: self.spec,
            status: self.status,
            result: self.result,
            error: self.error,
            created_at: from_ms(self.created_ms),
            updated_at: from_ms(updated_ms),
            started_at: self.started_ms.map(from_ms),
            ended_at: self.ended_ms.map(from_ms),
        }
    }
}

/// The body of a `tasks.completions` document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CompletionDoc {
    pub(crate) task_id: String,
    pub(crate) parent_key: String,
    pub(crate) agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    pub(crate) status: CompletionStatus,
    pub(crate) result: CompletionResult,
    pub(crate) finished_ms: i64,
    #[serde(default)]
    pub(crate) attempts: u32,
    pub(crate) state: CompletionState,
    #[serde(default)]
    pub(crate) notify_mode: NotifyMode,
}

impl CompletionDoc {
    pub(crate) fn from_record(record: &CompletionRecord) -> Self {
        Self {
            task_id: record.task_id.clone(),
            parent_key: record.parent_key.clone(),
            agent_id: record.agent_id.clone(),
            label: record.label.clone(),
            status: record.status,
            result: record.result.clone(),
            finished_ms: to_ms(record.finished_at),
            attempts: record.attempts,
            state: record.state,
            notify_mode: record.notify_mode,
        }
    }

    /// The record this body stores, last updated at `updated_ms` (the
    /// document's metadata).
    pub(crate) fn into_record(self, updated_ms: i64) -> CompletionRecord {
        CompletionRecord {
            task_id: self.task_id,
            parent_key: self.parent_key,
            agent_id: self.agent_id,
            label: self.label,
            status: self.status,
            result: self.result,
            finished_at: from_ms(self.finished_ms),
            updated_at: from_ms(updated_ms),
            attempts: self.attempts,
            state: self.state,
            notify_mode: self.notify_mode,
        }
    }
}
