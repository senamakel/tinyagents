//! The stored form of a [`HarnessRunStatus`] (`harness.run_status`).
//!
//! `HarnessRunStatus` is a wire type: it carries its times as whole Unix
//! *seconds*. The stored body is a separate DTO with snake_case fields and
//! `*_ms` times (spec schema-v2 §3), so the wire format does not change.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::status::is_active;
use crate::cost::CostTotals;
use crate::events::HarnessRunStatus;
use crate::ids::{CallId, ComponentId, EventId, ExecutionStatus, HarnessPhase, RunId, ThreadId};
use tinyinference_llm::usage::UsageTotals;

/// The body of one `harness.run_status` document.
///
/// `root_run_id`, `thread_id` and `active` are the indexed projections the
/// lineage and liveness queries filter on; the rest is the status itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunStatusRecord {
    pub(crate) run_id: RunId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent_run_id: Option<RunId>,
    pub(crate) root_run_id: RunId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) thread_id: Option<ThreadId>,
    pub(crate) active: bool,
    pub(crate) component: ComponentId,
    pub(crate) status: ExecutionStatus,
    pub(crate) current_phase: HarnessPhase,
    pub(crate) model_calls: usize,
    pub(crate) tool_calls: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) active_model_call: Option<CallId>,
    #[serde(default)]
    pub(crate) active_tool_calls: Vec<CallId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) last_event_id: Option<EventId>,
    pub(crate) usage: UsageTotals,
    pub(crate) cost: CostTotals,
    pub(crate) started_ms: i64,
    /// When the status last changed, as the writer reported it (the record
    /// metadata's own `updated_ms` is when the row was written).
    pub(crate) updated_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) ended_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(default)]
    pub(crate) metadata: Value,
}

fn to_ms(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .map(|elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn from_ms(ms: i64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(u64::try_from(ms).unwrap_or(0))
}

impl From<&HarnessRunStatus> for RunStatusRecord {
    fn from(status: &HarnessRunStatus) -> Self {
        Self {
            run_id: status.run_id.clone(),
            parent_run_id: status.parent_run_id.clone(),
            root_run_id: status.root_run_id.clone(),
            thread_id: status.thread_id.clone(),
            active: is_active(status),
            component: status.component.clone(),
            status: status.status,
            current_phase: status.current_phase,
            model_calls: status.model_calls,
            tool_calls: status.tool_calls,
            active_model_call: status.active_model_call.clone(),
            active_tool_calls: status.active_tool_calls.clone(),
            last_event_id: status.last_event_id.clone(),
            usage: status.usage,
            cost: status.cost,
            started_ms: to_ms(status.started_at),
            updated_ms: to_ms(status.updated_at),
            ended_ms: status.ended_at.map(to_ms),
            error: status.error.clone(),
            metadata: status.metadata.clone(),
        }
    }
}

impl From<RunStatusRecord> for HarnessRunStatus {
    fn from(record: RunStatusRecord) -> Self {
        Self {
            run_id: record.run_id,
            parent_run_id: record.parent_run_id,
            root_run_id: record.root_run_id,
            thread_id: record.thread_id,
            component: record.component,
            status: record.status,
            current_phase: record.current_phase,
            model_calls: record.model_calls,
            tool_calls: record.tool_calls,
            active_model_call: record.active_model_call,
            active_tool_calls: record.active_tool_calls,
            last_event_id: record.last_event_id,
            usage: record.usage,
            cost: record.cost,
            started_at: from_ms(record.started_ms),
            updated_at: from_ms(record.updated_ms),
            ended_at: record.ended_ms.map(from_ms),
            error: record.error,
            metadata: record.metadata,
        }
    }
}

/// The v2 `(id, body)` of a v1 `harness_run_status` document
/// (`{ runId, rootRunId, threadId, active, record }`), or `None` when its
/// `record` no longer decodes.
pub(crate) fn from_v1(body: &Value) -> Option<(String, Value)> {
    let status: HarnessRunStatus = serde_json::from_value(body.get("record")?.clone()).ok()?;
    let id = tinyagents_storage_kit::Key::of([status.run_id.as_str()])
        .ok()?
        .as_str()
        .to_owned();
    Some((
        id,
        serde_json::to_value(RunStatusRecord::from(&status)).ok()?,
    ))
}
