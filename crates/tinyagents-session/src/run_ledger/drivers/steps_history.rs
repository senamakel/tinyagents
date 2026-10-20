//! The v1 -> v2 steps of [`super::history_storage_schema`]. See `steps.rs` for
//! the v1 envelope. The v1 full-text rows have digest ids that cannot be mapped
//! to the new ones, so the session, message and tool-call steps emit the
//! `history.search` row of each record they migrate and the v1 `fts` step
//! drops the old rows.

use super::dto_history::FtsRow;
use super::history::{
    message_fts_id, message_fts_row, message_id, session_fts_id, session_fts_row, tool_call_id,
    tool_fts_id, tool_fts_row,
};
use super::steps::v1_record;
use super::storage::{
    HIST_FTS, HIST_MESSAGES, HIST_SESSIONS, HIST_TOOL_CALLS, SCHEMA_VERSION, encode, key,
};
use crate::types::{SessionMessage, SessionRecord, SessionToolCall};
use tinyagents_storage_kit::{Result, SourceRecord, Step, TargetRecord};

fn search_row(id: Result<String>, row: &FtsRow) -> Option<TargetRecord> {
    Some(TargetRecord::new(
        HIST_FTS,
        id.ok()?,
        serde_json::to_value(row).ok()?,
    ))
}

fn with_created(record: TargetRecord, source: &SourceRecord) -> TargetRecord {
    if source.meta.created_ms > 0 {
        record.created_ms(source.meta.created_ms)
    } else {
        record
    }
}

/// Every step of the `history` owner.
pub(super) fn steps() -> Vec<Step> {
    vec![
        Step::new("sessions_v2", HIST_SESSIONS, 1, SCHEMA_VERSION, |batch| {
            let mut out = Vec::new();
            for source in &batch {
                let Some(session) = v1_record::<SessionRecord>(&source.body) else {
                    continue;
                };
                let (Ok(id), Ok(body)) = (key(&[&session.id]), encode(&session)) else {
                    continue;
                };
                out.push(with_created(
                    TargetRecord::new(HIST_SESSIONS, id, body),
                    source,
                ));
                out.extend(search_row(
                    session_fts_id(&session.id),
                    &session_fts_row(&session),
                ));
            }
            Ok(out)
        })
        .source("session_history_sessions"),
        Step::new("messages_v2", HIST_MESSAGES, 1, SCHEMA_VERSION, |batch| {
            let mut out = Vec::new();
            for source in &batch {
                let Some(message) = v1_record::<SessionMessage>(&source.body) else {
                    continue;
                };
                let (Ok(id), Ok(body)) = (message_id(message.id), encode(&message)) else {
                    continue;
                };
                out.push(with_created(
                    TargetRecord::new(HIST_MESSAGES, id, body),
                    source,
                ));
                out.extend(search_row(
                    message_fts_id(message.id),
                    &message_fts_row(&message.session_id, &message.content),
                ));
            }
            Ok(out)
        })
        .source("session_history_messages"),
        Step::new(
            "tool_calls_v2",
            HIST_TOOL_CALLS,
            1,
            SCHEMA_VERSION,
            |batch| {
                let mut out = Vec::new();
                for source in &batch {
                    let Some(call) = v1_record::<SessionToolCall>(&source.body) else {
                        continue;
                    };
                    let (Ok(id), Ok(body)) = (tool_call_id(call.id), encode(&call)) else {
                        continue;
                    };
                    out.push(with_created(
                        TargetRecord::new(HIST_TOOL_CALLS, id, body),
                        source,
                    ));
                    out.extend(search_row(
                        tool_fts_id(call.id),
                        &tool_fts_row(&call.session_id, &call.tool_name),
                    ));
                }
                Ok(out)
            },
        )
        .source("session_history_tool_calls"),
        Step::new("search_v2", HIST_FTS, 1, SCHEMA_VERSION, |_| Ok(Vec::new()))
            .source("session_history_fts"),
    ]
}
