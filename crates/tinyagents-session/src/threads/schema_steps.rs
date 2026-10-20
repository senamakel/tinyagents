//! The v1 -> v2 steps of [`super::conversations_storage_schema`]. Each is a
//! pure function of its input, so a resumed run produces the same output.

use super::dto::{AppendIntent, MessageDto, ThreadDto, ThreadState, to_ms_or_epoch};
use super::schema::{APPENDS, MESSAGES, SCHEMA_VERSION, THREADS, stream_name};
use super::{ConversationThread, ThreadMessage};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tinyagents_storage_kit::{Key, StorageError};
use tinyagents_storage_kit::{SourceRecord, Step, TargetRecord};

/// The v1 thread collection.
const LEGACY_THREADS: &str = "conversation_threads_v1";
/// The v1 append-intent collection.
const LEGACY_APPENDS: &str = "conversation_appends_v1";
/// The prefix of a v1 message stream: `conversation_messages_v1:<sha>:<incarnation>`.
const LEGACY_STREAM_PREFIX: &str = "conversation_messages_v1:";

/// A v1 thread document: the wire thread beside its stream state.
#[derive(Deserialize)]
struct LegacyThread {
    incarnation: String,
    deleted: bool,
    visible_end: u64,
    thread: ConversationThread,
}

/// A v1 append intent: every field a string.
#[derive(Deserialize)]
struct LegacyIntent {
    token: String,
    thread_id: String,
    incarnation: String,
    offset: String,
    record: String,
}

/// A v1 message row, in a stream or inside an intent's `record`.
#[derive(Deserialize)]
struct LegacyRecord {
    schema: u32,
    incarnation: String,
    message: ThreadMessage,
}

/// The v1 thread key: the hex SHA-256 of the thread id.
pub(super) fn legacy_key(thread_id: &str) -> String {
    hex::encode(Sha256::digest(thread_id.as_bytes()))
}

fn lenient_ms(text: &str) -> Result<i64, StorageError> {
    Ok(to_ms_or_epoch(text))
}

fn target(item: &str, id: String, body: Value, source: &SourceRecord) -> TargetRecord {
    let record = TargetRecord::new(item, id, body);
    if source.meta.created_ms > 0 {
        record.created_ms(source.meta.created_ms)
    } else {
        record
    }
}

/// `conversation_threads_v1` -> `conversations.threads`: the id is the thread
/// id read from the body, and `messages_key` keeps the v1 stream's name.
fn threads() -> Step {
    Step::new("threads_v2", THREADS, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let legacy: LegacyThread = serde_json::from_value(source.body.clone()).ok()?;
                let id = Key::of([&legacy.thread.id]).ok()?.as_str().to_owned();
                let state = ThreadState {
                    incarnation: legacy.incarnation,
                    deleted: legacy.deleted,
                    visible_end: legacy.visible_end,
                    messages_key: Some(legacy_key(&legacy.thread.id)),
                };
                let dto = ThreadDto::from_wire(&legacy.thread, state, lenient_ms).ok()?;
                let body = serde_json::to_value(dto).ok()?;
                Some(target(THREADS, id, body, source))
            })
            .collect())
    })
    .source(LEGACY_THREADS)
}

/// A v1 message as a stored message; `None` when it is not a v1 record of `incarnation`.
fn message_body(row: &Value, incarnation: &str) -> Option<Value> {
    let record: LegacyRecord = serde_json::from_value(row.clone()).ok()?;
    if record.schema != 1 || record.incarnation != incarnation {
        return None;
    }
    let dto = MessageDto::from_message(&record.message, lenient_ms).ok()?;
    serde_json::to_value(dto).ok()
}

/// `conversation_appends_v1` -> `conversations.appends`: the offset becomes a
/// number and the record, a JSON string, becomes the object the stream row is.
fn appends() -> Step {
    Step::new("appends_v2", APPENDS, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let legacy: LegacyIntent = serde_json::from_value(source.body.clone()).ok()?;
                let row: Value = serde_json::from_str(&legacy.record).ok()?;
                let record = message_body(&row, &legacy.incarnation)?;
                let id = Key::of([&legacy.thread_id, &legacy.incarnation]).ok()?;
                let intent = AppendIntent {
                    token: legacy.token,
                    thread_id: legacy.thread_id,
                    incarnation: legacy.incarnation,
                    offset: legacy.offset.parse().ok()?,
                    record,
                };
                let body = serde_json::to_value(intent).ok()?;
                Some(target(APPENDS, id.as_str().to_owned(), body, source))
            })
            .collect())
    })
    .source(LEGACY_APPENDS)
}

/// `conversation_messages_v1:<sha>:<incarnation>` -> `conversations.messages/<sha>/<incarnation>`.
///
/// The thread id is not in the stream or its entries, so the thread segment of
/// the new name is the v1 digest; the thread step records it as `messages_key`.
/// The entries drop the `schema` and `incarnation` they repeated. A stream ends
/// at its first row that is not a v1 record of its incarnation.
fn messages() -> Step {
    Step::stream(
        "messages_v2",
        MESSAGES,
        1,
        SCHEMA_VERSION,
        |source, entries| {
            let Some((key, incarnation)) = source
                .strip_prefix(LEGACY_STREAM_PREFIX)
                .and_then(|rest| rest.split_once(':'))
            else {
                return Ok(None);
            };
            let mapped: Vec<Value> = entries
                .iter()
                .map_while(|row| message_body(row, incarnation))
                .collect();
            if mapped.len() < entries.len() {
                tracing::warn!(
                    target: "tinyagents_session::threads",
                    kept = mapped.len(),
                    dropped = entries.len() - mapped.len(),
                    "conversation stream ends at its first unreadable row"
                );
            }
            Ok(Some((stream_name(key, incarnation)?, mapped)))
        },
    )
    .source_prefix(LEGACY_STREAM_PREFIX)
}

/// Every step of the `conversations` owner.
pub(super) fn steps() -> Vec<Step> {
    vec![threads(), appends(), messages()]
}
