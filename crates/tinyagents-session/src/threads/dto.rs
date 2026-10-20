//! Storage bodies of the conversation store (schema v2).
//!
//! [`ConversationThread`] and [`ThreadMessage`] are wire types: camelCase, with
//! RFC 3339 strings, mirrored by the app. Their stored form is separate: flat
//! snake_case objects with epoch-millisecond `*_ms` times and no envelope, as
//! `docs/specs/schema-v2.md` of tinystoragedrivers requires. Only the stored
//! form differs; nothing here changes what [`ConversationPersistence`] returns.
//!
//! Times keep millisecond precision. A time that is not RFC 3339 is refused
//! when written ([`to_ms`]); a v1 record that holds one migrates as the epoch
//! ([`to_ms_or_epoch`]), so the migration never drops a conversation over it.
//!
//! [`ConversationPersistence`]: super::ConversationPersistence

use super::{ConversationThread, ThreadMessage};
use chrono::{DateTime, SecondsFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tinystoragedrivers_types::StorageError;

/// An RFC 3339 time as epoch milliseconds.
///
/// # Errors
/// [`StorageError`] of kind `InvalidInput` when `text` is not RFC 3339.
pub(super) fn to_ms(text: &str) -> Result<i64, StorageError> {
    DateTime::parse_from_rfc3339(text)
        .map(|time| time.timestamp_millis())
        .map_err(|error| {
            StorageError::invalid_input("conversation times must be RFC 3339").with_source(error)
        })
}

/// [`to_ms`], reading a time that does not parse as the epoch. For migrating
/// v1 records, which stored any string.
pub(super) fn to_ms_or_epoch(text: &str) -> i64 {
    to_ms(text).unwrap_or(0)
}

/// Epoch milliseconds as the RFC 3339 string the wire types carry: UTC with
/// the shortest fraction that holds the milliseconds exactly.
///
/// # Errors
/// [`StorageError`] of kind `Serialization` for a time outside the calendar.
pub(super) fn from_ms(ms: i64) -> Result<String, StorageError> {
    DateTime::from_timestamp_millis(ms)
        .map(|time| time.to_rfc3339_opts(SecondsFormat::AutoSi, true))
        .ok_or_else(|| StorageError::serialization("conversation time out of range"))
}

/// What a thread document records besides the thread itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ThreadState {
    /// The generation of the thread's message stream.
    pub incarnation: String,
    /// A deleted thread stays as a tombstone until purged.
    pub deleted: bool,
    /// How many messages of the stream are committed.
    pub visible_end: u64,
    /// The thread segment of the current stream's name when it is not the
    /// thread id: set by the v1 migration (the SHA-256 of the id, which names
    /// the migrated stream) until the next rewrite opens a canonical stream.
    pub messages_key: Option<String>,
}

/// `conversations.threads`: one thread and its stream state, flat.
///
/// The message count is `visible_end`; it is not stored twice.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ThreadDto {
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<i64>,
    pub is_active: bool,
    /// When the thread was created, as the caller said: not the record's
    /// `_created_ms`, which is when this document was first written.
    pub created_ms: i64,
    pub last_message_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_thread_id: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personality_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    pub incarnation: String,
    pub deleted: bool,
    pub visible_end: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages_key: Option<String>,
}

impl ThreadDto {
    /// The stored form of `thread`, with `times` reading its two time strings.
    pub fn from_wire(
        thread: &ConversationThread,
        state: ThreadState,
        times: fn(&str) -> Result<i64, StorageError>,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            id: thread.id.clone(),
            title: thread.title.clone(),
            chat_id: thread.chat_id,
            is_active: thread.is_active,
            created_ms: times(&thread.created_at)?,
            last_message_ms: times(&thread.last_message_at)?,
            parent_thread_id: thread.parent_thread_id.clone(),
            labels: thread.labels.clone(),
            personality_id: thread.personality_id.clone(),
            working_dir: thread.working_dir.clone(),
            incarnation: state.incarnation,
            deleted: state.deleted,
            visible_end: state.visible_end,
            messages_key: state.messages_key,
        })
    }

    /// The wire thread (its count is `visible_end`) and the stream state.
    pub fn into_wire(self) -> Result<(ConversationThread, ThreadState), StorageError> {
        let message_count = usize::try_from(self.visible_end)
            .map_err(|_| StorageError::serialization("conversation prefix out of range"))?;
        let thread = ConversationThread {
            id: self.id,
            title: self.title,
            chat_id: self.chat_id,
            is_active: self.is_active,
            message_count,
            last_message_at: from_ms(self.last_message_ms)?,
            created_at: from_ms(self.created_ms)?,
            parent_thread_id: self.parent_thread_id,
            labels: self.labels,
            personality_id: self.personality_id,
            working_dir: self.working_dir,
        };
        let state = ThreadState {
            incarnation: self.incarnation,
            deleted: self.deleted,
            visible_end: self.visible_end,
            messages_key: self.messages_key,
        };
        Ok((thread, state))
    }
}

/// One message: a `conversations.messages/…` stream entry, and the `record`
/// of a `conversations.appends` intent (an object, not a JSON string).
///
/// The entry does not repeat its thread or incarnation: the stream name has them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MessageDto {
    pub id: String,
    pub content: String,
    pub message_type: String,
    /// Caller metadata, kept verbatim (its keys are not renamed).
    #[serde(default)]
    pub extra_metadata: Value,
    pub sender: String,
    pub created_ms: i64,
}

impl MessageDto {
    /// The stored form of `message`, with `times` reading its time string.
    pub fn from_message(
        message: &ThreadMessage,
        times: fn(&str) -> Result<i64, StorageError>,
    ) -> Result<Self, StorageError> {
        Ok(Self {
            id: message.id.clone(),
            content: message.content.clone(),
            message_type: message.message_type.clone(),
            extra_metadata: message.extra_metadata.clone(),
            sender: message.sender.clone(),
            created_ms: times(&message.created_at)?,
        })
    }

    /// The wire message.
    pub fn into_message(self) -> Result<ThreadMessage, StorageError> {
        Ok(ThreadMessage {
            id: self.id,
            content: self.content,
            message_type: self.message_type,
            extra_metadata: self.extra_metadata,
            sender: self.sender,
            created_at: from_ms(self.created_ms)?,
        })
    }
}

/// `conversations.appends`: a message reserved at an offset of one
/// incarnation's stream, until it is published. The id is
/// `Key::of([thread_id, incarnation])`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AppendIntent {
    /// Identifies the attempt that reserved the slot.
    pub token: String,
    pub thread_id: String,
    pub incarnation: String,
    /// The stream offset the record goes to.
    pub offset: u64,
    /// The [`MessageDto`] object the stream row will hold, kept as the value
    /// written so the row can be compared with it byte for byte.
    pub record: Value,
}

#[cfg(test)]
#[path = "dto_tests.rs"]
mod tests;
