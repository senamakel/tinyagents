//! What the conversation store keeps, for the host's `Registry::compose`:
//! [`conversations_storage_schema`] (owner `conversations`, scope class Tenant).
//!
//! | v1 | v2 |
//! |---|---|
//! | `conversation_threads_v1` (id: hex SHA-256 of the thread id; `{schema, incarnation, deleted, visible_end, thread: {camelCase}}`) | `conversations.threads`, id `Key::of([thread_id])`, flat snake_case body with `created_ms` / `last_message_ms`; no `schema` field (the record's `_schema`), no stored count (`visible_end`) |
//! | `conversation_appends_v1` (id `<sha>:<incarnation>`; every field a string, `record` a JSON string) | `conversations.appends`, id `Key::of([thread_id, incarnation])`, `offset` a number, `record` an object |
//! | stream `conversation_messages_v1:<sha>:<incarnation>` (`{schema, incarnation, message: {camelCase}}`) | stream `conversations.messages/<thread>/<incarnation>`, entries `{id, content, message_type, extra_metadata, sender, created_ms}` |
//!
//! The conversation store is shared by a tenant's agents, so every item has
//! scope class Tenant. A host binds it to the tenant scope.
//!
//! # Recovering the thread id of a migrated stream
//!
//! A v1 stream is named by a digest of the thread id, and its entries do not
//! hold the id. A stream step sees only the stream, so it cannot name the
//! canonical `conversations.messages/<thread id>/<incarnation>`. It moves each
//! stream to `conversations.messages/<sha>/<incarnation>` instead, and the
//! thread step, which does hold the id, records that segment as `messages_key`
//! on the migrated thread document. The store reads the stream of a thread with
//! a `messages_key` from that segment. A later edit or tail cut copies the
//! log to a fresh incarnation whose stream carries the canonical segment, and
//! the document drops `messages_key`.
//!
//! # Behaviour changes
//!
//! - Times are stored as whole milliseconds, so a sub-millisecond fraction or a
//!   non-UTC offset does not survive a round trip. A time that is not RFC 3339
//!   is refused when written; one found in v1 data migrates as the epoch.
//! - A stream that stops being readable (a row that is not a v1 record, or one
//!   of another incarnation) migrates up to that row; the v1 reader refused it
//!   at that row too.
//! - The append intent no longer guards against a body TTL on its fields: a
//!   record's expiry is its `_expires_ms`, which the store never sets.

use super::schema_steps;
use tinyagents_storage_kit::{CollectionSpec, Key, StorageError};
use tinyagents_storage_kit::{Owner, Schema, ScopeClass};

/// The owner of the conversation store's collections and streams.
pub(super) const OWNER: &str = "conversations";
/// The body version the v1 -> v2 steps produce.
pub(super) const SCHEMA_VERSION: u32 = 2;
/// Thread documents.
pub(super) const THREADS: &str = "conversations.threads";
/// Pending append intents.
pub(super) const APPENDS: &str = "conversations.appends";
/// The per-incarnation message stream family: `conversations.messages/<thread>/<incarnation>`.
pub(super) const MESSAGES: &str = "conversations.messages";

/// The thread collection's declaration (no secondary indexes).
pub(super) fn thread_spec() -> CollectionSpec {
    CollectionSpec::new(THREADS)
}

/// The append-intent collection's declaration.
pub(super) fn append_spec() -> CollectionSpec {
    CollectionSpec::new(APPENDS)
}

/// The name of an incarnation's message stream, `conversations.messages/<thread>/<incarnation>`:
/// `thread` is the thread id, or the v1 digest of a migrated stream (see the
/// module docs).
///
/// # Errors
/// `InvalidInput` when `thread` is empty.
pub(super) fn stream_name(thread: &str, incarnation: &str) -> Result<String, StorageError> {
    Ok(format!(
        "{MESSAGES}/{}",
        Key::of([thread, incarnation])?.as_str()
    ))
}

/// What the conversation store keeps (owner `conversations`, scope class
/// Tenant), with the steps that bring the v1 names and shapes forward.
#[must_use]
pub fn conversations_storage_schema() -> Schema {
    let owner = Owner::new(OWNER).expect("`conversations` is a valid owner");
    let mut schema = Schema::new(owner)
        .collection_with(SCHEMA_VERSION, thread_spec(), ScopeClass::Tenant)
        .collection_with(SCHEMA_VERSION, append_spec(), ScopeClass::Tenant)
        .stream(MESSAGES, ScopeClass::Tenant);
    for step in schema_steps::steps() {
        schema = schema.step(step);
    }
    schema
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
