//! Wire/storage types for the workspace-backed conversation store: threads,
//! messages, create requests, and partial-update patches.
//!
//! These mirror one-to-one the JSONL records persisted under
//! `<workspace>/memory/conversations/`. The `camelCase` serde renames and the
//! `type`/`extraMetadata` wire keys are part of the OpenHuman on-disk contract
//! and must be preserved byte-for-byte so existing transcripts keep loading.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A persisted conversation thread, mirroring one entry in `threads.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConversationThread {
    /// Stable thread identifier; the JSONL key callers reference everywhere.
    pub id: String,
    /// Human-readable thread title.
    pub title: String,
    /// Optional host chat id (e.g. a messaging platform chat); omitted from the
    /// wire record when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_id: Option<i64>,
    /// Whether the thread is currently active (not archived/closed).
    pub is_active: bool,
    /// Cached count of messages in the thread's log.
    pub message_count: usize,
    /// ISO-8601 timestamp of the most recent message.
    pub last_message_at: String,
    /// ISO-8601 timestamp of thread creation.
    pub created_at: String,
    /// Parent thread id when this thread was branched from another; omitted from
    /// the wire record when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_thread_id: Option<String>,
    /// Free-form labels/tags; defaults to empty when absent on disk.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Optional personality id bound to the thread; omitted from the wire record
    /// when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub personality_id: Option<String>,
    /// Optional working directory the thread's agent acts in, chosen when the
    /// thread was started; omitted from the wire record when `None`, which
    /// means the host's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
}

/// A single message appended to a thread's JSONL log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMessage {
    /// Stable message identifier, unique within the thread's log.
    pub id: String,
    /// Message body text.
    pub content: String,
    /// Message kind; serialized under the wire key `type` (e.g. `"text"`).
    #[serde(rename = "type")]
    pub message_type: String,
    /// Arbitrary per-message metadata; serialized under the wire key
    /// `extraMetadata`. Defaults to JSON null when absent.
    #[serde(default)]
    pub extra_metadata: Value,
    /// Sender identifier/role for the message.
    pub sender: String,
    /// ISO-8601 timestamp of when the message was created.
    pub created_at: String,
}

/// Input payload to create-or-update a thread via
/// [`ConversationPersistence::ensure_thread`](super::ConversationPersistence::ensure_thread).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateConversationThread {
    /// Desired thread id; reused as-is if the thread already exists.
    pub id: String,
    /// Initial thread title.
    pub title: String,
    /// ISO-8601 creation timestamp to record.
    pub created_at: String,
    /// Optional parent thread id when branching from an existing thread.
    #[serde(default)]
    pub parent_thread_id: Option<String>,
    /// Optional initial labels; `None` leaves the thread's labels unset.
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    /// Optional personality id to bind to the thread.
    #[serde(default)]
    pub personality_id: Option<String>,
    /// Optional working directory to bind to the thread. Like
    /// `personality_id`, `None` on a later upsert keeps the bound value.
    #[serde(default)]
    pub working_dir: Option<String>,
}

/// Partial update to apply to a stored message (e.g. rewriting `extraMetadata`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMessagePatch {
    /// Replacement `extraMetadata` payload; `None` leaves existing metadata
    /// untouched.
    #[serde(default)]
    pub extra_metadata: Option<Value>,
    /// Recursively merges a JSON Merge Patch into the latest `extraMetadata`.
    /// Object members are merged, arrays/scalars are replaced, and `null`
    /// removes an object member. Applied inside the driver's compare-and-swap
    /// retry so concurrent patches are not based on a stale host-side read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merge_extra_metadata: Option<Value>,
}

/// A single match returned by
/// [`ConversationPersistence::search_cross_thread_messages`](super::ConversationPersistence::search_cross_thread_messages).
/// Carries the source `thread_id` so the caller can render provenance into the
/// `[Cross-chat context]` block (issue #1505).
#[derive(Debug, Clone, PartialEq)]
pub struct CrossThreadHit {
    /// Source thread the match came from; drives provenance rendering.
    pub thread_id: String,
    /// Id of the matched message within that thread.
    pub message_id: String,
    /// Sender role of the matched message.
    pub role: String,
    /// Matched message content.
    pub content: String,
    /// ISO-8601 timestamp of the matched message.
    pub created_at: String,
    /// Relevance score for the match; higher is more relevant.
    pub score: f64,
}

/// Prefix of the assistant reply ids the core mints from a request id.
///
/// [`is_deterministic_message_id`] identifies this assistant-only format for
/// transcript correlation. The append retry contract is broader and is
/// defined by [`is_idempotent_message_id`].
pub const DETERMINISTIC_MESSAGE_ID_PREFIX: &str = "agent:";

/// Whether a message id marks an append that may be retried safely.
///
/// Agent replies use `agent:<request-id>` and user rows submitted through a
/// queued/hosted path use `user:<request-id>`. Both identifiers are stable
/// across retries; ordinary UI ids and provider-specific sender ids remain
/// outside this contract unless their message carries `scope: channel`.
pub fn is_idempotent_message_id(id: &str) -> bool {
    id.starts_with(DETERMINISTIC_MESSAGE_ID_PREFIX) || id.starts_with("user:")
}

/// The id an autonomous run's closing reply is stored under.
///
/// Two writers legitimately persist that one reply — the core's
/// background-delivery path and the client that also persists the
/// `chat_done` the run announces (`ChatRuntimeProvider` mirrors this shape for
/// `client_id: "system"` turns) — so both must derive the same id and the store
/// must collapse the second write onto the first (#5933).
pub fn run_reply_message_id(run_id: &str) -> String {
    format!("{DETERMINISTIC_MESSAGE_ID_PREFIX}{run_id}")
}

/// Whether `id` is an assistant reply id minted from a run/request id.
///
/// This predicate is used for transcript correlation by
/// [`reply_run_id`]. Use [`is_idempotent_message_id`] to check the broader
/// append retry contract.
pub fn is_deterministic_message_id(id: &str) -> bool {
    id.starts_with(DETERMINISTIC_MESSAGE_ID_PREFIX)
}

/// The run/request id [`run_reply_message_id`] minted `id` from, or `None`
/// when `id` is not a deterministic reply id (see
/// [`is_deterministic_message_id`]).
///
/// Backs `threads.edit_message` / `threads.regenerate`: an assistant reply's
/// store id is the one place the conversation-store id space and the
/// model-facing transcript's `request_id` space provably correlate, so
/// recovering the run id from the store id is how a UI message id resolves
/// to a transcript cut point.
pub fn reply_run_id(id: &str) -> Option<&str> {
    id.strip_prefix(DETERMINISTIC_MESSAGE_ID_PREFIX)
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
