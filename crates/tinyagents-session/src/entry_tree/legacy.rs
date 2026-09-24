//! Deterministic derivation of tree [`Entry`] values from the two pre-tree
//! linear record shapes this crate already reads and writes: the JSONL
//! transcript ([`crate::transcript::SessionTranscript`]) and the SQLite
//! `session_messages` table ([`crate::types::SessionMessage`]).
//!
//! Both are "append a message, in file/row order" formats with no parent
//! pointer. The tree model needs one, so this module assigns it the only
//! sound way for data with no branching: entry *n* is the parent of entry
//! *n+1*, in the order the source already has. Ids are derived with
//! [`EntryId::derive`], so re-reading the same source (same session id, same
//! message order) always assigns the same ids and re-importing is a no-op
//! for a store that has already imported it (see
//! [`super::EntryTree::import_legacy`]).

use crate::transcript::{SessionTranscript, TranscriptMessage};
use crate::types::SessionMessage;

use super::types::{Entry, EntryId, EntryKind};

/// Derives an append-only linear chain of [`crate::entry_tree::EntryKind::Message`] nodes from a
/// parsed JSONL transcript's message array, in file order.
pub fn from_transcript(session_id: &str, transcript: &SessionTranscript) -> Vec<Entry> {
    from_messages(session_id, &transcript.messages)
}

/// Derives an append-only linear chain of [`crate::entry_tree::EntryKind::Message`] nodes from a
/// bare message slice, in the given order.
pub fn from_messages(session_id: &str, messages: &[TranscriptMessage]) -> Vec<Entry> {
    let mut entries = Vec::with_capacity(messages.len());
    let mut parent = None;
    for (ordinal, message) in messages.iter().enumerate() {
        let ordinal = ordinal as u64;
        let id = EntryId::derive(session_id, ordinal);
        let ts = message
            .turn_usage
            .as_ref()
            .map(|u| u.ts.clone())
            .filter(|ts| !ts.is_empty())
            .unwrap_or_default();
        entries.push(Entry {
            id: id.clone(),
            parent_id: parent.take(),
            ordinal,
            kind: EntryKind::Message(message.clone()),
            ts,
        });
        parent = Some(id);
    }
    entries
}

/// Derives an append-only linear chain of [`crate::entry_tree::EntryKind::Message`] nodes from
/// SQLite `session_messages` rows, in `id` (insertion) order.
///
/// Rows are converted to [`TranscriptMessage`] with `extra_metadata`
/// carrying the SQLite-only columns (`model`, token counts, cost) that have
/// no home on the neutral transcript record, so no information is dropped
/// by importing.
pub fn from_session_messages(session_id: &str, rows: &[SessionMessage]) -> Vec<Entry> {
    let messages: Vec<TranscriptMessage> = rows
        .iter()
        .map(|row| {
            let mut message = TranscriptMessage::new(row.role.clone(), row.content.clone());
            let extra = serde_json::json!({
                "sql_id": row.id,
                "model": row.model,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "cost_usd": row.cost_usd,
                "reasoning_content": row.reasoning_content,
            });
            message.extra_metadata = Some(extra);
            message
        })
        .collect();
    let mut entries = from_messages(session_id, &messages);
    for (entry, row) in entries.iter_mut().zip(rows.iter()) {
        entry.ts = row.created_at.to_rfc3339();
    }
    entries
}
