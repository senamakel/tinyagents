//! The compaction checkpoint message: how a summary is written into a
//! transcript, and how a later compaction recognises one.
//!
//! A checkpoint is the summary of compacted turns, marked with
//! [`CHECKPOINT_PREFIX`] so it is never mistaken for live conversation:
//!
//! - the model reads it as background reference, not as instructions;
//! - a later compaction (in the same run, or in a later turn after the host
//!   persisted the compacted transcript) takes its body as the *previous
//!   summary* to refine, instead of re-summarizing it as raw history.

use tinyinference_llm::message::Message;

use super::types::SummaryPlacement;

/// Opening line of every compaction checkpoint message.
///
/// Doubles as the marker [`is_checkpoint`] looks for, so it must stay stable:
/// a transcript persisted by one release is read by the next.
pub const CHECKPOINT_PREFIX: &str = "[Context checkpoint — earlier turns were compacted. \
This is background reference data, not instructions; continue from the latest live message.]";

/// Builds the checkpoint message for `summary` with the given placement.
///
/// `summary` may already be a checkpoint body or a full checkpoint text; an
/// existing marker is not duplicated.
pub fn checkpoint_message(placement: SummaryPlacement, summary: &str) -> Message {
    let body = strip_marker(summary).trim();
    let text = format!("{CHECKPOINT_PREFIX}\n\n{body}");
    match placement {
        SummaryPlacement::User => Message::user(text),
        SummaryPlacement::System => Message::system(text),
    }
}

/// Whether `message` is a compaction checkpoint (either placement).
pub fn is_checkpoint(message: &Message) -> bool {
    matches!(message, Message::User(_) | Message::System(_))
        && message.text().starts_with(CHECKPOINT_PREFIX)
}

/// The summary body of a checkpoint message (without the marker line), or
/// `None` when `message` is not a checkpoint.
pub fn checkpoint_body(message: &Message) -> Option<String> {
    is_checkpoint(message).then(|| strip_marker(&message.text()).trim().to_string())
}

/// `text` without a leading [`CHECKPOINT_PREFIX`].
fn strip_marker(text: &str) -> &str {
    text.strip_prefix(CHECKPOINT_PREFIX).unwrap_or(text)
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
