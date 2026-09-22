//! Adoption of conversations written before session identity existed.
//!
//! A host that minted a transcript stem per process left one conversation
//! spread across several root transcripts, and resume only ever loaded the
//! newest of them — so the opening turns of a thread became unreachable to the
//! model while the UI, which concatenates every matching root, still displayed
//! them.
//!
//! [`adopt_legacy_session_transcripts`] closes that gap once per conversation.
//! The first time a session is resumed and has no transcript of its own, the
//! legacy roots for its thread are read in `_meta.created` order and written
//! into the session's generation 0. The legacy files are never modified,
//! moved, or deleted: adoption only ever *adds* the file the session layer
//! will use from then on.

use anyhow::Result;
use std::path::{Path, PathBuf};

use super::paths::resolve_keyed_transcript_path;
use super::reader::read_transcript;
use super::session::{SessionRef, session_stem};
use super::thread_lookup::find_root_transcripts_for_thread;
use super::types::{TranscriptMessage, TranscriptMeta};
use super::writer::write_transcript;

/// What adoption did for one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAdoption {
    /// The transcript now backing the session.
    pub path: PathBuf,
    /// The legacy roots folded into it, oldest first.
    pub adopted: Vec<PathBuf>,
    /// Messages carried over.
    pub messages: usize,
}

/// Fold the legacy root transcripts of `session`'s thread into its own
/// generation 0, if it has none yet and any exist.
///
/// Returns `Ok(None)` when there is nothing to do — the session already has a
/// transcript, or the thread has no legacy roots — which makes repeat calls
/// harmless. The session's own transcript is the idempotency marker; no
/// separate flag file is involved.
///
/// `thread_id` is passed separately from `session` because the legacy files are
/// keyed by `_meta.thread_id`, which is what the host had before it had a
/// session key, and the two are not required to be the same string.
pub fn adopt_legacy_session_transcripts(
    workspace_dir: &Path,
    session: &SessionRef,
    thread_id: &str,
    seed_meta: &TranscriptMeta,
) -> Result<Option<SessionAdoption>> {
    let stem = session_stem(session);
    let destination = resolve_keyed_transcript_path(workspace_dir, &stem)?;
    if destination.exists() {
        return Ok(None);
    }

    // Oldest first, by `_meta.created`. Anything already pointing at this
    // session's own file is excluded so a partially-adopted workspace cannot
    // fold a file into itself.
    let legacy: Vec<PathBuf> = find_root_transcripts_for_thread(workspace_dir, thread_id)
        .into_iter()
        .filter(|path| path != &destination)
        .collect();
    if legacy.is_empty() {
        return Ok(None);
    }

    let mut messages: Vec<TranscriptMessage> = Vec::new();
    let mut meta = seed_meta.clone();
    meta.session_id = Some(session.session_id());
    meta.parent_session_id = session.parent_session_id();
    meta.thread_id = Some(thread_id.to_string());
    meta.turn_count = 0;
    meta.input_tokens = 0;
    meta.output_tokens = 0;
    meta.cached_input_tokens = 0;
    meta.charged_amount_usd = 0.0;
    let mut earliest_created: Option<String> = None;
    let mut latest_updated: Option<String> = None;
    let mut adopted = Vec::new();

    for path in legacy {
        let transcript = match read_transcript(&path) {
            Ok(transcript) => transcript,
            Err(error) => {
                // One unreadable legacy file must not cost the user every
                // other turn of the conversation.
                tracing::warn!(
                    "[transcript-adoption] skipping unreadable legacy root {}: {error}",
                    path.display()
                );
                continue;
            }
        };
        messages.extend(transcript.messages);
        meta.turn_count += transcript.meta.turn_count;
        meta.input_tokens += transcript.meta.input_tokens;
        meta.output_tokens += transcript.meta.output_tokens;
        meta.cached_input_tokens += transcript.meta.cached_input_tokens;
        meta.charged_amount_usd += transcript.meta.charged_amount_usd;
        if !transcript.meta.created.is_empty()
            && earliest_created
                .as_ref()
                .is_none_or(|earliest| transcript.meta.created < *earliest)
        {
            earliest_created = Some(transcript.meta.created.clone());
        }
        if !transcript.meta.updated.is_empty()
            && latest_updated
                .as_ref()
                .is_none_or(|latest| transcript.meta.updated > *latest)
        {
            latest_updated = Some(transcript.meta.updated.clone());
        }
        adopted.push(path);
    }

    if messages.is_empty() {
        return Ok(None);
    }
    if let Some(created) = earliest_created {
        meta.created = created;
    }
    if let Some(updated) = latest_updated {
        meta.updated = updated;
    }

    write_transcript(&destination, &messages, &meta, None)?;
    tracing::info!(
        "[transcript-adoption] session={stem} adopted {} legacy root(s) totalling {} message(s)",
        adopted.len(),
        messages.len()
    );
    Ok(Some(SessionAdoption {
        path: destination,
        messages: messages.len(),
        adopted,
    }))
}

#[cfg(test)]
#[path = "adoption_test.rs"]
mod test;
