//! Adoption of pre-identity roots into a session, for driver-backed
//! transcripts.
//!
//! Pre-identity timestamped roots remain under their old stems. Resume addresses a
//! conversation by [`SessionRef`], so without this those roots are never read
//! again. [`TranscriptLocator::adopt_legacy`] folds them into the session's
//! generation 0 once, with the semantics of the file adopter
//! ([`adopt_legacy_session_transcripts`](crate::transcript::adopt_legacy_session_transcripts)):
//!
//! - nothing happens when the session already has a transcript;
//! - candidates are the thread's written roots other than the session's own
//!   stem, skipping session-identified ones and, when the session names an
//!   agent, another agent's; an unreadable candidate defers the whole call;
//! - candidates come from the index **and** from the entry log, so a root
//!   whose durable write outran its index refresh is still folded (and its
//!   index repaired by the read) instead of being left out of a fold that
//!   never runs again; a failed log scan defers the call too;
//! - they are folded oldest first by `_meta.created` (messages, last recorded
//!   tools, summed totals, earliest `created`, latest `updated`, the first
//!   non-empty source's `prefix_message_count`);
//! - the legacy transcripts are never modified.
//!
//! The write is race-safe: the session's stem is reserved first (as a new
//! generation is) and written through the reserved handle, so a concurrent
//! first turn or adopter makes this call report `Ok(None)` instead of
//! overwriting it.

use std::collections::BTreeSet;

use serde_json::Value;
use tinyagents_storage_kit::{DocumentStoreExt, Filter, Query, StorageError};

use super::{DriverTranscriptLocator, ENTRIES, discovered_seed, is_subagent, reservation, run_on};
use crate::transcript::{
    SessionAdoption, SessionRef, TranscriptHistory, TranscriptLocator, TranscriptMessage,
    TranscriptMeta, TranscriptRead, TranscriptTurn, session_stem,
};

/// One legacy root read for folding.
struct Candidate {
    path: std::path::PathBuf,
    meta: TranscriptMeta,
    messages: Vec<TranscriptMessage>,
    tools: Option<serde_json::Value>,
}

impl DriverTranscriptLocator {
    /// The legacy roots of `thread_id` that `session` adopts, oldest first.
    fn adoption_candidates(
        &self,
        session: &SessionRef,
        thread_id: &str,
        own_stem: &str,
    ) -> anyhow::Result<Vec<Candidate>> {
        let filter = Filter::eq("thread_id", thread_id).and(Filter::eq("subagent", false));
        let mut stems: Vec<String> = self
            .indexed(filter, "adopt_legacy")
            .into_iter()
            .map(|indexed| indexed.stem)
            .collect();
        for logged in self.logged_roots(thread_id)? {
            if !stems.contains(&logged) {
                tracing::debug!(
                    target: "tinyagents_session::port::drivers",
                    stem = %logged,
                    "[session-store] adoption found a root missing from the index"
                );
                stems.push(logged);
            }
        }
        let mut found = Vec::new();
        for stem in stems {
            if stem == own_stem {
                continue;
            }
            let handle = self.handle(&stem, discovered_seed(&stem));
            let Some(transcript) = handle.read_session().map_err(|error| {
                error.context(format!(
                    "legacy root {stem} is unreadable; deferring adoption rather than finalizing a \
                     fold that would permanently drop its turns"
                ))
            })?
            else {
                continue;
            };
            if transcript.meta.session_id.is_some()
                || transcript.meta.thread_id.as_deref() != Some(thread_id)
            {
                continue;
            }
            if let Some(expected) = session.agent_id.as_deref()
                && transcript.meta.agent_id.as_deref() != Some(expected)
            {
                continue;
            }
            found.push(Candidate {
                path: handle.path().to_path_buf(),
                meta: transcript.meta,
                messages: transcript.messages,
                tools: transcript.tools,
            });
        }
        // Stable: equal `created` keeps the index's creation order.
        found.sort_by(|a, b| a.meta.created.cmp(&b.meta.created));
        Ok(found)
    }

    /// Root stems whose entry log names `thread_id` in a `_meta`, whether or
    /// not the index has caught up with them. An error when the log cannot be
    /// scanned: adoption must not finalize without it.
    fn logged_roots(&self, thread_id: &str) -> anyhow::Result<BTreeSet<String>> {
        let thread_id = thread_id.to_string();
        let entries = self.run(|docs, declared| async move {
            declared.ensure(&docs).await?;
            let query = Query::filter(Filter::eq("meta.thread_id", thread_id));
            docs.query_all(ENTRIES, &query).await
        });
        let entries = entries.map_err(|error| {
            anyhow::Error::new(error)
                .context("could not scan the transcript log for legacy roots; deferring adoption")
        })?;
        Ok(entries
            .iter()
            .filter_map(|entry| entry.doc.get("stem").and_then(Value::as_str))
            .filter(|stem| !is_subagent(stem))
            .map(str::to_string)
            .collect())
    }

    pub(super) fn adopt(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        if self.session_exists(session) {
            return Ok(None);
        }
        let stem = session_stem(session);
        let candidates = self.adoption_candidates(session, thread_id, &stem)?;

        let mut meta = seed.clone();
        meta.session_id = Some(session.session_id());
        meta.parent_session_id = session.parent_session_id();
        meta.thread_id = Some(thread_id.to_string());
        meta.turn_count = 0;
        meta.prefix_message_count = None;
        meta.input_tokens = 0;
        meta.output_tokens = 0;
        meta.cached_input_tokens = 0;
        meta.charged_amount_usd = 0.0;
        let mut messages: Vec<TranscriptMessage> = Vec::new();
        let mut tools = None;
        let mut earliest_created: Option<String> = None;
        let mut latest_updated: Option<String> = None;
        let mut adopted = Vec::new();
        for candidate in candidates {
            if messages.is_empty() && !candidate.messages.is_empty() {
                meta.prefix_message_count = candidate.meta.prefix_message_count;
            }
            messages.extend(candidate.messages);
            if candidate.tools.is_some() {
                tools = candidate.tools;
            }
            meta.turn_count += candidate.meta.turn_count;
            meta.input_tokens += candidate.meta.input_tokens;
            meta.output_tokens += candidate.meta.output_tokens;
            meta.cached_input_tokens += candidate.meta.cached_input_tokens;
            meta.charged_amount_usd += candidate.meta.charged_amount_usd;
            let created = &candidate.meta.created;
            if !created.is_empty() && earliest_created.as_ref().is_none_or(|e| created < e) {
                earliest_created = Some(created.clone());
            }
            let updated = &candidate.meta.updated;
            if !updated.is_empty() && latest_updated.as_ref().is_none_or(|l| updated > l) {
                latest_updated = Some(updated.clone());
            }
            adopted.push(candidate.path);
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

        let reservation = {
            let stem = stem.clone();
            let docs = std::sync::Arc::clone(&self.docs);
            let tables = std::sync::Arc::clone(&self.tables);
            let declared = std::sync::Arc::clone(&self.declared);
            run_on(&self.bridge, async move {
                declared.ensure(&docs).await?;
                Ok(reservation::reserve(&docs, &tables.leases, &stem).await)
            })?
        };
        let reservation = match reservation {
            Ok(reservation) => reservation,
            Err(error) if error.downcast_ref::<StorageError>().is_some() => return Err(error),
            Err(_) => {
                tracing::debug!(
                    target: "tinyagents_session::port::drivers",
                    session = %stem,
                    "[session-store] adoption lost the session stem to a concurrent write"
                );
                return Ok(None);
            }
        };
        let handle = self.reserved_handle(&stem, meta.clone(), Some(reservation.clone()));
        let written = handle.append_turn(TranscriptTurn {
            prev: &[],
            next: &messages,
            meta: &meta,
            turn_usage: None,
            request_id: None,
            tools: tools.as_ref(),
        });
        if let Err(error) = written {
            if self.session_exists(session) {
                return Ok(None);
            }
            self.release(&stem, &reservation);
            return Err(error);
        }
        tracing::info!(
            target: "tinyagents_session::port::drivers",
            session = %stem,
            adopted = adopted.len(),
            messages = messages.len(),
            "[session-store] adopted legacy roots into the session"
        );
        Ok(Some(SessionAdoption {
            path: handle.path().to_path_buf(),
            adopted,
            messages: messages.len(),
        }))
    }
}

#[cfg(test)]
#[path = "transcripts_adoption_tests.rs"]
mod tests;
