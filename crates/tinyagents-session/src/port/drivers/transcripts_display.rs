//! The display read and the thread listing of driver-backed transcripts.
//!
//! - **Display.** [`TranscriptRead::read_display`] replays a stem's whole
//!   entry log, not the reduced context: `extend` rows are message records, a
//!   `set` after the first write is a compaction marker carrying the new
//!   context (the JSONL writer records exactly that), a `clear` is an empty
//!   compaction, and a `partial` is an interrupted record at the position it
//!   was written. Entries keep no wall clock, so a marker's `ts` is `None` and
//!   intermediate tool-call rows carry no per-step timestamp.
//! - **Thread listing.** [`TranscriptLocator::thread_transcripts`] reads the
//!   index (first re-indexing any written transcript whose index document is
//!   missing, see [`DriverTranscriptLocator::repair_unindexed`]): the thread's
//!   written roots, ordered like the file view (by
//!   `_meta.created`, session chains oldest first, pre-identity roots dropped
//!   once a session root exists), and the sub-agent transcripts that descend
//!   from a root stem or (unscoped) name the thread.

use std::collections::HashSet;
use std::sync::Arc;

use serde_json::Value;
use tinyagents_storage_kit::{DocumentStoreExt, Filter, Query, Sort, StorageError};

use super::{
    DriverTranscriptHistory, DriverTranscriptLocator, ENTRIES, Entry, HistoryInner, INDEX,
    discovered_seed, normalized, oldest_first, run_on,
};
use crate::transcript::{
    CompactionMarker, DisplayRecord, DisplaySessionTranscript, SessionRef, ThreadTranscript,
    ThreadTranscripts, TranscriptLocator, TranscriptMeta, TranscriptPartial, TranscriptRead,
    display_message_of, partial_display_message, session_stem,
};

impl HistoryInner {
    /// The display projection of this stem's whole log; `None` when nothing
    /// was ever written to it.
    async fn display(&self) -> Result<Option<DisplaySessionTranscript>, StorageError> {
        self.declared.ensure(&self.docs).await?;
        {
            // As the model-context read does: a write whose index refresh
            // failed is repaired here, so a transcript read only for display
            // does not stay invisible to the thread listing and spend.
            let mut replay = self.replay.lock().await;
            self.refresh(&mut replay).await?;
            self.repair_index(&mut replay).await;
        }
        let query = Query::filter(Filter::eq("stem", self.stem.as_str())).sort(Sort::asc("seq"));
        let mut meta = None;
        let mut records: Vec<DisplayRecord> = Vec::new();
        let mut written = false;
        for stored in self.docs.query_all(ENTRIES, &query).await? {
            let entry: Entry = serde_json::from_value(stored.doc).map_err(|error| {
                StorageError::serialization(format!(
                    "transcript entry {} is unreadable: {error}",
                    stored.id
                ))
            })?;
            let was_written = written;
            written |= entry.written;
            if let Some(entry_meta) = entry.meta {
                meta = Some(entry_meta);
            }
            if let Some(set) = entry.set {
                let rows: Vec<_> = normalized(set).iter().map(display_message_of).collect();
                if was_written {
                    records.push(DisplayRecord::Compaction(CompactionMarker {
                        replacement: rows,
                        ts: None,
                        request_id: entry.request_id.clone(),
                    }));
                } else {
                    records.extend(
                        rows.into_iter()
                            .map(|row| DisplayRecord::Message(Box::new(row))),
                    );
                }
            }
            if let Some(extend) = entry.extend {
                records.extend(
                    normalized(extend)
                        .iter()
                        .map(|row| DisplayRecord::Message(Box::new(display_message_of(row)))),
                );
            }
            if let Some(partial) = entry.partial {
                let partial = TranscriptPartial::from(partial);
                records.push(DisplayRecord::Message(Box::new(partial_display_message(
                    &partial,
                    entry.request_id.as_deref(),
                ))));
            }
            if entry.clear {
                records.push(DisplayRecord::Compaction(CompactionMarker {
                    replacement: Vec::new(),
                    ts: None,
                    request_id: None,
                }));
            }
        }
        Ok(written.then(|| DisplaySessionTranscript {
            meta: meta.unwrap_or_else(|| self.seed.clone()),
            records,
        }))
    }
}

impl DriverTranscriptHistory {
    pub(super) fn display(&self) -> anyhow::Result<Option<DisplaySessionTranscript>> {
        let inner = Arc::clone(&self.inner);
        Ok(run_on(&self.bridge, async move { inner.display().await })?)
    }
}

/// One index document's lookup fields.
pub(super) struct Indexed {
    pub(super) stem: String,
    agent_id: Option<String>,
    thread_id: Option<String>,
}

impl Indexed {
    fn of(doc: &Value) -> Option<Self> {
        let field = |key: &str| doc.get(key).and_then(Value::as_str).map(str::to_string);
        Some(Self {
            stem: field("stem")?,
            agent_id: field("agent_id"),
            thread_id: field("thread_id"),
        })
    }
}

impl DriverTranscriptLocator {
    /// The written index documents matching `filter`, oldest first.
    pub(super) fn indexed(&self, filter: Filter, what: &'static str) -> Vec<Indexed> {
        let found = self.run(|docs, declared| async move {
            declared.ensure(&docs).await?;
            let query = oldest_first(Query::filter(filter.and(Filter::eq("written", true))));
            docs.query_all(INDEX, &query).await
        });
        match found {
            Ok(items) => items
                .iter()
                .filter_map(|item| Indexed::of(&item.doc))
                .collect(),
            Err(error) => {
                tracing::warn!(
                    target: "tinyagents_session::port::drivers",
                    locator = %self.label,
                    lookup = what,
                    %error,
                    "[session-store] transcript listing failed"
                );
                Vec::new()
            }
        }
    }

    /// Re-indexes every written transcript that has no index document.
    ///
    /// A write whose index refresh failed (see `HistoryInner::index`) leaves a
    /// durable entry log with no lookup document, and nothing repairs that
    /// until the transcript is next read or written; until then the thread
    /// listing, which reads the index alone, would not see it. Detecting this
    /// is two counts (written index documents against stems whose first entry
    /// is written); only a mismatch pays for finding the missing stems, and
    /// reading each one through its own handle runs the ordinary read-time
    /// index repair.
    fn repair_unindexed(&self) {
        let missing = self.run(|docs, declared| async move {
            declared.ensure(&docs).await?;
            let first_entries = Filter::eq("seq", 0).and(Filter::eq("written", true));
            let indexed = docs.count(INDEX, &Filter::eq("written", true)).await?;
            let logs = docs.count(ENTRIES, &first_entries).await?;
            if logs <= indexed {
                return Ok(Vec::new());
            }
            let indexed_stems: HashSet<String> = docs
                .query_all(INDEX, &Query::filter(Filter::eq("written", true)))
                .await?
                .iter()
                .filter_map(|item| Indexed::of(&item.doc))
                .map(|indexed| indexed.stem)
                .collect();
            Ok(docs
                .query_all(ENTRIES, &Query::filter(first_entries))
                .await?
                .iter()
                .filter_map(|item| item.doc.get("stem").and_then(Value::as_str))
                .filter(|stem| !indexed_stems.contains(*stem))
                .map(str::to_string)
                .collect::<Vec<_>>())
        });
        let missing = match missing {
            Ok(missing) => missing,
            Err(error) => {
                tracing::warn!(
                    target: "tinyagents_session::port::drivers",
                    locator = %self.label,
                    %error,
                    "[session-store] unindexed transcript scan failed"
                );
                return;
            }
        };
        for stem in missing {
            tracing::debug!(
                target: "tinyagents_session::port::drivers",
                locator = %self.label,
                %stem,
                "[session-store] re-indexing a transcript whose index document is missing"
            );
            if let Err(error) = self.handle(&stem, discovered_seed(&stem)).read_session() {
                tracing::warn!(
                    target: "tinyagents_session::port::drivers",
                    locator = %self.label,
                    %stem,
                    %error,
                    "[session-store] re-indexing an unindexed transcript failed"
                );
            }
        }
    }

    fn listed(&self, stem: String) -> ThreadTranscript {
        let transcript: Arc<dyn TranscriptRead> =
            Arc::new(self.handle(&stem, discovered_seed(&stem)));
        ThreadTranscript { stem, transcript }
    }

    /// The `_meta` of the transcript at `stem`, when it is readable.
    fn root_meta(&self, stem: &str) -> Option<TranscriptMeta> {
        self.handle(stem, discovered_seed(stem))
            .read_session()
            .ok()
            .flatten()
            .map(|session| session.meta)
    }

    /// The thread's roots in projection order; see the module docs.
    ///
    /// Like the file view, roots are first ordered by `_meta.created` (then
    /// generation, then stem), not by when their index document was written:
    /// an import writes files in name order, which need not be chronological.
    /// Once any root records a session identity, pre-identity roots are
    /// dropped — they were folded into that session — whether or not the
    /// session is agent-scoped.
    fn ordered_roots(&self, thread_id: &str, roots: Vec<Indexed>) -> Vec<String> {
        let mut roots: Vec<(Indexed, Option<TranscriptMeta>)> = roots
            .into_iter()
            .map(|root| {
                let meta = self.root_meta(&root.stem);
                (root, meta)
            })
            .collect();
        roots.sort_by(|(a, a_meta), (b, b_meta)| {
            let created = |meta: &Option<TranscriptMeta>| {
                meta.as_ref()
                    .map(|meta| meta.created.clone())
                    .unwrap_or_default()
            };
            created(a_meta)
                .cmp(&created(b_meta))
                .then_with(|| stem_generation(&a.stem).cmp(&stem_generation(&b.stem)))
                .then_with(|| a.stem.cmp(&b.stem))
        });
        let has_session_id = |meta: &Option<TranscriptMeta>| {
            meta.as_ref().is_some_and(|meta| meta.session_id.is_some())
        };
        if !roots.iter().any(|(_, meta)| has_session_id(meta)) {
            return roots.into_iter().map(|(root, _)| root.stem).collect();
        }
        let mut chains: Vec<Vec<String>> = Vec::new();
        let mut agents: HashSet<String> = HashSet::new();
        for (root, _) in &roots {
            if let Some(agent_id) = &root.agent_id
                && agents.insert(agent_id.clone())
            {
                let chain: Vec<String> = self
                    .session_chain(&SessionRef::scoped(thread_id, agent_id.as_str()))
                    .iter()
                    .map(session_stem)
                    .collect();
                if !chain.is_empty() {
                    chains.push(chain);
                }
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        let mut ordered = Vec::new();
        let mut dropped_legacy = 0usize;
        for (root, meta) in roots {
            if seen.contains(&root.stem) {
                continue;
            }
            if let Some(chain) = chains.iter().find(|chain| chain.contains(&root.stem)) {
                for link in chain {
                    if seen.insert(link.clone()) {
                        ordered.push(link.clone());
                    }
                }
            } else if has_session_id(&meta) {
                seen.insert(root.stem.clone());
                ordered.push(root.stem);
            } else {
                dropped_legacy += 1;
            }
        }
        if dropped_legacy > 0 {
            tracing::debug!(
                target: "tinyagents_session::port::drivers",
                thread_id,
                dropped_legacy,
                "[session-store] dropped adopted legacy roots in favour of the session chain"
            );
        }
        ordered
    }

    pub(super) fn list_thread(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<ThreadTranscripts> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return None;
        }
        self.repair_unindexed();
        let mut filter = Filter::eq("thread_id", thread_id).and(Filter::eq("subagent", false));
        if let Some(agent_id) = agent_id {
            filter = filter.and(Filter::eq("agent_id", agent_id));
        }
        let roots = self.indexed(filter, "thread_roots");
        if roots.is_empty() {
            return None;
        }
        let prefixes: Vec<String> = roots
            .iter()
            .map(|root| format!("{}__", root.stem))
            .collect();
        let roots = self.ordered_roots(thread_id, roots);
        // Bounded to this thread: each root's `{stem}__` range, plus
        // (unscoped) the sub-agents naming the thread. The in-memory filter
        // below stays as the exact rule.
        let by_prefix = prefixes
            .iter()
            .map(|prefix| prefix_range("stem", prefix))
            .reduce(Filter::or);
        let by_thread = Filter::eq("thread_id", thread_id);
        let wanted = match (by_prefix, agent_id) {
            (Some(ranges), None) => ranges.or(by_thread),
            (Some(ranges), Some(_)) => ranges,
            (None, _) => by_thread,
        };
        let mut subagents: Vec<String> = self
            .indexed(Filter::eq("subagent", true).and(wanted), "thread_subagents")
            .into_iter()
            .filter(|sub| {
                prefixes.iter().any(|prefix| sub.stem.starts_with(prefix))
                    || (agent_id.is_none() && sub.thread_id.as_deref() == Some(thread_id))
            })
            .map(|sub| sub.stem)
            .collect();
        subagents.sort();
        subagents.dedup();
        tracing::debug!(
            target: "tinyagents_session::port::drivers",
            thread_id,
            roots = roots.len(),
            subagents = subagents.len(),
            "[session-store] listed thread transcripts"
        );
        Some(ThreadTranscripts {
            roots: roots.into_iter().map(|stem| self.listed(stem)).collect(),
            subagents: subagents
                .into_iter()
                .map(|stem| self.listed(stem))
                .collect(),
        })
    }
}

/// `field` starts with `prefix`, as a range: `prefix <= field < upper`,
/// where `upper` is `prefix` with its last character incremented. A stem
/// prefix here always ends in `__`, so the bump is `_` to `` ` ``.
fn prefix_range(field: &str, prefix: &str) -> Filter {
    let mut upper: Vec<char> = prefix.chars().collect();
    let filter = Filter::gte(field, prefix);
    match upper.pop().and_then(|last| char::from_u32(last as u32 + 1)) {
        Some(next) => {
            upper.push(next);
            filter.and(Filter::lt(field, upper.into_iter().collect::<String>()))
        }
        None => filter,
    }
}

/// The compaction generation a stem encodes: `n` for `{base}.g{n}`, else 0.
fn stem_generation(stem: &str) -> u32 {
    stem.rsplit_once(".g")
        .filter(|(_, digits)| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|(_, digits)| digits.parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "transcripts_display_tests.rs"]
mod tests;
