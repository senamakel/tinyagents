//! Conversation entry tree: append-only, branchable session history.
//!
//! See the [`crate`] docs for how this relates to the JSONL transcript and
//! the `session_messages` SQLite index, and `docs/modules/session/README.md`
//! for the design write-up (entry kinds, the context-projection rule, fork
//! semantics, and the legacy import path). In short: every entry has an
//! [`EntryId`] and an optional `parent_id`; a session is a *tree*, not a
//! list; a tip is any entry with no children (or a [`Branch`] label on one);
//! [`EntryTree::build_context`] projects one tip's ancestor chain to a
//! model-ready message list, stopping at the newest compaction; and
//! [`EntryTree::fork`] creates a new tip without disturbing existing ones.
//!
//! Pre-tree data (a JSONL transcript or the `session_messages` table) has no
//! parent pointer. [`legacy::from_transcript`] and
//! [`legacy::from_session_messages`] derive one deterministically (entry *n*
//! parents entry *n+1*, in file/row order) so that data reads into this same
//! model; [`EntryTree::import_legacy`] persists the result idempotently.

pub mod legacy;
mod store;
mod types;

use std::path::Path;

use tinyagents_harness::error::Result;
use tinyagents_harness::tinyinference_llm::message::{CustomMessage, SystemMessage, UserMessage};
use tinyagents_harness::tinyinference_llm::{ContentBlock, Message};

use crate::context::StorageContext;
use crate::store::{with_connection, with_transaction};

pub use legacy::{from_messages, from_session_messages, from_transcript};
pub use types::{
    Branch, BranchSummaryEntry, CompactionEntry, CustomEntry, Entry, EntryId, EntryKind, Fork,
    ForkPosition, ForkScope, LabelEntry,
};

/// Handle to one session's entry tree, scoped to a workspace root and
/// session id. Cheap to construct; every method opens the shared, cached
/// session-database connection (see [`crate::store`]).
pub struct EntryTree<'a> {
    workspace_dir: &'a Path,
    session_id: String,
}

impl<'a> EntryTree<'a> {
    pub fn new(workspace_dir: &'a Path, session_id: impl Into<String>) -> Self {
        Self {
            workspace_dir,
            session_id: session_id.into(),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Appends `kind` as a new entry with the given `parent_id`.
    ///
    /// Passing `None` is only valid for a session's first entry; every
    /// following call should pass an explicit parent (typically the id just
    /// returned, or a fork's result) to grow that branch. For the common
    /// "always extend the latest thing written" case, use
    /// [`EntryTree::append_to_head`].
    pub fn append(&self, parent_id: Option<&EntryId>, kind: EntryKind) -> Result<EntryId> {
        with_transaction(self.workspace_dir, |conn| {
            let ordinal = store::next_ordinal(conn, &self.session_id)?;
            let id = EntryId::derive(&self.session_id, ordinal);
            let entry = Entry {
                id: id.clone(),
                parent_id: parent_id.cloned(),
                ordinal,
                kind,
                ts: chrono::Utc::now().to_rfc3339(),
            };
            store::insert_entry(conn, &self.session_id, &entry)?;
            Ok(id)
        })
    }

    /// Appends `kind` as a child of the current head (the entry with the
    /// greatest ordinal in the session, or the session root if empty).
    ///
    /// This is the "linear append still works" path: a caller that never
    /// forks and always calls this keeps writing a plain, non-branching
    /// chain, identical in shape to the pre-tree transcript.
    pub fn append_to_head(&self, kind: EntryKind) -> Result<EntryId> {
        with_transaction(self.workspace_dir, |conn| {
            let parent = store::head(conn, &self.session_id)?;
            let ordinal = store::next_ordinal(conn, &self.session_id)?;
            let id = EntryId::derive(&self.session_id, ordinal);
            let entry = Entry {
                id: id.clone(),
                parent_id: parent,
                ordinal,
                kind,
                ts: chrono::Utc::now().to_rfc3339(),
            };
            store::insert_entry(conn, &self.session_id, &entry)?;
            Ok(id)
        })
    }

    /// Returns the entry with the greatest ordinal (the current head), if
    /// any entries have been appended.
    pub fn head(&self) -> Result<Option<EntryId>> {
        with_connection(self.workspace_dir, |conn| store::head(conn, &self.session_id))
    }

    /// Fetches one entry by id.
    pub fn get(&self, id: &EntryId) -> Result<Option<Entry>> {
        with_connection(self.workspace_dir, |conn| {
            store::get_entry(conn, &self.session_id, id)
        })
    }

    /// The full root-to-`tip` ancestor chain, in chronological order.
    pub fn ancestor_chain(&self, tip: &EntryId) -> Result<Vec<Entry>> {
        with_connection(self.workspace_dir, |conn| {
            store::ancestor_chain(conn, &self.session_id, tip)
        })
    }

    /// Every current tip (entry with no children) in the session, in
    /// ordinal order.
    pub fn tips(&self) -> Result<Vec<EntryId>> {
        with_connection(self.workspace_dir, |conn| {
            store::leaf_entries(conn, &self.session_id)
        })
    }

    /// Every named branch (label → tip) in the session, in name order.
    pub fn branches(&self) -> Result<Vec<Branch>> {
        with_connection(self.workspace_dir, |conn| {
            store::list_branches(conn, &self.session_id)
        })
    }

    /// Appends a [`LabelEntry`] naming `tip`, and records `name` as a
    /// branch pointing at the new label entry (which becomes the new tip
    /// for that name). Re-labeling reassigns the name to the new entry;
    /// existing entries and other branches are untouched.
    pub fn label(&self, tip: &EntryId, name: &str) -> Result<EntryId> {
        with_transaction(self.workspace_dir, |conn| {
            let ordinal = store::next_ordinal(conn, &self.session_id)?;
            let id = EntryId::derive(&self.session_id, ordinal);
            let entry = Entry {
                id: id.clone(),
                parent_id: Some(tip.clone()),
                ordinal,
                kind: EntryKind::Label(LabelEntry {
                    name: name.to_string(),
                }),
                ts: chrono::Utc::now().to_rfc3339(),
            };
            store::insert_entry(conn, &self.session_id, &entry)?;
            store::insert_label(conn, &self.session_id, name, &id)?;
            Ok(id)
        })
    }

    /// Creates a new tip diverging from `tip`'s ancestor chain.
    ///
    /// - [`ForkScope::Branch`]: no entries are copied. The returned id is an
    ///   *existing* entry (the fork point itself); appending to it grows a
    ///   new sibling subtree in place.
    /// - [`ForkScope::Tree`]: the entire root-to-fork-point ancestor chain is
    ///   duplicated as brand-new entries (new ids, same kind/payload), and
    ///   the returned id is the copy of the fork point. Use this when the
    ///   caller needs a history that is independently addressable — nothing
    ///   reachable from the original tip changes.
    ///
    /// [`ForkPosition::At`] points the fork at `tip` itself;
    /// [`ForkPosition::Before`] points it at `tip`'s parent (dropping `tip`
    /// from the new branch). `Before` on a root entry (no parent) is an
    /// error.
    pub fn fork(&self, tip: &EntryId, fork: Fork) -> Result<EntryId> {
        with_transaction(self.workspace_dir, |conn| {
            let tip_entry = store::get_entry(conn, &self.session_id, tip)?
                .storage_context(&format!("fork: unknown tip {tip}"))?;
            let target = match fork.position {
                ForkPosition::At => tip.clone(),
                ForkPosition::Before => tip_entry.parent_id.clone().storage_context(&format!(
                    "fork: entry {tip} has no parent (position: before)"
                ))?,
            };
            match fork.scope {
                ForkScope::Branch => Ok(target),
                ForkScope::Tree => {
                    let chain = store::ancestor_chain(conn, &self.session_id, &target)?;
                    let mut new_parent: Option<EntryId> = None;
                    let mut copied_target = None;
                    for entry in &chain {
                        let ordinal = store::next_ordinal(conn, &self.session_id)?;
                        let new_id = EntryId::derive(&self.session_id, ordinal);
                        let copy = Entry {
                            id: new_id.clone(),
                            parent_id: new_parent.take(),
                            ordinal,
                            kind: entry.kind.clone(),
                            ts: chrono::Utc::now().to_rfc3339(),
                        };
                        store::insert_entry(conn, &self.session_id, &copy)?;
                        new_parent = Some(new_id.clone());
                        if entry.id == target {
                            copied_target = Some(new_id);
                        }
                    }
                    copied_target.storage_context("fork: tree copy produced no entries")
                }
            }
        })
    }

    /// Rebuilds the `branch_entries` materialized index for every current
    /// tip and named branch, from scratch, by walking `parent_id`.
    ///
    /// [`EntryTree::build_context`] uses the index opportunistically and
    /// falls back to a live walk when a tip has no (or a stale) index entry,
    /// so this is a performance operation, not a correctness prerequisite —
    /// call it after heavy branching/forking to keep lookups O(chain) via
    /// the index rather than O(depth) via repeated parent walks.
    pub fn rebuild_index(&self) -> Result<()> {
        with_transaction(self.workspace_dir, |conn| {
            store::rebuild_index(conn, &self.session_id)
        })
    }

    /// Imports legacy linear entries (from [`legacy::from_transcript`] or
    /// [`legacy::from_session_messages`]) into the tree, skipping any id
    /// that already exists so repeated imports of the same source are a
    /// no-op.
    pub fn import_legacy(&self, entries: &[Entry]) -> Result<()> {
        with_transaction(self.workspace_dir, |conn| {
            for entry in entries {
                if store::get_entry(conn, &self.session_id, &entry.id)?.is_some() {
                    continue;
                }
                store::insert_entry(conn, &self.session_id, entry)?;
            }
            Ok(())
        })
    }

    /// Projects `tip`'s ancestor chain to a model-ready message list.
    ///
    /// Walks the chain newest-first (logically; the index/walk both return
    /// chronological order and this reasons over it that way) to find the
    /// **newest** [`EntryKind::Compaction`] entry on the path, and never
    /// includes anything older than it: the result is
    /// `[summary as a system message] + kept entries in chronological
    /// order`, where "kept" is everything from the compaction's
    /// `first_kept_entry_id` (inclusive) to `tip`. When there is no
    /// compaction on the path, every entry from the root is kept.
    ///
    /// [`EntryKind::Label`] and [`EntryKind::BranchSummary`] entries are
    /// skipped — they are tree bookkeeping, not conversation content.
    /// [`EntryKind::Custom`] becomes `Message::Custom`.
    pub fn build_context(&self, tip: &EntryId) -> Result<Vec<Message>> {
        let chain = with_connection(self.workspace_dir, |conn| {
            if let Some(chain) = store::indexed_chain(conn, &self.session_id, tip)? {
                Ok(chain)
            } else {
                store::ancestor_chain(conn, &self.session_id, tip)
            }
        })?;

        let compaction = chain
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, entry)| match &entry.kind {
                EntryKind::Compaction(c) => Some((idx, c.clone())),
                _ => None,
            });

        let mut messages = Vec::new();
        let start_idx = match compaction {
            Some((idx, compaction)) => {
                messages.push(Message::System(SystemMessage {
                    content: vec![ContentBlock::Text(compaction.summary.clone())],
                }));
                chain
                    .iter()
                    .position(|e| e.id == compaction.first_kept_entry_id)
                    .unwrap_or(idx + 1)
            }
            None => 0,
        };

        for entry in &chain[start_idx..] {
            if let Some(message) = entry_to_message(&entry.kind) {
                messages.push(message);
            }
        }
        Ok(messages)
    }
}

/// Converts one entry's kind to a context message, or `None` for kinds that
/// carry no conversational content ([`EntryKind::Label`],
/// [`EntryKind::BranchSummary`], and a stray [`EntryKind::Compaction`] that
/// is not the boundary entry itself — [`EntryTree::build_context`] never
/// passes one of those in, but the match stays exhaustive for safety).
fn entry_to_message(kind: &EntryKind) -> Option<Message> {
    match kind {
        EntryKind::Message(message) => Some(transcript_message_to_message(message)),
        EntryKind::Custom(custom) => Some(Message::Custom(CustomMessage {
            kind: custom.kind.clone(),
            payload: custom.payload.clone(),
            display: custom.display.clone(),
        })),
        EntryKind::Label(_) | EntryKind::BranchSummary(_) | EntryKind::Compaction(_) => None,
    }
}

/// Best-effort mapping from the durable, provider-neutral
/// [`crate::transcript::TranscriptMessage`] to an inference [`Message`].
///
/// `system`/`user`/`assistant` map directly to their typed counterparts as a
/// single text content block (transcript rows do not carry structured
/// content blocks, tool calls, or a `tool_call_id`, so richer providers'
/// round trip is necessarily lossy here — callers that need full fidelity
/// should keep their own typed message alongside the transcript row). Any
/// other role, including `tool` (no `tool_call_id` is recoverable from a
/// bare transcript row), is carried through as `Message::Custom` tagged
/// `legacy:{role}` so no content is silently dropped.
fn transcript_message_to_message(message: &crate::transcript::TranscriptMessage) -> Message {
    let text = message.content.clone();
    match message.role.as_str() {
        "system" => Message::System(SystemMessage {
            content: vec![ContentBlock::Text(text)],
        }),
        "user" => Message::User(UserMessage {
            content: vec![ContentBlock::Text(text)],
        }),
        "assistant" => Message::Assistant(tinyagents_harness::tinyinference_llm::AssistantMessage {
            id: message.id.clone(),
            content: vec![ContentBlock::Text(text)],
            tool_calls: Vec::new(),
            usage: None,
        }),
        other => Message::Custom(CustomMessage {
            kind: format!("legacy:{other}"),
            payload: serde_json::json!({ "content": text }),
            display: Some(text),
        }),
    }
}

#[cfg(test)]
mod test;
