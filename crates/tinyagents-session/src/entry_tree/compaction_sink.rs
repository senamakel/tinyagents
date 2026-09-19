//! [`tinyagents_harness::summarization::CompactionSink`] implementation
//! backed by the entry tree.
//!
//! `tinyagents-harness` cannot depend on `tinyagents-session` (the
//! dependency runs the other way — see that trait's docs), so the harness
//! only defines the interface; this is the durable, session-backed
//! implementation a host attaches to a [`RunContext`][tinyagents_harness::context::RunContext]
//! via `RunContext::with_compaction_sink`.

use std::path::Path;
use std::sync::Mutex;

use tinyagents_harness::error::Result;
use tinyagents_harness::summarization::{CompactionRecord, CompactionSink};

use super::types::{CompactionEntry, EntryId, EntryKind};
use super::EntryTree;

/// A [`CompactionSink`] that persists every [`CompactionRecord`] a run
/// produces into a session's entry tree as a durable
/// [`CompactionEntry`][crate::entry_tree::CompactionEntry], anchored to the
/// tree's current head at the moment [`Self::persist`] is called.
///
/// # Index translation
///
/// [`CompactionRecord::first_kept_index`] is a position in the *harness's*
/// non-system message slice, not an [`EntryId`] the entry tree understands.
/// `persist` translates it by re-walking the ancestor chain of the current
/// tip and counting only the entries
/// [`EntryTree::build_context`] itself turns into messages
/// ([`EntryKind::Message`] and [`EntryKind::Custom`] — the same filter
/// `entry_to_message` applies internally), so the index lines up with what a
/// harness compaction step actually operated on. A record whose index falls
/// outside that count (for example, one produced from a transcript this
/// session never saw) is skipped rather than persisted as a boundary that
/// would corrupt [`EntryTree::build_context`].
pub struct SessionCompactionSink<'a> {
    tree: EntryTree<'a>,
    tip: Mutex<Option<EntryId>>,
}

impl<'a> SessionCompactionSink<'a> {
    /// Builds a sink anchored to `session_id`'s current head at construction
    /// time. The tip advances to each new compaction entry as
    /// [`Self::persist`] is called, so a run that compacts more than once
    /// keeps appending to the same branch instead of re-anchoring at the
    /// original head every time.
    pub fn new(workspace_dir: &'a Path, session_id: impl Into<String>) -> Result<Self> {
        let tree = EntryTree::new(workspace_dir, session_id);
        let tip = tree.head()?;
        Ok(Self {
            tree,
            tip: Mutex::new(tip),
        })
    }

    /// The entry tree this sink writes to.
    pub fn tree(&self) -> &EntryTree<'a> {
        &self.tree
    }

    /// The current tip this sink will anchor its next compaction entry to.
    pub fn tip(&self) -> Option<EntryId> {
        self.tip.lock().expect("tip mutex poisoned").clone()
    }
}

impl CompactionSink for SessionCompactionSink<'_> {
    fn persist(&self, record: &CompactionRecord) -> Result<()> {
        let mut tip_guard = self.tip.lock().expect("tip mutex poisoned");
        let Some(tip) = tip_guard.clone() else {
            // Nothing has been appended to this session yet — there is
            // nothing to anchor a compaction boundary to.
            return Ok(());
        };

        let chain = self.tree.ancestor_chain(&tip)?;
        let message_entry_ids: Vec<EntryId> = chain
            .iter()
            .filter(|entry| matches!(entry.kind, EntryKind::Message(_) | EntryKind::Custom(_)))
            .map(|entry| entry.id.clone())
            .collect();

        let Some(first_kept_entry_id) = message_entry_ids.get(record.first_kept_index).cloned()
        else {
            return Ok(());
        };

        let entry = EntryKind::Compaction(CompactionEntry {
            summary: record.summary.clone(),
            first_kept_entry_id,
            tokens_before: record.tokens_before,
            usage: record.usage.clone(),
            details: record.details.clone(),
        });
        let new_tip = self.tree.append(Some(&tip), entry)?;
        *tip_guard = Some(new_tip);
        Ok(())
    }
}
