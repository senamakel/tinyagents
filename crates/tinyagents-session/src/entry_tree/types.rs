//! Public types for the conversation entry tree: [`EntryId`], [`Entry`],
//! [`EntryKind`], and the fork request/response shapes.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use tinyagents_harness::tinyinference_llm::Usage;

use crate::transcript::TranscriptMessage;

/// A stable identifier for one node in the entry tree.
///
/// Ids are opaque strings. The store allocates them deterministically as
/// `"{session_id}:{ordinal}"`, so re-reading the same session (or replaying
/// a legacy linear transcript into the tree) always assigns the same ids —
/// see [`crate::entry_tree::legacy`] for the derivation used for pre-tree
/// data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntryId(pub String);

impl EntryId {
    /// Builds the deterministic id for the `ordinal`-th entry of `session_id`
    /// (0-based).
    pub fn derive(session_id: &str, ordinal: u64) -> Self {
        Self(format!("{session_id}:{ordinal}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for EntryId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for EntryId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// One append-only node in a session's entry tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: EntryId,
    /// `None` only for the root entry of a session.
    pub parent_id: Option<EntryId>,
    /// Monotonic per-session sequence, also used to derive [`EntryId`] for
    /// entries created by this store (legacy-derived entries reuse their
    /// source file order the same way — see [`crate::entry_tree::legacy`]).
    pub ordinal: u64,
    pub kind: EntryKind,
    /// RFC-3339 timestamp this entry was appended.
    pub ts: String,
}

/// The payload carried by an [`Entry`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntryKind {
    /// An ordinary conversation message.
    Message(TranscriptMessage),
    /// A durable record of a context-compaction operation. Context
    /// projection ([`crate::entry_tree::EntryTree::build_context`]) never
    /// reads past the newest entry of this kind on the path to a tip.
    Compaction(CompactionEntry),
    /// A note written at a navigation point summarizing the path that was
    /// left behind (pi's "abandoned branch" bookmark).
    BranchSummary(BranchSummaryEntry),
    /// A named bookmark on a tip.
    Label(LabelEntry),
    /// A host-defined out-of-band record, e.g. a tool-execution or
    /// notification entry that is not itself a conversation message. Mirrors
    /// `tinyinference_llm::message::CustomMessage` and is projected to
    /// `Message::Custom` by [`crate::entry_tree::EntryTree::build_context`].
    Custom(CustomEntry),
}

/// Durable record of a compaction operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionEntry {
    /// The replacement summary installed as the new context.
    pub summary: String,
    /// The first entry (by id) that survives the compaction unsummarized;
    /// context projection includes everything from this entry to the tip,
    /// in addition to the summary.
    pub first_kept_entry_id: EntryId,
    /// Token count of the context immediately before compaction.
    pub tokens_before: u64,
    /// Usage/cost of the summarization call, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Additional host-defined provenance (cut-point rule, split-turn
    /// bookkeeping, hook that authored the summary, ...).
    #[serde(default)]
    pub details: Value,
}

/// A note summarizing the branch abandoned at a navigation point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchSummaryEntry {
    /// The entry the navigation moved away from.
    pub from_id: EntryId,
    pub summary: String,
}

/// A named bookmark on a tip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelEntry {
    pub name: String,
}

/// A host-defined out-of-band record carried in the tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomEntry {
    /// Host-defined discriminator, e.g. `"compaction"` or `"notification"`.
    pub kind: String,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

/// Which part of the tree a [`Fork`] duplicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForkScope {
    /// In-place branching: the returned tip is an existing entry in the
    /// ancestor chain (a shared parent pointer). Appending to it grows a new
    /// sibling subtree without copying anything.
    Branch,
    /// Path copy: the entire root-to-target ancestor chain is duplicated as
    /// new entries with new ids, and the returned tip is the copy of the
    /// fork point. Use when the caller needs an independently addressable
    /// history (e.g. before an edit that must not perturb ids reachable from
    /// the original tip).
    Tree,
}

/// Where in the ancestor chain a [`Fork`] points, relative to the given tip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForkPosition {
    /// The fork point is the tip's parent (the tip itself is dropped from
    /// the new branch).
    Before,
    /// The fork point is the tip itself.
    At,
}

/// A fork request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fork {
    pub scope: ForkScope,
    pub position: ForkPosition,
}

/// One named branch: a label pointing at a tip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub tip_id: EntryId,
}
