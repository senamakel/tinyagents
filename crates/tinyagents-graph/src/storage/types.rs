//! The stored bodies of the one graph checkpointer (spec schema-v2 §5).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tinyagents_storage_kit::Key;

use crate::checkpoint::PendingWrite;

/// The body of one `graph.checkpoints` document.
///
/// `checkpoint` is the opaque checkpoint payload, kept as a typed field rather
/// than a JSON string; everything else is an indexed projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CheckpointBody {
    pub(crate) ns: String,
    pub(crate) thread: String,
    pub(crate) subgraph: String,
    pub(crate) seq: u64,
    pub(crate) checkpoint_id: String,
    pub(crate) checkpoint: Value,
}

/// The body of one `graph.threads` document: the next insertion sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ThreadBody {
    pub(crate) ns: String,
    pub(crate) thread: String,
    pub(crate) next: u64,
}

/// The body of one `graph.writes` document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WritesBody {
    pub(crate) ns: String,
    pub(crate) thread: String,
    pub(crate) subgraph: String,
    pub(crate) checkpoint_id: String,
    pub(crate) writes: Vec<PendingWrite>,
}

/// A subgraph namespace as one string, injectively: the root is empty and any
/// other namespace is `Key::of(["n", parts...])`, so `["a", "b"]`, `["a/b"]`
/// and `[""]` never meet.
pub(crate) fn subgraph_key(namespace: &[String]) -> String {
    if namespace.is_empty() {
        return String::new();
    }
    Key::of(std::iter::once("n").chain(namespace.iter().map(String::as_str)))
        .map(|key| key.as_str().to_owned())
        .unwrap_or_default()
}

/// The id of the checkpoint inserted at `seq` for `thread` in `ns`.
pub(crate) fn checkpoint_id_key(ns: &str, thread: &str, seq: u64) -> String {
    key(&[ns, thread, &format!("{seq:020}")])
}

/// The id of `thread`'s counter document in `ns`.
pub(crate) fn thread_id_key(ns: &str, thread: &str) -> String {
    key(&[ns, thread])
}

/// The id of the pending writes of `checkpoint_id` in `subgraph`.
pub(crate) fn writes_id_key(ns: &str, thread: &str, subgraph: &str, checkpoint_id: &str) -> String {
    key(&[ns, thread, subgraph, checkpoint_id])
}

/// The lease key of `thread` in `ns` (owner `graph`).
pub(crate) fn lease_key(ns: &str, thread: &str) -> String {
    key(&[ns, thread])
}

/// `Key::of(parts)` as text; the parts are never empty, so it cannot fail.
fn key(parts: &[&str]) -> String {
    Key::of(parts)
        .map(|key| key.as_str().to_owned())
        .unwrap_or_default()
}
