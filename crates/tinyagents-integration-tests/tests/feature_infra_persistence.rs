//! Feature/integration tests for the harness store infrastructure.
//!
//! Covers the durable substrate that outlives a single run: key-value stores
//! (in-memory, file-backed with path-traversal guards), append-only journals
//! (offset semantics, retention/eviction, JSONL durability), the store
//! registry.
//!
//! Deterministic and offline. File-backed cases use a unique temp directory.

use std::path::PathBuf;

use serde_json::json;
use tinyagents_harness::store::{
    AppendStore, FileStore, InMemoryAppendStore, InMemoryStore, JsonlAppendStore, Store,
    StoreRegistry,
};

/// A process-unique temp directory for file-backed cases.
fn temp_dir(tag: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    dir.push(format!("tinyagents_feature_infra_{tag}_{nanos}"));
    dir
}

// ── Store: key-value semantics ──────────────────────────────────────────────

#[tokio::test]
async fn in_memory_store_round_trips_namespaced_values() {
    let store = InMemoryStore::new();
    assert_eq!(store.get("cache", "missing").await.unwrap(), None);

    store.put("cache", "k1", json!({"n": 1})).await.unwrap();
    store.put("cache", "k2", json!({"n": 2})).await.unwrap();
    // A different namespace is an independent bucket.
    store.put("events", "k1", json!("hi")).await.unwrap();

    assert_eq!(
        store.get("cache", "k1").await.unwrap(),
        Some(json!({"n": 1}))
    );

    let mut keys = store.list("cache").await.unwrap();
    keys.sort();
    assert_eq!(keys, vec!["k1".to_string(), "k2".to_string()]);
    assert_eq!(store.list("events").await.unwrap(), vec!["k1".to_string()]);

    store.delete("cache", "k1").await.unwrap();
    assert_eq!(store.get("cache", "k1").await.unwrap(), None);
    // Deleting a missing key is a no-op, not an error.
    assert!(store.delete("cache", "gone").await.is_ok());
}

#[tokio::test]
async fn file_store_persists_and_survives_reopen() {
    let dir = temp_dir("filestore");
    {
        let store = FileStore::new(&dir);
        store
            .put("threads", "t1", json!({"msg": "hi"}))
            .await
            .unwrap();
    }
    // A fresh handle over the same root sees the durable value.
    let reopened = FileStore::new(&dir);
    assert_eq!(
        reopened.get("threads", "t1").await.unwrap(),
        Some(json!({"msg": "hi"}))
    );
    assert_eq!(
        reopened.list("threads").await.unwrap(),
        vec!["t1".to_string()]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn file_store_rejects_path_traversal_names() {
    let dir = temp_dir("traversal");
    let store = FileStore::new(&dir);
    // `..` and slashes must be rejected to keep writes inside the root.
    assert!(store.get("..", "k").await.is_err());
    assert!(store.put("ns", "../escape", json!(1)).await.is_err());
    assert!(store.get("ns", "a/b").await.is_err());
    // Empty names are rejected too.
    assert!(store.put("", "k", json!(1)).await.is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

// ── AppendStore: journal semantics ──────────────────────────────────────────

#[tokio::test]
async fn append_store_assigns_dense_offsets_and_reads_from_offset() {
    let store = InMemoryAppendStore::new();
    assert_eq!(store.len("log").await.unwrap(), 0);

    assert_eq!(store.append("log", json!("a")).await.unwrap(), 0);
    assert_eq!(store.append("log", json!("b")).await.unwrap(), 1);
    assert_eq!(store.append("log", json!("c")).await.unwrap(), 2);
    assert_eq!(store.len("log").await.unwrap(), 3);

    // Reading from 0 replays the whole stream, paired with offsets.
    let all = store.read_from("log", 0).await.unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0], (0, json!("a")));
    assert_eq!(all[2], (2, json!("c")));

    // Reading from a mid-stream offset returns the tail.
    let tail = store.read_from("log", 2).await.unwrap();
    assert_eq!(tail, vec![(2, json!("c"))]);
    // Reading at/after len is empty; an unknown stream is empty, not an error.
    assert!(store.read_from("log", 3).await.unwrap().is_empty());
    assert!(store.read_from("unknown", 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn append_store_retention_evicts_oldest_but_keeps_offsets_monotonic() {
    let store = InMemoryAppendStore::new().with_max_entries_per_stream(2);
    for c in ["a", "b", "c", "d"] {
        store.append("log", json!(c)).await.unwrap();
    }
    // Logical length keeps advancing even though only 2 entries are retained.
    assert_eq!(store.len("log").await.unwrap(), 4);

    // Only the newest two remain, at their original monotonic offsets.
    let retained = store.read_from("log", 0).await.unwrap();
    assert_eq!(retained, vec![(2, json!("c")), (3, json!("d"))]);
}

#[tokio::test]
async fn jsonl_append_store_is_durable_across_handles() {
    let dir = temp_dir("jsonl");
    let store = JsonlAppendStore::new(&dir);
    assert_eq!(store.append("run1", json!({"i": 0})).await.unwrap(), 0);
    assert_eq!(store.append("run1", json!({"i": 1})).await.unwrap(), 1);

    // A fresh handle learns the length from disk and reads entries back.
    let reopened = JsonlAppendStore::new(&dir);
    assert_eq!(reopened.len("run1").await.unwrap(), 2);
    let entries = reopened.read_from("run1", 1).await.unwrap();
    assert_eq!(entries, vec![(1, json!({"i": 1}))]);
    // The next append continues the offset sequence.
    assert_eq!(reopened.append("run1", json!({"i": 2})).await.unwrap(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── StoreRegistry ───────────────────────────────────────────────────────────

#[tokio::test]
async fn store_registry_resolves_named_and_default_stores() {
    let mut registry = StoreRegistry::new();
    assert!(registry.get("events").is_none());

    let events: std::sync::Arc<dyn Store> = std::sync::Arc::new(InMemoryStore::new());
    registry.register("events", events);
    assert!(registry.get("events").is_some());

    // The built-in default store is always available.
    let default = registry.default_store();
    default.put("ns", "k", json!(true)).await.unwrap();
    assert_eq!(default.get("ns", "k").await.unwrap(), Some(json!(true)));
}
