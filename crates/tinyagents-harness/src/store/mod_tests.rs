//! Tests for the harness store backends.
//!
//! Cover [`super::InMemoryStore`] put/get/delete/list round-tripping and
//! namespace isolation, [`super::StoreRegistry`] default and named-store
//! lookup/replacement, and [`super::InMemoryAppendStore`] offset monotonicity,
//! `read_from` tailing, per-stream isolation and retention eviction. The
//! driver-backed stores are covered in `drivers_tests.rs`.

use std::sync::Arc;

use super::*;
use serde_json::json;
use tinystoragedrivers_testkit::MemoryBackend;

#[tokio::test]
async fn in_memory_store_put_get_delete() {
    let store = InMemoryStore::new();

    // Key absent before first write.
    assert_eq!(store.get("ns", "k").await.unwrap(), None);

    // Write and read back.
    store.put("ns", "k", json!({"x": 1})).await.unwrap();
    assert_eq!(store.get("ns", "k").await.unwrap(), Some(json!({"x": 1})));

    // Delete removes the key.
    store.delete("ns", "k").await.unwrap();
    assert_eq!(store.get("ns", "k").await.unwrap(), None);
}

#[tokio::test]
async fn in_memory_store_list() {
    let store = InMemoryStore::new();
    store.put("ns", "a", json!(1)).await.unwrap();
    store.put("ns", "b", json!(2)).await.unwrap();
    let mut keys = store.list("ns").await.unwrap();
    keys.sort();
    assert_eq!(keys, vec!["a", "b"]);
}

#[tokio::test]
async fn store_registry_default_store_is_usable() {
    let reg = StoreRegistry::new();
    let store = reg.default_store();
    store.put("ns", "key", json!("value")).await.unwrap();
    assert_eq!(store.get("ns", "key").await.unwrap(), Some(json!("value")));
}

#[tokio::test]
async fn store_registry_named_store() {
    let mut reg = StoreRegistry::new();
    reg.register("cache", Arc::new(InMemoryStore::new()));
    let cache = reg.get("cache").expect("cache store registered");
    cache.put("ns", "k", json!(42)).await.unwrap();
    assert_eq!(cache.get("ns", "k").await.unwrap(), Some(json!(42)));
    assert!(reg.get("missing").is_none());
}

#[tokio::test]
async fn in_memory_delete_missing_is_noop() {
    let store = InMemoryStore::new();
    // Deleting a key (or whole namespace) that was never written is a no-op.
    store.delete("ns", "never").await.unwrap();
    // list on an unwritten namespace returns empty.
    assert!(store.list("empty-ns").await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_namespaces_are_isolated() {
    let store = InMemoryStore::new();
    store.put("a", "k", json!(1)).await.unwrap();
    store.put("b", "k", json!(2)).await.unwrap();
    assert_eq!(store.get("a", "k").await.unwrap(), Some(json!(1)));
    assert_eq!(store.get("b", "k").await.unwrap(), Some(json!(2)));
    assert_eq!(store.list("a").await.unwrap(), vec!["k"]);
}

#[tokio::test]
async fn store_registry_default_store_is_stable() {
    let reg = StoreRegistry::new();
    // Two handles to the default store share the same backing data (Arc clone).
    reg.default_store().put("ns", "k", json!(1)).await.unwrap();
    assert_eq!(
        reg.default_store().get("ns", "k").await.unwrap(),
        Some(json!(1))
    );
}

#[tokio::test]
async fn store_registry_register_replaces() {
    let mut reg = StoreRegistry::new();
    reg.register("s", Arc::new(InMemoryStore::new()));
    reg.get("s")
        .unwrap()
        .put("ns", "k", json!(1))
        .await
        .unwrap();

    // Re-registering under the same name replaces the previous store.
    reg.register("s", Arc::new(InMemoryStore::new()));
    assert_eq!(reg.get("s").unwrap().get("ns", "k").await.unwrap(), None);
}

// ── InMemoryAppendStore ────────────────────────────────────────────────────────

#[tokio::test]
async fn in_memory_append_returns_increasing_offsets() {
    let store = InMemoryAppendStore::new();
    assert_eq!(store.len("evts").await.unwrap(), 0);

    assert_eq!(store.append("evts", json!({"n": 0})).await.unwrap(), 0);
    assert_eq!(store.append("evts", json!({"n": 1})).await.unwrap(), 1);
    assert_eq!(store.append("evts", json!({"n": 2})).await.unwrap(), 2);

    // len equals the offset the next append will receive.
    assert_eq!(store.len("evts").await.unwrap(), 3);
}

#[tokio::test]
async fn in_memory_read_from_returns_tail() {
    let store = InMemoryAppendStore::new();
    for n in 0..5 {
        store.append("s", json!(n)).await.unwrap();
    }

    // Whole stream from offset 0.
    let all = store.read_from("s", 0).await.unwrap();
    assert_eq!(all.len(), 5);
    assert_eq!(all[0], (0, json!(0)));
    assert_eq!(all[4], (4, json!(4)));

    // Tail from a mid offset.
    let tail = store.read_from("s", 3).await.unwrap();
    assert_eq!(tail, vec![(3, json!(3)), (4, json!(4))]);

    // Reading from len (or beyond) yields nothing.
    assert!(store.read_from("s", 5).await.unwrap().is_empty());
    assert!(store.read_from("s", 99).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_append_streams_are_isolated() {
    let store = InMemoryAppendStore::new();
    store.append("a", json!("a0")).await.unwrap();
    store.append("b", json!("b0")).await.unwrap();
    store.append("a", json!("a1")).await.unwrap();

    assert_eq!(store.len("a").await.unwrap(), 2);
    assert_eq!(store.len("b").await.unwrap(), 1);
    // An unwritten stream reads back empty without error.
    assert!(store.read_from("missing", 0).await.unwrap().is_empty());
    assert_eq!(store.len("missing").await.unwrap(), 0);
}

#[tokio::test]
async fn in_memory_append_bounded_evicts_oldest_and_keeps_offsets_monotonic() {
    let store = InMemoryAppendStore::new().with_max_entries_per_stream(3);
    for i in 0..5u64 {
        // Offsets keep advancing past the cap.
        assert_eq!(store.append("s", json!(i)).await.unwrap(), i);
    }

    // Logical length is unaffected by eviction: the next append gets offset 5.
    assert_eq!(store.len("s").await.unwrap(), 5);

    // Only the newest 3 entries remain, at their original offsets.
    let all = store.read_from("s", 0).await.unwrap();
    assert_eq!(
        all,
        vec![(2, json!(2)), (3, json!(3)), (4, json!(4))],
        "evicted offsets are no longer readable"
    );

    // Reading from a retained offset returns exactly the tail.
    assert_eq!(
        store.read_from("s", 3).await.unwrap(),
        vec![(3, json!(3)), (4, json!(4))]
    );
    // Reading past the end is empty, not an error.
    assert!(store.read_from("s", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_append_bound_applies_per_stream() {
    let store = InMemoryAppendStore::new().with_max_entries_per_stream(2);
    for i in 0..3u64 {
        store.append("a", json!(i)).await.unwrap();
    }
    store.append("b", json!("b0")).await.unwrap();

    // Stream `a` was trimmed; stream `b` is untouched.
    assert_eq!(store.read_from("a", 0).await.unwrap().len(), 2);
    assert_eq!(
        store.read_from("b", 0).await.unwrap(),
        vec![(0, json!("b0"))]
    );
}

#[tokio::test]
async fn in_memory_append_unbounded_by_default() {
    let store = InMemoryAppendStore::new();
    for i in 0..100u64 {
        store.append("s", json!(i)).await.unwrap();
    }
    assert_eq!(store.read_from("s", 0).await.unwrap().len(), 100);
    assert_eq!(store.len("s").await.unwrap(), 100);
}

#[tokio::test]
async fn read_window_returns_a_bounded_slice_for_both_backends() {
    use tinyagents_storage_kit::{Scope, StorageBackend};
    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    let driver = DriverAppendStore::new(Arc::clone(scoped.streams()));
    let memory = InMemoryAppendStore::new();
    for n in 0..5 {
        driver.append("evts", json!({ "n": n })).await.unwrap();
        memory.append("evts", json!({ "n": n })).await.unwrap();
    }
    let expected = vec![(1, json!({"n": 1})), (2, json!({"n": 2}))];
    assert_eq!(driver.read_window("evts", 1, 2).await.unwrap(), expected);
    assert_eq!(memory.read_window("evts", 1, 2).await.unwrap(), expected);
    assert!(driver.read_window("evts", 1, 0).await.unwrap().is_empty());
    assert_eq!(driver.read_window("evts", 4, 10).await.unwrap().len(), 1);
    assert!(
        driver
            .read_window("missing", 0, 10)
            .await
            .unwrap()
            .is_empty()
    );
}
