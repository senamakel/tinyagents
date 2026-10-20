use super::*;
use crate::store::conformance::run_store_conformance;
use serde_json::json;
use tinyagents_storage_kit::{Kv, Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn scoped(storage: &MemoryBackend, scope: &str) -> tinyagents_storage_kit::ScopedStorage {
    storage.for_scope(&Scope::new(scope).unwrap()).unwrap()
}

#[tokio::test]
async fn driver_store_passes_the_store_conformance_suite() {
    let storage = MemoryBackend::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    run_store_conformance(&store).await;
}

#[tokio::test]
async fn namespaces_and_keys_may_hold_any_characters() {
    let storage = MemoryBackend::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    store.put("a/b", "c", json!(1)).await.unwrap();
    store.put("a", "b/c", json!(2)).await.unwrap();
    store.put("odd ns:é", "k y", json!(3)).await.unwrap();
    assert_eq!(store.get("a/b", "c").await.unwrap(), Some(json!(1)));
    assert_eq!(store.get("a", "b/c").await.unwrap(), Some(json!(2)));
    assert_eq!(store.get("odd ns:é", "k y").await.unwrap(), Some(json!(3)));
    assert_eq!(store.list("a").await.unwrap(), vec!["b/c".to_owned()]);
}

#[tokio::test]
async fn stores_in_different_scopes_and_owners_are_isolated() {
    let storage = MemoryBackend::new();
    let alice = DriverStore::new(Arc::clone(scoped(&storage, "alice").documents()));
    let bob = DriverStore::new(Arc::clone(scoped(&storage, "bob").documents()));
    let other = DriverStore::with_owner(Arc::clone(scoped(&storage, "alice").documents()), "other");
    alice.put("ns", "k", json!("a")).await.unwrap();
    assert_eq!(bob.get("ns", "k").await.unwrap(), None);
    assert_eq!(other.get("ns", "k").await.unwrap(), None);
    assert!(bob.list("ns").await.unwrap().is_empty());
}

#[tokio::test]
async fn a_list_follows_every_page() {
    let storage = MemoryBackend::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    for i in 0..1_005 {
        store
            .put("big", &format!("k{i:04}"), json!(i))
            .await
            .unwrap();
    }
    assert_eq!(store.list("big").await.unwrap().len(), 1_005);
}

#[tokio::test]
async fn an_overlong_key_is_stored_under_a_hashed_id_and_still_listed() {
    let storage = MemoryBackend::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    let long = "k".repeat(600);
    store.put("ns", &long, json!(1)).await.unwrap();
    assert_eq!(store.get("ns", &long).await.unwrap(), Some(json!(1)));
    assert_eq!(store.list("ns").await.unwrap(), vec![long]);
}

#[tokio::test]
async fn an_empty_namespace_is_a_validation_error() {
    let storage = MemoryBackend::new();
    let store = DriverStore::new(Arc::clone(scoped(&storage, "local").documents()));
    let error = store.put("", "k", json!(1)).await.unwrap_err();
    assert!(matches!(error, TinyAgentsError::Validation(_)), "{error:?}");
}

/// The v2 shape: entries are `tsd.kv` documents of the store's owner, with the
/// value unwrapped into the primitive's own `value` field.
#[tokio::test]
async fn entries_are_tsd_kv_documents_of_the_owner() {
    let storage = MemoryBackend::new();
    let docs = Arc::clone(scoped(&storage, "local").documents());
    let store = DriverStore::new(Arc::clone(&docs));
    store.put("ns", "k", json!({"v": 1})).await.unwrap();

    let kv = Kv::ns(&docs, "harness", "ns").unwrap();
    assert_eq!(kv.get::<Value>("k").await.unwrap(), Some(json!({"v": 1})));
    let raw = docs
        .get(tinyagents_storage_kit::primitives::KV, "harness/ns/k")
        .await
        .unwrap()
        .expect("the entry sits under Key::of([owner, ns, key])");
    assert_eq!(
        raw.doc,
        json!({"owner": "harness", "ns": "ns", "key": "k", "value": {"v": 1}})
    );
    assert!(
        docs.collections("harness_store").await.unwrap().is_empty(),
        "the v1 collection is not created"
    );
}

#[tokio::test]
async fn a_store_reads_only_its_own_owner() {
    let storage = MemoryBackend::new();
    let docs = Arc::clone(scoped(&storage, "local").documents());
    let harness = DriverStore::new(Arc::clone(&docs));
    let session = DriverStore::with_owner(Arc::clone(&docs), "session");
    harness.put("ns", "k", json!("h")).await.unwrap();
    session.put("ns", "k", json!("s")).await.unwrap();
    assert_eq!(harness.get("ns", "k").await.unwrap(), Some(json!("h")));
    assert_eq!(session.get("ns", "k").await.unwrap(), Some(json!("s")));
    session.delete("ns", "k").await.unwrap();
    assert_eq!(harness.get("ns", "k").await.unwrap(), Some(json!("h")));
}

#[test]
fn backend_failures_map_to_storage_errors() {
    let error = map_error(StorageError::unavailable("busy"));
    assert!(matches!(error, TinyAgentsError::Storage(ref m) if m.contains("busy")));
}

#[tokio::test]
async fn driver_append_store_keeps_dense_offsets() {
    let storage = MemoryBackend::new();
    let journal = DriverAppendStore::new(Arc::clone(scoped(&storage, "local").streams()));
    assert_eq!(journal.len("run-1").await.unwrap(), 0);
    assert!(journal.read_from("run-1", 0).await.unwrap().is_empty());
    for i in 0..3 {
        assert_eq!(journal.append("run-1", json!({ "i": i })).await.unwrap(), i);
    }
    assert_eq!(journal.len("run-1").await.unwrap(), 3);
    assert_eq!(
        journal.read_from("run-1", 1).await.unwrap(),
        vec![(1, json!({ "i": 1 })), (2, json!({ "i": 2 }))]
    );
    assert_eq!(
        journal.read_window("run-1", 0, 2).await.unwrap(),
        vec![(0, json!({ "i": 0 })), (1, json!({ "i": 1 }))]
    );
    assert!(journal.read_from("run-1", 9).await.unwrap().is_empty());
}

#[tokio::test]
async fn read_from_follows_long_streams_and_prefixes_isolate() {
    let storage = MemoryBackend::new();
    let streams = Arc::clone(scoped(&storage, "local").streams());
    let journal = DriverAppendStore::with_prefix(Arc::clone(&streams), "journal/");
    let other = DriverAppendStore::with_prefix(Arc::clone(&streams), "other/");
    for i in 0..(READ_PAGE as u64 + 5) {
        journal.append("s", json!(i)).await.unwrap();
    }
    let all = journal.read_from("s", 0).await.unwrap();
    assert_eq!(all.len(), READ_PAGE + 5);
    assert_eq!(all.last().unwrap().0, READ_PAGE as u64 + 4);
    assert_eq!(other.len("s").await.unwrap(), 0);
    let error = journal.append("", json!(1)).await;
    assert!(error.is_ok(), "a prefix makes the empty stream name valid");
}

#[tokio::test]
async fn streams_are_named_by_prefix_and_stream_without_a_length_prefix() {
    let storage = MemoryBackend::new();
    let scoped = scoped(&storage, "local");
    let journal = DriverAppendStore::with_prefix(Arc::clone(scoped.streams()), "harness.journal/");
    journal.append("run.1", json!(1)).await.unwrap();
    let bare = DriverAppendStore::new(Arc::clone(scoped.streams()));
    bare.append("plain", json!(2)).await.unwrap();
    assert_eq!(
        scoped.streams().streams("").await.unwrap(),
        vec!["harness.journal/run.1".to_owned(), "plain".to_owned()]
    );
}

#[tokio::test]
async fn sealing_closes_a_stream_without_changing_what_a_reader_sees() {
    let storage = MemoryBackend::new();
    let scoped = scoped(&storage, "local");
    let journal = DriverAppendStore::with_prefix(Arc::clone(scoped.streams()), "harness.journal/");
    for i in 0..(READ_PAGE as u64 + 3) {
        journal.append("run.1", json!({ "i": i })).await.unwrap();
    }
    let before_all = journal.read_from("run.1", 0).await.unwrap();
    let before_window = journal.read_window("run.1", 5, 7).await.unwrap();
    let before_len = journal.len("run.1").await.unwrap();

    journal.seal("run.1").await.unwrap();
    journal.seal("run.1").await.unwrap();

    assert_eq!(journal.read_from("run.1", 0).await.unwrap(), before_all);
    assert_eq!(
        journal.read_window("run.1", 5, 7).await.unwrap(),
        before_window
    );
    assert_eq!(journal.len("run.1").await.unwrap(), before_len);
    let info = scoped
        .streams()
        .info("harness.journal/run.1")
        .await
        .unwrap()
        .unwrap();
    assert!(info.sealed_ms.is_some(), "the driver stream is sealed");
    assert!(
        journal.append("run.1", json!("late")).await.is_err(),
        "a sealed stream refuses appends"
    );
}

#[tokio::test]
async fn sealing_a_stream_nothing_was_written_to_is_a_no_op() {
    let storage = MemoryBackend::new();
    let journal = DriverAppendStore::new(Arc::clone(scoped(&storage, "local").streams()));
    journal.seal("never-written").await.unwrap();
    assert_eq!(journal.len("never-written").await.unwrap(), 0);
}
