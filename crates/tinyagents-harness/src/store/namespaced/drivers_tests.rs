use super::super::FlatNamespacedStore;
use super::*;
use crate::store::Store as FlatStore;
use crate::store::conformance::run_namespaced_store_conformance;
use serde_json::json;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn docs(storage: &dyn StorageBackend, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn ns(segments: &[&str]) -> Namespace {
    Namespace::new(segments.iter().copied()).expect("valid namespace")
}

#[tokio::test]
async fn memory_backend_passes_the_namespaced_conformance_suite() {
    let storage = MemoryBackend::new();
    run_namespaced_store_conformance(&DriverNamespacedStore::new(docs(&storage, "local"))).await;
}

#[tokio::test]
async fn testkit_backend_passes_the_namespaced_conformance_suite() {
    let storage = MemoryBackend::new();
    run_namespaced_store_conformance(&DriverNamespacedStore::new(docs(&storage, "local"))).await;
}

#[tokio::test]
async fn items_keep_timestamps_when_the_driver_is_reopened() {
    let storage = MemoryBackend::new();
    let created;
    {
        let store = DriverNamespacedStore::new(docs(&storage, "local"));
        store
            .put(&ns(&["users", "alice"]), "k", json!({"v": 1}))
            .await
            .unwrap();
        created = store
            .get(&ns(&["users", "alice"]), "k")
            .await
            .unwrap()
            .unwrap()
            .created_at_ms;
    }
    let store = DriverNamespacedStore::new(docs(&storage, "local"));
    let item = store
        .get(&ns(&["users", "alice"]), "k")
        .await
        .unwrap()
        .expect("persisted");
    assert_eq!(item.value, json!({"v": 1}));
    assert_eq!(item.created_at_ms, created);
    // Overwriting keeps the original creation time and moves the update time.
    store
        .put(&ns(&["users", "alice"]), "k", json!({"v": 2}))
        .await
        .unwrap();
    let again = store
        .get(&ns(&["users", "alice"]), "k")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(again.created_at_ms, created);
    assert!(again.updated_at_ms >= item.updated_at_ms);
}

/// An expired item is invisible on read before any sweep, and a sweep reclaims it.
#[tokio::test]
async fn expired_items_are_invisible_and_reclaimed_by_a_sweep() {
    let store = DriverNamespacedStore::new(docs(&MemoryBackend::new(), "local"));
    let n = ns(&["cache"]);
    store
        .put_with_ttl(&n, "gone", json!(1), Some(f64::MIN_POSITIVE))
        .await
        .unwrap();
    store
        .put_with_ttl(&n, "keep", json!(2), None)
        .await
        .unwrap();
    store
        .put_with_ttl(&n, "also-gone", json!(3), Some(f64::MIN_POSITIVE))
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    assert!(store.get(&n, "gone").await.unwrap().is_none());
    let found = store
        .search(SearchQuery {
            namespace_prefix: vec!["cache".into()],
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(found.len(), 1, "search skips expired items");
    // A backend with native TTL has already reclaimed the other expired item
    // (it reads as absent), so a sweep finds at most one; a second finds none.
    assert!(store.sweep_expired().await.unwrap() <= 1);
    assert_eq!(store.sweep_expired().await.unwrap(), 0);
    assert!(store.get(&n, "also-gone").await.unwrap().is_none());
    assert!(store.get(&n, "keep").await.unwrap().is_some());
}

#[tokio::test]
async fn default_ttl_applies_and_a_sliding_window_extends_on_read() {
    let sliding =
        DriverNamespacedStore::new(docs(&MemoryBackend::new(), "local")).with_ttl(TtlConfig {
            default_ttl_minutes: Some(60.0),
            refresh_on_read: true,
        });
    assert_eq!(sliding.ttl_config().default_ttl_minutes, Some(60.0));
    let n = ns(&["cache"]);
    sliding.put(&n, "k", json!(1)).await.unwrap();
    let first = sliding.get(&n, "k").await.unwrap().unwrap();
    assert!(first.expires_at_ms.is_some(), "the store default applies");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let second = sliding.get(&n, "k").await.unwrap().unwrap();
    assert!(
        second.expires_at_ms > first.expires_at_ms,
        "a read slides the window forward"
    );
    // A per-read override beats the policy.
    let results = sliding
        .batch(&[StoreOp::Get {
            namespace: n.clone(),
            key: "k".into(),
            refresh_ttl: Some(false),
        }])
        .await
        .unwrap();
    let held = results[0].clone().into_item().unwrap().unwrap();
    assert_eq!(held.expires_at_ms, second.expires_at_ms);
}

#[tokio::test]
async fn batch_results_align_positionally_with_the_request() {
    let store = DriverNamespacedStore::new(docs(&MemoryBackend::new(), "local"));
    let n = ns(&["b"]);
    let results = store
        .batch(&[
            StoreOp::Put {
                namespace: n.clone(),
                key: "k".into(),
                value: Some(json!("v")),
                ttl_minutes: None,
            },
            StoreOp::Get {
                namespace: n.clone(),
                key: "k".into(),
                refresh_ttl: None,
            },
            StoreOp::Search(SearchQuery {
                namespace_prefix: vec!["b".into()],
                ..SearchQuery::default()
            }),
            StoreOp::ListNamespaces(ListNamespacesQuery::default()),
            StoreOp::Put {
                namespace: n.clone(),
                key: "k".into(),
                value: None,
                ttl_minutes: None,
            },
            StoreOp::Get {
                namespace: n,
                key: "k".into(),
                refresh_ttl: None,
            },
        ])
        .await
        .unwrap();
    assert_eq!(results.len(), 6);
    assert!(matches!(results[0], StoreResult::Ack));
    assert!(matches!(results[1], StoreResult::Item(Some(_))));
    assert!(matches!(&results[2], StoreResult::Items(v) if v.len() == 1));
    assert!(matches!(&results[3], StoreResult::Namespaces(v) if v.len() == 1));
    assert!(matches!(results[4], StoreResult::Ack));
    assert!(matches!(results[5], StoreResult::Item(None)));
}

#[tokio::test]
async fn invalid_namespaces_are_rejected_before_any_write() {
    let store = DriverNamespacedStore::new(docs(&MemoryBackend::new(), "local"));
    let bad = Namespace(vec!["langgraph".into()]);
    assert!(store.put(&bad, "k", json!(1)).await.is_err());
    assert!(store.get(&Namespace(vec![]), "k").await.is_err());
}

#[tokio::test]
async fn scopes_and_owners_keep_items_apart() {
    let storage = MemoryBackend::new();
    let alice = DriverNamespacedStore::new(docs(&storage, "alice"));
    let bob = DriverNamespacedStore::new(docs(&storage, "bob"));
    let other = DriverNamespacedStore::with_owner(docs(&storage, "alice"), "other");
    let n = ns(&["users", "x"]);
    alice.put(&n, "k", json!(1)).await.unwrap();
    assert!(bob.get(&n, "k").await.unwrap().is_none());
    assert!(other.get(&n, "k").await.unwrap().is_none());
    assert!(
        bob.list_namespaces(ListNamespacesQuery::default())
            .await
            .unwrap()
            .is_empty()
    );
    // Segments and keys that would collide under naive joining stay distinct.
    alice
        .put(&ns(&["a", "b"]), "c", json!("one"))
        .await
        .unwrap();
    alice.put(&ns(&["a"]), "b/c", json!("two")).await.unwrap();
    assert_eq!(
        alice
            .get(&ns(&["a", "b"]), "c")
            .await
            .unwrap()
            .unwrap()
            .value,
        json!("one")
    );
    assert_eq!(
        alice.get(&ns(&["a"]), "b/c").await.unwrap().unwrap().value,
        json!("two")
    );
}

#[tokio::test]
async fn the_flat_store_surface_works_over_the_driver() {
    let store = FlatNamespacedStore::new(DriverNamespacedStore::new(docs(
        &MemoryBackend::new(),
        "local",
    )));
    store.put("events", "e1", json!({"a": 1})).await.unwrap();
    assert_eq!(
        FlatStore::get(&store, "events", "e1").await.unwrap(),
        Some(json!({"a": 1}))
    );
    assert_eq!(store.list("events").await.unwrap(), vec!["e1".to_string()]);
    FlatStore::delete(&store, "events", "e1").await.unwrap();
    assert!(store.list("events").await.unwrap().is_empty());
}

/// The v2 shape: an item is a `tsd.kv` document of owner `harness` whose
/// namespace is the segments joined with `.` under a reserved prefix, the
/// creation and expiry times live in the record metadata, not the body.
#[tokio::test]
async fn items_are_tsd_kv_documents_with_times_in_metadata() {
    let storage = MemoryBackend::new();
    let documents = docs(&storage, "local");
    let store = DriverNamespacedStore::new(Arc::clone(&documents));
    store
        .put_with_ttl(
            &ns(&["users", "alice"]),
            "likes",
            json!({"food": "tea"}),
            Some(60.0),
        )
        .await
        .unwrap();

    let row = documents
        .get("tsd.kv", "harness/namespaced:users.alice/likes")
        .await
        .unwrap()
        .expect("the item sits under Key::of([owner, ns, key])");
    assert_eq!(
        row.doc,
        json!({
            "owner": "harness",
            "ns": "namespaced:users.alice",
            "key": "likes",
            "value": {"food": "tea"},
        })
    );
    assert!(row.meta.created_ms > 0);
    let expires = row.meta.expires_ms.expect("the ttl is the record expiry");
    assert!(expires > row.meta.created_ms);

    let item = store
        .get(&ns(&["users", "alice"]), "likes")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(item.created_at_ms as i64, row.meta.created_ms);
    assert_eq!(item.expires_at_ms, Some(expires as u64));
}

/// A subtree search narrows by root without surfacing a sibling root whose
/// name merely sorts nearby, nor the flat store's entries of the same owner.
#[tokio::test]
async fn a_root_search_ignores_lookalike_roots_and_flat_entries() {
    let storage = MemoryBackend::new();
    let documents = docs(&storage, "local");
    let store = DriverNamespacedStore::new(Arc::clone(&documents));
    store.put(&ns(&["a", "x"]), "k", json!(1)).await.unwrap();
    store.put(&ns(&["a-b"]), "k", json!(2)).await.unwrap();
    store.put(&ns(&["ab"]), "k", json!(3)).await.unwrap();
    crate::store::DriverStore::new(Arc::clone(&documents))
        .put("a", "k", json!(4))
        .await
        .unwrap();
    let found = store
        .search(SearchQuery {
            namespace_prefix: vec!["a".into()],
            ..SearchQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].value, json!(1));
}

/// Expiry is the driver's: an expired item reads as absent, and a sweep never fails.
#[tokio::test]
async fn expired_items_read_as_absent_and_a_sweep_is_harmless() {
    let store = DriverNamespacedStore::new(docs(&MemoryBackend::new(), "local"));
    let n = ns(&["cache"]);
    store
        .put_with_ttl(&n, "gone", json!(1), Some(f64::MIN_POSITIVE))
        .await
        .unwrap();
    store
        .put_with_ttl(&n, "keep", json!(2), None)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    assert!(store.get(&n, "gone").await.unwrap().is_none());
    store.sweep_expired().await.unwrap();
    assert!(store.get(&n, "keep").await.unwrap().is_some());
}
