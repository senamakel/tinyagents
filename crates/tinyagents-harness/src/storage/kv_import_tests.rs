use super::*;
use serde_json::json;
use tinyagents_storage_kit::{Kv, Migrations, Precondition, Scope, ScopedStorage, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn scoped() -> ScopedStorage {
    MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap()
}

async fn seed(storage: &ScopedStorage, collection: &str, id: &str, body: serde_json::Value) {
    storage
        .documents()
        .put(collection, id, body, Precondition::None)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_flat_v1_store_moves_into_tsd_kv_with_the_value_unwrapped() {
    let storage = scoped();
    // The v1 id was `<len>:<ns>/<key>`; the body is what matters.
    seed(
        &storage,
        "harness_store",
        "2:ns/k1",
        json!({"ns": "ns", "key": "k1", "value": {"v": 1}}),
    )
    .await;
    seed(
        &storage,
        "harness_store",
        "2:ns/k2",
        json!({"ns": "ns", "key": "k2", "value": "two"}),
    )
    .await;

    let moved = import_legacy_kv(&storage, "harness", "harness_store", LegacyKv::Flat)
        .await
        .unwrap();
    assert_eq!(moved, 2);

    let kv = Kv::ns(storage.documents(), "harness", "ns").unwrap();
    assert_eq!(
        kv.get::<serde_json::Value>("k1").await.unwrap(),
        Some(json!({"v": 1}))
    );
    assert_eq!(
        kv.get::<serde_json::Value>("k2").await.unwrap(),
        Some(json!("two"))
    );
    let left = storage
        .documents()
        .count("harness_store", &tinyagents_storage_kit::Filter::default())
        .await
        .unwrap();
    assert_eq!(left, 0, "the v1 rows are removed once copied");
}

#[tokio::test]
async fn the_import_runs_once_and_never_overwrites_newer_v2_values() {
    let storage = scoped();
    seed(
        &storage,
        "session_kv",
        "3:ctl/a",
        json!({"ns": "ctl", "key": "a", "value": "old"}),
    )
    .await;
    // A v2 write that landed before the import finished must survive it.
    let kv = Kv::ns(storage.documents(), "session", "ctl").unwrap();
    kv.put("a", &"new").await.unwrap();

    let moved = import_legacy_kv(&storage, "session", "session_kv", LegacyKv::Flat)
        .await
        .unwrap();
    assert_eq!(moved, 0, "an existing v2 entry is kept, not counted");
    assert_eq!(kv.get::<String>("a").await.unwrap().as_deref(), Some("new"));
    assert!(
        Migrations::new(&storage)
            .done("session", "kv_import_session_kv")
            .await
            .unwrap()
    );

    // Done is recorded: new legacy rows are not picked up by a second call.
    seed(
        &storage,
        "session_kv",
        "3:ctl/b",
        json!({"ns": "ctl", "key": "b", "value": "late"}),
    )
    .await;
    assert_eq!(
        import_legacy_kv(&storage, "session", "session_kv", LegacyKv::Flat)
            .await
            .unwrap(),
        0
    );
    assert_eq!(kv.get::<String>("b").await.unwrap(), None);
}

#[tokio::test]
async fn namespaced_items_keep_their_segments_creation_time_and_expiry() {
    let storage = scoped();
    let far = 4_102_444_800_000_i64;
    seed(
        &storage,
        "harness_namespaced_store",
        "deadbeef",
        json!({
            "root": "memories", "ns": ["memories", "alice"], "key": "likes",
            "value": {"food": "tea"}, "createdAtMs": 1_700_000_000_000_u64,
            "updatedAtMs": 1_700_000_100_000_u64, "expiresAtMs": far,
        }),
    )
    .await;
    // Already expired: not worth carrying over.
    seed(
        &storage,
        "harness_namespaced_store",
        "gone",
        json!({
            "root": "memories", "ns": ["memories", "bob"], "key": "x",
            "value": 1, "createdAtMs": 1, "updatedAtMs": 1, "expiresAtMs": 1_000,
        }),
    )
    .await;

    let moved = import_legacy_kv(
        &storage,
        "harness",
        "harness_namespaced_store",
        LegacyKv::Namespaced,
    )
    .await
    .unwrap();
    assert_eq!(moved, 1);

    let row = storage
        .documents()
        .get("tsd.kv", "harness/namespaced:memories.alice/likes")
        .await
        .unwrap()
        .expect("namespaced entries sit under the reserved namespaced: ns");
    assert_eq!(
        row.doc,
        json!({
            "owner": "harness", "ns": "namespaced:memories.alice",
            "key": "likes", "value": {"food": "tea"},
        })
    );
    assert_eq!(row.meta.created_ms, 1_700_000_000_000);
    assert_eq!(row.meta.expires_ms, Some(far));
}

#[tokio::test]
async fn response_cache_entries_move_under_the_cache_namespace() {
    let storage = scoped();
    seed(
        &storage,
        "harness_response_cache",
        "abc",
        json!({"ns": "default", "key": "req-1", "value": {"text": "hi"}, "expiresAtMs": null}),
    )
    .await;
    let moved = import_legacy_kv(
        &storage,
        "harness",
        "harness_response_cache",
        LegacyKv::ResponseCache,
    )
    .await
    .unwrap();
    assert_eq!(moved, 1);
    let kv = Kv::ns(storage.documents(), "harness", "cache:default").unwrap();
    assert_eq!(
        kv.get::<serde_json::Value>("req-1").await.unwrap(),
        Some(json!({"text": "hi"}))
    );
}

#[tokio::test]
async fn a_missing_legacy_collection_is_a_completed_import() {
    let storage = scoped();
    let moved = import_legacy_kv(&storage, "harness", "harness_store", LegacyKv::Flat)
        .await
        .unwrap();
    assert_eq!(moved, 0);
    assert!(
        Migrations::new(&storage)
            .done("harness", "kv_import_harness_store")
            .await
            .unwrap()
    );
}
