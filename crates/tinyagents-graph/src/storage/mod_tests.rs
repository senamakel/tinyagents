use super::*;
use crate::cache::{DriverTaskCache, TaskCache, TaskCacheKey};
use crate::checkpoint::{Checkpoint, Checkpointer, DriverCheckpointer};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use tinyagents_harness::ids::{GraphId, NodeId, now_ms};
use tinyagents_storage_kit::{
    DocumentStoreExt, MigrateOptions, Precondition, Query, Registry, Scope, ScopeClass,
    ScopedStorage, StorageBackend,
};
use tinystoragedrivers_testkit::MemoryBackend;

fn scoped() -> ScopedStorage {
    MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap()
}

async fn put_v1(storage: &ScopedStorage, collection: &str, id: &str, body: Value) {
    storage
        .documents()
        .put(collection, id, body, Precondition::None)
        .await
        .unwrap();
}

async fn migrate(storage: &ScopedStorage) {
    let registry = Registry::compose([storage_schema()]).unwrap();
    for item in storage_schema().items {
        if let tinyagents_storage_kit::Item::Collection { spec, .. } = item {
            storage.documents().ensure_collection(&spec).await.unwrap();
        }
    }
    for spec in [
        tinyagents_storage_kit::primitives::leases_collection_spec(),
        tinyagents_storage_kit::primitives::kv_collection_spec(),
    ] {
        storage.documents().ensure_collection(&spec).await.unwrap();
    }
    registry
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap();
}

/// A v1 checkpoint row as `DriverCheckpointer` wrote it under `<prefix>_checkpoints`.
async fn v1_checkpoint(
    storage: &ScopedStorage,
    prefix: &str,
    thread: &str,
    seq: u64,
    id: &str,
    state: i32,
) {
    let checkpoint = Checkpoint::new(state, Vec::new())
        .with_thread_id(thread.to_string())
        .with_checkpoint_id(id.to_string());
    put_v1(
        storage,
        &format!("{prefix}_checkpoints"),
        &format!("{}:{thread}/20:{seq:020}", thread.len()),
        json!({
            "thread": thread, "namespace": "", "seq": seq, "checkpoint_id": id,
            "record": serde_json::to_value(&checkpoint).unwrap(),
        }),
    )
    .await;
    put_v1(
        storage,
        &format!("{prefix}_threads"),
        &format!("{}:{thread}", thread.len()),
        json!({"thread": thread, "next": seq + 1}),
    )
    .await;
}

fn checkpointer(storage: &ScopedStorage, ns: &str) -> DriverCheckpointer<i32> {
    DriverCheckpointer::with_namespace(Arc::clone(storage.documents()), ns)
}

#[test]
fn the_schema_composes_and_names_the_one_checkpointer() {
    let schema = storage_schema();
    assert_eq!(schema.owner.as_str(), "graph");
    let names: Vec<&str> = schema.items.iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        ["graph.checkpoints", "graph.threads", "graph.writes"]
    );
    assert!(
        schema
            .items
            .iter()
            .all(|item| item.scope() == ScopeClass::Agent)
    );
    Registry::compose([schema]).unwrap();
}

#[tokio::test]
async fn each_v1_prefix_becomes_its_own_namespace_of_the_same_collections() {
    let storage = scoped();
    // The same thread id under all three prefixes must not collide.
    v1_checkpoint(&storage, "graph", "t", 0, "g1", 1).await;
    v1_checkpoint(&storage, "delegation_graph", "t", 0, "d1", 2).await;
    v1_checkpoint(&storage, "flows_graph", "t", 0, "f1", 3).await;
    migrate(&storage).await;

    for (ns, id, state) in [
        ("graph", "g1", 1),
        ("delegation", "d1", 2),
        ("flows", "f1", 3),
    ] {
        let cp = checkpointer(&storage, ns);
        let found = cp.get("t", None).await.unwrap().expect(ns);
        assert_eq!(
            (found.checkpoint_id.as_str(), found.state),
            (id, state),
            "{ns}"
        );
        assert_eq!(cp.list_threads().await.unwrap(), ["t"], "{ns}");
        // The counter survived: the next checkpoint continues after the migrated one.
        let next = Checkpoint::new(9, Vec::new())
            .with_thread_id("t".to_string())
            .with_checkpoint_id("next".to_string());
        cp.put(next).await.unwrap();
        let history = cp.get_thread("t").await.unwrap();
        assert_eq!(history.len(), 2, "{ns}");
        assert_eq!(history[1].checkpoint_id, "next", "{ns}");
    }

    let row = storage
        .documents()
        .query_all("graph.checkpoints", &Query::all())
        .await
        .unwrap()
        .into_iter()
        .find(|row| row.doc["ns"] == "delegation" && row.doc["checkpoint_id"] == "d1")
        .unwrap();
    assert_eq!(row.doc["thread"], "t");
    assert_eq!(row.doc["seq"], 0);
    assert!(
        row.doc["checkpoint"].is_object(),
        "typed, not a JSON string"
    );
    assert!(row.doc.get("record").is_none());
    // A row carries the version its step ends at: one step per prefix.
    assert_eq!(row.meta.schema, 3);
}

#[tokio::test]
async fn v1_pending_writes_follow_their_checkpoint_into_the_namespace() {
    let storage = scoped();
    put_v1(
        &storage,
        "delegation_graph_writes",
        "3:t/4:::c1",
        json!({"thread": "t", "namespace": ["sub"], "checkpoint_id": "c1",
               "writes": [{"node": "n", "task_id": "task", "idx": 0, "channel": "out", "payload": "x"}]}),
    )
    .await;
    migrate(&storage).await;
    let cp = checkpointer(&storage, "delegation");
    let config = crate::checkpoint::CheckpointConfig {
        thread_id: "t".to_string(),
        checkpoint_id: Some("c1".to_string()),
        namespace: vec!["sub".to_string()],
    };
    let writes = cp.get_writes(&config).await.unwrap();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].payload, json!("x"));
    assert!(
        checkpointer(&storage, "graph")
            .get_writes(&config)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn migrating_twice_changes_nothing() {
    let storage = scoped();
    v1_checkpoint(&storage, "graph", "t", 0, "c1", 1).await;
    migrate(&storage).await;
    migrate(&storage).await;
    let rows = storage
        .documents()
        .query_all("graph.checkpoints", &Query::all())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
}

#[tokio::test]
async fn a_live_v1_lease_is_carried_over_and_an_expired_one_dropped() {
    let storage = scoped();
    let now = now_ms();
    put_v1(
        &storage,
        "delegation_graph_leases",
        "live",
        json!({"owner": "worker-a", "expires_at_ms": now + 60_000}),
    )
    .await;
    // Ids are the v1 `key([thread])`: length-prefixed.
    put_v1(
        &storage,
        "delegation_graph_leases",
        "1:a",
        json!({"owner": "worker-a", "expires_at_ms": now + 60_000}),
    )
    .await;
    put_v1(
        &storage,
        "flows_graph_leases",
        "1:b",
        json!({"owner": "worker-b", "expires_at_ms": now.saturating_sub(1)}),
    )
    .await;
    assert_eq!(import_legacy_leases(&storage).await.unwrap(), 1);

    let delegation = checkpointer(&storage, "delegation");
    assert!(
        !delegation
            .try_claim("a", "other", Duration::from_secs(5))
            .await
            .unwrap()
    );
    assert!(
        delegation
            .renew("a", "worker-a", Duration::from_secs(5))
            .await
            .unwrap()
    );
    let flows = checkpointer(&storage, "flows");
    assert!(
        flows
            .try_claim("b", "other", Duration::from_secs(5))
            .await
            .unwrap()
    );
    // The legacy collections are gone and the import is recorded.
    assert!(
        storage
            .documents()
            .query_all("delegation_graph_leases", &Query::all())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(import_legacy_leases(&storage).await.unwrap(), 0);
}

#[tokio::test]
async fn v1_task_cache_rows_move_into_the_graph_kv_namespace() {
    let storage = scoped();
    let now = now_ms();
    put_v1(
        &storage,
        "graph_task_cache",
        "h1",
        json!({"graph": "g", "node": "n", "hash": "abc", "value": {"v": 1}, "expiresAtMs": null}),
    )
    .await;
    put_v1(
        &storage,
        "graph_task_cache",
        "h2",
        json!({"graph": "g", "node": "n", "hash": "old", "value": 2, "expiresAtMs": now.saturating_sub(5)}),
    )
    .await;
    put_v1(
        &storage,
        "graph_task_cache",
        "h3",
        json!({"graph": "g", "node": "n", "hash": "soon", "value": 3, "expiresAtMs": now + 60_000}),
    )
    .await;
    assert_eq!(import_legacy_task_cache(&storage).await.unwrap(), 2);
    let cache = DriverTaskCache::new(Arc::clone(storage.documents()));
    let key = |hash: &str| TaskCacheKey::new(GraphId::new("g"), NodeId::from("n"), hash);
    assert_eq!(cache.get(&key("abc")).await.unwrap(), Some(json!({"v": 1})));
    assert_eq!(cache.get(&key("soon")).await.unwrap(), Some(json!(3)));
    assert_eq!(cache.get(&key("old")).await.unwrap(), None);
    let soon = storage
        .documents()
        .query_all("tsd.kv", &Query::all())
        .await
        .unwrap()
        .into_iter()
        .find(|row| row.doc["key"].as_str().is_some_and(|k| k.ends_with("soon")))
        .unwrap();
    assert_eq!(soon.doc["owner"], "graph");
    assert!(
        soon.meta.expires_ms.is_some(),
        "the deadline is metadata now"
    );
    assert_eq!(import_legacy_task_cache(&storage).await.unwrap(), 0);
}
