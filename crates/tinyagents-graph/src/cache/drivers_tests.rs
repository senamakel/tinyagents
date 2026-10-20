use super::*;
use crate::testkit::conformance::task_cache_contract;
use serde_json::json;
use tinyagents_harness::ids::NodeId;
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

fn key(graph: &str, node: &str, hash: &str) -> TaskCacheKey {
    TaskCacheKey::new(GraphId::new(graph), NodeId::from(node), hash)
}

#[tokio::test]
async fn memory_backend_passes_the_task_cache_contract() {
    let storage = MemoryBackend::new();
    task_cache_contract(DriverTaskCache::new(docs(&storage, "local"))).await;
}

#[tokio::test]
async fn testkit_backend_passes_the_task_cache_contract() {
    let storage = MemoryBackend::new();
    task_cache_contract(DriverTaskCache::new(docs(&storage, "local"))).await;
}

#[tokio::test]
async fn entries_persist_across_driver_handles_on_one_backend() {
    let storage = MemoryBackend::new();
    {
        let cache = DriverTaskCache::new(docs(&storage, "local"));
        cache
            .put(&key("g", "n", "h"), json!(7), None)
            .await
            .unwrap();
    }
    let reopened = DriverTaskCache::new(docs(&storage, "local"));
    assert_eq!(
        reopened.get(&key("g", "n", "h")).await.unwrap(),
        Some(json!(7))
    );
}

#[tokio::test]
async fn scopes_and_namespaces_keep_entries_apart() {
    let storage = MemoryBackend::new();
    let alice = DriverTaskCache::new(docs(&storage, "alice"));
    let bob = DriverTaskCache::new(docs(&storage, "bob"));
    let other = DriverTaskCache::with_namespace(docs(&storage, "alice"), "other_cache");
    alice
        .put(&key("g", "n", "h"), json!(1), None)
        .await
        .unwrap();
    assert_eq!(bob.get(&key("g", "n", "h")).await.unwrap(), None);
    assert_eq!(other.get(&key("g", "n", "h")).await.unwrap(), None);
    // Clearing one tenant's graph never reaches another tenant's entry.
    bob.put(&key("g", "n", "h"), json!(2), None).await.unwrap();
    alice.clear(&GraphId::new("g")).await.unwrap();
    assert_eq!(bob.get(&key("g", "n", "h")).await.unwrap(), Some(json!(2)));
}

#[tokio::test]
async fn long_input_hashes_fit_the_document_id_limit() {
    let storage = MemoryBackend::new();
    let cache = DriverTaskCache::new(docs(&storage, "local"));
    let long = key("g", "n", &"x".repeat(10_000));
    cache.put(&long, json!("big"), None).await.unwrap();
    assert_eq!(cache.get(&long).await.unwrap(), Some(json!("big")));
}

#[tokio::test]
async fn a_backend_that_cannot_keep_expiry_refuses_a_ttl_instead_of_caching_forever() {
    let storage = MemoryBackend::new();
    let wrapped: Arc<dyn DocumentStore> = Arc::new(tinyagents_harness::testkit::WithoutNativeTtl(
        docs(&storage, "local"),
    ));
    let cache = DriverTaskCache::new(wrapped);
    let error = cache
        .put(&key("g", "n", "h"), json!(1), Some(Duration::from_secs(60)))
        .await
        .unwrap_err();
    assert!(matches!(error, TinyAgentsError::Graph(_)), "{error:?}");
    assert_eq!(cache.get(&key("g", "n", "h")).await.unwrap(), None);
    // Without a TTL the entry needs no metadata and is stored.
    cache
        .put(&key("g", "n", "h"), json!(1), None)
        .await
        .unwrap();
}
