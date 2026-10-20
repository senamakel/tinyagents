use super::*;
use crate::store::Store as _;
use std::time::Duration;
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

async fn round_trips_expires_and_counts(cache: DriverResponseCache) {
    assert!(cache.get("k").await.unwrap().is_none());
    cache
        .put("k", ModelResponse::assistant("durable"))
        .await
        .unwrap();
    assert_eq!(cache.get("k").await.unwrap().unwrap().text(), "durable");

    cache
        .put_with_ttl(
            "short",
            ModelResponse::assistant("stale"),
            Some(Duration::from_millis(20)),
        )
        .await
        .unwrap();
    assert_eq!(cache.get("short").await.unwrap().unwrap().text(), "stale");
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(
        cache.get("short").await.unwrap().is_none(),
        "an expired entry reads as a miss"
    );
    assert!(
        cache.get("short").await.unwrap().is_none(),
        "and stays gone"
    );
    // A rewrite revives the key.
    cache
        .put("short", ModelResponse::assistant("fresh"))
        .await
        .unwrap();
    assert_eq!(cache.get("short").await.unwrap().unwrap().text(), "fresh");

    let stats = cache.stats();
    assert_eq!(stats.writes, 3);
    // A backend with native TTL hides the expired entry itself, so only a
    // backend without it lets this process observe (and count) the expiry.
    assert!(stats.expirations <= 1);
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 3);
}

#[tokio::test]
async fn memory_backend_round_trips_expires_and_counts() {
    round_trips_expires_and_counts(DriverResponseCache::new(docs(
        &MemoryBackend::new(),
        "local",
    )))
    .await;
}

#[tokio::test]
async fn testkit_backend_round_trips_expires_and_counts() {
    let storage = MemoryBackend::new();
    round_trips_expires_and_counts(DriverResponseCache::new(docs(&storage, "local"))).await;
}

#[tokio::test]
async fn entries_survive_reopening_the_driver_on_one_backend() {
    let storage = MemoryBackend::new();
    {
        DriverResponseCache::new(docs(&storage, "local"))
            .put("k", ModelResponse::assistant("persisted"))
            .await
            .unwrap();
    }
    let reopened = DriverResponseCache::new(docs(&storage, "local"));
    assert_eq!(
        reopened.get("k").await.unwrap().unwrap().text(),
        "persisted"
    );
}

#[tokio::test]
async fn namespaces_do_not_cross_serve_and_clear_independently() {
    let base = DriverResponseCache::new(docs(&MemoryBackend::new(), "local"));
    let tenant_a = base.with_namespace("tenant-a");
    let tenant_b = base.with_namespace("tenant-b");
    assert_eq!(tenant_a.namespace(), "tenant-a");
    tenant_a
        .put("k", ModelResponse::assistant("a"))
        .await
        .unwrap();
    assert!(tenant_b.get("k").await.unwrap().is_none());
    tenant_b
        .put("k", ModelResponse::assistant("b"))
        .await
        .unwrap();
    tenant_a.clear().await.unwrap();
    assert!(tenant_a.get("k").await.unwrap().is_none());
    assert_eq!(tenant_b.get("k").await.unwrap().unwrap().text(), "b");
}

#[tokio::test]
async fn scopes_and_owners_keep_entries_apart_and_long_keys_fit() {
    let storage = MemoryBackend::new();
    let alice = DriverResponseCache::new(docs(&storage, "alice"));
    let bob = DriverResponseCache::new(docs(&storage, "bob"));
    let other = DriverResponseCache::with_owner(docs(&storage, "alice"), "other");
    let long = "k".repeat(10_000);
    alice
        .put(&long, ModelResponse::assistant("mine"))
        .await
        .unwrap();
    assert_eq!(alice.get(&long).await.unwrap().unwrap().text(), "mine");
    assert!(bob.get(&long).await.unwrap().is_none());
    assert!(other.get(&long).await.unwrap().is_none());
}

/// The v2 shape: an entry is a `tsd.kv` document of owner `harness` in the
/// namespace `cache:<namespace>`, its value the response and its deadline the
/// record's `expires_ms`, not a body field.
#[tokio::test]
async fn entries_are_tsd_kv_documents_with_the_deadline_in_metadata() {
    let storage = MemoryBackend::new();
    let documents = docs(&storage, "local");
    let cache = DriverResponseCache::new(Arc::clone(&documents)).with_namespace("arm-a");
    cache
        .put_with_ttl(
            "req-1",
            ModelResponse::assistant("hi"),
            Some(Duration::from_secs(60)),
        )
        .await
        .unwrap();
    cache
        .put("forever", ModelResponse::assistant("kept"))
        .await
        .unwrap();

    let row = documents
        .get("tsd.kv", "harness/cache:arm-a/req-1")
        .await
        .unwrap()
        .expect("the entry sits under Key::of([owner, ns, key])");
    assert_eq!(row.doc["owner"], "harness");
    assert_eq!(row.doc["ns"], "cache:arm-a");
    assert_eq!(row.doc["key"], "req-1");
    assert!(row.doc["value"].is_object(), "the response is unwrapped");
    assert!(row.doc.get("expiresAtMs").is_none());
    assert!(row.meta.expires_ms.unwrap() > row.meta.created_ms);
    let forever = documents
        .get("tsd.kv", "harness/cache:arm-a/forever")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(forever.meta.expires_ms, None);
}

/// Clearing one namespace leaves other `tsd.kv` entries alone, including the
/// flat store's of the same owner.
#[tokio::test]
async fn clear_drops_only_its_own_namespace_of_the_shared_collection() {
    let storage = MemoryBackend::new();
    let documents = docs(&storage, "local");
    let flat = crate::store::DriverStore::new(Arc::clone(&documents));
    flat.put("default", "k", serde_json::json!("flat"))
        .await
        .unwrap();
    let cache = DriverResponseCache::new(Arc::clone(&documents));
    cache.put("k", ModelResponse::assistant("c")).await.unwrap();
    cache.clear().await.unwrap();
    assert!(cache.get("k").await.unwrap().is_none());
    assert_eq!(
        flat.get("default", "k").await.unwrap(),
        Some(serde_json::json!("flat"))
    );
}
