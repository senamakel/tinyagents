//! Unit tests for the task cache backends: in-memory round trip, TTL
//! expiry, per-graph `clear`, and (behind `sqlite`) the SQLite backend.

use super::*;
use serde_json::json;
use std::time::Duration;
use tinyagents_harness::ids::{GraphId, NodeId};

fn key(graph: &str, node: &str, hash: &str) -> TaskCacheKey {
    TaskCacheKey::new(GraphId::new(graph), NodeId::from(node), hash)
}

#[tokio::test]
async fn in_memory_put_then_get_round_trips() {
    let cache = InMemoryTaskCache::new();
    let k = key("g", "n", "h1");
    assert_eq!(cache.get(&k).await.unwrap(), None);
    cache.put(&k, json!({"v": 1}), None).await.unwrap();
    assert_eq!(cache.get(&k).await.unwrap(), Some(json!({"v": 1})));
    // A different hash is a different entry.
    assert_eq!(cache.get(&key("g", "n", "h2")).await.unwrap(), None);
}

#[tokio::test]
async fn in_memory_entry_expires_after_ttl() {
    let cache = InMemoryTaskCache::new();
    let k = key("g", "n", "h");
    cache
        .put(&k, json!(1), Some(Duration::from_millis(30)))
        .await
        .unwrap();
    assert_eq!(cache.get(&k).await.unwrap(), Some(json!(1)));
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert_eq!(cache.get(&k).await.unwrap(), None, "expired after TTL");
}

#[tokio::test]
async fn in_memory_clear_only_drops_one_graph() {
    let cache = InMemoryTaskCache::new();
    cache.put(&key("a", "n", "h"), json!(1), None).await.unwrap();
    cache.put(&key("b", "n", "h"), json!(2), None).await.unwrap();
    cache.clear(&GraphId::new("a")).await.unwrap();
    assert_eq!(cache.get(&key("a", "n", "h")).await.unwrap(), None);
    assert_eq!(cache.get(&key("b", "n", "h")).await.unwrap(), Some(json!(2)));
}
