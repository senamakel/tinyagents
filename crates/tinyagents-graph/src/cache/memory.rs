//! Process-local, TTL-aware [`TaskCache`].

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::{TaskCache, TaskCacheKey};
use crate::{Result, TinyAgentsError};
use tinyagents_harness::ids::GraphId;

/// An in-memory [`TaskCache`]: a mutex-guarded map honoring per-entry TTL
/// on read. Entries never leave memory on their own; `clear` (or dropping
/// the cache) is what reclaims them.
#[derive(Default)]
pub struct InMemoryTaskCache {
    entries: Mutex<HashMap<TaskCacheKey, (serde_json::Value, Option<Instant>)>>,
}

impl InMemoryTaskCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<TaskCacheKey, (serde_json::Value, Option<Instant>)>>>
    {
        self.entries
            .lock()
            .map_err(|_| TinyAgentsError::Graph("task cache lock poisoned".to_string()))
    }
}

#[async_trait]
impl TaskCache for InMemoryTaskCache {
    async fn get(&self, key: &TaskCacheKey) -> Result<Option<serde_json::Value>> {
        let mut entries = self.lock()?;
        match entries.get(key) {
            Some((_, Some(expires_at))) if *expires_at <= Instant::now() => {
                entries.remove(key);
                Ok(None)
            }
            Some((value, _)) => Ok(Some(value.clone())),
            None => Ok(None),
        }
    }

    async fn put(
        &self,
        key: &TaskCacheKey,
        value: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let expires_at = ttl.map(|ttl| Instant::now() + ttl);
        self.lock()?.insert(key.clone(), (value, expires_at));
        Ok(())
    }

    async fn clear(&self, graph_id: &GraphId) -> Result<()> {
        self.lock()?.retain(|key, _| &key.graph_id != graph_id);
        Ok(())
    }
}
