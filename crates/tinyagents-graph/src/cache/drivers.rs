//! A [`TaskCache`] over `tinystoragedrivers` document ports.
//!
//! [`DriverTaskCache`] keeps cached node results on whatever backend the host
//! opened. The document handle is already bound to a tenant scope, so two
//! tenants never see each other's entries even when graph, node and input hash
//! coincide.
//!
//! # Layout
//!
//! Entries are documents of the shared `tsd.kv` collection, owned by `graph`:
//! `{ owner, ns: "task_cache:<graph>", key, value }` with `key` =
//! `Key::of([node, hash])`, under `Key::of([owner, ns, key])`, so any hash
//! length is accepted and no two entries meet. A graph's entries share a
//! namespace, which is what [`clear`](TaskCache::clear) drops. A cache built
//! with [`with_namespace`](DriverTaskCache::with_namespace) uses
//! `task_cache:<namespace>:<graph>` so independent caches can share a backend.
//!
//! # Expiry
//!
//! The deadline is the record's `expires_ms` metadata, an absolute Unix epoch
//! millisecond instant that outlives the process. The driver enforces it: an
//! expired entry reads as absent, which this cache reports as a miss.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, Filter, IndexSpec, Key, Repo, StorageError, WriteOptions,
    primitives,
};
use tokio::sync::OnceCell;

use super::{TaskCache, TaskCacheKey};
use crate::storage::{DEFAULT_CACHE_NAMESPACE, KV_OWNER, task_cache_ns};
use crate::{Result, TinyAgentsError};
use tinyagents_harness::ids::{GraphId, now_ms};

/// Map a storage driver failure onto the graph error type.
fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Graph(format!("task cache storage: {error}"))
}

/// A `tsd.kv` document body.
#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    owner: String,
    ns: String,
    key: String,
    value: Value,
}

/// A [`TaskCache`] that stores entries in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverTaskCache {
    docs: Arc<dyn DocumentStore>,
    namespace: String,
    declared: Arc<OnceCell<()>>,
}

impl DriverTaskCache {
    /// Cache in the default namespace.
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_namespace(docs, DEFAULT_CACHE_NAMESPACE)
    }

    /// Cache in `namespace`, so several independent caches can share a backend.
    pub fn with_namespace(docs: Arc<dyn DocumentStore>, namespace: impl Into<String>) -> Self {
        Self {
            docs,
            namespace: namespace.into(),
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// The declaration every cache needs: the shared `tsd.kv` collection.
    pub fn collection_spec() -> CollectionSpec {
        CollectionSpec::new(primitives::KV)
            .index(IndexSpec::new("kv_by_key", ["owner", "ns", "key"]))
    }

    /// The `tsd.kv` namespace of `graph`'s entries.
    fn kv_ns(&self, graph: &str) -> String {
        task_cache_ns(&self.namespace, graph)
    }

    /// `(entry id, kv key)` of `key`.
    fn ids(&self, key: &TaskCacheKey) -> Result<(String, String)> {
        let kv_key = Key::of([key.node_id.as_str(), key.hash.as_str()])
            .map_err(map_error)?
            .as_str()
            .to_owned();
        let id = Key::of([KV_OWNER, &self.kv_ns(key.graph_id.as_str()), &kv_key])
            .map_err(map_error)?
            .as_str()
            .to_owned();
        Ok((id, kv_key))
    }

    fn repo(&self, options: WriteOptions) -> Result<Repo<Entry>> {
        Ok(Repo::new(&self.docs, primitives::KV)
            .map_err(map_error)?
            .with_options(options))
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                self.docs
                    .ensure_collection(&Self::collection_spec())
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl TaskCache for DriverTaskCache {
    async fn get(&self, key: &TaskCacheKey) -> Result<Option<Value>> {
        self.declared().await?;
        let (id, _) = self.ids(key)?;
        Ok(self
            .repo(WriteOptions::new())?
            .get(&id)
            .await
            .map_err(map_error)?
            .map(|found| found.doc.value))
    }

    async fn put(&self, key: &TaskCacheKey, value: Value, ttl: Option<Duration>) -> Result<()> {
        self.declared().await?;
        let (id, kv_key) = self.ids(key)?;
        let mut options = WriteOptions::new();
        if let Some(ttl) = ttl {
            let at = now_ms().saturating_add(ttl.as_millis() as u64);
            options = options.expires_ms(i64::try_from(at).unwrap_or(i64::MAX));
        }
        let entry = Entry {
            owner: KV_OWNER.to_owned(),
            ns: self.kv_ns(key.graph_id.as_str()),
            key: kv_key,
            value,
        };
        self.repo(options)?
            .put(&id, &entry)
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    async fn clear(&self, graph_id: &GraphId) -> Result<()> {
        self.declared().await?;
        let filter = Filter::eq("owner", KV_OWNER)
            .and(Filter::eq("ns", self.kv_ns(graph_id.as_str()).as_str()));
        self.docs
            .delete_where(primitives::KV, &filter)
            .await
            .map(|_| ())
            .map_err(map_error)
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
