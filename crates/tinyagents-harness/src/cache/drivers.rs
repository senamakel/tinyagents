//! A [`ResponseCache`] over `tinystoragedrivers` document ports.
//!
//! [`DriverResponseCache`] keeps cached model responses on whatever backend the
//! host opened (SQLite on a desktop, MongoDB in the cloud, memory in tests), so
//! a restart does not pay the whole provider bill again. The document handle is
//! already bound to a tenant scope; the namespace below separates populations
//! *within* a tenant (a control and an experiment arm, say).
//!
//! # Layout
//!
//! Entries are documents of the shared `tsd.kv` collection, one per
//! `(namespace, key)`, owned by [`DriverResponseCache::DEFAULT_OWNER`]
//! (`harness`): `{ owner, ns: "cache:<namespace>", key, value }` under
//! `Key::of([owner, ns, key])`, so any key length is accepted and no two entries
//! meet. The `cache:` prefix keeps them apart from the flat
//! [`DriverStore`](crate::store::DriverStore) of the same owner.
//!
//! # Expiry
//!
//! The deadline is the record's `expires_ms` metadata, an absolute Unix epoch
//! millisecond instant that outlives the process. The driver enforces it: an
//! expired entry reads as absent, which this cache reports as a miss.
//!
//! # Statistics
//!
//! [`ResponseCache::stats`] is synchronous, so entry and byte counts (which need
//! a query) stay zero; hits, misses and writes are counted in this process and
//! shared by every clone of one handle. The driver hides an expired entry itself,
//! so this cache can no longer tell an expired entry from one never written and
//! `expirations` stays zero.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tinyagents_storage_kit::{CollectionSpec, DocumentStore, Filter, Repo, WriteOptions};
use tinyinference_llm::model::ModelResponse;
use tokio::sync::OnceCell;

use super::types::{CacheStats, ResponseCache};
use crate::error::{Result, TinyAgentsError};
use crate::ids::now_ms;
use crate::storage::{CACHE_NS_PREFIX, KV, KvEntry, kv_id};
use crate::store::{DriverStore, SchemaAdmission, map_driver_error as map_error};

#[derive(Debug, Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    writes: AtomicU64,
}

/// A durable [`ResponseCache`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverResponseCache {
    docs: Arc<dyn DocumentStore>,
    owner: String,
    namespace: String,
    declared: Arc<OnceCell<()>>,
    counters: Arc<Counters>,
}

impl DriverResponseCache {
    /// The `tsd.kv` owner used by [`DriverResponseCache::new`].
    pub const DEFAULT_OWNER: &'static str = "harness";
    /// The namespace used when none is given.
    pub const DEFAULT_NAMESPACE: &'static str = "default";

    /// Cache under [`Self::DEFAULT_OWNER`] and [`Self::DEFAULT_NAMESPACE`].
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_owner(docs, Self::DEFAULT_OWNER)
    }

    /// Cache under `owner`, so independent caches can share a backend.
    pub fn with_owner(docs: Arc<dyn DocumentStore>, owner: impl Into<String>) -> Self {
        Self {
            docs,
            owner: owner.into(),
            namespace: Self::DEFAULT_NAMESPACE.to_string(),
            declared: Arc::new(OnceCell::new()),
            counters: Arc::new(Counters::default()),
        }
    }

    /// A handle scoped to `namespace`, sharing this handle's backend.
    ///
    /// Two harnesses that must not cross-serve point at the same backend with
    /// different namespaces; [`clear`](ResponseCache::clear) then drops only the
    /// handle's own population.
    #[must_use]
    pub fn with_namespace(&self, namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            ..self.clone()
        }
    }

    /// This handle's namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The declaration every cache needs: the shared `tsd.kv` collection.
    pub fn collection_spec() -> CollectionSpec {
        DriverStore::collection_spec()
    }

    /// The `tsd.kv` namespace of this handle.
    fn kv_ns(&self) -> String {
        format!("{CACHE_NS_PREFIX}{}", self.namespace)
    }

    fn id(&self, key: &str) -> Result<String> {
        kv_id(&self.owner, &self.kv_ns(), key).map_err(map_error)
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                SchemaAdmission::Declare
                    .admit(self.docs.as_ref(), &Self::collection_spec())
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl ResponseCache for DriverResponseCache {
    async fn get(&self, key: &str) -> Result<Option<ModelResponse>> {
        self.declared().await?;
        let Some(found) = self.docs.get(KV, &self.id(key)?).await.map_err(map_error)? else {
            self.counters.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };
        let value =
            found.doc.get("value").cloned().ok_or_else(|| {
                TinyAgentsError::Storage("response cache entry has no value".into())
            })?;
        let response: ModelResponse = serde_json::from_value(value)
            .map_err(|e| TinyAgentsError::Storage(format!("decode response cache entry: {e}")))?;
        self.counters.hits.fetch_add(1, Ordering::Relaxed);
        Ok(Some(response))
    }

    async fn put(&self, key: &str, value: ModelResponse) -> Result<()> {
        self.put_with_ttl(key, value, None).await
    }

    async fn put_with_ttl(
        &self,
        key: &str,
        value: ModelResponse,
        ttl: Option<Duration>,
    ) -> Result<()> {
        self.declared().await?;
        let mut options = WriteOptions::new();
        if let Some(ttl) = ttl {
            let at = now_ms().saturating_add(ttl.as_millis() as u64);
            options = options.expires_ms(i64::try_from(at).unwrap_or(i64::MAX));
        }
        let entry = KvEntry::new(
            &self.owner,
            &self.kv_ns(),
            key,
            serde_json::to_value(&value).map_err(|e| {
                TinyAgentsError::Storage(format!("encode response cache entry: {e}"))
            })?,
        );
        Repo::<KvEntry>::new(&self.docs, KV)
            .map_err(map_error)?
            .with_options(options)
            .put(&self.id(key)?, &entry)
            .await
            .map_err(map_error)?;
        self.counters.writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.declared().await?;
        let filter =
            Filter::eq("owner", self.owner.as_str()).and(Filter::eq("ns", self.kv_ns().as_str()));
        let dropped = self
            .docs
            .delete_where(KV, &filter)
            .await
            .map_err(map_error)?;
        tracing::debug!(
            namespace = %self.namespace,
            dropped,
            "[cache] cleared the driver response cache namespace"
        );
        Ok(())
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.counters.hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            writes: self.counters.writes.load(Ordering::Relaxed),
            ..CacheStats::default()
        }
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
