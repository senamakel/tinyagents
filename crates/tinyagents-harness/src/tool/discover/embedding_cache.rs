//! An [`EmbeddingCache`] over `tinystoragedrivers` document ports.
//!
//! [`DriverEmbeddingCache`] keeps rebuildable catalogue vectors on whatever
//! backend the host opened, so a cold process pays for the catalogue once.
//! The document handle is already bound to a tenant scope; the embedding-space
//! signature keeps vectors of different models apart within it.
//!
//! # Layout
//!
//! The collection `harness.embedding_cache` with one document per
//! `(signature, key)`, id `Key::of([signature, key])`: `{ sig, key, vector }`. `key` is the
//! ranker's 64-bit hash, written as 16 hex digits because a JSON/BSON number
//! cannot carry a full `u64`. Writes are per-entry upserts, so concurrent
//! writers with disjoint deltas never replace each other's entries and a
//! repeated store is idempotent. A `signature` is never compared against
//! another's vectors.
//!
//! A vector with a non-finite component cannot be represented in JSON and is
//! skipped rather than stored: the ranker would reject it on load anyway.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, IndexSpec, Key, Precondition, WriteOptions,
};
use tokio::sync::OnceCell;

use super::EmbeddingCache;
use crate::error::Result;
use crate::store::map_driver_error as map_error;

/// Document ids fetched per concurrent wave when loading.
const LOAD_CONCURRENCY: usize = 32;

/// A durable [`EmbeddingCache`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverEmbeddingCache {
    docs: Arc<dyn DocumentStore>,
    declared: Arc<OnceCell<()>>,
}

impl DriverEmbeddingCache {
    /// The collection every cache uses.
    pub const COLLECTION: &'static str = crate::storage::EMBEDDING_CACHE;

    /// Cache in [`Self::COLLECTION`].
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self {
            docs,
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// The declaration a cache needs.
    pub fn collection_spec() -> CollectionSpec {
        CollectionSpec::new(Self::COLLECTION).index(IndexSpec::new("by_signature", ["sig"]))
    }

    /// `Key::of([signature, key as 16 hex digits])`.
    pub(crate) fn id(signature: &str, key: &str) -> Result<String> {
        Ok(Key::of([signature, key])
            .map_err(map_error)?
            .as_str()
            .to_owned())
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
impl EmbeddingCache for DriverEmbeddingCache {
    async fn load(&self, signature: &str, keys: &[u64]) -> Result<HashMap<u64, Vec<f32>>> {
        self.declared().await?;
        let mut found = HashMap::new();
        for wave in keys.chunks(LOAD_CONCURRENCY) {
            let reads = wave.iter().map(|key| {
                let id = Self::id(signature, &format!("{key:016x}"));
                async move {
                    let row = match id {
                        Ok(id) => self
                            .docs
                            .get(Self::COLLECTION, &id)
                            .await
                            .map_err(map_error),
                        Err(error) => Err(error),
                    };
                    (*key, row)
                }
            });
            for (key, row) in futures::future::join_all(reads).await {
                let Some(row) = row? else {
                    continue;
                };
                // A document from another signature can only collide by hash
                // accident; the stored signature is the authority.
                if row.doc.get("sig").and_then(Value::as_str) != Some(signature) {
                    continue;
                }
                let Some(vector) = row.doc.get("vector").and_then(Value::as_array) else {
                    continue;
                };
                let vector: Option<Vec<f32>> = vector
                    .iter()
                    .map(|component| component.as_f64().map(|value| value as f32))
                    .collect();
                if let Some(vector) = vector {
                    found.insert(key, vector);
                }
            }
        }
        Ok(found)
    }

    async fn store(&self, signature: &str, values: &[(u64, Vec<f32>)]) -> Result<()> {
        self.declared().await?;
        for (key, vector) in values {
            if vector.iter().any(|component| !component.is_finite()) {
                tracing::debug!(
                    key,
                    "[tool-search] not persisting a vector with a non-finite component"
                );
                continue;
            }
            let doc = json!({
                "sig": signature,
                "key": format!("{key:016x}"),
                "vector": vector,
            });
            self.docs
                .put_with(
                    Self::COLLECTION,
                    &Self::id(signature, &format!("{key:016x}"))?,
                    doc,
                    Precondition::None,
                    WriteOptions::new().schema(crate::storage::SCHEMA_VERSION),
                )
                .await
                .map_err(map_error)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "embedding_cache_tests.rs"]
mod tests;
