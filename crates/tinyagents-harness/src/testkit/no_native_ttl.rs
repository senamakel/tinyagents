//! A [`DocumentStore`] that hides its backend's native TTL support.
//!
//! The driver-backed caches declare an expiry field only when the backend
//! advertises [`Capability::Ttl`], and enforce expiry on read either way. Every
//! bundled backend has native TTL, so this wrapper is how a test (or a host
//! certifying a custom backend) exercises the read-side path a backend without
//! it would take.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinyagents_storage_kit::{
    Capabilities, Capability, CollectionSpec, DocumentStore, Filter, Page, Precondition, Query,
    ReadCheck, Result, SearchHit, Sort, Version, Versioned, WriteOp, WriteResult,
};

/// Forwards every call to `inner` but reports no [`Capability::Ttl`].
///
/// A collection spec asking for native expiry is declared without it, so
/// documents stay readable after their expiry instant until the caller removes
/// them, exactly as on a backend that cannot expire.
#[derive(Debug, Clone)]
pub struct WithoutNativeTtl(pub Arc<dyn DocumentStore>);

#[async_trait]
impl DocumentStore for WithoutNativeTtl {
    fn capabilities(&self) -> Capabilities {
        self.0.capabilities().without(Capability::Ttl)
    }

    async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<()> {
        let mut spec = spec.clone();
        spec.ttl_field = None;
        self.0.ensure_collection(&spec).await
    }

    async fn get(&self, collection: &str, id: &str) -> Result<Option<Versioned<Value>>> {
        self.0.get(collection, id).await
    }

    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: Value,
        precondition: Precondition,
    ) -> Result<Version> {
        self.0.put(collection, id, doc, precondition).await
    }

    async fn delete(&self, collection: &str, id: &str, precondition: Precondition) -> Result<bool> {
        self.0.delete(collection, id, precondition).await
    }

    async fn query(&self, collection: &str, query: &Query) -> Result<Page<Versioned<Value>>> {
        self.0.query(collection, query).await
    }

    async fn count(&self, collection: &str, filter: &Filter) -> Result<u64> {
        self.0.count(collection, filter).await
    }

    async fn delete_where(&self, collection: &str, filter: &Filter) -> Result<u64> {
        self.0.delete_where(collection, filter).await
    }

    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &Value,
    ) -> Result<Option<Versioned<Value>>> {
        self.0.claim(collection, filter, sort, patch).await
    }

    async fn atomic_batch(&self, ops: Vec<WriteOp>) -> Result<Vec<WriteResult>> {
        self.0.atomic_batch(ops).await
    }

    async fn atomic_batch_checked(
        &self,
        checks: Vec<ReadCheck>,
        ops: Vec<WriteOp>,
    ) -> Result<Vec<WriteResult>> {
        self.0.atomic_batch_checked(checks, ops).await
    }

    async fn search(&self, collection: &str, text: &str, limit: usize) -> Result<Vec<SearchHit>> {
        self.0.search(collection, text, limit).await
    }

    async fn drop_collection(&self, collection: &str) -> Result<()> {
        self.0.drop_collection(collection).await
    }

    async fn collections(&self, prefix: &str) -> Result<Vec<String>> {
        self.0.collections(prefix).await
    }
}
