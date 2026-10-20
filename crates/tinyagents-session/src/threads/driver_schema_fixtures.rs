use super::*;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyagents_storage_kit::{CollectionSpec, DocumentStore, Page, Sort};

#[derive(Debug)]
pub(super) struct RecordingDocs {
    pub inner: Arc<dyn DocumentStore>,
    pub validates: AtomicUsize,
    pub declares: AtomicUsize,
    pub refuse: AtomicBool,
    pub marker: Arc<std::io::Error>,
}
#[async_trait]
impl DocumentStore for RecordingDocs {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    async fn ensure_collection(
        &self,
        spec: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.declares.fetch_add(1, Ordering::SeqCst);
        self.inner.ensure_collection(spec).await
    }
    async fn validate_collection_spec(
        &self,
        spec: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.validates.fetch_add(1, Ordering::SeqCst);
        if self.refuse.load(Ordering::SeqCst) {
            return Err(StorageError::invalid_input("native validation refused")
                .with_source(self.marker.clone()));
        }
        self.inner.validate_collection_spec(spec).await
    }
    async fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<serde_json::Value>>> {
        self.inner.get(collection, id).await
    }
    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        condition: Precondition,
    ) -> tinystoragedrivers_types::Result<tinystoragedrivers_types::Version> {
        self.inner.put(collection, id, doc, condition).await
    }
    async fn put_with(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        condition: Precondition,
        options: tinystoragedrivers_types::WriteOptions,
    ) -> tinystoragedrivers_types::Result<tinystoragedrivers_types::Version> {
        self.inner
            .put_with(collection, id, doc, condition, options)
            .await
    }
    async fn delete(
        &self,
        collection: &str,
        id: &str,
        condition: Precondition,
    ) -> tinystoragedrivers_types::Result<bool> {
        self.inner.delete(collection, id, condition).await
    }
    async fn query(
        &self,
        collection: &str,
        query: &Query,
    ) -> tinystoragedrivers_types::Result<Page<Versioned<serde_json::Value>>> {
        self.inner.query(collection, query).await
    }
    async fn count(
        &self,
        collection: &str,
        filter: &Filter,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.count(collection, filter).await
    }
    async fn delete_where(
        &self,
        collection: &str,
        filter: &Filter,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.delete_where(collection, filter).await
    }
    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &serde_json::Value,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<serde_json::Value>>> {
        self.inner.claim(collection, filter, sort, patch).await
    }
    async fn drop_collection(&self, collection: &str) -> tinystoragedrivers_types::Result<()> {
        self.inner.drop_collection(collection).await
    }
}
