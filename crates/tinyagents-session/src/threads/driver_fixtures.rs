use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyagents_storage_kit::{CollectionSpec, DocumentStore, Page, Sort};

#[derive(Debug)]
pub(super) struct ConflictingDocs {
    pub inner: Arc<dyn DocumentStore>,
    pub attempts: AtomicUsize,
    pub barrier: tokio::sync::Barrier,
    pub refuse_get: bool,
}
#[async_trait]
impl DocumentStore for ConflictingDocs {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    async fn ensure_collection(
        &self,
        spec: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.inner.ensure_collection(spec).await
    }
    async fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<serde_json::Value>>> {
        if self.refuse_get {
            return Err(StorageError::unavailable("configured backend refused"));
        }
        self.inner.get(collection, id).await
    }
    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        condition: Precondition,
    ) -> tinystoragedrivers_types::Result<tinystoragedrivers_types::Version> {
        if self.attempts.fetch_add(1, Ordering::SeqCst) < 2 {
            self.barrier.wait().await;
        }
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
        if self.attempts.fetch_add(1, Ordering::SeqCst) < 2 {
            self.barrier.wait().await;
        }
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

#[derive(Debug)]
pub(super) struct PausedStreams {
    pub inner: Arc<dyn StreamStore>,
    pub calls: AtomicUsize,
    pub max_limit: AtomicUsize,
    pub reached: tokio::sync::Notify,
    pub release: tokio::sync::Notify,
    pub pause_first: bool,
}
#[async_trait]
impl StreamStore for PausedStreams {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }

    async fn append(
        &self,
        stream: &str,
        value: serde_json::Value,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.append(stream, value).await
    }
    async fn append_batch(
        &self,
        stream: &str,
        values: Vec<serde_json::Value>,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.append_batch(stream, values).await
    }
    async fn read_window(
        &self,
        stream: &str,
        from: u64,
        limit: usize,
    ) -> tinystoragedrivers_types::Result<Vec<tinystoragedrivers_types::StreamEntry>> {
        self.max_limit.fetch_max(limit, Ordering::SeqCst);
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 && self.pause_first {
            self.reached.notify_one();
            self.release.notified().await;
        }
        self.inner.read_window(stream, from, limit).await
    }
    async fn len(&self, stream: &str) -> tinystoragedrivers_types::Result<u64> {
        self.inner.len(stream).await
    }
    async fn truncate_before(
        &self,
        stream: &str,
        offset: u64,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.truncate_before(stream, offset).await
    }
    async fn delete_stream(&self, stream: &str) -> tinystoragedrivers_types::Result<bool> {
        self.inner.delete_stream(stream).await
    }
    async fn streams(&self, prefix: &str) -> tinystoragedrivers_types::Result<Vec<String>> {
        self.inner.streams(prefix).await
    }
}
