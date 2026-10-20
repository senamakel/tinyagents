use super::*;
use crate::testkit::MemoryStorage;
use serde_json::Value;
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, ErrorKind, Page, Precondition, ReadCheck, Version, Versioned,
    WriteOp, WriteResult,
};
type NativeResult<T> = std::result::Result<T, StorageError>;

#[derive(Debug)]
struct BoundedQueries {
    inner: Arc<dyn DocumentStore>,
    limits: parking_lot::Mutex<Vec<Option<usize>>>,
    refuse: bool,
}
#[async_trait::async_trait]
impl DocumentStore for BoundedQueries {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    async fn ensure_collection(&self, spec: &CollectionSpec) -> NativeResult<()> {
        self.inner.ensure_collection(spec).await
    }
    async fn get(&self, collection: &str, id: &str) -> NativeResult<Option<Versioned<Value>>> {
        self.inner.get(collection, id).await
    }
    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: Value,
        precondition: Precondition,
    ) -> NativeResult<Version> {
        self.inner.put(collection, id, doc, precondition).await
    }

    async fn delete(
        &self,
        collection: &str,
        id: &str,
        precondition: Precondition,
    ) -> NativeResult<bool> {
        self.inner.delete(collection, id, precondition).await
    }
    async fn query(&self, collection: &str, query: &Query) -> NativeResult<Page<Versioned<Value>>> {
        if collection == storage::EVENTS {
            self.limits.lock().push(query.limit);
            if self.refuse {
                return Err(StorageError::unavailable("event admission refused"));
            }
            if query.limit.is_none_or(|limit| limit > 3) {
                return Err(StorageError::invalid_input(
                    "event query exceeded caller budget",
                ));
            }
            let mut page = query.clone();
            page.limit = Some(query.limit.unwrap().min(2));
            return self.inner.query(collection, &page).await;
        }
        self.inner.query(collection, query).await
    }
    async fn count(&self, collection: &str, filter: &Filter) -> NativeResult<u64> {
        self.inner.count(collection, filter).await
    }
    async fn delete_where(&self, collection: &str, filter: &Filter) -> NativeResult<u64> {
        self.inner.delete_where(collection, filter).await
    }
    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &Value,
    ) -> NativeResult<Option<Versioned<Value>>> {
        self.inner.claim(collection, filter, sort, patch).await
    }
    async fn drop_collection(&self, collection: &str) -> NativeResult<()> {
        self.inner.drop_collection(collection).await
    }
    async fn atomic_batch_checked(
        &self,
        checks: Vec<ReadCheck>,
        writes: Vec<WriteOp>,
    ) -> NativeResult<Vec<WriteResult>> {
        self.inner.atomic_batch_checked(checks, writes).await
    }
}

#[derive(Debug)]
struct ArrivingReceipts {
    inner: Arc<dyn tinystoragedrivers_api::StreamStore>,
    ledger: DriverRunLedger,
    generated: std::sync::atomic::AtomicUsize,
}
#[async_trait::async_trait]
impl tinystoragedrivers_api::StreamStore for ArrivingReceipts {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }

    async fn append(
        &self,
        stream: &str,
        value: serde_json::Value,
    ) -> std::result::Result<u64, StorageError> {
        self.inner.append(stream, value).await
    }
    async fn append_batch(
        &self,
        stream: &str,
        values: Vec<serde_json::Value>,
    ) -> std::result::Result<u64, StorageError> {
        self.inner.append_batch(stream, values).await
    }
    async fn append_batch_if_offset(
        &self,
        stream: &str,
        expected: u64,
        values: Vec<serde_json::Value>,
    ) -> std::result::Result<u64, StorageError> {
        let offset = self
            .inner
            .append_batch_if_offset(stream, expected, values)
            .await?;
        if self
            .generated
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            < 64
        {
            self.ledger
                .append_run_event(event())
                .await
                .map_err(|error| match error {
                    super::super::LedgerError::Driver(error) => *error,
                    other => panic!("native fixture error: {other:?}"),
                })?;
        }
        Ok(offset)
    }
    async fn read_window(
        &self,
        stream: &str,
        from: u64,
        limit: usize,
    ) -> std::result::Result<Vec<tinystoragedrivers_types::StreamEntry>, StorageError> {
        self.inner.read_window(stream, from, limit).await
    }
    async fn len(&self, stream: &str) -> std::result::Result<u64, StorageError> {
        self.inner.len(stream).await
    }
    async fn truncate_before(
        &self,
        stream: &str,
        offset: u64,
    ) -> std::result::Result<u64, StorageError> {
        self.inner.truncate_before(stream, offset).await
    }
    async fn delete_stream(&self, stream: &str) -> std::result::Result<bool, StorageError> {
        self.inner.delete_stream(stream).await
    }
    async fn streams(&self, prefix: &str) -> std::result::Result<Vec<String>, StorageError> {
        self.inner.streams(prefix).await
    }
}

fn event() -> RunEventAppend {
    RunEventAppend {
        run_id: "run".into(),
        event_type: "tick".into(),
        payload: serde_json::json!({}),
    }
}
#[tokio::test]
async fn event_pages_follow_native_cursors_with_the_requested_limit() {
    let backend = MemoryStorage::new();
    let scoped = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let native = DriverRunLedger::new(scoped.clone());
    for _ in 0..5 {
        native.append_run_event(event()).await.unwrap();
    }
    let documents = Arc::new(BoundedQueries {
        inner: scoped.documents().clone(),
        limits: Default::default(),
        refuse: false,
    });
    let actor = DriverRunLedger::new(crate::testkit::with_documents(&scoped, documents.clone()));
    let page = actor
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            after_sequence: None,
            limit: Some(3),
        })
        .await
        .unwrap();
    assert_eq!(
        page.events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(*documents.limits.lock(), vec![Some(3), Some(1)]);
}
#[tokio::test]
async fn zero_limit_event_reads_still_require_native_admission() {
    let backend = MemoryStorage::new();
    let scoped = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let documents = Arc::new(BoundedQueries {
        inner: scoped.documents().clone(),
        limits: Default::default(),
        refuse: true,
    });
    let actor = DriverRunLedger::new(crate::testkit::with_documents(&scoped, documents));
    let error = actor
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            after_sequence: None,
            limit: Some(0),
        })
        .await
        .unwrap_err();
    assert!(
        matches!(error, super::super::LedgerError::Driver(error) if error.kind() == ErrorKind::Unavailable)
    );
}
#[tokio::test]
async fn successful_projection_progress_does_not_exhaust_the_contention_budget() {
    let backend = MemoryStorage::new();
    let scoped = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let receipts = DriverRunLedger::new(scoped.clone());
    receipts.append_run_event(event()).await.unwrap();
    let streams = Arc::new(ArrivingReceipts {
        inner: scoped.streams().clone(),
        ledger: receipts,
        generated: std::sync::atomic::AtomicUsize::new(0),
    });
    let actor = DriverRunLedger::new(crate::testkit::with_documents_and_streams(
        &scoped,
        scoped.documents().clone(),
        streams,
    ));
    assert_eq!(actor.reconcile_run_events("run").await.unwrap(), 65);
    assert_eq!(actor.reconcile_run_events("run").await.unwrap(), 65);
}
