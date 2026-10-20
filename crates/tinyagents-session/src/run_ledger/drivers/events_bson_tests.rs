use super::*;
use crate::testkit::MemoryStorage;
use serde_json::Value;
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, ErrorKind, Page, Precondition, ReadCheck, Version, Versioned,
    WriteOp, WriteResult,
};
type NativeResult<T> = std::result::Result<T, StorageError>;

/// Matches canonical Mongo convert::number_bson: signed integers or floating
/// values are representable; unsigned integers above i64::MAX are rejected.
/// No database behavior is copied into production code.
fn signed_bson(value: &Value) -> NativeResult<()> {
    match value {
        Value::Number(number) if number.is_u64() && number.as_i64().is_none() => {
            Err(StorageError::serialization(
                "unsigned JSON integer is not representable in native BSON",
            ))
        }
        Value::Array(values) => {
            for value in values {
                signed_bson(value)?;
            }
            Ok(())
        }
        Value::Object(values) => {
            for value in values.values() {
                signed_bson(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[derive(Debug)]
struct SignedBsonDocuments {
    inner: Arc<dyn DocumentStore>,
}
#[async_trait::async_trait]
impl DocumentStore for SignedBsonDocuments {
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
        signed_bson(&doc)?;
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
        for write in &writes {
            if let WriteOp::Put { doc, .. } = write {
                signed_bson(doc)?;
            }
        }
        self.inner.atomic_batch_checked(checks, writes).await
    }
}

pub(super) fn wrap(native: ScopedStorage) -> DriverRunLedger {
    let wrapped = crate::testkit::with_documents(
        &native,
        Arc::new(SignedBsonDocuments {
            inner: native.documents().clone(),
        }),
    );
    DriverRunLedger::new(wrapped)
}

#[tokio::test]
async fn last_signed_event_sequence_commits_with_bson_representable_head() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let ledger = wrap(native.clone());
    ledger.declare().await.unwrap();
    native
        .documents()
        .put(
            storage::EVENT_HEADS,
            &storage::key(&["run"]).unwrap(),
            serde_json::json!({"next":i64::MAX}),
            Precondition::None,
        )
        .await
        .unwrap();
    let input = || RunEventAppend {
        run_id: "run".into(),
        event_type: "last".into(),
        payload: serde_json::json!({}),
    };
    assert_eq!(
        ledger.append_run_event(input()).await.unwrap().sequence,
        i64::MAX as u64
    );
    let head = native
        .documents()
        .get(storage::EVENT_HEADS, &storage::key(&["run"]).unwrap())
        .await
        .unwrap()
        .unwrap();
    signed_bson(&head.doc).unwrap();
    assert!(
        matches!(ledger.append_run_event(input()).await.unwrap_err(), super::super::LedgerError::Driver(error) if error.kind() == ErrorKind::Serialization)
    );
    assert_eq!(
        ledger
            .list_recent_run_events(&RunEventListRequest {
                run_id: "run".into(),
                ..Default::default()
            })
            .await
            .unwrap()
            .events
            .len(),
        1
    );
}
