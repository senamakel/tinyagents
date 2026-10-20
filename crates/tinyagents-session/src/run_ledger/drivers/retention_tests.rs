use super::*;
use crate::testkit::MemoryStorage;
use chrono::Duration;
use tinyagents_storage_kit::{Precondition, Scope, ScopedStorage};
use tinystoragedrivers_api::StorageBackend;

async fn seed(ledger: &DriverRunLedger, id: &str, at: DateTime<Utc>) {
    let mut row = ledger
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: id.into(),
            input_tokens: Some(17),
            ..Default::default()
        })
        .await
        .unwrap();
    row.updated_at = Some(at);
    ledger
        .scoped
        .documents()
        .put(
            TELEMETRY,
            &storage::key(&[id]).unwrap(),
            storage::encode(&row).unwrap(),
            Precondition::None,
        )
        .await
        .unwrap();
}

async fn prunes_only_old_telemetry(backend: &dyn StorageBackend) {
    let a_scope = Scope::new("a").unwrap();
    let a_storage = ScopedStorage::new(backend.for_scope(&a_scope).unwrap(), a_scope.clone());
    let a = DriverRunLedger::new(a_storage.clone());
    let peer = DriverRunLedger::new(a_storage);
    let b_scope = Scope::new("b").unwrap();
    let sibling = DriverRunLedger::new(ScopedStorage::new(
        backend.for_scope(&b_scope).unwrap(),
        b_scope,
    ));
    let cutoff = DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    seed(&a, "old", cutoff - Duration::seconds(1)).await;
    seed(&a, "boundary", cutoff).await;
    seed(&a, "new", cutoff + Duration::milliseconds(1)).await;
    seed(&sibling, "old", cutoff - Duration::seconds(1)).await;
    seed(&a, "undated", cutoff - Duration::days(1)).await;
    let mut undated = a
        .read::<RunTelemetry>(TELEMETRY, "undated")
        .await
        .unwrap()
        .unwrap()
        .doc;
    undated.updated_at = None;
    a.scoped
        .documents()
        .put(
            TELEMETRY,
            &storage::key(&["undated"]).unwrap(),
            storage::encode(&undated).unwrap(),
            Precondition::None,
        )
        .await
        .unwrap();
    assert_eq!(a.prune_run_telemetry_before(cutoff).await.unwrap(), 1);
    assert!(
        peer.read::<RunTelemetry>(TELEMETRY, "old")
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        peer.read::<RunTelemetry>(TELEMETRY, "boundary")
            .await
            .unwrap()
            .unwrap()
            .doc
            .input_tokens,
        17
    );
    assert!(
        peer.read::<RunTelemetry>(TELEMETRY, "new")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        sibling
            .read::<RunTelemetry>(TELEMETRY, "old")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        peer.read::<RunTelemetry>(TELEMETRY, "undated")
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(peer.prune_run_telemetry_before(cutoff).await.unwrap(), 0);
}

#[tokio::test]
async fn memory_telemetry_retention_preserves_cutoff_and_other_scopes() {
    prunes_only_old_telemetry(&MemoryStorage::new()).await;
}

#[tokio::test]
async fn reconstructed_memory_telemetry_retention_preserves_cutoff_and_other_scopes() {
    let backend = MemoryStorage::new();
    prunes_only_old_telemetry(&backend).await;
}

use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, ErrorKind, Page, ReadCheck, Version, Versioned, WriteOp,
    WriteResult,
};
type NativeResult<T> = std::result::Result<T, StorageError>;

#[derive(Debug)]
struct RetentionBoundary {
    inner: Arc<dyn DocumentStore>,
    refresh: Option<Value>,
    raced: AtomicBool,
    refuse: bool,
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl DocumentStore for RetentionBoundary {
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(value) = &self.refresh
            && !self.raced.swap(true, Ordering::SeqCst)
        {
            self.inner
                .put(
                    TELEMETRY,
                    &storage::key(&["old"]).unwrap(),
                    value.clone(),
                    Precondition::None,
                )
                .await?;
        }
        if self.refuse {
            return Err(StorageError::unavailable("retention admission expired")
                .with_source(std::io::Error::other("retained native source")));
        }
        self.inner.atomic_batch_checked(checks, writes).await
    }
}

fn wrap(
    native: &ScopedStorage,
    refresh: Option<Value>,
    refuse: bool,
) -> (DriverRunLedger, Arc<RetentionBoundary>) {
    let boundary = Arc::new(RetentionBoundary {
        inner: native.documents().clone(),
        refresh,
        raced: AtomicBool::new(false),
        refuse,
        calls: AtomicUsize::new(0),
    });
    let scoped = crate::testkit::with_documents(native, boundary.clone());
    (DriverRunLedger::new(scoped), boundary)
}

#[tokio::test]
async fn concurrent_telemetry_refresh_survives_retention_retry() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("a").unwrap()).unwrap();
    let peer = DriverRunLedger::new(native.clone());
    let cutoff = now();
    seed(&peer, "old", cutoff - Duration::days(1)).await;
    seed(&peer, "other", cutoff - Duration::days(1)).await;
    let mut refreshed = peer
        .read::<RunTelemetry>(TELEMETRY, "old")
        .await
        .unwrap()
        .unwrap()
        .doc;
    refreshed.updated_at = Some(cutoff);
    refreshed.input_tokens = 91;
    let (ledger, boundary) = wrap(&native, Some(storage::encode(&refreshed).unwrap()), false);
    assert_eq!(ledger.prune_run_telemetry_before(cutoff).await.unwrap(), 1);
    assert_eq!(boundary.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        peer.read::<RunTelemetry>(TELEMETRY, "old")
            .await
            .unwrap()
            .unwrap()
            .doc,
        refreshed
    );
    assert!(
        peer.read::<RunTelemetry>(TELEMETRY, "other")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn native_retention_refusal_preserves_all_candidates_and_source() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("a").unwrap()).unwrap();
    let peer = DriverRunLedger::new(native.clone());
    let cutoff = now();
    for id in ["old", "other"] {
        seed(&peer, id, cutoff - Duration::days(1)).await;
    }
    let (ledger, _) = wrap(&native, None, true);
    let error = ledger.prune_run_telemetry_before(cutoff).await.unwrap_err();
    assert!(
        matches!(&error, super::super::LedgerError::Driver(native) if native.kind() == ErrorKind::Unavailable)
    );
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "retained native source"
    );
    for id in ["old", "other"] {
        assert!(
            peer.read::<RunTelemetry>(TELEMETRY, id)
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[tokio::test]
async fn empty_retention_still_requires_native_admission() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("a").unwrap()).unwrap();
    let (ledger, boundary) = wrap(&native, None, true);
    let error = ledger.prune_run_telemetry_before(now()).await.unwrap_err();
    assert!(
        matches!(error, super::super::LedgerError::Driver(native) if native.kind() == ErrorKind::Unavailable)
    );
    assert_eq!(boundary.calls.load(Ordering::SeqCst), 1);
}
