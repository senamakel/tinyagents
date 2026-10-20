use super::*;
use crate::testkit::MemoryStorage;
use crate::testkit::conformance::{session_store_conformance, session_store_isolation_conformance};
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyagents_storage_kit::{CollectionSpec, DocumentStore, ErrorKind};
use tinystoragedrivers_api::StorageBackend;

#[derive(Debug, Default)]
struct Observations {
    declarations: AtomicUsize,
    validated: Mutex<Vec<CollectionSpec>>,
    refusal: AtomicUsize,
}

#[derive(Debug)]
struct ProbeDocs {
    inner: Arc<dyn DocumentStore>,
    observed: Arc<Observations>,
}

#[derive(Debug, thiserror::Error)]
#[error("original retired actor")]
struct Retired;

#[async_trait::async_trait]
impl DocumentStore for ProbeDocs {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<(), StorageError> {
        self.observed.declarations.fetch_add(1, Ordering::SeqCst);
        self.inner.ensure_collection(spec).await
    }
    async fn validate_collection_spec(&self, spec: &CollectionSpec) -> Result<(), StorageError> {
        self.observed.validated.lock().unwrap().push(spec.clone());
        match self.observed.refusal.load(Ordering::SeqCst) {
            1 => Err(StorageError::unsupported(
                tinystoragedrivers_types::Capability::CollectionValidation,
                "unsupported adapter",
            )),
            2 => Err(StorageError::unavailable("retired actor").with_source(Retired)),
            _ => self.inner.validate_collection_spec(spec).await,
        }
    }
    async fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<tinystoragedrivers_types::Versioned<serde_json::Value>>, StorageError> {
        self.inner.get(collection, id).await
    }
    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        precondition: tinystoragedrivers_types::Precondition,
    ) -> Result<tinystoragedrivers_types::Version, StorageError> {
        self.inner.put(collection, id, doc, precondition).await
    }
    async fn put_with(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        precondition: tinystoragedrivers_types::Precondition,
        options: tinystoragedrivers_types::WriteOptions,
    ) -> Result<tinystoragedrivers_types::Version, StorageError> {
        self.inner
            .put_with(collection, id, doc, precondition, options)
            .await
    }
    async fn delete(
        &self,
        collection: &str,
        id: &str,
        precondition: tinystoragedrivers_types::Precondition,
    ) -> Result<bool, StorageError> {
        self.inner.delete(collection, id, precondition).await
    }
    async fn query(
        &self,
        collection: &str,
        query: &tinystoragedrivers_types::Query,
    ) -> Result<
        tinystoragedrivers_types::Page<tinystoragedrivers_types::Versioned<serde_json::Value>>,
        StorageError,
    > {
        self.inner.query(collection, query).await
    }
    async fn count(
        &self,
        collection: &str,
        filter: &tinystoragedrivers_types::Filter,
    ) -> Result<u64, StorageError> {
        self.inner.count(collection, filter).await
    }
    async fn delete_where(
        &self,
        collection: &str,
        filter: &tinystoragedrivers_types::Filter,
    ) -> Result<u64, StorageError> {
        self.inner.delete_where(collection, filter).await
    }
    async fn claim(
        &self,
        collection: &str,
        filter: &tinystoragedrivers_types::Filter,
        sort: &[tinystoragedrivers_types::Sort],
        patch: &serde_json::Value,
    ) -> Result<Option<tinystoragedrivers_types::Versioned<serde_json::Value>>, StorageError> {
        self.inner.claim(collection, filter, sort, patch).await
    }
    async fn drop_collection(&self, collection: &str) -> Result<(), StorageError> {
        self.inner.drop_collection(collection).await
    }
}

#[derive(Debug)]
struct ProbeBackend {
    inner: Arc<dyn StorageBackend>,
    observed: Arc<Observations>,
}
impl StorageBackend for ProbeBackend {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    fn for_scope(&self, scope: &Scope) -> Result<tinyagents_storage_kit::Storage, StorageError> {
        let scoped = self.inner.for_scope(scope)?;
        Ok(tinyagents_storage_kit::Storage::new(
            Arc::new(ProbeDocs {
                inner: scoped.documents().clone(),
                observed: self.observed.clone(),
            }),
            scoped.streams().clone(),
            scoped.kv().clone(),
            scoped.blobs().clone(),
            scoped.queues().clone(),
            scoped.secrets().clone(),
        ))
    }
}

async fn prepared() -> (Arc<dyn StorageBackend>, Arc<Observations>) {
    let storage: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    storage.prepare_schema().await.unwrap();
    DriverSessionStores::prepare_schema(
        storage
            .for_scope(&Scope::new("local").unwrap())
            .unwrap()
            .documents()
            .as_ref(),
    )
    .await
    .unwrap();
    let observed = Arc::new(Observations::default());
    (
        Arc::new(ProbeBackend {
            inner: storage.existing_schema().unwrap(),
            observed: observed.clone(),
        }),
        observed,
    )
}

#[tokio::test]
async fn validation_only_conformance_never_attempts_schema_declaration() {
    let (backend, observed) = prepared().await;
    let provider = DriverSessionStores::new_with_schema_admission(
        backend,
        Scope::new("local").unwrap(),
        SchemaAdmission::ValidateOnly,
    )
    .unwrap()
    .recover_on_open(true);
    session_store_conformance(&provider).await;
    session_store_isolation_conformance(&provider).await;
    assert_eq!(observed.declarations.load(Ordering::SeqCst), 0);
    let checked = observed.validated.lock().unwrap();
    for spec in DriverSessionStores::collection_specs() {
        assert!(
            checked.contains(&spec),
            "original declaration not validated: {spec:?}"
        );
    }
}

#[tokio::test]
async fn validation_only_refusals_keep_native_sources_and_retry_without_caching() {
    use std::error::Error;
    let (backend, observed) = prepared().await;
    let provider = DriverSessionStores::new_with_schema_admission(
        backend,
        Scope::new("local").unwrap(),
        SchemaAdmission::ValidateOnly,
    )
    .unwrap();
    for (refusal, kind) in [
        (
            1,
            ErrorKind::Unsupported(tinystoragedrivers_types::Capability::CollectionValidation),
        ),
        (2, ErrorKind::Unavailable),
    ] {
        observed.refusal.store(refusal, Ordering::SeqCst);
        let error = provider.try_for_agent("agent").unwrap_err();
        assert_eq!(error.kind(), kind);
        if refusal == 2 {
            assert!(error.source().unwrap().is::<Retired>());
        }
        assert!(provider.agents.lock().unwrap().is_empty());
        assert!(provider.recovered.lock().unwrap().is_empty());
    }
    observed.refusal.store(0, Ordering::SeqCst);
    let first = provider.try_for_agent("agent").unwrap();
    let cached = provider.try_for_agent("agent").unwrap();
    assert!(Arc::ptr_eq(&first.turn_states, &cached.turn_states));
    assert_eq!(observed.declarations.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn explicit_actor_factory_selects_validation_without_changing_original_provider() {
    let (backend, observed) = prepared().await;
    let original = Arc::new(
        DriverSessionStores::new(crate::testkit::fixed_scope_factory(
            backend.clone(),
            "local",
        ))
        .unwrap()
        .recover_on_open(true),
    );
    let provider: Arc<dyn SessionStoreProvider> = original.clone();
    let actor = provider
        .bind_actor_with_schema_admission(backend, SchemaAdmission::ValidateOnly)
        .unwrap()
        .unwrap();
    assert!(actor.try_for_agent("agent").is_ok());
    assert_eq!(observed.declarations.load(Ordering::SeqCst), 0);
    assert_eq!(original.schema_admission, SchemaAdmission::Declare);
    assert!(original.agents.lock().unwrap().is_empty());
    assert!(
        original.try_for_agent("agent").is_err(),
        "original declaration policy must still refuse restricted backend"
    );
    assert!(actor.try_for_agent("agent").is_ok());
}

#[tokio::test]
async fn validation_only_missing_or_incompatible_schema_refuses_before_recovery_or_cache() {
    for incompatible in [false, true] {
        let storage: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
        storage.prepare_schema().await.unwrap();
        let scoped = storage.for_scope(&Scope::new("local").unwrap()).unwrap();
        let mut specs = DriverSessionStores::collection_specs();
        let at = specs.iter().position(|spec| spec.name == "tsd.kv").unwrap();
        let required = specs.remove(at);
        for spec in specs {
            scoped.documents().ensure_collection(&spec).await.unwrap();
        }
        if incompatible {
            let mut wrong = required.clone();
            wrong.indexes.clear();
            scoped.documents().ensure_collection(&wrong).await.unwrap();
        }
        let provider = DriverSessionStores::new_with_schema_admission(
            storage.existing_schema().unwrap(),
            Scope::new("local").unwrap(),
            SchemaAdmission::ValidateOnly,
        )
        .unwrap()
        .recover_on_open(true);
        let error = provider.try_for_agent("agent").unwrap_err();
        assert_eq!(
            error.kind(),
            if incompatible {
                ErrorKind::InvalidInput
            } else {
                ErrorKind::NotFound
            }
        );
        assert!(provider.agents.lock().unwrap().is_empty());
        assert!(provider.recovered.lock().unwrap().is_empty());
        if !incompatible {
            scoped
                .documents()
                .ensure_collection(&required)
                .await
                .unwrap();
            assert!(
                provider.try_for_agent("agent").is_ok(),
                "failed validation must not poison retry"
            );
        }
    }
}

#[tokio::test]
async fn legacy_actor_factory_inherits_validation_policy_without_declaring_schema() {
    let (backend, observed) = prepared().await;
    let original: Arc<dyn SessionStoreProvider> = Arc::new(
        DriverSessionStores::new_with_schema_admission(
            backend.clone(),
            Scope::new("local").unwrap(),
            SchemaAdmission::ValidateOnly,
        )
        .unwrap()
        .recover_on_open(true),
    );
    let actor = original.bind_actor(backend.clone()).unwrap().unwrap();
    assert!(actor.try_for_agent("agent").is_ok());
    session_store_conformance(actor.as_ref()).await;
    let rebound = actor.bind_actor(backend).unwrap().unwrap();
    assert!(rebound.try_for_agent("agent").is_ok());
    assert_eq!(observed.declarations.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn validation_only_provider_ledger_uses_prepared_schema_without_declaration() {
    let storage: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    storage.prepare_schema().await.unwrap();
    crate::run_ledger::DriverRunLedger::prepare_schema(
        storage
            .for_scope(&Scope::new("local").unwrap())
            .unwrap()
            .documents()
            .as_ref(),
    )
    .await
    .unwrap();
    let observed = Arc::new(Observations::default());
    let backend = Arc::new(ProbeBackend {
        inner: storage.existing_schema().unwrap(),
        observed: observed.clone(),
    });
    let provider = DriverSessionStores::new_with_schema_admission(
        backend,
        Scope::new("local").unwrap(),
        SchemaAdmission::ValidateOnly,
    )
    .unwrap();
    let ledger = provider.run_ledger_for_agent("alice").unwrap().unwrap();
    let result = ledger
        .list_recent_run_events(&crate::run_ledger::RunEventListRequest {
            run_id: "run".into(),
            ..Default::default()
        })
        .await;
    assert_eq!(observed.declarations.load(Ordering::SeqCst), 0);
    assert_eq!(result.unwrap().count, 0);
    assert!(!observed.validated.lock().unwrap().is_empty());
}
