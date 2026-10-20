use super::*;
use crate::testkit::MemoryStorage;
use crate::testkit::conformance::{session_store_conformance, session_store_isolation_conformance};
use crate::turn_state::{TurnLifecycle, TurnState};
use tinystoragedrivers_api::StorageBackend;

fn storage_factory(backend: Arc<dyn StorageBackend>) -> AgentStorageFactory {
    let bindings = [
        ("plain", "test-bound-plain"),
        ("a", "test-bound-a"),
        ("b", "test-bound-b"),
        ("agent", "test-bound-agent"),
        ("agent-1", "test-bound-agent-1"),
        ("other", "test-bound-other"),
        ("alice", "test-bound-alice"),
        ("bob", "test-bound-bob"),
        ("prepared-actor", "test-bound-prepared-actor"),
        ("contract-store-agent", "test-bound-contract-store"),
        ("contract-iso-a", "test-bound-contract-iso-a"),
        ("contract-iso-b", "test-bound-contract-iso-b"),
    ]
    .into_iter()
    .map(|(agent, label)| (agent.to_owned(), Scope::new(label).unwrap()))
    .collect();
    crate::testkit::mapped_storage_factory(backend, bindings)
}

fn memory() -> DriverSessionStores {
    DriverSessionStores::new(storage_factory(Arc::new(MemoryStorage::new()))).unwrap()
}

#[test]
fn bootstrap_manifest_contains_all_session_document_indexes() {
    use tinystoragedrivers_types::IndexSpec;
    let specs = DriverSessionStores::collection_specs();
    assert_eq!(specs.len(), 5);
    assert_eq!(specs[0].name, "session.transcript_entries");
    assert_eq!(
        specs[0].indexes,
        vec![
            IndexSpec::new("by_stem", ["stem", "seq"]),
            IndexSpec::new("by_stem_written", ["stem", "written"]),
        ]
    );
    assert_eq!(specs[1].name, "session.transcripts");
    assert_eq!(
        specs[1].indexes,
        vec![
            IndexSpec::new("by_thread", ["thread_id", "created_ms"]),
            IndexSpec::new("by_agent_name", ["agent_name", "created_ms"]),
            IndexSpec::new("by_agent_id", ["agent_id", "created_ms"]),
        ]
    );
    assert_eq!(specs[2].name, "session.turn_states");
    assert_eq!(
        specs[2].indexes,
        vec![IndexSpec::new("by_thread", ["thread_id"])]
    );
    assert_eq!(specs[3], DriverStore::collection_spec());
    assert_eq!(specs[4].name, "tsd.leases", "generation reservations");
    for spec in specs {
        spec.validate().unwrap();
        assert!(spec.ttl_field.is_none());
        assert!(spec.search.is_none());
    }
}

#[tokio::test]
async fn bootstrap_preparation_is_idempotent_and_preserves_native_session_data() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    {
        let scoped = backend
            .for_scope(&Scope::new("host-prepared-test").unwrap())
            .unwrap();
        DriverSessionStores::prepare_schema(scoped.documents().as_ref())
            .await
            .unwrap();
        scoped
            .documents()
            .put(
                "tsd.kv",
                "bootstrap-control",
                serde_json::json!({"owner": "session", "ns": "control", "key": "k", "value": "retained"}),
                tinystoragedrivers_types::Precondition::Absent,
            )
            .await
            .unwrap();
        DriverSessionStores::prepare_schema(scoped.documents().as_ref())
            .await
            .unwrap();
        assert_eq!(
            scoped
                .documents()
                .get("tsd.kv", "bootstrap-control")
                .await
                .unwrap()
                .unwrap()
                .doc["value"],
            "retained"
        );
        let provider = DriverSessionStores::new(storage_factory(Arc::clone(&backend))).unwrap();
        session_store_conformance(&provider).await;
        session_store_isolation_conformance(&provider).await;
    }
}

#[tokio::test]
async fn memory_backed_stores_meet_the_session_store_contract() {
    session_store_conformance(&memory()).await;
}

#[tokio::test]
async fn memory_backed_stores_keep_agents_apart() {
    session_store_isolation_conformance(&memory()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn stores_work_from_a_multi_threaded_runtime() {
    session_store_conformance(&memory()).await;
}

#[test]
fn stores_work_outside_any_runtime() {
    let stores = memory().for_agent("plain");
    let turn = TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z");
    stores.turn_states.put(&turn).unwrap();
    assert_eq!(stores.turn_states.get("t").unwrap(), Some(turn));
}

#[test]
fn host_bound_factory_keeps_agents_in_their_supplied_scopes() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let bindings = [
        (
            "agent-a".to_owned(),
            Scope::new("host-selected-alpha").unwrap(),
        ),
        (
            "agent-b".to_owned(),
            Scope::new("host-selected-beta").unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    let provider = DriverSessionStores::new(crate::testkit::mapped_storage_factory(
        Arc::clone(&backend),
        bindings,
    ))
    .unwrap();
    let alice = provider.for_agent("agent-a");
    let bob = provider.for_agent("agent-b");
    alice
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();
    assert!(bob.turn_states.get("t").unwrap().is_none());
    assert!(alice.turn_states.get("t").unwrap().is_some());
}

#[test]
fn two_backends_never_share_a_destination() {
    let one = memory().for_agent("a");
    let two = memory().for_agent("a");
    assert_ne!(
        one.transcripts.destination_key(),
        two.transcripts.destination_key()
    );
}

#[test]
fn an_agent_gets_the_same_stores_every_time() {
    let provider = memory();
    let first = provider.for_agent("a");
    let again = provider.for_agent("a");
    assert!(Arc::ptr_eq(&first.turn_states, &again.turn_states));
    assert_eq!(
        first.transcripts.destination_key(),
        again.transcripts.destination_key()
    );
    assert_ne!(
        first.transcripts.destination_key(),
        provider.for_agent("b").transcripts.destination_key()
    );
    let debug = format!("{provider:?}");
    assert!(debug.contains("agents: 2"), "{debug}");
    assert!(
        provider
            .destination_key()
            .is_some_and(|key| key.starts_with("storage://"))
    );
}

#[test]
fn data_survives_a_new_provider_on_the_same_backend() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let turn = TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z");
    DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::clone(&backend),
        "local",
    ))
    .unwrap()
    .for_agent("a")
    .turn_states
    .put(&turn)
    .unwrap();
    let reopened = DriverSessionStores::new(crate::testkit::fixed_scope_factory(backend, "local"))
        .unwrap()
        .for_agent("a");
    assert_eq!(reopened.turn_states.get("t").unwrap(), Some(turn));
}

#[test]
fn recover_marks_open_agents_turns_interrupted() {
    let provider = memory();
    let stores = provider.for_agent("a");
    stores
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();
    provider.recover().unwrap();
    assert_eq!(
        stores.turn_states.get("t").unwrap().unwrap().lifecycle,
        TurnLifecycle::Interrupted
    );
}

#[test]
fn recover_on_open_interrupts_turns_an_earlier_process_left() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let earlier = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::clone(&backend),
        "local",
    ))
    .unwrap();
    earlier
        .for_agent("a")
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();

    let plain = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::clone(&backend),
        "local",
    ))
    .unwrap();
    assert_eq!(
        plain
            .for_agent("a")
            .turn_states
            .get("t")
            .unwrap()
            .unwrap()
            .lifecycle,
        TurnLifecycle::Started,
        "without the option nothing is touched"
    );

    let recovering =
        DriverSessionStores::new(crate::testkit::fixed_scope_factory(backend, "local"))
            .unwrap()
            .recover_on_open(true);
    assert!(format!("{recovering:?}").contains("recover_on_open: true"));
    let stores = recovering.for_agent("a");
    assert_eq!(
        stores.turn_states.get("t").unwrap().unwrap().lifecycle,
        TurnLifecycle::Interrupted
    );

    // Only the first open recovers: a turn started afterwards stays live.
    stores
        .turn_states
        .put(&TurnState::started("t2", "r", 8, "2026-01-01T00:01:00Z"))
        .unwrap();
    recovering.agents.lock().unwrap().clear();
    assert_eq!(
        recovering
            .for_agent("a")
            .turn_states
            .get("t2")
            .unwrap()
            .unwrap()
            .lifecycle,
        TurnLifecycle::Started
    );
}

/// A backend that cannot bind any scope.
#[derive(Debug)]
struct Unbindable;

impl StorageBackend for Unbindable {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        tinystoragedrivers_types::Capabilities::default()
    }

    fn for_scope(&self, _scope: &Scope) -> Result<tinyagents_storage_kit::Storage, StorageError> {
        Err(StorageError::unavailable("database is down"))
    }
}

#[tokio::test]
async fn an_unbindable_scope_fails_closed() {
    let provider = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::new(Unbindable),
        "local",
    ))
    .unwrap();
    assert!(provider.try_for_agent("a").is_err());
    let stores = provider.for_agent("a");

    let error = stores
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap_err();
    assert!(error.contains("database is down"), "{error}");
    assert!(stores.turn_states.list().is_err());
    assert!(stores.kv.get("ns", "k").await.is_err());
    assert!(
        stores
            .journal
            .append("s", serde_json::json!(1))
            .await
            .is_err()
    );
    assert!(stores.journal.len("s").await.is_err());

    let session = crate::transcript::SessionRef::scoped("t", "a");
    assert!(!stores.transcripts.session_exists(&session));
    assert!(stores.transcripts.root_for_thread("t").is_none());
    assert!(stores.transcripts.latest_for_agent("a").is_none());
    let handle = stores
        .transcripts
        .open_stem("stem", transcripts::tests::meta("t"))
        .unwrap();
    assert!(handle.read_session().is_err());
    assert!(handle.messages().is_err());
    assert!(
        stores
            .transcripts
            .begin_generation(&session, transcripts::tests::meta("t"))
            .is_err()
    );

    assert!(
        provider.agents.lock().unwrap().is_empty(),
        "refusals are not cached"
    );
    provider.recover().unwrap();
}

/// A memory backend whose documents fail every query while `down` is set.
#[derive(Debug)]
struct Flaky {
    storage: MemoryStorage,
    down: Arc<std::sync::atomic::AtomicBool>,
    reject_declaration: bool,
}

#[derive(Debug)]
struct FlakyDocs {
    inner: Arc<dyn tinystoragedrivers_api::DocumentStore>,
    down: Arc<std::sync::atomic::AtomicBool>,
    reject_declaration: bool,
}

impl FlakyDocs {
    /// Injects a typed refusal while the recovery backend is unavailable.
    fn check(&self) -> Result<(), StorageError> {
        if self.down.load(std::sync::atomic::Ordering::SeqCst) {
            Err(StorageError::unavailable("flaky backend is down").with_source(NativeFailure))
        } else {
            Ok(())
        }
    }
}

#[async_trait::async_trait]
impl tinystoragedrivers_api::DocumentStore for FlakyDocs {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    /// Fails declaration when selected, otherwise delegates the native schema operation.
    async fn ensure_collection(
        &self,
        spec: &tinystoragedrivers_types::CollectionSpec,
    ) -> Result<(), StorageError> {
        if self.reject_declaration {
            self.check()?;
        }
        self.inner.ensure_collection(spec).await
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
        self.check()?;
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
    async fn atomic_batch_checked(
        &self,
        checks: Vec<tinystoragedrivers_types::ReadCheck>,
        writes: Vec<tinystoragedrivers_types::WriteOp>,
    ) -> Result<Vec<tinystoragedrivers_types::WriteResult>, StorageError> {
        self.inner.atomic_batch_checked(checks, writes).await
    }
}

impl StorageBackend for Flaky {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.storage.capabilities()
    }
    fn for_scope(&self, scope: &Scope) -> Result<tinyagents_storage_kit::Storage, StorageError> {
        let inner = tinystoragedrivers_api::StorageBackend::for_scope(&self.storage, scope)?;
        Ok(tinyagents_storage_kit::Storage::new(
            Arc::new(FlakyDocs {
                inner: Arc::clone(inner.documents()),
                down: Arc::clone(&self.down),
                reject_declaration: self.reject_declaration,
            }),
            Arc::clone(inner.streams()),
            Arc::clone(inner.kv()),
            Arc::clone(inner.blobs()),
            Arc::clone(inner.queues()),
            Arc::clone(inner.secrets()),
        ))
    }
}

#[test]
fn a_failed_recovery_fails_closed_and_is_retried() {
    let down = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let provider = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::new(Flaky {
            storage: MemoryStorage::new(),
            down: Arc::clone(&down),
            reject_declaration: false,
        }),
        "local",
    ))
    .unwrap()
    .recover_on_open(true);

    let error = provider.try_for_agent("a").unwrap_err();
    assert!(error.message().contains("flaky backend is down"), "{error}");
    let refused = provider.for_agent("a");
    assert!(
        refused
            .turn_states
            .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
            .is_err(),
        "no turn can start before recovery ran"
    );
    assert!(provider.agents.lock().unwrap().is_empty());

    down.store(false, std::sync::atomic::Ordering::SeqCst);
    let stores = provider.for_agent("a");
    stores
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();
    assert_eq!(
        provider
            .for_agent("a")
            .turn_states
            .get("t")
            .unwrap()
            .unwrap()
            .lifecycle,
        TurnLifecycle::Started,
        "a recovered agent is never swept again"
    );
}

#[derive(Debug)]
struct NativeFailure;
impl std::fmt::Display for NativeFailure {
    /// Displays the marker without changing its native error identity.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native failure marker")
    }
}
impl std::error::Error for NativeFailure {}

#[test]
/// Pins native scope error and source identity through object-safe checked opening.
fn checked_open_preserves_native_scope_refusal_through_dyn_provider() {
    use std::error::Error;
    let marker = Arc::new(NativeFailure);
    let provider: Arc<dyn SessionStoreProvider> = Arc::new(
        DriverSessionStores::new({
            let marker = marker.clone();
            Arc::new(move |_| {
                Err(StorageError::unavailable("scope rejected").with_source(marker.clone()))
            })
        })
        .unwrap(),
    );
    // Unusual keys get the hashed principal segment, and are refused alike.
    for key in ["agent", "unusual key", "sha256:reserved"] {
        let error = provider.try_for_agent(key).unwrap_err();
        assert_eq!(
            error.kind(),
            tinystoragedrivers_types::ErrorKind::Unavailable
        );
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<Arc<NativeFailure>>()
            .unwrap();
        assert!(Arc::ptr_eq(source, &marker));
    }
}

#[test]
/// Pins native query errors, uncached failure and successful recovery retry.
fn checked_open_preserves_native_recovery_query_refusal_and_retries() {
    use std::error::Error;
    let down = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let provider = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::new(Flaky {
            storage: MemoryStorage::new(),
            down: down.clone(),
            reject_declaration: false,
        }),
        "local",
    ))
    .unwrap()
    .recover_on_open(true);
    let error = provider.try_for_agent("agent").unwrap_err();
    assert_eq!(
        error.kind(),
        tinystoragedrivers_types::ErrorKind::Unavailable
    );
    assert!(error.source().unwrap().is::<NativeFailure>());
    assert!(provider.agents.lock().unwrap().is_empty());
    assert!(provider.recovered.lock().unwrap().is_empty());
    down.store(false, std::sync::atomic::Ordering::SeqCst);
    let opened = provider.try_for_agent("agent").unwrap();
    let cached = provider.try_for_agent("agent").unwrap();
    assert!(Arc::ptr_eq(&opened.turn_states, &cached.turn_states));
}

#[test]
/// Pins native declaration errors before turn recovery can proceed.
fn checked_open_preserves_native_recovery_declaration_refusal() {
    use std::error::Error;
    let provider = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::new(Flaky {
            storage: MemoryStorage::new(),
            down: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            reject_declaration: true,
        }),
        "local",
    ))
    .unwrap()
    .recover_on_open(true);
    let error = provider.try_for_agent("agent").unwrap_err();
    assert_eq!(
        error.kind(),
        tinystoragedrivers_types::ErrorKind::Unavailable
    );
    assert!(error.source().unwrap().is::<NativeFailure>());
    assert!(provider.agents.lock().unwrap().is_empty());
}

#[test]
/// Pins cache reuse, actor cache independence and durable scope isolation.
fn checked_open_caches_success_but_actor_rebinding_has_independent_handles() {
    use std::error::Error;
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let provider: Arc<dyn SessionStoreProvider> = Arc::new(
        DriverSessionStores::new(crate::testkit::fixed_scope_factory(
            backend.clone(),
            "local",
        ))
        .unwrap(),
    );
    let first = provider.try_for_agent("agent").unwrap();
    let cached = provider.try_for_agent("agent").unwrap();
    assert!(Arc::ptr_eq(&first.turn_states, &cached.turn_states));
    let rebound = provider
        .bind_actor(crate::testkit::fixed_scope_factory(backend, "local"))
        .unwrap()
        .unwrap();
    let independent = rebound.try_for_agent("agent").unwrap();
    assert!(!Arc::ptr_eq(&first.turn_states, &independent.turn_states));
    first
        .turn_states
        .put(&TurnState::started("t", "r", 8, "2026-01-01T00:00:00Z"))
        .unwrap();
    assert!(independent.turn_states.get("t").unwrap().is_some());
    assert!(
        rebound
            .try_for_agent("other")
            .unwrap()
            .turn_states
            .get("t")
            .unwrap()
            .is_none()
    );

    let marker = Arc::new(NativeFailure);
    let refused = provider
        .bind_actor({
            let marker = marker.clone();
            Arc::new(move |_| {
                Err(StorageError::unavailable("scope rejected").with_source(marker.clone()))
            })
        })
        .unwrap()
        .unwrap();
    let error = refused.try_for_agent("agent").unwrap_err();
    assert!(Arc::ptr_eq(
        error
            .source()
            .unwrap()
            .downcast_ref::<Arc<NativeFailure>>()
            .unwrap(),
        &marker
    ));
    assert!(Arc::ptr_eq(
        &first.turn_states,
        &provider.try_for_agent("agent").unwrap().turn_states
    ));
}

#[tokio::test]
async fn native_provider_returns_stable_bound_ledger_and_rebinds_actor_independently() {
    let provider =
        DriverSessionStores::new(storage_factory(Arc::new(MemoryStorage::new()))).unwrap();
    let one = provider.run_ledger_for_agent("alice").unwrap().unwrap();
    let same = provider.run_ledger_for_agent("alice").unwrap().unwrap();
    assert!(Arc::ptr_eq(&one, &same));
    one.append_run_event(crate::run_ledger::RunEventAppend {
        run_id: "run".into(),
        event_type: "one".into(),
        payload: serde_json::json!({}),
    })
    .await
    .unwrap();
    let bob = provider.run_ledger_for_agent("bob").unwrap().unwrap();
    assert_eq!(
        bob.list_recent_run_events(&crate::run_ledger::RunEventListRequest {
            run_id: "run".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .count,
        0
    );
    let actor = provider
        .bind_actor(storage_factory(Arc::new(MemoryStorage::new())))
        .unwrap()
        .unwrap();
    let rebound = actor.run_ledger_for_agent("alice").unwrap().unwrap();
    assert!(!Arc::ptr_eq(&one, &rebound));
    assert_eq!(
        rebound
            .list_recent_run_events(&crate::run_ledger::RunEventListRequest {
                run_id: "run".into(),
                ..Default::default()
            })
            .await
            .unwrap()
            .count,
        0
    );
}

async fn prepared_restricted_actor_contract(backend: Arc<dyn StorageBackend>) {
    let factory = storage_factory(Arc::clone(&backend));
    let bootstrap = factory("prepared-actor").unwrap();
    DriverSessionStores::prepare_schema(bootstrap.documents().as_ref())
        .await
        .unwrap();
    let actor =
        DriverSessionStores::new_with_schema_admission(factory, SchemaAdmission::ValidateOnly)
            .unwrap()
            .recover_on_open(true);
    let opened = actor.try_for_agent("prepared-actor");
    assert!(
        opened.is_ok(),
        "prepared restricted actor recovery refused: {:?}",
        opened.err()
    );
    session_store_conformance(&actor).await;
    session_store_isolation_conformance(&actor).await;
}

#[tokio::test]
async fn prepared_restricted_actor_recovers_and_uses_memory_stores() {
    prepared_restricted_actor_contract(Arc::new(MemoryStorage::new())).await;
}

#[tokio::test]
async fn the_journal_lives_in_the_harness_journal_stream_family() {
    let backend = MemoryStorage::new();
    let provider = DriverSessionStores::new(storage_factory(Arc::new(backend.clone()))).unwrap();
    let stores = provider.for_agent("agent-1");
    stores
        .journal
        .append("run.abc", serde_json::json!({"kind": "x"}))
        .await
        .unwrap();

    let scoped = backend
        .for_scope(&Scope::new("test-bound-agent-1").unwrap())
        .unwrap();
    assert_eq!(
        scoped.streams().streams("").await.unwrap(),
        vec!["harness.journal/run.abc".to_owned()]
    );
    stores.journal.seal("run.abc").await.unwrap();
    assert!(
        scoped
            .streams()
            .info("harness.journal/run.abc")
            .await
            .unwrap()
            .unwrap()
            .sealed_ms
            .is_some()
    );
}
