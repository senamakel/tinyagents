use super::*;
use crate::testkit::MemoryStorage;
use serde_json::Value;
use tinystoragedrivers_api::StorageBackend;

#[tokio::test]
async fn retained_and_cloned_handles_follow_rebind_without_crossing_agent_scopes() {
    let registry = SessionStoreRegistry::default();
    let old_backend = Arc::new(MemoryStorage::new());
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(old_backend.clone(), "old"),
        false,
    )
    .unwrap();
    let first = provider.for_agent("agent-one");
    let first_clone = first.clone();
    let second = provider.for_agent("agent-two");
    first
        .kv
        .put("prefs", "key", serde_json::json!("old-one"))
        .await
        .unwrap();
    second
        .kv
        .put("prefs", "key", serde_json::json!("old-two"))
        .await
        .unwrap();

    let new_backend = Arc::new(MemoryStorage::new());
    assert_eq!(
        registry
            .rebind(crate::testkit::fixed_scope_factory(
                new_backend.clone(),
                "new"
            ))
            .unwrap(),
        1
    );
    assert_eq!(first_clone.kv.get("prefs", "key").await.unwrap(), None);
    first_clone
        .kv
        .put("prefs", "key", serde_json::json!("new-one"))
        .await
        .unwrap();
    second
        .kv
        .put("prefs", "key", serde_json::json!("new-two"))
        .await
        .unwrap();

    assert_eq!(
        first.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("new-one"))
    );
    assert_eq!(
        second.kv.get("prefs", "key").await.unwrap(),
        Some(serde_json::json!("new-two"))
    );
    let fresh_driver =
        DriverSessionStores::new(crate::testkit::fixed_scope_factory(new_backend, "new")).unwrap();
    assert_eq!(
        fresh_driver
            .for_agent("agent-one")
            .kv
            .get("prefs", "key")
            .await
            .unwrap(),
        Some(serde_json::json!("new-one"))
    );
    let old_driver =
        DriverSessionStores::new(crate::testkit::fixed_scope_factory(old_backend, "old")).unwrap();
    assert_eq!(
        old_driver
            .for_agent("agent-one")
            .kv
            .get("prefs", "key")
            .await
            .unwrap(),
        Some(serde_json::json!("old-one"))
    );
}

#[tokio::test]
async fn retained_journal_handle_routes_later_appends_to_new_backend() {
    let registry = SessionStoreRegistry::default();
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(Arc::new(MemoryStorage::new()), "initial"),
        false,
    )
    .unwrap();
    let stores = provider.for_agent("agent");
    stores
        .journal
        .append("turns", serde_json::json!({"where":"old"}))
        .await
        .unwrap();
    provider
        .rebind(crate::testkit::fixed_scope_factory(
            Arc::new(MemoryStorage::new()),
            "replacement",
        ))
        .unwrap();
    assert_eq!(stores.journal.len("turns").await.unwrap(), 0);
    stores
        .journal
        .append("turns", serde_json::json!({"where":"new"}))
        .await
        .unwrap();
    assert_eq!(stores.journal.len("turns").await.unwrap(), 1);
}

struct BlockingStore {
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

#[derive(Debug)]
struct GateBackend {
    inner: MemoryStorage,
    gate: Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>>,
}

#[async_trait::async_trait]
impl StorageBackend for GateBackend {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    fn for_scope(
        &self,
        scope: &tinystoragedrivers_types::Scope,
    ) -> std::result::Result<tinyagents_storage_kit::Storage, StorageError> {
        if let Some((entered, release)) = self.gate.lock().unwrap().take() {
            let _ = entered.send(());
            let _ = release.recv();
        }
        tinystoragedrivers_api::StorageBackend::for_scope(&self.inner, scope)
    }
}

#[async_trait::async_trait]
impl Store for BlockingStore {
    async fn get(&self, _: &str, _: &str) -> HarnessResult<Option<Value>> {
        if let Some(entered) = self.entered.lock().unwrap().take() {
            let _ = entered.send(());
        }
        let release = { self.release.lock().unwrap().take() };
        if let Some(release) = release {
            let _ = release.await;
        }
        Ok(Some(serde_json::json!("old-operation")))
    }
    async fn put(&self, _: &str, _: &str, _: Value) -> HarnessResult<()> {
        Ok(())
    }
    async fn delete(&self, _: &str, _: &str) -> HarnessResult<()> {
        Ok(())
    }
    async fn list(&self, _: &str) -> HarnessResult<Vec<String>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn rebind_during_a_blocked_store_call_keeps_that_call_old_and_routes_the_next_new() {
    let base_provider = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        Arc::new(MemoryStorage::new()),
        "local",
    ))
    .unwrap();
    let mut old = base_provider.for_agent("agent");
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    old.kv = Arc::new(BlockingStore {
        entered: Mutex::new(Some(entered_tx)),
        release: Mutex::new(Some(release_rx)),
    });
    let binding = Arc::new(LiveAgentStores {
        current: RwLock::new(old),
    });
    let operation = {
        let store = LiveStore(Arc::clone(&binding));
        tokio::spawn(async move { store.get("n", "k").await.unwrap() })
    };
    entered_rx.await.unwrap();

    let mut next = base_provider.for_agent("agent");
    next.kv = Arc::new(tinyagents_harness::store::InMemoryStore::new());
    next.kv
        .put("n", "k", serde_json::json!("new-operation"))
        .await
        .unwrap();
    *binding.current.write().unwrap() = next;
    release_tx.send(()).unwrap();

    assert_eq!(
        operation.await.unwrap(),
        Some(serde_json::json!("old-operation"))
    );
    assert_eq!(
        LiveStore(binding).get("n", "k").await.unwrap(),
        Some(serde_json::json!("new-operation"))
    );
}

#[test]
fn registration_waits_for_rebind_snapshot_and_then_binds_to_the_new_backend() {
    let registry = SessionStoreRegistry::default();
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(Arc::new(MemoryStorage::new()), "initial"),
        false,
    )
    .unwrap();
    let _already_live = provider.for_agent("already-live");
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let new_backend = Arc::new(GateBackend {
        inner: MemoryStorage::new(),
        gate: Mutex::new(Some((entered_tx, release_rx))),
    });
    let rebind_provider = Arc::clone(&provider);
    let rebind_backend: Arc<dyn StorageBackend> = new_backend.clone();
    let rebind = std::thread::spawn(move || {
        rebind_provider
            .rebind(crate::testkit::fixed_scope_factory(rebind_backend, "gated"))
            .unwrap()
    });

    // The gate is reached while rebinding the already-live agent, after the
    // provider has captured its live registry. A registration at this point
    // must not be able to slip onto the old provider.
    entered_rx.recv().unwrap();
    assert!(provider.live.try_lock().is_err());
    release_tx.send(()).unwrap();
    rebind.join().unwrap();

    let late = provider.for_agent("late-agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        late.kv
            .put("n", "k", serde_json::json!("new"))
            .await
            .unwrap();
        let direct =
            DriverSessionStores::new(crate::testkit::fixed_scope_factory(new_backend, "gated"))
                .unwrap();
        assert_eq!(
            direct
                .for_agent("late-agent")
                .kv
                .get("n", "k")
                .await
                .unwrap(),
            Some(serde_json::json!("new"))
        );
    });
}

#[test]
fn concurrent_registration_and_rebind_leave_all_retained_handles_on_current_backend() {
    let registry = SessionStoreRegistry::default();
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(Arc::new(MemoryStorage::new()), "initial"),
        false,
    )
    .unwrap();
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let registration_provider = Arc::clone(&provider);
    let registration_barrier = Arc::clone(&barrier);
    let registration = std::thread::spawn(move || {
        registration_barrier.wait();
        registration_provider.for_agent("racing-agent")
    });
    let rebind_provider = Arc::clone(&provider);
    let rebind_barrier = Arc::clone(&barrier);
    let new_backend = Arc::new(MemoryStorage::new());
    let expected_backend = Arc::clone(&new_backend);
    let rebind = std::thread::spawn(move || {
        rebind_barrier.wait();
        rebind_provider
            .rebind(crate::testkit::fixed_scope_factory(new_backend, "new"))
            .unwrap();
    });
    barrier.wait();
    let registered = registration.join().unwrap();
    rebind.join().unwrap();

    let fresh = provider.for_agent("racing-agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        registered
            .kv
            .put("n", "k", serde_json::json!("retained"))
            .await
            .unwrap();
        let independent =
            DriverSessionStores::new(crate::testkit::fixed_scope_factory(expected_backend, "new"))
                .unwrap()
                .for_agent("racing-agent");
        assert_eq!(
            independent.kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("retained"))
        );
        assert_eq!(
            fresh.kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("retained"))
        );
    });
}

#[test]
fn concurrent_rebinds_publish_one_consistent_backend_for_existing_and_new_handles() {
    let registry = SessionStoreRegistry::default();
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(Arc::new(MemoryStorage::new()), "initial"),
        false,
    )
    .unwrap();
    let retained = provider.for_agent("agent");
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let backends = [
        Arc::new(MemoryStorage::new()),
        Arc::new(MemoryStorage::new()),
    ];
    let candidate_backends = backends.clone();
    let mut workers = Vec::new();
    for backend in &backends {
        let provider = Arc::clone(&provider);
        let barrier = Arc::clone(&barrier);
        let backend = Arc::clone(backend);
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            provider
                .rebind(crate::testkit::fixed_scope_factory(backend, "candidate"))
                .unwrap();
        }));
    }
    barrier.wait();
    for worker in workers {
        worker.join().unwrap();
    }
    let fresh = provider.for_agent("agent");
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        retained
            .kv
            .put("n", "k", serde_json::json!("same-backend"))
            .await
            .unwrap();
        let mut winner = None;
        for (index, backend) in candidate_backends.into_iter().enumerate() {
            let direct =
                DriverSessionStores::new(crate::testkit::fixed_scope_factory(backend, "candidate"))
                    .unwrap();
            if direct
                .for_agent("agent")
                .kv
                .get("n", "k")
                .await
                .unwrap()
                .is_some()
            {
                winner = Some(index);
            }
        }
        let winner = winner.expect("one concurrent rebind backend has the retained write");
        fresh
            .kv
            .put("n", "fresh", serde_json::json!("current"))
            .await
            .unwrap();
        let direct = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
            backends[winner].clone(),
            "candidate",
        ))
        .unwrap();
        assert_eq!(
            direct.for_agent("agent").kv.get("n", "k").await.unwrap(),
            Some(serde_json::json!("same-backend"))
        );
        assert_eq!(
            direct
                .for_agent("agent")
                .kv
                .get("n", "fresh")
                .await
                .unwrap(),
            Some(serde_json::json!("current"))
        );
    });
}

#[tokio::test]
async fn host_supplied_scope_bindings_survive_rebind_without_crossing_scopes() {
    let registry = SessionStoreRegistry::default();
    let provider = RebindableSessionStores::new_in(
        &registry,
        crate::testkit::fixed_scope_factory(Arc::new(MemoryStorage::new()), "host-alice"),
        false,
    )
    .unwrap();
    let retained = provider.for_agent("agent");
    let next = Arc::new(MemoryStorage::new());
    registry
        .rebind(crate::testkit::fixed_scope_factory(
            next.clone(),
            "host-alice-rebound",
        ))
        .unwrap();
    retained
        .kv
        .put("n", "k", serde_json::json!("alice"))
        .await
        .unwrap();
    let alice = DriverSessionStores::new(crate::testkit::fixed_scope_factory(
        next.clone(),
        "host-alice-rebound",
    ))
    .unwrap()
    .for_agent("agent");
    let bob = DriverSessionStores::new(crate::testkit::fixed_scope_factory(next.clone(), "u:bob"))
        .unwrap()
        .for_agent("agent");
    let local = DriverSessionStores::new(crate::testkit::fixed_scope_factory(next, "local"))
        .unwrap()
        .for_agent("agent");
    assert_eq!(
        alice.kv.get("n", "k").await.unwrap(),
        Some(serde_json::json!("alice"))
    );
    assert_eq!(bob.kv.get("n", "k").await.unwrap(), None);
    assert_eq!(local.kv.get("n", "k").await.unwrap(), None);
}
