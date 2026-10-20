use super::*;
use crate::testkit::MemoryStorage;
use crate::turn_state::TurnState;

fn factory(storage: ScopedStorage) -> AgentStorageFactory {
    Arc::new(move |_| Ok(storage.clone()))
}

#[test]
fn actor_factory_uses_host_bound_storage_and_keeps_handles_independent() {
    let original_scope = Scope::new("host-bound-original").unwrap();
    let original_storage = MemoryStorage::new().for_scope(&original_scope).unwrap();
    let original = DriverSessionStores::new(factory(original_storage)).unwrap();
    let old = original.for_agent("arbitrary-agent-id");
    let old_state = TurnState::started("old", "old-r", 1, "2026-01-01T00:00:00Z");
    old.turn_states.put(&old_state).unwrap();

    let actor_scope = Scope::new("host-bound-actor").unwrap();
    let actor_storage = MemoryStorage::new().for_scope(&actor_scope).unwrap();
    let actor = original
        .bind_actor(factory(actor_storage))
        .unwrap()
        .expect("driver providers support host-bound actor storage");
    let fresh = actor.for_agent("arbitrary-agent-id");

    assert!(fresh.turn_states.get("old").unwrap().is_none());
    let fresh_state = TurnState::started("fresh", "fresh-r", 1, "2026-01-01T00:00:00Z");
    fresh.turn_states.put(&fresh_state).unwrap();
    assert!(old.turn_states.get("fresh").unwrap().is_none());
    assert_eq!(old.turn_states.get("old").unwrap(), Some(old_state));
    assert!(Arc::ptr_eq(
        &fresh.turn_states,
        &actor.for_agent("arbitrary-agent-id").turn_states
    ));
}

#[test]
fn arbitrary_provider_reports_actor_factory_absence_without_fallback() {
    let provider = crate::port::InMemorySessionStores::default();
    let scope = Scope::new("host-bound-test").unwrap();
    let storage = MemoryStorage::new().for_scope(&scope).unwrap();
    assert!(
        provider
            .bind_actor(factory(storage.clone()))
            .unwrap()
            .is_none()
    );

    let shared: Arc<dyn SessionStoreProvider> = Arc::new(provider);
    assert!(shared.bind_actor(factory(storage)).unwrap().is_none());
}
