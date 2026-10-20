use super::*;
use crate::testkit::MemoryStorage;
use crate::testkit::conformance::session_store_conformance;
use tinyagents_harness::store::{InMemoryAppendStore, InMemoryStore};

fn host_factory() -> AgentStorageFactory {
    let scope = tinyagents_storage_kit::Scope::new("host-bound-test").unwrap();
    let storage = MemoryStorage::new().for_scope(&scope).unwrap();
    Arc::new(move |_| Ok(storage.clone()))
}

/// A provider that shares one in-memory transcript locator between every
/// agent (with in-memory turn-state, key-value and journal stores), the way a
/// single-user host shares one store.
struct SharedStores {
    transcripts: Arc<InMemoryTranscriptLocator>,
}

impl Default for SharedStores {
    fn default() -> Self {
        Self {
            transcripts: Arc::new(InMemoryTranscriptLocator::new("shared-stores")),
        }
    }
}

impl SessionStoreProvider for SharedStores {
    fn for_agent(&self, _agent_id: &str) -> AgentStores {
        AgentStores {
            transcripts: self.transcripts.clone(),
            turn_states: Arc::new(InMemoryTurnStates::default()),
            kv: Arc::new(InMemoryStore::new()),
            journal: Arc::new(InMemoryAppendStore::new()),
        }
    }
}

#[tokio::test]
async fn the_shared_in_memory_building_blocks_meet_the_session_store_contract() {
    let provider = SharedStores::default();
    session_store_conformance(&provider).await;
    assert!(provider.recover().is_ok(), "recovery defaults to a no-op");
    assert!(provider.destination_key().is_none());
    assert!(
        provider.workspace_dir().is_none(),
        "only a provider that says so is file-backed"
    );
}

#[test]
fn the_in_memory_turn_states_serve_the_port() {
    let turns: Arc<dyn TurnStates> = Arc::new(InMemoryTurnStates::default());
    let state = TurnState::started("t", "r", 4, "2026-01-01T00:00:00Z");
    turns.put(&state).unwrap();
    assert!(turns.put_unless_completed(&state).unwrap());
    assert_eq!(turns.get("t").unwrap().unwrap().request_id, "r");
    assert_eq!(turns.list().unwrap().len(), 1);
    assert_eq!(
        turns.mark_all_interrupted("2026-01-01T00:01:00Z").unwrap(),
        1
    );
    assert!(turns.delete("t").unwrap());
    assert_eq!(turns.clear_all().unwrap(), 0);
}

#[test]
fn shared_providers_explicitly_leave_actor_rebinding_unsupported() {
    let provider = SharedStores::default();
    assert!(provider.bind_actor(host_factory()).unwrap().is_none());
}

#[test]
fn actor_factory_arc_forwarding_preserves_native_refusal_and_source() {
    #[derive(Debug, thiserror::Error)]
    #[error("actor admission retired")]
    struct Retired;
    struct RefusingProvider;
    impl SessionStoreProvider for RefusingProvider {
        fn for_agent(&self, id: &str) -> AgentStores {
            InMemorySessionStores::default().for_agent(id)
        }
        fn bind_actor(
            &self,
            _: AgentStorageFactory,
        ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError>
        {
            Err(
                tinystoragedrivers_types::StorageError::unavailable("actor refused")
                    .with_source(Retired),
            )
        }
    }
    let provider: Arc<dyn SessionStoreProvider> = Arc::new(RefusingProvider);
    let error = provider
        .bind_actor(host_factory())
        .err()
        .expect("configured refusal must remain an error");
    assert_eq!(
        error.kind(),
        tinystoragedrivers_types::ErrorKind::Unavailable
    );
    assert!(std::error::Error::source(&error).unwrap().is::<Retired>());
}

#[test]
/// Pins the default checked accessor for an unchanged file provider.
fn checked_accessor_keeps_legacy_provider_compatible() {
    let provider: &dyn SessionStoreProvider = &SharedStores::default();
    let stores = provider.try_for_agent("a").unwrap();
    assert!(stores.turn_states.list().unwrap().is_empty());
}

#[test]
/// Pins Arc forwarding rather than selecting the infallible default.
fn checked_accessor_arc_forwards_native_refusal() {
    use std::error::Error;
    #[derive(Debug)]
    struct Native;
    impl std::fmt::Display for Native {
        /// Displays the native test marker.
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("native marker")
        }
    }
    impl Error for Native {}
    struct Checked;
    impl SessionStoreProvider for Checked {
        /// Returns usable legacy stores to expose an incorrect default fallback.
        fn for_agent(&self, key: &str) -> AgentStores {
            InMemorySessionStores::default().for_agent(key)
        }
        /// Refuses checked opening with an identifiable native input error.
        fn try_for_agent(
            &self,
            _: &str,
        ) -> Result<AgentStores, tinystoragedrivers_types::StorageError> {
            Err(
                tinystoragedrivers_types::StorageError::invalid_input("invalid key")
                    .with_source(Native),
            )
        }
    }
    let provider = Arc::new(Checked);
    // Dispatch through the Arc implementation, rather than deref coercion.
    let erased: &dyn SessionStoreProvider = &provider;
    let error = erased.try_for_agent("invalid").unwrap_err();
    assert_eq!(
        error.kind(),
        tinystoragedrivers_types::ErrorKind::InvalidInput
    );
    assert!(error.source().unwrap().is::<Native>());
}

/// Providers opt into a ledger explicitly; an Arc forwards the selected handle.
#[test]
fn optional_run_ledger_provider_preserves_handle_and_defaults_to_none() {
    use crate::run_ledger::{DriverRunLedger, RunLedgerStore};
    use tinyagents_storage_kit::Scope;
    struct LedgerProvider {
        sessions: InMemorySessionStores,
        ledger: Arc<dyn RunLedgerStore>,
    }
    impl SessionStoreProvider for LedgerProvider {
        fn for_agent(&self, id: &str) -> AgentStores {
            self.sessions.for_agent(id)
        }
        fn run_ledger_for_agent(
            &self,
            id: &str,
        ) -> crate::run_ledger::LedgerResult<Option<Arc<dyn RunLedgerStore>>> {
            if id == "denied" {
                return Err(tinyagents_harness::TinyAgentsError::Storage(
                    "admission refused".into(),
                )
                .into());
            }
            Ok((id == "allowed").then(|| self.ledger.clone()))
        }
    }
    let ledger: Arc<dyn RunLedgerStore> = Arc::new(DriverRunLedger::new(
        MemoryStorage::new()
            .for_scope(&Scope::new("local").unwrap())
            .unwrap(),
    ));
    let provider = Arc::new(LedgerProvider {
        sessions: InMemorySessionStores::new(),
        ledger: ledger.clone(),
    });
    let selected = provider.run_ledger_for_agent("allowed").unwrap().unwrap();
    assert!(Arc::ptr_eq(&selected, &ledger));
    assert!(provider.run_ledger_for_agent("denied").is_err());
    assert!(
        InMemorySessionStores::new()
            .run_ledger_for_agent("allowed")
            .unwrap()
            .is_none()
    );
    let default = SharedStores::default();
    assert!(default.run_ledger_for_agent("allowed").unwrap().is_none());
}

#[test]
fn explicit_actor_policy_defaults_to_refusal_and_arc_forwards_native_error() {
    use tinyagents_harness::store::SchemaAdmission;
    struct LegacyFactory;
    impl SessionStoreProvider for LegacyFactory {
        fn for_agent(&self, id: &str) -> AgentStores {
            InMemorySessionStores::default().for_agent(id)
        }
        fn bind_actor(
            &self,
            _: AgentStorageFactory,
        ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError>
        {
            Err(tinystoragedrivers_types::StorageError::unavailable(
                "legacy declaration factory",
            ))
        }
    }
    let provider: Arc<dyn SessionStoreProvider> = Arc::new(LegacyFactory);
    assert!(
        provider
            .bind_actor_with_schema_admission(host_factory(), SchemaAdmission::ValidateOnly)
            .unwrap()
            .is_none()
    );
    let error = provider
        .bind_actor_with_schema_admission(host_factory(), SchemaAdmission::Declare)
        .err()
        .unwrap();
    assert_eq!(
        error.kind(),
        tinystoragedrivers_types::ErrorKind::Unavailable
    );
    assert!(error.to_string().contains("legacy declaration factory"));
}

#[test]
fn optional_conversation_provider_forwards_selected_handle_and_preserves_absence() {
    use crate::threads::{ConversationPersistence, ConversationResult, DriverConversationStore};
    struct Provider {
        sessions: InMemorySessionStores,
        conversations: Arc<dyn ConversationPersistence>,
    }
    impl SessionStoreProvider for Provider {
        fn for_agent(&self, id: &str) -> AgentStores {
            self.sessions.for_agent(id)
        }
        fn conversations_for_agent(
            &self,
            id: &str,
        ) -> ConversationResult<Option<Arc<dyn ConversationPersistence>>> {
            Ok((id == "allowed").then(|| self.conversations.clone()))
        }
    }
    let scope = tinyagents_storage_kit::Scope::new("conversation-fixture").unwrap();
    let scoped = crate::testkit::MemoryStorage::new()
        .for_scope(&scope)
        .unwrap();
    let conversations: Arc<dyn ConversationPersistence> =
        Arc::new(DriverConversationStore::new(scoped));
    let provider = Arc::new(Provider {
        sessions: InMemorySessionStores::new(),
        conversations: conversations.clone(),
    });
    let selected = provider
        .conversations_for_agent("allowed")
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&selected, &conversations));
    assert!(provider.conversations_for_agent("other").unwrap().is_none());
    assert!(
        InMemorySessionStores::new()
            .conversations_for_agent("allowed")
            .unwrap()
            .is_none()
    );
}

#[test]
fn optional_conversation_provider_keeps_native_refusal_kind_and_source_through_arc() {
    use crate::threads::{ConversationError, ConversationPersistence, ConversationResult};
    use tinyagents_storage_kit::{ErrorKind, StorageError};
    struct Refused(InMemorySessionStores);
    impl SessionStoreProvider for Refused {
        fn for_agent(&self, id: &str) -> AgentStores {
            self.0.for_agent(id)
        }
        fn conversations_for_agent(
            &self,
            _: &str,
        ) -> ConversationResult<Option<Arc<dyn ConversationPersistence>>> {
            Err(StorageError::unavailable("retired profile").into())
        }
    }
    let provider = Arc::new(Refused(InMemorySessionStores::new()));
    let error = match provider.conversations_for_agent("one") {
        Err(error) => error,
        Ok(_) => panic!("refusal became absence"),
    };
    assert!(std::error::Error::source(&error).is_some());
    let ConversationError::Driver(native) = error else {
        panic!("native source flattened");
    };
    assert_eq!(native.kind(), ErrorKind::Unavailable);
}
