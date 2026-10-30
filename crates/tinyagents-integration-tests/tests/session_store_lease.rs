//! A host-defined [`SessionStoreProvider`] can return a pinned lease.
//!
//! The crate's own unit tests cannot prove this: `pub(crate)` items are
//! reachable from inside `tinyagents-session`, so the lease constructor is
//! exercised here, from a separate crate, the way a host's provider uses it.

use std::sync::Arc;

use tinyagents_session::port::{
    AgentStores, InMemorySessionStores, SessionStoreLease, SessionStoreLeaseError,
    SessionStoreProvider,
};
use tinyagents_session::testkit::conformance::session_store_lease_conformance;

/// A provider defined outside `tinyagents-session` whose handles never move
/// off the one backend they were created over.
struct FixedBackendStores {
    backend: InMemorySessionStores,
}

impl SessionStoreProvider for FixedBackendStores {
    fn for_agent(&self, agent_id: &str) -> AgentStores {
        self.backend.for_agent(agent_id)
    }

    fn lease_for_agent(&self, agent_id: &str) -> Result<SessionStoreLease, SessionStoreLeaseError> {
        Ok(SessionStoreLease::pinned(self.for_agent(agent_id)))
    }
}

/// A provider that keeps the default and so cannot promise pinned handles.
struct DefaultLeaseStores(InMemorySessionStores);

impl SessionStoreProvider for DefaultLeaseStores {
    fn for_agent(&self, agent_id: &str) -> AgentStores {
        self.0.for_agent(agent_id)
    }
}

#[test]
fn an_external_provider_can_return_a_lease_over_its_own_stores() {
    let provider = FixedBackendStores {
        backend: InMemorySessionStores::new(),
    };
    let opened = provider.for_agent("agent-a");

    let lease = provider
        .lease_for_agent("agent-a")
        .expect("a provider with fixed handles can lease them");

    assert!(Arc::ptr_eq(
        &opened.turn_states,
        &lease.stores().turn_states
    ));
}

#[tokio::test]
async fn an_external_lease_passes_the_lease_conformance_suite() {
    let provider = FixedBackendStores {
        backend: InMemorySessionStores::new(),
    };

    session_store_lease_conformance(&provider).await;
}

#[test]
fn a_provider_that_keeps_the_default_still_refuses_a_lease() {
    let provider = DefaultLeaseStores(InMemorySessionStores::new());

    let refused = provider.lease_for_agent("agent-a");

    assert!(matches!(refused, Err(SessionStoreLeaseError::Unsupported)));
}
