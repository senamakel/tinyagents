//! The bundle of stores one agent persists through.

use std::sync::Arc;

use tinyagents_harness::store::{AppendStore, Store};

use super::TurnStates;
use crate::transcript::TranscriptLocator;

/// Every store one agent uses, as handed out by
/// [`SessionStoreProvider::for_agent`](super::SessionStoreProvider::for_agent).
///
/// Cloning is cheap: each field is a shared handle.
#[derive(Clone)]
pub struct AgentStores {
    /// The agent's conversation transcripts: resume reads, turn commits and
    /// compaction generations.
    pub transcripts: Arc<dyn TranscriptLocator>,
    /// Snapshots of the agent's in-flight turns.
    pub turn_states: Arc<dyn TurnStates>,
    /// Namespaced key-value records: run status, goals, todos, session
    /// descriptors.
    pub kv: Arc<dyn Store>,
    /// Append-only event streams: each run's journal.
    pub journal: Arc<dyn AppendStore>,
}

/// A set of session stores whose scoped storage was captured at open time.
///
/// A runtime that must keep persisting through the backend it admitted a
/// session on asks for a lease with
/// [`SessionStoreProvider::lease_for_agent`](super::SessionStoreProvider::lease_for_agent)
/// and holds it for as long as the session lives. Only a provider that can
/// promise this builds one, with [`SessionStoreLease::pinned`]; the default
/// `lease_for_agent` refuses with [`SessionStoreLeaseError::Unsupported`].
#[derive(Clone)]
pub struct SessionStoreLease {
    stores: AgentStores,
}

impl SessionStoreLease {
    /// The stores pinned by this lease.
    pub fn stores(&self) -> &AgentStores {
        &self.stores
    }

    /// Pins `stores` as a lease the provider guarantees.
    ///
    /// By calling this a provider promises that, for as long as the lease or
    /// any clone of it lives, every handle in `stores` keeps reading and
    /// writing the one backend it was opened over. A later change to the
    /// provider (a rebind, a failover, a swapped storage factory, a tenant
    /// that moves) must not redirect these handles; it may only affect stores
    /// opened afterwards. A provider whose binding can change underneath
    /// handles it already returned must not use this constructor and should
    /// keep the default [`SessionStoreProvider::lease_for_agent`](super::SessionStoreProvider::lease_for_agent),
    /// which refuses with [`SessionStoreLeaseError::Unsupported`]. The runtime
    /// trusts the promise and does not re-check it.
    ///
    /// # Examples
    ///
    /// A provider over one fixed backend hands out the same stores it would
    /// for [`for_agent`](super::SessionStoreProvider::for_agent) as a lease:
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use tinyagents_session::port::{
    ///     AgentStores, SessionStoreLease, SessionStoreLeaseError, SessionStoreProvider,
    /// };
    ///
    /// /// Every agent's stores were built over one backend when the host started
    /// /// and are never rebound.
    /// struct FixedStores(Arc<dyn Fn(&str) -> AgentStores + Send + Sync>);
    ///
    /// impl SessionStoreProvider for FixedStores {
    ///     fn for_agent(&self, agent_id: &str) -> AgentStores {
    ///         (self.0)(agent_id)
    ///     }
    ///
    ///     fn lease_for_agent(
    ///         &self,
    ///         agent_id: &str,
    ///     ) -> Result<SessionStoreLease, SessionStoreLeaseError> {
    ///         Ok(SessionStoreLease::pinned(self.for_agent(agent_id)))
    ///     }
    /// }
    /// ```
    pub fn pinned(stores: AgentStores) -> Self {
        Self { stores }
    }
}

/// Why a provider could not return a pinned set of stores.
#[derive(Debug, thiserror::Error)]
pub enum SessionStoreLeaseError {
    /// The provider cannot guarantee that its handles retain one backend.
    #[error("provider cannot guarantee pinned session stores")]
    Unsupported,
    /// Opening the pinned stores failed.
    #[error("opening pinned session stores failed: {0}")]
    Open(#[source] tinystoragedrivers_types::StorageError),
}

impl std::fmt::Debug for AgentStores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentStores")
            .field("transcripts", &self.transcripts.destination_key())
            .finish_non_exhaustive()
    }
}
