//! The session store port: where a host plugs in its own storage.
//!
//! Everything an agent persists while it runs — its transcripts, its
//! turn-state snapshots, its key-value records and its event journal — is
//! reached through one [`SessionStoreProvider`]. The host installs a provider
//! once; the runtime asks it for each agent's [`AgentStores`] and never names
//! a file, a database or a directory itself. The provider may additionally return
//! an optional bound [`crate::run_ledger::RunLedgerStore`] independently of this
//! bundle, through [`SessionStoreProvider::run_ledger_for_agent`].
//!
//! Hosts resolve each identity and supply a [`tinyagents_storage_kit::ScopedStorage`]
//! already bound to the authorized scope. The session library does not choose
//! scope labels or open a backend.
//!
//! # Isolation
//!
//! [`SessionStoreProvider`] provides handles bound to one agent. Hosts that
//! must propagate store-opening or recovery errors should use `try_for_agent`;
//! the legacy `for_agent` accessor can return refusing handles instead. A provider that serves several users
//! from one backend must make it impossible for one agent's
//! handles to read or write another's data; the explicitly enabled
//! `testkit::InMemorySessionStores` does, and the conformance suite
//! ([`crate::testkit::conformance::session_store_isolation_conformance`])
//! checks it. A single-user host (the desktop app) may legitimately share one
//! store between its agents, and simply does not claim isolation. The optional
//! [`SessionStoreProvider::run_ledger_for_agent`] handle follows the same identity
//! boundary; selecting it is independent of transcript-store selection.
//!
//! # Sync and async
//!
//! The transcript and turn-state seams are synchronous, as they already were:
//! the turn path that persists a transcript is a chain of sync `&mut self`
//! methods. An implementation over an async client bridges at its own
//! boundary. The key-value and journal seams are the harness's async
//! [`Store`] and [`AppendStore`].
//!
//! # Bundled implementations
//!
//! - `testkit::InMemorySessionStores`: per-agent, process-lifetime storage for
//!   tests and explicit test-only hosts.
//! - `DriverSessionStores`: opens each agent's stores from the host-supplied
//!   bound-storage factory and runs the session crate's owner migrations
//!   before exposing them. Hosts may use
//!   [`collections`] for declaration-only backend preparation.
//! - Workspace files, JSONL transcripts and database engines belong to the
//!   host; `docs/legacy-import-handoff.md` records the source migration handoff.

mod drivers;
mod helpers;
#[cfg(any(test, feature = "testkit"))]
mod memory;
mod types;

use std::sync::Arc;

pub use drivers::{
    DriverSessionStores, DriverTranscriptHistory, DriverTranscriptLocator, DriverTurnStates,
    RebindableSessionStores, SessionStoreRegistry, collections, open_agent_storage, storage_schema,
};
#[cfg(any(test, feature = "testkit"))]
pub use memory::{InMemorySessionStores, InMemoryTranscriptLocator, InMemoryTurnStates};
pub use types::AgentStores;

pub use tinyagents_harness::store::{AppendStore, Store};

/// Host-owned mapping from an agent identity to its already-bound storage.
/// The session library does not choose tenants, construct scopes, or bind a
/// backend. Hosts resolve identity and authorization before returning storage.
pub type AgentStorageFactory = Arc<
    dyn Fn(
            &str,
        )
            -> Result<tinyagents_storage_kit::ScopedStorage, tinystoragedrivers_types::StorageError>
        + Send
        + Sync,
>;

use crate::turn_state::{TurnLifecycle, TurnState};

/// Hands out each agent's stores. Installed once per runtime by the host.
pub trait SessionStoreProvider: Send + Sync {
    /// The stores of the agent `agent_id`.
    ///
    /// Called whenever the runtime needs an agent's stores, so it should be
    /// cheap: hand out shared handles rather than opening connections.
    fn for_agent(&self, agent_id: &str) -> AgentStores;

    /// Opens an agent's stores while retaining native admission and recovery errors.
    ///
    /// Legacy providers default to their infallible accessor. Driver providers
    /// override this method so a configured refusal cannot become missing stores
    /// or select a file fallback.
    ///
    /// # Errors
    ///
    /// The native backend error, including its source, when opening is refused.
    fn try_for_agent(
        &self,
        agent_id: &str,
    ) -> Result<AgentStores, tinystoragedrivers_types::StorageError> {
        Ok(self.for_agent(agent_id))
    }

    /// Optionally create an independent actor provider over a host-admitted backend.
    ///
    /// The backend must already embody the actor authority supplied by the
    /// host. This factory does not acquire a lease, resolve a tenant or repair
    /// a mutable backend that borrows another actor grant. Driver providers
    /// reuse their blocking bridge and layout but create independent caches and
    /// recovery bookkeeping. Agent IDs and durable transcript stems stay the same.
    /// Retained old providers keep their originally supplied backend.
    ///
    /// `Ok(None)` means this provider does not support actor rebinding. It does
    /// not authorize selecting a different backend or a file fallback. Errors
    /// retain their native sources; hosts must propagate a configured refusal.
    fn bind_actor(
        &self,
        _factory: AgentStorageFactory,
    ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError> {
        Ok(None)
    }

    /// Optional bound run ledger for this agent, independent of transcript storage.
    ///
    /// Defaults to absent so existing providers retain their behavior. Absence
    /// is not permission to create a local database: hosts decide whether and
    /// how to supply a driver-backed implementation.
    /// Multi-tenant providers must resolve identity before returning the handle
    /// and must not derive scope from ledger payloads. This seam alone neither
    /// routes existing workspace-based callers nor imports workspace data.
    ///
    /// Scope/admission or backend errors must be returned as errors, distinct
    /// from an unsupported ledger (`Ok(None)`), so callers can fail closed.
    fn run_ledger_for_agent(
        &self,
        _agent_id: &str,
    ) -> crate::run_ledger::LedgerResult<Option<Arc<dyn crate::run_ledger::RunLedgerStore>>> {
        Ok(None)
    }

    /// Rebinds an actor with an explicit immutable document-schema policy.
    ///
    /// `Declare` delegates to the legacy factory. `ValidateOnly` is unsupported
    /// by default: a provider must explicitly implement read-only schema
    /// admission. `Ok(None)` never authorizes a backend or file fallback.
    /// This selects schema behavior only; the host must supply actor authority.
    ///
    /// # Errors
    ///
    /// Native actor admission errors, retaining their sources.
    fn bind_actor_with_schema_admission(
        &self,
        factory: AgentStorageFactory,
        admission: tinyagents_harness::store::SchemaAdmission,
    ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError> {
        match admission {
            tinyagents_harness::store::SchemaAdmission::Declare => self.bind_actor(factory),
            tinyagents_harness::store::SchemaAdmission::ValidateOnly => Ok(None),
        }
    }

    /// Optional conversation handle admitted by the host for this agent.
    ///
    /// Conversations may span agents within one trusted profile scope. Providers
    /// must bind that scope themselves, never derive authority from a thread ID.
    /// Absence leaves selection to the host; a configured refusal remains Err
    /// and must never cause local file fallback. Existing AgentStores literals
    /// and classic providers remain compatible.
    fn conversations_for_agent(
        &self,
        _agent_id: &str,
    ) -> crate::threads::ConversationResult<Option<Arc<dyn crate::threads::ConversationPersistence>>>
    {
        Ok(None)
    }

    /// Repairs state an unclean shutdown left behind, once, before any agent
    /// runs: turns that were in flight are marked interrupted, and so on.
    ///
    /// Defaults to doing nothing, for stores that keep no in-flight state or
    /// whose host repairs it elsewhere.
    ///
    /// # Errors
    ///
    /// Whatever the backend reports; the runtime logs it and carries on.
    fn recover(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// A stable name for where this provider keeps its data, for logs and
    /// for telling two providers apart. `None` when it cannot say.
    fn destination_key(&self) -> Option<String> {
        None
    }

    /// The workspace directory a file-backed provider keeps the classic
    /// on-disk layout in (`session_raw/`, `tinyagents_store/`, …), or `None`
    /// for any other store.
    ///
    /// A host uses it to keep file-era companions of that layout running —
    /// mirrors that read transcript files back, say — only where there are
    /// files to read.
    fn workspace_dir(&self) -> Option<std::path::PathBuf> {
        None
    }
}

/// Restart-survivable snapshots of an agent's in-flight turns.
///
/// The contract of a turn-state store. Every method is scoped to the handle's own agent.
pub trait TurnStates: Send + Sync {
    /// Writes `state` as the snapshot of its turn.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn put(&self, state: &TurnState) -> Result<(), String>;

    /// Writes `state` unless its turn is already recorded as completed;
    /// returns whether it wrote.
    ///
    /// # Errors
    ///
    /// When the stored snapshot cannot be read or the backend cannot write.
    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String>;

    /// The latest turn's snapshot on `thread_id`.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String>;

    /// The snapshot of one turn.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String>;

    /// Removes every snapshot on `thread_id`; returns whether any existed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn delete(&self, thread_id: &str) -> Result<bool, String>;

    /// Removes one turn's snapshot; returns whether it existed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String>;

    /// The latest snapshot of every thread.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn list(&self) -> Result<Vec<TurnState>, String>;

    /// Every turn snapshot on `thread_id`.
    ///
    /// # Errors
    ///
    /// When the backend cannot read.
    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String>;

    /// Removes every snapshot; returns how many there were.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn clear_all(&self) -> Result<usize, String>;

    /// Marks every non-terminal snapshot interrupted, as of `now_rfc3339`;
    /// returns how many changed.
    ///
    /// # Errors
    ///
    /// When the backend cannot write.
    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String>;

    /// Records a terminal `lifecycle` on a turn that is still in flight;
    /// returns whether it changed anything.
    ///
    /// # Errors
    ///
    /// When the backend cannot read or write.
    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String>;
}

/// A provider shared behind an [`Arc`] is still a provider.
impl<P: SessionStoreProvider + ?Sized> SessionStoreProvider for Arc<P> {
    /// Forwards checked opening to the shared provider, including its native errors.
    fn try_for_agent(
        &self,
        agent_id: &str,
    ) -> Result<AgentStores, tinystoragedrivers_types::StorageError> {
        (**self).try_for_agent(agent_id)
    }

    fn bind_actor(
        &self,
        factory: AgentStorageFactory,
    ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError> {
        (**self).bind_actor(factory)
    }

    fn conversations_for_agent(
        &self,
        agent_id: &str,
    ) -> crate::threads::ConversationResult<Option<Arc<dyn crate::threads::ConversationPersistence>>>
    {
        (**self).conversations_for_agent(agent_id)
    }

    fn run_ledger_for_agent(
        &self,
        agent_id: &str,
    ) -> crate::run_ledger::LedgerResult<Option<Arc<dyn crate::run_ledger::RunLedgerStore>>> {
        (**self).run_ledger_for_agent(agent_id)
    }

    fn bind_actor_with_schema_admission(
        &self,
        factory: AgentStorageFactory,
        admission: tinyagents_harness::store::SchemaAdmission,
    ) -> Result<Option<Arc<dyn SessionStoreProvider>>, tinystoragedrivers_types::StorageError> {
        (**self).bind_actor_with_schema_admission(factory, admission)
    }

    fn for_agent(&self, agent_id: &str) -> AgentStores {
        (**self).for_agent(agent_id)
    }

    fn recover(&self) -> anyhow::Result<()> {
        (**self).recover()
    }

    fn destination_key(&self) -> Option<String> {
        (**self).destination_key()
    }

    fn workspace_dir(&self) -> Option<std::path::PathBuf> {
        (**self).workspace_dir()
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
