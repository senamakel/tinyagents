//! Session-persistence test building blocks.
//!
//! This is the session-layer counterpart to the harness
//! [`testkit`](tinyagents_harness::testkit) and the graph
//! [`testkit`](../../tinyagents_graph/testkit/index.html): reusable doubles and
//! **conformance** (contract) suites so a downstream backend author can prove
//! their implementation behaves like the bundled ones, and the bundled ones
//! can be pinned against each other.
//!
//! # Conformance suites
//!
//! See [`conformance`] for the run-ledger and transcript-history contract
//! suites. Call them from a `#[test]` against any implementation:
//!
//! ```rust
//! use std::sync::Arc;
//! use tinyagents_session::run_ledger::DriverRunLedger;
//! use tinyagents_session::testkit::conformance::run_ledger_conformance;
//! use tinystoragedrivers_testkit::memory_backend;
//! use tinyagents_storage_kit::{Scope, StorageBackend};
//!
//! # tokio::runtime::Builder::new_multi_thread().enable_time().build().unwrap().block_on(async {
//! let backend = memory_backend();
//! let scope = Scope::new("test-ledger").unwrap();
//! let storage = backend.for_scope(&scope).unwrap();
//! run_ledger_conformance(Arc::new(DriverRunLedger::new(storage))).await;
//! # });
//! ```
//!
//! # In-memory transcript double
//!
//! [`InMemoryTranscriptHistory`] implements
//! [`TranscriptHistory`](crate::transcript::TranscriptHistory) purely in
//! memory, so the transcript-history conformance suite (and any test that
//! needs a cheap stand-in for a real `session_raw/*.jsonl` file) does not have
//! to touch a filesystem.

pub mod conformance;

/// Makes a test-only host factory from explicit agent-to-scope bindings.
///
/// Scope labels in this map are arbitrary test values; this helper does not
/// encode or validate product identity conventions.
#[cfg(test)]
pub(crate) fn mapped_storage_factory(
    backend: std::sync::Arc<dyn tinystoragedrivers_api::StorageBackend>,
    scopes: std::collections::HashMap<String, tinyagents_storage_kit::Scope>,
) -> crate::port::AgentStorageFactory {
    std::sync::Arc::new(move |agent_id| {
        let scope = scopes.get(agent_id).cloned().ok_or_else(|| {
            tinyagents_storage_kit::StorageError::invalid_input("test agent scope is not mapped")
        })?;
        let storage = backend.for_scope(&scope)?;
        Ok(tinyagents_storage_kit::ScopedStorage::new(storage, scope))
    })
}

/// Makes an explicit test-host binding factory for every requested agent.
/// Labels are arbitrary fixtures and are never derived from product scope rules.
#[cfg(test)]
pub(crate) fn scoped_storage_factory(
    backend: std::sync::Arc<dyn tinystoragedrivers_api::StorageBackend>,
    labels: std::collections::HashMap<String, String>,
) -> crate::port::AgentStorageFactory {
    let scopes = labels
        .into_iter()
        .map(|(agent, label)| (agent, tinyagents_storage_kit::Scope::new(&label).unwrap()))
        .collect();
    mapped_storage_factory(backend, scopes)
}

/// Rebuilds a test scope with a decorated document port and all other real
/// ports forwarded unchanged.
#[cfg(test)]
pub(crate) fn with_documents(
    native: &tinyagents_storage_kit::ScopedStorage,
    documents: std::sync::Arc<dyn tinystoragedrivers_api::DocumentStore>,
) -> tinyagents_storage_kit::ScopedStorage {
    with_documents_and_streams(native, documents, native.streams().clone())
}

/// Rebuilds a test scope with decorated document/stream ports while forwarding
/// the remaining real ports unchanged.
#[cfg(test)]
pub(crate) fn with_documents_and_streams(
    native: &tinyagents_storage_kit::ScopedStorage,
    documents: std::sync::Arc<dyn tinystoragedrivers_api::DocumentStore>,
    streams: std::sync::Arc<dyn tinystoragedrivers_api::StreamStore>,
) -> tinyagents_storage_kit::ScopedStorage {
    let storage = tinystoragedrivers_api::Storage::new(
        documents,
        streams,
        native.kv().clone(),
        native.blobs().clone(),
        native.queues().clone(),
        native.secrets().clone(),
    );
    tinyagents_storage_kit::ScopedStorage::new(storage, native.scope().clone())
}

/// Test-only replacement for a fixed backend/scope pair. The finite identity
/// list is an explicit fixture map, not a production identity convention.
#[cfg(test)]
pub(crate) fn fixed_scope_factory(
    backend: std::sync::Arc<dyn tinystoragedrivers_api::StorageBackend>,
    label: &str,
) -> crate::port::AgentStorageFactory {
    let labels = [
        "a",
        "b",
        "alice",
        "bob",
        "agent",
        "agent-1",
        "other",
        "local",
        "allowed",
        "denied",
        "one",
        "two",
        "profile",
        "host-prepared-test",
        "prepared-actor",
        "agent-one",
        "agent-two",
        "old-agent",
        "new-agent",
        "racing-agent",
        "late-agent",
        "already-live",
        "retained",
        "contract-store-agent",
        "contract-iso-a",
        "contract-iso-b",
        "winner",
        "fixture",
    ]
    .into_iter()
    .map(|agent| (agent.to_owned(), format!("test-{label}-{agent}")))
    .collect();
    scoped_storage_factory(backend, labels)
}

/// Legacy-compatible constructor for tests that need a real, six-port memory
/// backend. This adapter is compiled only with the session testkit.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct MemoryStorage(std::sync::Arc<tinystoragedrivers_testkit::MemoryBackend>);

#[cfg(test)]
impl MemoryStorage {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn for_scope(
        &self,
        scope: &tinyagents_storage_kit::Scope,
    ) -> tinyagents_storage_kit::error::Result<tinyagents_storage_kit::ScopedStorage> {
        let storage = tinystoragedrivers_api::StorageBackend::for_scope(self, scope)?;
        Ok(tinyagents_storage_kit::ScopedStorage::new(
            storage,
            scope.clone(),
        ))
    }
}

#[cfg(test)]
impl tinystoragedrivers_api::StorageBackend for MemoryStorage {
    fn capabilities(&self) -> tinyagents_storage_kit::Capabilities {
        tinystoragedrivers_api::StorageBackend::capabilities(self.0.as_ref())
    }

    fn for_scope(
        &self,
        scope: &tinyagents_storage_kit::Scope,
    ) -> tinyagents_storage_kit::error::Result<tinystoragedrivers_api::Storage> {
        tinystoragedrivers_api::StorageBackend::for_scope(self.0.as_ref(), scope)
    }
}

mod in_memory_transcript;
pub use in_memory_transcript::InMemoryTranscriptHistory;

/// Creates an empty driver-backed transcript locator over in-memory storage.
///
/// Use this when a test needs the same locator interface used by production
/// session stores, but should not touch the filesystem.
#[cfg(test)]
pub fn in_memory_transcript() -> std::sync::Arc<dyn crate::transcript::TranscriptLocator> {
    use crate::port::{DriverSessionStores, SessionStoreProvider};
    use std::sync::Arc;
    use tinyagents_storage_kit::Scope;
    use tinystoragedrivers_testkit::memory_backend;

    let scope = Scope::new("test-transcript").unwrap();
    let factory = mapped_storage_factory(
        Arc::new(memory_backend()),
        [("test".to_owned(), scope)].into_iter().collect(),
    );
    DriverSessionStores::new(factory)
        .expect("driver session stores over memory")
        .for_agent("test")
        .transcripts
}

#[cfg(test)]
#[path = "in_memory_transcript_tests.rs"]
mod tests;
