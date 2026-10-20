//! Storage fixtures backed by the published storage ports and adapters.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;

use tinyagents_graph::checkpoint::DriverCheckpointer;
use tinyagents_session::port::AgentStorageFactory;
use tinyagents_storage_kit::{DocumentStore, Scope, ScopedStorage, StreamStore};
use tinystoragedrivers_api::{Storage, StorageBackend};
use tinystoragedrivers_document::sqlite::{SqliteDocuments, SqliteStreams};
use tinystoragedrivers_testkit::MemoryBackend;

/// A fresh in-memory scope using the real testkit backend.
pub fn memory() -> ScopedStorage {
    let scope = Scope::new("integration-memory").unwrap();
    let backend = MemoryBackend::new();
    ScopedStorage::new(backend.for_scope(&scope).unwrap(), scope)
}

/// The SQLite database `name` in `dir`, opened afresh on every call, so two
/// calls with the same arguments are two handles on one database. The other
/// four ports remain explicit testkit ports because these integration cases
/// exercise durable document/stream state only.
pub fn sqlite(dir: &Path, name: &str) -> ScopedStorage {
    let scope = Scope::new("integration-sqlite").unwrap();
    let connection =
        tinystoragedrivers_connect::sqlite::SqliteConnection::open(dir.join(name)).unwrap();
    let documents =
        Arc::new(SqliteDocuments::new(connection.clone(), scope.clone())) as Arc<dyn DocumentStore>;
    let streams = Arc::new(SqliteStreams::new(connection, scope.clone())) as Arc<dyn StreamStore>;
    let memory = MemoryBackend::new().for_scope(&scope).unwrap();
    let storage = Storage::new(
        documents,
        streams,
        memory.kv().clone(),
        memory.blobs().clone(),
        memory.queues().clone(),
        memory.secrets().clone(),
    );
    ScopedStorage::new(storage, scope)
}

/// The document port of `storage`.
pub fn docs(storage: &ScopedStorage) -> Arc<dyn DocumentStore> {
    Arc::clone(storage.documents())
}

/// A test-host binding to one already-open storage scope.
pub fn fixed_factory(storage: ScopedStorage) -> AgentStorageFactory {
    Arc::new(move |_| Ok(storage.clone()))
}

/// A durable checkpointer over `checkpoints.sqlite` in `dir`.
pub fn checkpointer<State>(dir: &Path) -> DriverCheckpointer<State> {
    DriverCheckpointer::new(docs(&sqlite(dir, "checkpoints.sqlite")))
}
