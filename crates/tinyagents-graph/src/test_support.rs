//! Shared testkit fixtures for storage-port conformance and reopen tests.

use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

use crate::checkpoint::DriverCheckpointer;

/// A checkpointer over the explicit testkit backend owned by one test.
pub(crate) fn dir_checkpointer<State>(backend: &MemoryBackend) -> DriverCheckpointer<State> {
    let scoped = backend
        .for_scope(&Scope::new("local").unwrap())
        .expect("local scope on a single-operator database");
    DriverCheckpointer::new(std::sync::Arc::clone(scoped.documents()))
}
