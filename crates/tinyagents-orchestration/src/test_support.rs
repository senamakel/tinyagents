//! Shared testkit fixtures for the orchestration drivers.

use tinyagents_session::run_ledger::DriverRunLedger;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

/// Opens a run ledger over the explicit testkit backend owned by one test.
pub(crate) fn ledger_in(backend: &MemoryBackend) -> DriverRunLedger {
    let scoped = backend
        .for_scope(&Scope::new("local").unwrap())
        .expect("local scope");
    DriverRunLedger::new(scoped)
}
