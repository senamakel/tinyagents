//! Harness store conformance: the same contract suite applied to every
//! built-in `Store` / `NamespacedStore` backend, proving they behave
//! interchangeably (gap #17, harness half — see `tests/conformance.rs` and
//! `tests/persistence_conformance.rs` for the graph checkpointer/task-store
//! side and `tests/session_conformance.rs` for the session side).

use tinyagents_harness::store::conformance::{
    run_namespaced_store_conformance, run_store_conformance,
};
use tinyagents_harness::store::namespaced::InMemoryNamespacedStore;
use tinyagents_harness::store::{FileStore, InMemoryStore};

#[tokio::test]
async fn in_memory_store_satisfies_the_conformance_suite() {
    run_store_conformance(&InMemoryStore::new()).await;
}

#[tokio::test]
async fn file_store_satisfies_the_conformance_suite() {
    let dir = tempfile::tempdir().unwrap();
    run_store_conformance(&FileStore::new(dir.path())).await;
}

#[tokio::test]
async fn in_memory_namespaced_store_satisfies_the_conformance_suite() {
    run_namespaced_store_conformance(&InMemoryNamespacedStore::new()).await;
}
