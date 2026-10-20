//! Shared primitives (owner `tsd`): leases, key-value, counters and the typed
//! [`Repo`] with its compare-and-swap [`update`](Repo::update) loop.
//!
//! Every owner needs the same four things, and each used to rebuild them. They
//! all run on a [`DocumentStore`](crate::DocumentStore) handle the caller opened for the right
//! scope (a tenant for [`Kv`] and [`Counters`], usually `cluster` for
//! [`Leases`] that place processes), so scope isolation and conditional writes
//! apply to them exactly as to any other record.
//!
//! The three collections they use, [`LEASES`], [`KV`] and [`COUNTERS`], are
//! declared by [`collection_specs`]; each storage owner calls [`ensure`] for
//! its already-bound scope during open.
//!
//! ```
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use std::sync::Arc;
//! use tinystoragedrivers_api::StorageBackend;
//! use tinystoragedrivers_testkit::memory_backend;
//! use tinyagents_storage_kit::primitives::Counters;
//! use tinystoragedrivers_types::{CollectionSpec, Scope};
//!
//! let storage = memory_backend().for_scope(&Scope::new("local")?)?;
//! for spec in [CollectionSpec::new("tsd.counters")] {
//!     storage.documents().ensure_collection(&spec).await?;
//! }
//! let counters = Counters::new(storage.documents());
//! assert_eq!(counters.next("cron", "runs").await?, 1);
//! assert_eq!(counters.next("cron", "runs").await?, 2);
//! # Ok::<(), tinystoragedrivers_types::StorageError>(())
//! # }).unwrap();
//! ```

mod counters;
mod kv;
mod leases;
mod repo;

use tinystoragedrivers_types::{CollectionSpec, IndexSpec};

pub use counters::Counters;
pub use kv::Kv;
pub use leases::{Acquire, Acquisition, Clock, Lease, LeaseRecord, Leases};
pub use repo::{Repo, RetryPolicy};

/// The lease collection.
pub const LEASES: &str = "tsd.leases";
/// The key-value collection.
pub const KV: &str = "tsd.kv";
/// The counter collection.
pub const COUNTERS: &str = "tsd.counters";

/// The declaration needed by [`Kv`] and owners that migrate legacy key-value
/// rows into `tsd.kv`.
#[must_use]
pub fn kv_collection_spec() -> CollectionSpec {
    CollectionSpec::new(KV).index(IndexSpec::new("kv_by_key", ["owner", "ns", "key"]))
}

/// The declaration needed by [`Leases`].
#[must_use]
pub fn leases_collection_spec() -> CollectionSpec {
    CollectionSpec::new(LEASES)
}

/// The collections the primitives need, with their indexes.
///
/// [`Kv`] declares an index over `(owner, ns, key)` so a namespace lists in
/// key order without scanning other owners' entries. Leases and counters are
/// only ever read by id.
#[must_use]
pub fn collection_specs() -> Vec<CollectionSpec> {
    vec![
        leases_collection_spec(),
        kv_collection_spec(),
        CollectionSpec::new(COUNTERS),
    ]
}

/// Declare the shared primitive collections in this scope.
pub async fn ensure(storage: &crate::ScopedStorage) -> tinystoragedrivers_types::Result<()> {
    for spec in collection_specs() {
        storage.documents().ensure_collection(&spec).await?;
    }
    Ok(())
}
