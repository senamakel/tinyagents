//! The graph crate's v2 storage declarations: its `storage_schema()`, the
//! stored bodies of the one graph checkpointer and the steps and importers
//! that bring the v1 collections forward.

mod import;
mod steps;
mod types;

use tinyagents_storage_kit::{
    CollectionSpec, IndexSpec, MigrateOptions, Owner, Registry, Schema, ScopeClass, ScopedStorage,
    primitives,
};

pub use import::{import_legacy_leases, import_legacy_task_cache};
pub(crate) use import::{import_legacy_leases_with_lease, import_legacy_task_cache_with_lease};
pub(crate) use types::{
    CheckpointBody, ThreadBody, WritesBody, checkpoint_id_key, lease_key, subgraph_key,
    thread_id_key, writes_id_key,
};

/// The body version of every `graph.*` collection.
///
/// A collection takes one v1 -> v2 step per legacy prefix (see
/// [`storage_schema`]) and a step chain cannot fork, so the chain runs 1 -> 2
/// -> 3 -> 4 and the declared version is the end of it. The body shape itself
/// has not changed since the first step.
pub(crate) const SCHEMA_VERSION: u32 = 4;

/// Stored checkpoints, one document per checkpoint.
pub(crate) const CHECKPOINTS: &str = "graph.checkpoints";

/// The per-thread insertion counters.
pub(crate) const THREADS: &str = "graph.threads";

/// The merged pending writes of a checkpoint.
pub(crate) const WRITES: &str = "graph.writes";

/// The namespace [`DriverCheckpointer::new`](crate::checkpoint::DriverCheckpointer::new) uses.
pub(crate) const DEFAULT_NAMESPACE: &str = "graph";

/// The `tsd.leases` owner of thread execution leases.
pub(crate) const LEASE_OWNER: &str = "graph";

/// The `tsd.kv` owner of the node-result cache.
pub(crate) const KV_OWNER: &str = "graph";

/// The `tsd.kv` namespace of the node-result cache when none is given.
pub(crate) const DEFAULT_CACHE_NAMESPACE: &str = "task_cache";

/// The `tsd.kv` namespace of `graph`'s node results in the cache `base`.
pub(crate) fn task_cache_ns(base: &str, graph: &str) -> String {
    format!("{base}:{graph}")
}

/// `graph.checkpoints`: every read filters on the namespace and thread.
pub(crate) fn checkpoints_spec() -> CollectionSpec {
    CollectionSpec::new(CHECKPOINTS)
        .index(IndexSpec::new("by_thread_seq", ["ns", "thread", "seq"]))
        .index(IndexSpec::new(
            "by_thread_id",
            ["ns", "thread", "checkpoint_id"],
        ))
        .index(IndexSpec::new(
            "by_thread_subgraph",
            ["ns", "thread", "subgraph", "seq"],
        ))
}

/// `graph.threads`.
pub(crate) fn threads_spec() -> CollectionSpec {
    CollectionSpec::new(THREADS).index(IndexSpec::new("by_ns", ["ns"]))
}

/// `graph.writes`.
pub(crate) fn writes_spec() -> CollectionSpec {
    CollectionSpec::new(WRITES).index(IndexSpec::new("by_thread", ["ns", "thread"]))
}

/// `tsd.leases`, which the checkpointer claims through `Leases`.
pub(crate) fn leases_spec() -> CollectionSpec {
    CollectionSpec::new(primitives::LEASES)
}

/// The graph crate's storage declaration for its checkpointer and the steps
/// that bring the v1 names and shapes forward. Graph open runs this
/// owner-local migration against the already-bound scope before returning the
/// handle.
///
/// | v1 | v2 |
/// |---|---|
/// | `graph_{checkpoints,threads,writes}` | `graph.{checkpoints,threads,writes}`, `ns` = `graph` |
/// | `delegation_graph_{checkpoints,threads,writes}` | the same collections, `ns` = `delegation` |
/// | `flows_graph_{checkpoints,threads,writes}` | the same collections, `ns` = `flows` |
/// | `<prefix>_leases` | `tsd.leases`, owner `graph` |
/// | `graph_task_cache` | `tsd.kv`, owner `graph` |
///
/// The old prefix becomes the `ns` body field and is part of every id and
/// index, so two namespaces never collide. The checkpoint record stays a typed
/// field (`checkpoint`), not a JSON string. A collection gets one step per
/// prefix, so the steps chain through versions 1 to 4.
///
/// `tsd.leases` and `tsd.kv` belong to `tsd`, so a document step cannot write
/// them: the host calls [`import_legacy_leases`] and
/// [`import_legacy_task_cache`] once per scope. The goal and todo lists use the
/// harness `Store`, which lives in `tsd.kv` under the store's own owner.
#[must_use]
pub fn storage_schema() -> Schema {
    let owner = Owner::new("graph").expect("`graph` is a valid owner");
    let mut schema = Schema::new(owner)
        .collection_with(SCHEMA_VERSION, checkpoints_spec(), ScopeClass::Agent)
        .collection_with(SCHEMA_VERSION, threads_spec(), ScopeClass::Agent)
        .collection_with(SCHEMA_VERSION, writes_spec(), ScopeClass::Agent);
    for step in steps::all() {
        schema = schema.step(step);
    }
    schema
}

/// Collections owned by graph; callers declare these on their chosen backend.
#[must_use]
pub fn collections() -> Vec<CollectionSpec> {
    storage_schema().collections()
}

/// Open graph storage for one scope, applying graph-owned migrations and
/// importing the graph's legacy cache and lease records before use.
pub async fn open_storage(
    storage: ScopedStorage,
) -> tinyagents_storage_kit::error::Result<ScopedStorage> {
    let owned_storage = storage.clone();
    tinyagents_storage_kit::with_migration_lease(&storage, "graph", move |lease| async move {
        let storage = owned_storage;
        let registry = Registry::compose([storage_schema()])?;
        for spec in collections() {
            storage.documents().ensure_collection(&spec).await?;
        }
        for spec in [
            primitives::leases_collection_spec(),
            primitives::kv_collection_spec(),
        ] {
            storage.documents().ensure_collection(&spec).await?;
        }
        registry.migrate(&storage, &MigrateOptions::new()).await?;
        import_legacy_leases_with_lease(&storage, &lease).await?;
        import_legacy_task_cache_with_lease(&storage, &lease).await?;
        Ok(storage)
    })
    .await
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
