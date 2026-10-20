//! Moving v1 data into `tsd`-owned collections.
//!
//! A document step may only write its own owner's collections, and the graph's
//! leases and task cache now live in `tsd.leases` and `tsd.kv`, which `tsd`
//! owns. So the move is an importer the host calls once per scope after
//! `Registry::migrate`, in the style of a legacy-file importer. Each records
//! completion in `tsd.migrations` (`Migrations::done(owner, step)`), so a
//! second call does nothing.

use serde_json::Value;
use tinyagents_harness::ids::now_ms;
use tinyagents_storage_kit::{
    CollectionSpec, Cursor, ErrorKind, Filter, ID_FIELD, Key, Leases, Migrations, Precondition,
    Query, Result, ScopedStorage, Sort, WriteOptions, primitives,
};

use super::steps::LEGACY;
use super::{KV_OWNER, LEASE_OWNER, task_cache_ns};
use crate::storage::types::lease_key;

/// Rows read per page while importing.
const PAGE: usize = 200;

/// Every row of the legacy collection `collection`, oldest id first.
async fn legacy_rows(
    storage: &ScopedStorage,
    collection: &str,
) -> Result<Vec<tinyagents_storage_kit::Versioned<Value>>> {
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(collection))
        .await?;
    let mut rows = Vec::new();
    let mut cursor: Option<Cursor> = None;
    loop {
        let mut query = Query::filter(Filter::default())
            .sort(Sort::asc(ID_FIELD))
            .limit(PAGE);
        if let Some(after) = cursor.take() {
            query = query.after(after);
        }
        let page = docs.query(collection, &query).await?;
        rows.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(rows),
        }
    }
}

/// The thread a v1 lease id belongs to: ids were `len:thread`.
fn v1_lease_thread(id: &str) -> Option<&str> {
    let (len, thread) = id.split_once(':')?;
    (len.parse::<usize>().ok()? == thread.len()).then_some(thread)
}

/// Carry the live execution leases of the v1 checkpointers
/// (`<prefix>_leases`: `{owner, expires_at_ms}`) into `tsd.leases` for the
/// scope `storage`, keeping the holder and the time left, then remove the
/// legacy collections. A lease that has already lapsed is dropped. Returns how
/// many leases were carried over.
///
/// A completed import is skipped; a missing collection counts as empty.
///
/// # Errors
///
/// The first storage error; the import resumes on the next call because the
/// legacy collections are removed only after every lease was carried over.
pub async fn import_legacy_leases(storage: &ScopedStorage) -> Result<u64> {
    tinyagents_storage_kit::with_migration_lease(storage, "graph", |lease| async move {
        import_legacy_leases_with_lease(storage, &lease).await
    })
    .await
}

pub(crate) async fn import_legacy_leases_with_lease(
    storage: &ScopedStorage,
    lease: &tinyagents_storage_kit::MigrationLease,
) -> Result<u64> {
    let migrations = Migrations::new(storage);
    if migrations.done(LEASE_OWNER, "lease_import").await? {
        tracing::debug!("[graph-storage] lease import already done");
        return Ok(0);
    }
    storage
        .documents()
        .ensure_collection(&primitives::leases_collection_spec())
        .await?;
    let leases = Leases::new(storage.documents());
    let now = now_ms() as i64;
    let mut carried = 0_u64;
    for legacy in &LEGACY {
        for row in legacy_rows(storage, legacy.leases).await? {
            let (Some(holder), Some(expires)) = (
                row.doc.get("owner").and_then(Value::as_str),
                row.doc.get("expires_at_ms").and_then(Value::as_i64),
            ) else {
                continue;
            };
            let Some(thread) = v1_lease_thread(&row.id) else {
                continue;
            };
            if expires <= now {
                continue;
            }
            lease.renew().await?;
            let held = leases
                .acquire(
                    LEASE_OWNER,
                    &lease_key(legacy.ns, thread),
                    holder,
                    expires - now,
                )
                .await?;
            carried += u64::from(held.is_some());
        }
    }
    for legacy in &LEGACY {
        lease.renew().await?;
        storage
            .documents()
            .delete_where(legacy.leases, &Filter::default())
            .await?;
    }
    lease.mark_done(LEASE_OWNER, "lease_import").await?;
    tracing::debug!(carried, "[graph-storage] lease import complete");
    Ok(carried)
}

/// Carry the v1 node-result cache (`graph_task_cache`:
/// `{graph, node, hash, value, expiresAtMs}`) into `tsd.kv` under owner
/// `graph` for the scope `storage`, moving the deadline into the entry's
/// `expires_ms` metadata, then remove the legacy collection. Expired rows are
/// dropped. Returns how many entries were written.
///
/// A completed import is skipped; an entry already present in `tsd.kv` is kept.
///
/// # Errors
///
/// The first storage error; the import resumes on the next call.
pub async fn import_legacy_task_cache(storage: &ScopedStorage) -> Result<u64> {
    tinyagents_storage_kit::with_migration_lease(storage, "graph", |lease| async move {
        import_legacy_task_cache_with_lease(storage, &lease).await
    })
    .await
}

pub(crate) async fn import_legacy_task_cache_with_lease(
    storage: &ScopedStorage,
    lease: &tinyagents_storage_kit::MigrationLease,
) -> Result<u64> {
    const LEGACY_COLLECTION: &str = "graph_task_cache";
    const STEP: &str = "kv_import_graph_task_cache";
    let migrations = Migrations::new(storage);
    if migrations.done(KV_OWNER, STEP).await? {
        tracing::debug!("[graph-storage] task cache import already done");
        return Ok(0);
    }
    storage
        .documents()
        .ensure_collection(&primitives::kv_collection_spec())
        .await?;
    let now = now_ms() as i64;
    let mut written = 0_u64;
    for row in legacy_rows(storage, LEGACY_COLLECTION).await? {
        let (Some(graph), Some(node), Some(hash), Some(value)) = (
            row.doc.get("graph").and_then(Value::as_str),
            row.doc.get("node").and_then(Value::as_str),
            row.doc.get("hash").and_then(Value::as_str),
            row.doc.get("value"),
        ) else {
            continue;
        };
        let expires = row.doc.get("expiresAtMs").and_then(Value::as_i64);
        if expires.is_some_and(|at| at <= now) {
            continue;
        }
        let ns = task_cache_ns(super::DEFAULT_CACHE_NAMESPACE, graph);
        let key = Key::of([node, hash])?.as_str().to_owned();
        let id = Key::of([KV_OWNER, &ns, &key])?.as_str().to_owned();
        let mut options = WriteOptions::new();
        if let Some(at) = expires {
            options = options.expires_ms(at);
        }
        let body = serde_json::json!({"owner": KV_OWNER, "ns": ns, "key": key, "value": value});
        lease.renew().await?;
        match storage
            .documents()
            .put_with(primitives::KV, &id, body, Precondition::Absent, options)
            .await
        {
            Ok(_) => written += 1,
            Err(error)
                if matches!(error.kind(), ErrorKind::AlreadyExists | ErrorKind::Conflict) => {}
            Err(error) => return Err(error),
        }
    }
    lease.renew().await?;
    storage
        .documents()
        .delete_where(LEGACY_COLLECTION, &Filter::default())
        .await?;
    lease.mark_done(KV_OWNER, STEP).await?;
    tracing::debug!(written, "[graph-storage] task cache import complete");
    Ok(written)
}
