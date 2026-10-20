//! Moving a v1 key-value collection into `tsd.kv`.
//!
//! A document step may only write collections of its own owner, and the key-value
//! stores of the harness (`harness_store`, `harness_namespaced_store`,
//! `harness_response_cache`) and of the session crate (`session_kv`) now live in
//! `tsd.kv`, which `tsd` owns. So the move is an importer the host calls once per
//! scope after `Registry::migrate`, in the style of a legacy-file importer:
//! [`import_legacy_kv`] copies the rows, removes them, and records completion in
//! `tsd.migrations` (`Migrations::done(owner, "kv_import_<collection>")`).
//!
//! The copy is idempotent and resumable: entries are written with
//! `Precondition::Absent`, so a crash re-runs harmlessly, and an entry the v2 store has
//! already written (a newer value) is never overwritten by the stale v1 row.

use serde_json::Value;
use tinyagents_storage_kit::{
    CollectionSpec, Cursor, ErrorKind, Filter, ID_FIELD, Migrations, Precondition, Query, Result,
    ScopedStorage, Sort, WriteOptions,
};

use super::kv::{CACHE_NS_PREFIX, KV, NAMESPACED_NS_PREFIX, kv_body, kv_id};

/// Rows read per page while importing.
const PAGE: usize = 200;

/// The v1 body shape of a key-value collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyKv {
    /// `{ ns, key, value }`: `harness_store` and the session crate's `session_kv`.
    Flat,
    /// `{ root, ns: [segments], key, value, createdAtMs, updatedAtMs, expiresAtMs }`:
    /// `harness_namespaced_store`.
    Namespaced,
    /// `{ ns, key, value, expiresAtMs }`: `harness_response_cache`.
    ResponseCache,
}

/// One v1 row as the `tsd.kv` entry it becomes.
struct Converted {
    ns: String,
    key: String,
    value: Value,
    created_ms: Option<i64>,
    expires_ms: Option<i64>,
}

impl LegacyKv {
    /// The entry a v1 body becomes, or `None` for a row that is malformed or
    /// already expired.
    fn convert(self, body: &Value, now_ms: i64) -> Option<Converted> {
        let key = body.get("key")?.as_str()?.to_owned();
        let value = body.get("value")?.clone();
        let expires_ms = body
            .get("expiresAtMs")
            .and_then(Value::as_i64)
            .filter(|_| self != Self::Flat);
        if expires_ms.is_some_and(|at| at <= now_ms) {
            return None;
        }
        let (ns, created_ms) = match self {
            Self::Flat => (body.get("ns")?.as_str()?.to_owned(), None),
            Self::ResponseCache => (
                format!("{CACHE_NS_PREFIX}{}", body.get("ns")?.as_str()?),
                None,
            ),
            Self::Namespaced => {
                let segments: Vec<&str> = body
                    .get("ns")?
                    .as_array()?
                    .iter()
                    .filter_map(Value::as_str)
                    .collect();
                (
                    format!("{NAMESPACED_NS_PREFIX}{}", segments.join(".")),
                    body.get("createdAtMs").and_then(Value::as_i64),
                )
            }
        };
        Some(Converted {
            ns,
            key,
            value,
            created_ms,
            expires_ms,
        })
    }
}

/// Copy the v1 collection `legacy_collection` into `tsd.kv` under `owner`, then
/// remove it, and record the import in `tsd.migrations` (owner `owner`, step
/// `kv_import_<legacy_collection>`). Returns how many entries were written.
///
/// A completed import is skipped. A missing collection counts as completed.
/// Rows that are malformed or already expired are dropped, an entry already
/// present in `tsd.kv` is kept, and neither is counted.
///
/// # Errors
///
/// The first storage error; the import then resumes where it stopped on the next
/// call, because nothing is removed before every row has been copied.
pub async fn import_legacy_kv(
    storage: &ScopedStorage,
    owner: &str,
    legacy_collection: &str,
    kind: LegacyKv,
) -> Result<u64> {
    tinyagents_storage_kit::with_migration_lease(storage, owner, |lease| async move {
        import_legacy_kv_with_lease(storage, owner, legacy_collection, kind, &lease).await
    })
    .await
}

pub(crate) async fn import_legacy_kv_with_lease(
    storage: &ScopedStorage,
    owner: &str,
    legacy_collection: &str,
    kind: LegacyKv,
    lease: &tinyagents_storage_kit::MigrationLease,
) -> Result<u64> {
    let step = format!("kv_import_{legacy_collection}");
    let migrations = Migrations::new(storage);
    if migrations.done(owner, &step).await? {
        tracing::debug!(owner, step = %step, "[storage] kv import already done");
        return Ok(0);
    }
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(legacy_collection))
        .await?;
    let now_ms = crate::ids::now_ms() as i64;
    let mut written = 0_u64;
    let mut cursor: Option<Cursor> = None;
    loop {
        let mut query = Query::filter(Filter::default())
            .sort(Sort::asc(ID_FIELD))
            .limit(PAGE);
        if let Some(after) = cursor.take() {
            query = query.after(after);
        }
        let page = docs.query(legacy_collection, &query).await?;
        for row in page.items {
            let Some(entry) = kind.convert(&row.doc, now_ms) else {
                tracing::debug!(
                    collection = legacy_collection,
                    id = %row.id,
                    "[storage] kv import dropped a malformed or expired row"
                );
                continue;
            };
            let mut options = WriteOptions::new();
            if let Some(created) = entry.created_ms {
                options = options.created_ms(created);
            }
            if let Some(expires) = entry.expires_ms {
                options = options.expires_ms(expires);
            }
            let id = kv_id(owner, &entry.ns, &entry.key)?;
            let body = kv_body(owner, &entry.ns, &entry.key, entry.value);
            lease.renew().await?;
            match docs
                .put_with(KV, &id, body, Precondition::Absent, options)
                .await
            {
                Ok(_) => written += 1,
                Err(error)
                    if matches!(error.kind(), ErrorKind::AlreadyExists | ErrorKind::Conflict) => {}
                Err(error) => return Err(error),
            }
        }
        match page.next {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    lease.renew().await?;
    docs.delete_where(legacy_collection, &Filter::default())
        .await?;
    lease.mark_done(owner, &step).await?;
    tracing::debug!(owner, step = %step, written, "[storage] kv import complete");
    Ok(written)
}

#[cfg(test)]
#[path = "kv_import_tests.rs"]
mod tests;
