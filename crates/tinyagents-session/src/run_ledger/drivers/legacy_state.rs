//! Moving the v1 state that cannot be a document step into the shared `tsd.*`
//! primitives: a document step writes only its own owner's collections.
//!
//! [`import_legacy_ledger`] is this owner's once-per-scope self-migration. The
//! host invokes it on the already-bound scope before `Registry::migrate`. It moves:
//!
//! - the live driver leases embedded in `run_ledger_workflow_runs` to
//!   `tsd.leases` (`ledger`, `Key::of(["workflow", run_id])`). Call it **before**
//!   `Registry::migrate`: the workflow step removes the v1 rows (and their
//!   lease fields). Run afterwards it finds nothing, and a lease that was
//!   held at the time simply lapses, which a restart implies anyway.
//! - the dense id counters of `session_history_counters` to `tsd.counters`
//!   (`history`, `messages` / `tool_calls`).
//!
//! The agent-import receipts (`run_ledger_agent_imports`) are dropped: their v1
//! id is a digest of the operation id, which a `tsd.migrations` record keyed by
//! operation id cannot be rebuilt from. The rows such an import wrote are
//! migrated, so re-running the same operation id after the upgrade is refused
//! with a conflict on the existing rows instead of being a no-op.
//!
//! A renewable `ledger` migration lease protects the scan, writes, deletes and
//! completion marker in `tsd.migrations` (step `legacy_state_import`); the
//! importer is idempotent and resumable.

use super::storage::{HISTORY_OWNER, LEDGER_OWNER, key, workflow_lease_key};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tinyagents_storage_kit::primitives::{self, LEASES};
use tinyagents_storage_kit::{
    CollectionSpec, Cursor, ErrorKind, Filter, ID_FIELD, Precondition, Query, Result, Sort,
};
use tinyagents_storage_kit::{
    Counters, MIGRATIONS_COLLECTION, Migrations, ScopedStorage, with_migration_lease,
};

const STEP: &str = "legacy_state_import";
const PAGE: usize = 200;

/// What [`import_legacy_ledger`] moved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegacyLedgerImport {
    /// Workflow leases written to `tsd.leases` (live ones only).
    pub leases: u64,
    /// History counters raised in `tsd.counters`.
    pub counters: u64,
}

/// The v1 id of a record: a digest of the length-prefixed parts.
fn legacy_key(parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part.as_bytes());
    }
    hex::encode(digest.finalize())
}

async fn rows(
    storage: &ScopedStorage,
    collection: &str,
) -> Result<Vec<tinystoragedrivers_types::Versioned<Value>>> {
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(collection))
        .await?;
    let mut out = Vec::new();
    let mut cursor: Option<Cursor> = None;
    loop {
        let mut query = Query::filter(Filter::default())
            .sort(Sort::asc(ID_FIELD))
            .limit(PAGE);
        if let Some(after) = cursor.take() {
            query = query.after(after);
        }
        let page = docs.query(collection, &query).await?;
        out.extend(page.items);
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(out),
        }
    }
}

/// Move the v1 workflow leases and history counters of `storage`'s scope into
/// `tsd.leases` and `tsd.counters`; see the module docs.
///
/// # Errors
/// The first storage error; the import resumes on the next call.
pub async fn import_legacy_ledger(storage: &ScopedStorage) -> Result<LegacyLedgerImport> {
    with_migration_lease(storage, LEDGER_OWNER, |lease| async move {
        let migrations = Migrations::new(storage);
        primitives::ensure(storage).await?;
        storage
            .documents()
            .ensure_collection(&CollectionSpec::new(MIGRATIONS_COLLECTION))
            .await?;
        let mut report = LegacyLedgerImport::default();
        if migrations.done(LEDGER_OWNER, STEP).await? {
            return Ok(report);
        }
        let docs = storage.documents();
        let now_ms = chrono::Utc::now().timestamp_millis();
        for row in rows(storage, "run_ledger_workflow_runs").await? {
            let Some(record) = row.doc.get("record") else {
                continue;
            };
            let (Some(run_id), Some(holder), Some(expires)) = (
                record.get("id").and_then(Value::as_str),
                record.get("leaseOwner").and_then(Value::as_str),
                record
                    .get("leaseExpiresAt")
                    .and_then(Value::as_str)
                    .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok()),
            ) else {
                continue;
            };
            if expires.timestamp_millis() <= now_ms {
                continue;
            }
            let id = key(&[LEDGER_OWNER, &workflow_lease_key(run_id)?])?;
            let body =
                json!({"holder": holder, "epoch": 1, "expires_ms": expires.timestamp_millis()});
            match docs.put(LEASES, &id, body, Precondition::Absent).await {
                Ok(_) => report.leases += 1,
                Err(error)
                    if matches!(error.kind(), ErrorKind::Conflict | ErrorKind::AlreadyExists) => {}
                Err(error) => return Err(error),
            }
        }
        let counters = Counters::new(docs);
        let names = [
            ("messages", super::history::MESSAGE_COUNTER),
            ("tool_calls", super::history::TOOL_CALL_COUNTER),
        ];
        for row in rows(storage, "session_history_counters").await? {
            for (v1_name, name) in names {
                if row.id != legacy_key(&[v1_name]) {
                    continue;
                }
                // The v1 counter held the next id to hand out; the primitive holds the last.
                let next = row
                    .doc
                    .pointer("/record/next")
                    .and_then(Value::as_u64)
                    .unwrap_or(1);
                counters
                    .set_at_least(HISTORY_OWNER, name, next.saturating_sub(1))
                    .await?;
                report.counters += 1;
            }
        }
        docs.delete_where("session_history_counters", &Filter::default())
            .await?;
        docs.delete_where("run_ledger_agent_imports", &Filter::default())
            .await
            .ok();
        lease.mark_done(LEDGER_OWNER, STEP).await?;
        Ok(report)
    })
    .await
}

#[cfg(test)]
#[path = "legacy_state_tests.rs"]
mod tests;
