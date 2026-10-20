//! Durable [`HarnessStatusStore`] over `tinystoragedrivers` document ports, plus
//! the run-id and redaction helpers a host needs to attach a durable journal to
//! a run.
//!
//! The crate ships [`super::types::InMemoryStatusStore`], which does not survive
//! a process restart. [`DriverStatusStore`] keeps one document per run on
//! whatever backend the host opened and answers the lineage and liveness
//! queries with indexed filters, which is what lets a supervisor reattach after
//! the UI died and find every active descendant of a root run.

use std::sync::Arc;

use async_trait::async_trait;
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, DocumentStoreExt, Filter, IndexSpec, Key, Precondition, Query,
    WriteOptions,
};
use tokio::sync::OnceCell;

use super::HarnessStatusStore;
use super::status_record::RunStatusRecord;
use crate::error::Result;
use crate::events::HarnessRunStatus;
use crate::ids::{ExecutionStatus, RunId};
use crate::store::map_driver_error as map_error;

/// Name fragments that mark an environment variable as credential material.
const SECRET_NAME_MARKERS: [&str; 7] = [
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "BEARER",
];

/// Shortest value (in bytes, after trimming) worth masking.
const MIN_SECRET_LEN: usize = 8;

/// Mints a fresh, slash-free, process-unique run id (`run.<32-hex>`).
///
/// One id serves a whole turn: the `EventSink::with_stream_id` prefix (so
/// persisted `event_id`s are the restart-stable `{run_id}-evt-{offset}`), the
/// journal stream key, and the status-store key. The `simple()` uuid form (no
/// hyphens) keeps the id readable in logs and file names.
pub fn mint_run_id() -> RunId {
    RunId::new(format!("run.{}", uuid::Uuid::new_v4().simple()))
}

/// Values of process environment variables whose name looks like a secret
/// (contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL` or
/// `BEARER`, case-insensitively) and whose trimmed value is at least eight
/// bytes long.
///
/// Seed a [`super::RedactingSink`] with this so an API key or bearer token
/// echoed into model text, a tool argument fragment or an error string is never
/// persisted in the clear. Short values are left alone so unrelated config is
/// not masked.
pub fn process_env_secrets() -> Vec<String> {
    secrets_from_vars(std::env::vars())
}

/// Pure core of [`process_env_secrets`], over an explicit variable list.
pub fn secrets_from_vars(vars: impl IntoIterator<Item = (String, String)>) -> Vec<String> {
    let mut secrets = Vec::new();
    for (name, value) in vars {
        let upper = name.to_ascii_uppercase();
        if SECRET_NAME_MARKERS.iter().any(|m| upper.contains(m))
            && value.trim().len() >= MIN_SECRET_LEN
        {
            secrets.push(value);
        }
    }
    tracing::debug!(
        count = secrets.len(),
        "[journal] redaction policy seeded with secret value(s)"
    );
    secrets
}

/// A durable [`HarnessStatusStore`] stored in a driver [`DocumentStore`].
///
/// One `harness.run_status` document per run, id `Key::of([run_id])`: the
/// status as a snake_case body with `*_ms` times, plus the indexed projections
/// (`root_run_id`, `thread_id`, `active`) the queries filter on. Each snapshot
/// is compact (ids, phase, counters, timestamps, error), never prompts or
/// payloads. The handle is already bound to a tenant scope, so runs of two
/// tenants never meet.
#[derive(Debug, Clone)]
pub struct DriverStatusStore {
    docs: Arc<dyn DocumentStore>,
    declared: Arc<OnceCell<()>>,
}

impl DriverStatusStore {
    /// The collection every store uses.
    pub const COLLECTION: &'static str = crate::storage::RUN_STATUS;

    /// Store statuses in [`Self::COLLECTION`].
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self {
            docs,
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// The declaration a store needs.
    pub fn collection_spec() -> CollectionSpec {
        CollectionSpec::new(Self::COLLECTION)
            .index(IndexSpec::new("by_root", ["root_run_id"]))
            .index(IndexSpec::new("by_thread", ["thread_id"]))
            .index(IndexSpec::new("by_active", ["active"]))
    }

    fn id(run_id: &str) -> Result<String> {
        Ok(Key::of([run_id]).map_err(map_error)?.as_str().to_owned())
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                self.docs
                    .ensure_collection(&Self::collection_spec())
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }

    /// Every status matching `filter`. A document that no longer decodes (a
    /// record from an incompatible build) is skipped, never fatal.
    async fn matching(&self, filter: Filter) -> Result<Vec<HarnessRunStatus>> {
        self.declared().await?;
        let rows = self
            .docs
            .query_all(Self::COLLECTION, &Query::filter(filter))
            .await
            .map_err(map_error)?;
        Ok(rows
            .into_iter()
            .filter_map(
                |row| match serde_json::from_value::<RunStatusRecord>(row.doc) {
                    Ok(record) => Some(HarnessRunStatus::from(record)),
                    Err(err) => {
                        tracing::debug!(
                            "[journal] skipping undecodable run status id={} err={err}",
                            row.id
                        );
                        None
                    }
                },
            )
            .collect())
    }
}

#[async_trait]
impl HarnessStatusStore for DriverStatusStore {
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()> {
        self.declared().await?;
        let body = serde_json::to_value(RunStatusRecord::from(&status))?;
        self.docs
            .put_with(
                Self::COLLECTION,
                &Self::id(status.run_id.as_str())?,
                body,
                Precondition::None,
                WriteOptions::new().schema(crate::storage::SCHEMA_VERSION),
            )
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    async fn get_status(&self, run_id: &str) -> Result<Option<HarnessRunStatus>> {
        self.declared().await?;
        let Some(found) = self
            .docs
            .get(Self::COLLECTION, &Self::id(run_id)?)
            .await
            .map_err(map_error)?
        else {
            return Ok(None);
        };
        Ok(serde_json::from_value::<RunStatusRecord>(found.doc)
            .ok()
            .map(HarnessRunStatus::from))
    }

    async fn list_by_thread(&self, thread_id: &str) -> Result<Vec<HarnessRunStatus>> {
        self.matching(Filter::eq("thread_id", thread_id)).await
    }

    async fn list_by_root(&self, root_run_id: &str) -> Result<Vec<HarnessRunStatus>> {
        self.matching(Filter::eq("root_run_id", root_run_id)).await
    }

    async fn list_active(&self) -> Result<Vec<HarnessRunStatus>> {
        self.matching(Filter::eq("active", true)).await
    }
}

/// Whether a run is still live (`Pending`, `Running` or `Interrupted`).
pub fn is_active(status: &HarnessRunStatus) -> bool {
    matches!(
        status.status,
        ExecutionStatus::Pending | ExecutionStatus::Running | ExecutionStatus::Interrupted
    )
}

#[cfg(test)]
#[path = "status_tests.rs"]
mod tests;
