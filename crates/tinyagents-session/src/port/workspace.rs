//! Session stores for a host that only has a workspace directory, or that
//! already owns a storage backend.
//!
//! [`WorkspaceStores::from_backend`] builds the agent's stores on the supplied
//! backend. The deprecated [`WorkspaceStores::open`] convenience opens SQLite
//! at `<workspace>/storage/`, carries forward records from the former private
//! `<workspace>/session_store/` backend, and imports classic
//! `session_raw/*.jsonl` files once. Callers receive store ports; they never
//! name a transcript file.
//!
//! Handles are cached per `(workspace, agent)` so a call site that used to
//! construct a file locator on every use does not reopen the database.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use tinystoragedrivers_core::{DocumentStoreExt, Precondition, Query, Scope, StorageBackend};
use tinystoragedrivers_sqlite::SqliteStorage;

use super::AgentStores;
use super::types::WorkspaceInner;
use super::{DriverSessionStores, WorkspaceStores};
use crate::legacy::{OnceOutcome, import_transcripts_once};
use crate::transcript::TranscriptLocator;

/// Former directory under the workspace holding the private SQLite store.
/// Kept for detecting data to import from older releases.
pub const STORE_DIR: &str = "session_store";

const STORAGE_DIR: &str = "storage";
const LEGACY_COLLECTIONS: [&str; 5] = [
    "session.transcripts",
    "session.transcript_entries",
    "session.turn_states",
    "tsd.kv",
    "tsd.migrations",
];
const JOURNAL_PREFIX: &str = "harness.journal/";

type Cache = Mutex<HashMap<(PathBuf, String), Weak<WorkspaceInner>>>;

/// Weak entries: a backend is released once every handle drops, while live
/// handles are still shared.
fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Serialises first opens so two threads never run the legacy import for the
/// same workspace concurrently (one could see the other's half-written
/// transcript as complete).
static OPEN_LOCK: Mutex<()> = Mutex::new(());

fn cached(key: &(PathBuf, String)) -> Option<WorkspaceStores> {
    let mut map = cache().lock().unwrap_or_else(PoisonError::into_inner);
    match map.get(key).and_then(Weak::upgrade) {
        Some(inner) => Some(WorkspaceStores { inner }),
        None => {
            map.remove(key);
            None
        }
    }
}

impl WorkspaceStores {
    /// Builds the stores for `agent_id` on the supplied backend.
    ///
    /// This constructor does not open a private database or import files. The
    /// backend must already be configured for the tenant whose agent scope is
    /// `local/agent:<agent_id>`.
    ///
    /// # Errors
    ///
    /// The provider cannot be initialized or the agent scope cannot be bound.
    pub fn from_backend(backend: Arc<dyn StorageBackend>, agent_id: &str) -> anyhow::Result<Self> {
        let provider = Arc::new(DriverSessionStores::new(backend, Scope::local())?);
        let stores = provider.try_for_agent(agent_id)?;
        let inner = Arc::new(WorkspaceInner {
            provider,
            stores,
            outcome: OnceOutcome::NotAttempted,
        });
        Ok(Self { inner })
    }

    /// Opens the stores of `agent_id` under `<workspace>/storage/` and imports
    /// classic workspace transcripts on first open.
    ///
    /// Existing data written by earlier versions under `<workspace>/session_store/`
    /// is copied into the shared backend before the classic file import. The
    /// source database and JSONL files remain untouched, and the copy is
    /// idempotent so an interrupted open can be retried.
    ///
    /// Prefer [`Self::from_backend`] when the host already owns a storage
    /// backend.
    ///
    /// # Errors
    ///
    /// SQLite cannot be opened, data cannot be copied, or the classic import
    /// is refused. Nothing is cached then, so the next call retries.
    #[deprecated(note = "use WorkspaceStores::from_backend with the host storage backend")]
    pub fn open(workspace_dir: &Path, agent_id: &str) -> anyhow::Result<Self> {
        let key = (workspace_dir.to_path_buf(), agent_id.to_string());
        if let Some(found) = cached(&key) {
            return Ok(found);
        }
        let _opening = OPEN_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        // Another thread may have finished while this one waited.
        if let Some(found) = cached(&key) {
            return Ok(found);
        }
        let backend: Arc<dyn StorageBackend> =
            Arc::new(SqliteStorage::open(workspace_dir.join(STORAGE_DIR))?);
        let provider = Arc::new(DriverSessionStores::new(
            Arc::clone(&backend),
            Scope::local(),
        )?);
        let legacy_db = workspace_dir.join(STORE_DIR).join("storage.db");
        if legacy_db.is_file() {
            let legacy_backend: Arc<dyn StorageBackend> =
                Arc::new(SqliteStorage::open(workspace_dir.join(STORE_DIR))?);
            let legacy_provider = DriverSessionStores::new(legacy_backend, Scope::local())?;
            legacy_provider.try_for_agent(agent_id)?;
            provider.try_for_agent(agent_id)?;
            copy_agent_storage(&legacy_provider, &provider, agent_id)?;
        }
        let stores = provider.try_for_agent(agent_id)?;
        let outcome = import_transcripts_once(&provider, agent_id, workspace_dir)?;
        tracing::debug!(
            target: "tinyagents_session::port::workspace",
            agent_id,
            "[session-store] workspace stores opened"
        );
        // An import that skipped an unreadable file is not complete: do not
        // cache it, so the next open retries once the file is fixed.
        let complete = matches!(
            outcome,
            OnceOutcome::AlreadyImported | OnceOutcome::Imported { marked: true, .. }
        );
        let inner = Arc::new(WorkspaceInner {
            provider,
            stores,
            outcome,
        });
        if complete {
            cache()
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key, Arc::downgrade(&inner));
        }
        Ok(Self { inner })
    }

    /// The transcript port.
    pub fn transcripts(&self) -> Arc<dyn TranscriptLocator> {
        Arc::clone(&self.inner.stores.transcripts)
    }

    /// Every store of the agent.
    pub fn stores(&self) -> &AgentStores {
        &self.inner.stores
    }

    /// The provider the stores come from.
    pub fn provider(&self) -> &Arc<DriverSessionStores> {
        &self.inner.provider
    }

    /// What the first open's legacy import did.
    pub fn import_outcome(&self) -> &OnceOutcome {
        &self.inner.outcome
    }
}

/// Copies the old private SQLite store through the storage ports. Documents
/// are copied only when absent; streams resume from the destination's next
/// offset. Keeping the source intact makes retries safe after a partial copy.
fn copy_agent_storage(
    source: &DriverSessionStores,
    destination: &DriverSessionStores,
    agent_id: &str,
) -> anyhow::Result<()> {
    let (source_scope, _) = source.scoped_for(agent_id)?;
    let (destination_scope, _) = destination.scoped_for(agent_id)?;
    source.bridge().run(async move {
        let source_documents = source_scope.documents();
        let destination_documents = destination_scope.documents();
        for collection in LEGACY_COLLECTIONS {
            for row in source_documents
                .query_all(collection, &Query::all())
                .await?
            {
                if destination_documents
                    .get(collection, &row.id)
                    .await?
                    .is_none()
                {
                    destination_documents
                        .put(collection, &row.id, row.doc, Precondition::Absent)
                        .await?;
                }
            }
        }

        let source_streams = source_scope.streams();
        let destination_streams = destination_scope.streams();
        for stream in source_streams.streams(JOURNAL_PREFIX).await? {
            let Some(info) = source_streams.info(&stream).await? else {
                continue;
            };
            let destination_info = destination_streams.info(&stream).await?;
            let mut next = destination_info.map_or(0, |found| found.len);
            if let Some(found) = destination_info {
                if found.base != 0 && found.base != info.base {
                    return Err(tinystoragedrivers_core::StorageError::backend(format!(
                        "cannot safely import journal {stream}: source and destination retain different offsets"
                    )));
                }
                // A matching retained prefix identifies an interrupted copy;
                // a different prefix is newer or unrelated destination data.
                if next > info.base {
                    let overlap_end = next.min(info.len);
                    let old = source_streams
                        .read_window(&stream, info.base, (overlap_end - info.base) as usize)
                        .await?;
                    let copied = destination_streams
                        .read_window(&stream, info.base, (overlap_end - info.base) as usize)
                        .await?;
                    if old != copied {
                        return Err(tinystoragedrivers_core::StorageError::backend(format!(
                            "cannot safely import journal {stream}: destination prefix differs"
                        )));
                    }
                } else if next > 0 {
                    let placeholders = destination_streams.read_window(&stream, 0, next as usize).await?;
                    if placeholders.len() != next as usize
                        || placeholders
                            .iter()
                            .any(|entry| !entry.value.is_null())
                    {
                        return Err(tinystoragedrivers_core::StorageError::backend(format!(
                            "cannot safely resume journal {stream}: destination prefix differs"
                        )));
                    }
                }
            }

            // Truncated streams retain their original next offset. Null
            // placeholders recreate only the already-discarded prefix, which
            // is immediately truncated after the retained events are copied.
            while next < info.base {
                let count = (info.base - next).min(256) as usize;
                destination_streams
                    .append_batch(&stream, vec![serde_json::Value::Null; count])
                    .await?;
                next += count as u64;
            }
            while next < info.len {
                let entries = source_streams
                    .read_window(&stream, next, 256)
                    .await?;
                if entries.is_empty() {
                    break;
                }
                let values = entries.into_iter().map(|entry| entry.value).collect();
                let count = values.len() as u64;
                destination_streams.append_batch(&stream, values).await?;
                next += count;
            }
            if info.base > 0 {
                destination_streams.truncate_before(&stream, info.base).await?;
            }
            if info.is_sealed() {
                destination_streams.seal(&stream).await?;
            }
        }
        Ok::<(), tinystoragedrivers_core::StorageError>(())
    })??;
    Ok(())
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;
