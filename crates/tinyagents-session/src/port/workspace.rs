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

use tinyagents_harness::store::SchemaAdmission;
use tinystoragedrivers_core::{Scope, StorageBackend};
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
mod workspace_migration;

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
            let legacy_backend: Arc<dyn StorageBackend> = Arc::new(
                SqliteStorage::open_existing_read_only(workspace_dir.join(STORE_DIR))?,
            );
            let legacy_provider = DriverSessionStores::new_with_schema_admission(
                legacy_backend,
                Scope::local(),
                SchemaAdmission::ValidateOnly,
            )?;
            legacy_provider.try_for_agent(agent_id)?;
            provider.try_for_agent(agent_id)?;
            workspace_migration::copy_agent_storage(&legacy_provider, &provider, agent_id)?;
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

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;
