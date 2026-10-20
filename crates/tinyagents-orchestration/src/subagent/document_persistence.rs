//! Compare-and-swap subagent lifecycle state over scoped storage documents.
//!
//! One `tsd.kv` value contains both pause and terminal state, so independent
//! processes cannot commit competing lifecycle transitions. The neutral
//! `subagents/lifecycle-neutral` namespace leaves host-specific legacy records
//! available for the host to convert through [`DocumentSubagentPersistence::import_record_at`].

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use tinyagents_storage_kit::{Blocking, DocumentStore, Key, Kv, Repo, RetryPolicy, StorageError};
use tokio::sync::OnceCell;

use super::{
    PersistedSubagentPause, SubagentError, SubagentOutcome, SubagentOutcomeKind,
    SubagentPausePersistenceDisposition, SubagentPersistence, SubagentResume, SubagentTaskKey,
    SubagentTerminalPersistenceDisposition,
};

type Result<T> = std::result::Result<T, SubagentError>;
const OWNER: &str = "subagents";
const NS: &str = "lifecycle-neutral";

#[derive(Clone, Default, Serialize, Deserialize)]
struct LifecycleRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pause: Option<PersistedSubagentPause>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    terminal: Option<SubagentOutcome>,
}

/// The documented body of a `tsd.kv` document.
#[derive(Serialize, Deserialize)]
struct KvEntry {
    owner: String,
    ns: String,
    key: String,
    value: serde_json::Value,
}

/// Durable lifecycle persistence over an already tenant/agent-scoped handle.
///
/// Synchronous accessors and async trait calls share a blocking bridge. A
/// transition performs its complete read/CAS before returning, with no async
/// suspension after the commit boundary. Task lookup reads the live records
/// themselves, avoiding a separately committed index that can lose a key when
/// a process exits between two writes.
#[derive(Clone)]
pub struct DocumentSubagentPersistence {
    docs: Arc<dyn DocumentStore>,
    bridge: Blocking,
    declared: Arc<OnceCell<()>>,
}

impl DocumentSubagentPersistence {
    /// Opens a store over `docs`; its caller chooses the storage scope.
    ///
    /// # Errors
    /// Returns an error when the bridge runtime cannot start.
    pub fn new(docs: Arc<dyn DocumentStore>) -> std::result::Result<Self, StorageError> {
        Ok(Self {
            docs,
            bridge: Blocking::new()?,
            declared: Arc::new(OnceCell::new()),
        })
    }

    /// Stable component used for neutral and legacy lifecycle record lookup.
    pub fn key_component(key: &SubagentTaskKey) -> String {
        let hex = |value: &str| {
            value
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        let encoded = format!(
            "{}-{}-{}-{}",
            hex(&key.root_run_id),
            hex(&key.parent_run_id),
            hex(key.thread_id.as_deref().unwrap_or("")),
            hex(&key.task_id)
        );
        if encoded.len() <= 180 {
            encoded
        } else {
            format!(
                "sha256-{}",
                Sha256::digest(encoded.as_bytes())
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )
        }
    }

    fn name(key: &SubagentTaskKey) -> String {
        format!("record/{}", Self::key_component(key))
    }

    async fn declare(&self) -> std::result::Result<(), StorageError> {
        self.declared
            .get_or_try_init(|| async {
                self.docs
                    .ensure_collection(&tinyagents_storage_kit::primitives::kv_collection_spec())
                    .await?;
                Ok(())
            })
            .await
            .map(|_| ())
    }

    fn run<T: Send + 'static>(
        &self,
        work: impl std::future::Future<Output = std::result::Result<T, StorageError>> + Send + 'static,
    ) -> Result<T> {
        self.bridge.run(work).map_err(failure)?.map_err(failure)
    }

    fn read(&self, key: &SubagentTaskKey) -> Result<LifecycleRecord> {
        let this = self.clone();
        let name = Self::name(key);
        self.run(async move {
            this.declare().await?;
            Ok(Kv::ns(&this.docs, OWNER, NS)?
                .get(&name)
                .await?
                .unwrap_or_default())
        })
    }

    fn update<T, R>(
        &self,
        name: String,
        change: impl Fn(&mut T) -> (R, bool) + Send + Sync + 'static,
    ) -> Result<R>
    where
        T: DeserializeOwned + Serialize + Default + Send + 'static,
        R: Send + 'static,
    {
        let this = self.clone();
        self.run(async move {
            this.declare().await?;
            let repo = Repo::<KvEntry>::new(&this.docs, tinyagents_storage_kit::primitives::KV)?
                .with_retry(RetryPolicy::default());
            let id = Key::of([OWNER, NS, &name])?;
            let mut result = None;
            repo.update(id.as_str(), |current| {
                let mut record = current
                    .map(|entry| serde_json::from_value(entry.value.clone()))
                    .transpose()
                    .map_err(|error| StorageError::invalid_input(error.to_string()))?
                    .unwrap_or_default();
                let (outcome, changed) = change(&mut record);
                result = Some(outcome);
                if !changed {
                    return Ok(None);
                }
                Ok(Some(KvEntry {
                    owner: OWNER.into(),
                    ns: NS.into(),
                    key: name.clone(),
                    value: serde_json::to_value(record)
                        .map_err(|error| StorageError::invalid_input(error.to_string()))?,
                }))
            })
            .await?;
            result.ok_or_else(|| StorageError::backend("lifecycle CAS returned no result"))
        })
    }

    /// Returns live scoped task keys, preserving ambiguous bare task ids.
    pub fn keys_for_task(&self, task_id: &str) -> Result<Vec<SubagentTaskKey>> {
        let this = self.clone();
        let task_id = task_id.to_owned();
        self.run(async move {
            this.declare().await?;
            let records = Kv::ns(&this.docs, OWNER, NS)?
                .list::<LifecycleRecord>("record/")
                .await?;
            let mut keys = Vec::new();
            for (_, record) in records {
                if record.terminal.is_none()
                    && let Some(pause) = record.pause
                    && pause.key.task_id == task_id
                    && !keys.contains(&pause.key)
                {
                    keys.push(pause.key);
                }
            }
            Ok(keys)
        })
    }

    /// Reads an already committed terminal from synchronous host paths.
    pub fn terminal_outcome(&self, key: &SubagentTaskKey) -> Result<Option<SubagentOutcome>> {
        Ok(self.read(key)?.terminal)
    }

    /// Reads the live pause outcome from synchronous host recovery paths.
    pub fn pause_outcome(&self, key: &SubagentTaskKey) -> Result<Option<SubagentOutcome>> {
        let record = self.read(key)?;
        if record.terminal.is_some() {
            return Ok(None);
        }
        Ok(record
            .pause
            .filter(|pause| pause_resume(pause).is_some())
            .map(|pause| pause.outcome))
    }

    /// Imports converted legacy state, keeping newer committed parts.
    pub fn import_record(
        &self,
        key: &SubagentTaskKey,
        pause: Option<PersistedSubagentPause>,
        terminal: Option<SubagentOutcome>,
    ) -> Result<()> {
        if pause.as_ref().is_some_and(|pause| &pause.key != key) {
            return Err(SubagentError::Persistence(
                "imported pause key mismatch".into(),
            ));
        }
        self.import_record_at(&Self::key_component(key), pause, terminal)
    }

    /// Imports a legacy component whose full task key is unavailable.
    ///
    /// The host converts its old pause/terminal types and supplies their
    /// existing component. Terminal-only records need no reconstructed key;
    /// normal lookup uses the same component. Existing terminals always win.
    pub fn import_record_at(
        &self,
        component: &str,
        pause: Option<PersistedSubagentPause>,
        terminal: Option<SubagentOutcome>,
    ) -> Result<()> {
        if let Some(pause) = &pause
            && (Self::key_component(&pause.key) != component || pause_resume(pause).is_none())
        {
            return Err(SubagentError::Persistence(
                "imported pause component or status mismatch".into(),
            ));
        }
        self.update::<LifecycleRecord, ()>(format!("record/{component}"), move |record| {
            if record.terminal.is_some() {
                return ((), false);
            }
            if let Some(terminal) = &terminal {
                record.terminal = Some(terminal.clone());
                record.pause = None;
                return ((), true);
            }
            if record.pause.is_none() && pause.is_some() {
                record.pause.clone_from(&pause);
                return ((), true);
            }
            ((), false)
        })
    }
}

fn failure(error: impl std::fmt::Display) -> SubagentError {
    SubagentError::Persistence(error.to_string())
}

fn pause_resume(pause: &PersistedSubagentPause) -> Option<&SubagentResume> {
    match &pause.outcome.status {
        SubagentOutcomeKind::AwaitingInput(state) => Some(&state.resume),
        _ => None,
    }
}

#[async_trait]
impl SubagentPersistence for DocumentSubagentPersistence {
    async fn load_terminal(&self, key: &SubagentTaskKey) -> Result<Option<SubagentOutcome>> {
        self.terminal_outcome(key)
    }

    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>> {
        let record = self.read(key)?;
        if record.terminal.is_some() {
            return Ok(None);
        }
        Ok(record.pause.as_ref().and_then(pause_resume).cloned())
    }

    async fn load_pause(&self, key: &SubagentTaskKey) -> Result<Option<SubagentOutcome>> {
        self.pause_outcome(key)
    }

    async fn save_pause(
        &self,
        pause: PersistedSubagentPause,
    ) -> Result<SubagentPausePersistenceDisposition> {
        if pause_resume(&pause).is_none() {
            return Err(SubagentError::Persistence(
                "cannot persist a non-paused outcome as a pause".into(),
            ));
        }
        self.update::<LifecycleRecord, _>(Self::name(&pause.key), move |record| {
            if record.terminal.is_some() {
                return (SubagentPausePersistenceDisposition::TerminalExisting, false);
            }
            let matches = match (&record.pause, &pause.replaces) {
                (None, None) => true,
                (Some(current), Some(expected)) => {
                    pause_resume(current).is_some_and(|resume| resume == expected)
                }
                _ => false,
            };
            if !matches {
                return (SubagentPausePersistenceDisposition::Existing, false);
            }
            let replaced = record.pause.is_some();
            record.pause = Some(pause.clone());
            (
                if replaced {
                    SubagentPausePersistenceDisposition::Replaced
                } else {
                    SubagentPausePersistenceDisposition::Inserted
                },
                true,
            )
        })
    }

    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
        replaces: Option<&SubagentResume>,
    ) -> Result<SubagentTerminalPersistenceDisposition> {
        let outcome = outcome.clone();
        let replaces = replaces.cloned();
        self.update::<LifecycleRecord, _>(Self::name(key), move |record| {
            if record.terminal.is_some() {
                return (SubagentTerminalPersistenceDisposition::Existing, false);
            }
            let matches = match (&record.pause, &replaces) {
                (None, None) => true,
                (Some(current), Some(expected)) => {
                    pause_resume(current).is_some_and(|resume| resume == expected)
                }
                _ => false,
            };
            if !matches {
                return (SubagentTerminalPersistenceDisposition::PauseExisting, false);
            }
            record.pause = None;
            record.terminal = Some(outcome.clone());
            (SubagentTerminalPersistenceDisposition::Inserted, true)
        })
    }
}

#[cfg(test)]
#[path = "document_persistence_tests.rs"]
mod tests;
