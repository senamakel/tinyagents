//! The whole classic workspace for one agent, in one call.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::source::{QuiescentSource, SourceOwner};
use super::transcripts::{TranscriptImportReport, import_transcripts};
use super::turn_states::{TurnStateImportReport, import_turn_states};
use crate::port::DriverSessionStores;
use crate::run_ledger::drivers::LegacyDatabaseReport;

/// Where the classic files of one workspace live, relative to its root.
#[derive(Debug, Clone)]
pub struct WorkspaceLayout {
    /// The workspace root.
    pub workspace_dir: PathBuf,
}

impl WorkspaceLayout {
    /// The classic layout under `workspace_dir`.
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }

    /// `session_db/sessions.db`.
    pub fn sessions_db(&self) -> PathBuf {
        self.workspace_dir.join("session_db").join("sessions.db")
    }
}

/// What importing one workspace did.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LegacyWorkspaceReport {
    /// The database, when the workspace had one.
    pub database: Option<LegacyDatabaseReport>,
    /// The JSONL transcripts.
    pub transcripts: TranscriptImportReport,
    /// The per-turn snapshots.
    pub turn_states: TurnStateImportReport,
}

/// Import a classic workspace's database, transcripts and turn states into the
/// stores of `agent_id`.
///
/// Calling this is the host's assertion that nothing is writing the workspace
/// (see [`QuiescentSource`]) and that the workspace belongs to `agent_id`
/// (see [`SourceOwner`]). The opened database's contents and schema and the
/// classic input-file contents are unchanged; SQLite may create or update
/// WAL/SHM coordination files for the read-only database handle. Every record
/// keeps its id and is written only if the destination lacks it, so the call is safe to
/// repeat and safe after a partial run. A missing database, raw directory or
/// turn-state directory simply contributes nothing.
///
/// # Errors
///
/// A source that cannot be admitted (including an uncheckpointed WAL), a record
/// that does not decode, or a destination refusal. Work already done stays.
pub async fn import_workspace(
    provider: &DriverSessionStores,
    agent_id: &str,
    workspace_dir: &Path,
) -> anyhow::Result<LegacyWorkspaceReport> {
    let layout = WorkspaceLayout::new(workspace_dir);
    let stores = provider.try_for_agent(agent_id)?;
    let mut report = LegacyWorkspaceReport::default();

    let db = layout.sessions_db();
    if db.is_file() {
        let ledger = provider.driver_run_ledger(agent_id)?;
        let source = QuiescentSource::asserted_by_trusted_host(
            db,
            SourceOwner::asserted_by_trusted_host(ledger.scope().clone()),
        );
        report.database = Some(ledger.import_legacy_database(&source).await?);
    }

    // The file stores speak the synchronous seams, which run their port calls
    // on the provider's blocking bridge; keep that off the async workers.
    let workspace = workspace_dir.to_path_buf();
    let transcripts = stores.transcripts.clone();
    report.transcripts =
        tokio::task::spawn_blocking(move || import_transcripts(&workspace, transcripts.as_ref()))
            .await??;
    let workspace = workspace_dir.to_path_buf();
    let turn_states = stores.turn_states.clone();
    report.turn_states =
        tokio::task::spawn_blocking(move || import_turn_states(&workspace, turn_states.as_ref()))
            .await??;
    Ok(report)
}
