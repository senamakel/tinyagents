use async_trait::async_trait;

use super::{
    PersistedSubagentPause, SubagentError, SubagentOutcome, SubagentResume, SubagentTaskKey,
};

/// Host boundary for durable pause, resume, and terminal lifecycle state.
///
/// Implementations must make `record_terminal` idempotent by
/// [`SubagentTaskKey`] across process boundaries. The key is scoped by root
/// run, immediate parent run, and (when supplied) thread, so a bare task id is
/// never a global lifecycle identity. The driver additionally suppresses
/// duplicate records from repeated calls made through the same driver instance. A persistence
/// future's successful return is its commit boundary: implementations must not
/// make a write visible and then await again before returning `Ok(())`. The
/// driver races that boundary with cancellation and, when cancellation wins,
/// records one truthful `Cancelled` terminal outcome instead.
#[async_trait]
pub trait SubagentPersistence: Send + Sync {
    /// Loads the most recent resumable state, if a caller did not supply one.
    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError>;

    /// Saves one resumable pause. The driver never also records a terminal for
    /// that same committed outcome. A paused outcome is deliberately not cached
    /// by the driver; a later call reloads this state and resumes execution.
    async fn save_pause(&self, pause: PersistedSubagentPause) -> Result<(), SubagentError>;

    /// Records a non-pause terminal outcome exactly once per scoped task.
    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
    ) -> Result<(), SubagentError>;
}
