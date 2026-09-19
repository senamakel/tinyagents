use async_trait::async_trait;

use crate::{RuntimeError, SessionTerminal, SessionTurnOutcome, SessionTurnRequest};

/// Host observation/preparation around a session turn.
///
/// Hooks do not grant tools, select models, compose product prompts, or own
/// transcript state.  They can prepare the input and observe committed results.
#[async_trait]
pub trait SessionHooks: Send + Sync {
    /// Runs before the driver sees the request.
    async fn before_turn(&self, request: &mut SessionTurnRequest) -> Result<(), RuntimeError>;
    /// Runs after the driver has produced a candidate and before it commits.
    /// Returning an error or observing cancellation therefore leaves no
    /// durable session mutation behind.
    async fn after_turn(&self, outcome: &SessionTurnOutcome) -> Result<(), RuntimeError>;
    /// Runs exactly once after a successful durable transcript commit.
    ///
    /// Errors and cancellation observed here are deliberately observational:
    /// the result has already become durable and remains successful.
    async fn after_commit(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }
    /// Runs exactly once for every terminal turn result.
    async fn on_terminal(&self, terminal: &SessionTerminal) -> Result<(), RuntimeError>;
}

/// A no-op hook set for hosts that need no lifecycle observation.
#[derive(Default)]
pub struct NoopSessionHooks;

#[async_trait]
impl SessionHooks for NoopSessionHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn after_commit(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: &SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}
