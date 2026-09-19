//! Shared helper for running blocking (synchronous file/DB) work off the
//! tokio runtime.
//!
//! Several backends — [`crate::store::FileStore`], the JSONL append store, and
//! [`crate::cache::SqliteResponseCache`] under the `sqlite` feature — perform
//! blocking I/O (`std::fs::*`, `rusqlite` calls) that must never run directly
//! inside an `async fn` body, since that stalls whichever tokio worker thread
//! happens to poll it. [`run_blocking`] offloads the work via
//! `tokio::task::spawn_blocking` when a runtime is present, and falls back to
//! running it inline when there is none (e.g. a synchronous caller outside any
//! runtime, such as some test harnesses).

use crate::error::{Result, TinyAgentsError};

/// Runs `work` off the async runtime via `spawn_blocking`, falling back to
/// running it inline when no tokio runtime is currently entered.
pub(crate) async fn run_blocking<F, T>(work: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle
            .spawn_blocking(work)
            .await
            .map_err(|e| TinyAgentsError::Validation(format!("blocking task error: {e}")))?,
        Err(_) => work(),
    }
}
