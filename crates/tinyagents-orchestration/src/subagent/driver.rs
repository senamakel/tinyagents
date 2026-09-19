use std::{collections::HashMap, future::Future, sync::Arc};

use tokio::sync::{Mutex, Notify};

use super::{
    PersistedSubagentPause, SubagentError, SubagentExecution, SubagentExecutor, SubagentOutcome,
    SubagentPersistence, SubagentPlanner, SubagentRequest, SubagentStatus, SubagentTaskKey,
};
use tinyagents_harness::CancellationToken;

/// Optional host seams accepted by [`SubagentDriver::new`].
///
/// Hosts that cannot provide every seam must receive a typed construction error
/// rather than accidentally executing a partial lifecycle.
pub struct SubagentCapabilities<C: Send + 'static = (), H: Send + 'static = ()> {
    /// Host planner, which resolves policy and explicit execution inputs.
    pub planner: Option<Arc<dyn SubagentPlanner<C, H>>>,
    /// Host executor, which drives the prepared run.
    pub executor: Option<Arc<dyn SubagentExecutor<C>>>,
    /// Host persistence for resume and one lifecycle record.
    pub persistence: Option<Arc<dyn SubagentPersistence>>,
}

/// Generic lifecycle driver for one host's subagent runs.
///
/// Concurrent calls for one scoped task key coalesce, while task ids from
/// distinct parents, roots, or threads remain independent. Hosts still need
/// idempotent persistence for multiple processes/drivers.
pub struct SubagentDriver<C: Send + 'static = (), H: Send + 'static = ()> {
    planner: Arc<dyn SubagentPlanner<C, H>>,
    executor: Arc<dyn SubagentExecutor<C>>,
    persistence: Arc<dyn SubagentPersistence>,
    terminal_outcomes: Mutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    in_flight: Mutex<HashMap<SubagentTaskKey, Arc<InFlight>>>,
}

/// Result shared by callers that arrived while the same task was executing.
///
/// The map only protects reservation and removal. Planner, executor, and
/// persistence futures never run while it is locked.
struct InFlight {
    result: Mutex<Option<Result<SubagentOutcome, SubagentError>>>,
    notify: Notify,
}

impl InFlight {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            notify: Notify::new(),
        }
    }

    async fn wait(
        &self,
        cancellation: &CancellationToken,
        task_id: &str,
    ) -> Result<SubagentOutcome, SubagentError> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.result.lock().await.clone() {
                return result;
            }
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Ok(SubagentOutcome::cancelled(task_id)),
                _ = &mut notified => {}
            }
        }
    }

    async fn complete(&self, result: Result<SubagentOutcome, SubagentError>) {
        *self.result.lock().await = Some(result);
        self.notify.notify_waiters();
    }
}

impl<C: Send + 'static, H: Send + 'static> SubagentDriver<C, H> {
    /// Validates host capability availability before exposing a runnable driver.
    pub fn new(capabilities: SubagentCapabilities<C, H>) -> Result<Self, SubagentError> {
        Ok(Self {
            planner: capabilities
                .planner
                .ok_or(SubagentError::MissingCapability("planner"))?,
            executor: capabilities
                .executor
                .ok_or(SubagentError::MissingCapability("executor"))?,
            persistence: capabilities
                .persistence
                .ok_or(SubagentError::MissingCapability("persistence"))?,
            terminal_outcomes: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashMap::new()),
        })
    }

    /// Runs `load -> prepare -> execute -> one persistence action`.
    ///
    /// A caller-supplied resume bypasses loading. Cancellation is checked before
    /// preparation, after every awaited lifecycle stage, and after execution;
    /// a cancellation observed after an executor result preserves its output,
    /// history, usage, and artifact references while changing only the status.
    pub async fn run(
        &self,
        request: SubagentRequest<C, H>,
        cancellation: CancellationToken,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_key = request.task_key().clone();
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            return Ok(outcome);
        }

        let (entry, is_leader) = {
            let mut in_flight = self.in_flight.lock().await;
            match in_flight.get(&task_key) {
                Some(entry) => (entry.clone(), false),
                None => {
                    let entry = Arc::new(InFlight::new());
                    in_flight.insert(task_key.clone(), entry.clone());
                    (entry, true)
                }
            }
        };
        if !is_leader {
            return entry.wait(&cancellation, &task_key.task_id).await;
        }
        // A preceding caller may have committed a terminal result between the
        // initial cache check and this reservation. Do not reopen that task
        // after its in-flight entry has been removed.
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            entry.complete(Ok(outcome.clone())).await;
            let mut in_flight = self.in_flight.lock().await;
            in_flight.remove(&task_key);
            return Ok(outcome);
        }

        let result = self
            .run_reserved(request, task_key.clone(), cancellation)
            .await;
        entry.complete(result.clone()).await;
        let mut in_flight = self.in_flight.lock().await;
        if in_flight
            .get(&task_key)
            .is_some_and(|current| Arc::ptr_eq(current, &entry))
        {
            in_flight.remove(&task_key);
        }
        result
    }

    async fn run_reserved(
        &self,
        mut request: SubagentRequest<C, H>,
        task_key: SubagentTaskKey,
        cancellation: CancellationToken,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_id = request.task_id().to_owned();

        if cancellation.is_cancelled() {
            return self.persist_cancelled(task_key, task_id).await;
        }

        if request.resume().is_none() {
            request.set_resume(self.persistence.load(&task_key).await?);
        }
        if cancellation.is_cancelled() {
            return self.persist_cancelled(task_key, task_id).await;
        }

        let mut prepared = self.planner.prepare(request).await?;
        if prepared.task_id != task_id {
            return Err(SubagentError::TaskIdMismatch {
                expected: task_id,
                actual: prepared.task_id,
            });
        }
        if cancellation.is_cancelled() {
            return self.persist_cancelled(task_key, prepared.task_id).await;
        }
        prepared.run_context = prepared.run_context.with_cancellation(cancellation.clone());

        let executed = self
            .executor
            .execute(SubagentExecution {
                prepared,
                cancellation: cancellation.clone(),
            })
            .await;

        let outcome = match executed {
            Ok(outcome) => {
                if outcome.task_id != task_id {
                    return Err(SubagentError::TaskIdMismatch {
                        expected: task_id,
                        actual: outcome.task_id,
                    });
                }
                if cancellation.is_cancelled() {
                    outcome.cancelled_preserving()
                } else {
                    outcome
                }
            }
            Err(SubagentError::Cancelled) => SubagentOutcome::cancelled(task_id),
            Err(_) if cancellation.is_cancelled() => SubagentOutcome::cancelled(task_id),
            Err(error) => return Err(error),
        };
        self.persist(task_key, outcome, &cancellation).await
    }

    async fn persist_cancelled(
        &self,
        task_key: SubagentTaskKey,
        task_id: String,
    ) -> Result<SubagentOutcome, SubagentError> {
        let outcome = SubagentOutcome::cancelled(task_id);
        // Once cancellation has won, this is the one terminal action. Do not
        // race it with the already-latched token or a caller could observe an
        // indeterminate terminal write.
        self.persistence
            .record_terminal(&task_key, &outcome)
            .await?;
        self.cache_terminal(task_key, &outcome).await;
        Ok(outcome)
    }

    async fn persist(
        &self,
        task_key: SubagentTaskKey,
        outcome: SubagentOutcome,
        cancellation: &CancellationToken,
    ) -> Result<SubagentOutcome, SubagentError> {
        if matches!(&outcome.status, SubagentStatus::Cancelled) {
            self.persistence
                .record_terminal(&task_key, &outcome)
                .await?;
            self.cache_terminal(task_key, &outcome).await;
            return Ok(outcome);
        }
        let committed = match &outcome.status {
            SubagentStatus::AwaitingInput(pause) => {
                self.commit_or_cancel(
                    self.persistence.save_pause(PersistedSubagentPause {
                        key: task_key.clone(),
                        pause: pause.clone(),
                    }),
                    cancellation,
                )
                .await?
            }
            SubagentStatus::Completed | SubagentStatus::Incomplete(_) => {
                self.commit_or_cancel(
                    self.persistence.record_terminal(&task_key, &outcome),
                    cancellation,
                )
                .await?
            }
            SubagentStatus::Cancelled => unreachable!("handled before persistence race"),
        };
        if !committed {
            return self.persist_cancelled(task_key, outcome.task_id).await;
        }
        if matches!(
            &outcome.status,
            SubagentStatus::Completed | SubagentStatus::Incomplete(_)
        ) {
            self.cache_terminal(task_key, &outcome).await;
        }
        Ok(outcome)
    }

    async fn commit_or_cancel<F>(
        &self,
        operation: F,
        cancellation: &CancellationToken,
    ) -> Result<bool, SubagentError>
    where
        F: Future<Output = Result<(), SubagentError>>,
    {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Ok(false),
            result = operation => {
                result?;
                Ok(true)
            }
        }
    }

    async fn cache_terminal(&self, key: SubagentTaskKey, outcome: &SubagentOutcome) {
        self.terminal_outcomes
            .lock()
            .await
            .insert(key, outcome.clone());
    }
}
