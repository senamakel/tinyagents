use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex},
};

use tokio::sync::{Mutex as AsyncMutex, Notify};

use super::{
    PersistedSubagentPause, SubagentError, SubagentExecution, SubagentExecutor, SubagentOutcome,
    SubagentPausePersistenceDisposition, SubagentPersistence, SubagentPersistenceDisposition,
    SubagentPlanner, SubagentRequest, SubagentRunResult, SubagentStatus, SubagentTaskKey,
    SubagentTerminalPersistenceDisposition,
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
    terminal_outcomes: AsyncMutex<HashMap<SubagentTaskKey, SubagentOutcome>>,
    in_flight: Arc<Mutex<HashMap<SubagentTaskKey, Arc<InFlight>>>>,
}

/// Result shared by callers that arrived while the same task was executing.
///
/// The map only protects reservation and removal. Planner, executor, and
/// persistence futures never run while it is locked.
struct InFlight {
    result: Mutex<Option<Result<SubagentRunResult, SubagentError>>>,
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
    ) -> Result<SubagentRunResult, SubagentError> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self
                .result
                .lock()
                .expect("in-flight result poisoned")
                .clone()
            {
                return result;
            }
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => return Ok(SubagentRunResult::new(SubagentOutcome::cancelled(task_id), SubagentPersistenceDisposition::ObserverCancelled)),
                _ = &mut notified => {}
            }
        }
    }

    fn complete(&self, result: Result<SubagentRunResult, SubagentError>) {
        *self.result.lock().expect("in-flight result poisoned") = Some(result);
        self.notify.notify_waiters();
    }
}

/// Owns the leader's in-flight reservation until its result is published.
///
/// Dropping a caller future must also release its reservation: otherwise a
/// later caller becomes a follower of work that will never publish a result.
struct LeaderReservation {
    in_flight: Arc<Mutex<HashMap<SubagentTaskKey, Arc<InFlight>>>>,
    task_key: SubagentTaskKey,
    entry: Arc<InFlight>,
    finished: bool,
}

impl LeaderReservation {
    fn finish(mut self, result: Result<SubagentRunResult, SubagentError>) {
        self.entry.complete(result);
        let mut in_flight = self.in_flight.lock().expect("in-flight map poisoned");
        if in_flight
            .get(&self.task_key)
            .is_some_and(|current| Arc::ptr_eq(current, &self.entry))
        {
            in_flight.remove(&self.task_key);
        }
        self.finished = true;
    }
}

impl Drop for LeaderReservation {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let in_flight = self.in_flight.clone();
        let task_key = self.task_key.clone();
        let entry = self.entry.clone();
        // Publish a typed result before removal so existing followers cannot
        // wait indefinitely, even if the future is dropped outside a runtime.
        entry.complete(Err(SubagentError::Cancelled));
        let mut in_flight = in_flight.lock().expect("in-flight map poisoned");
        if in_flight
            .get(&task_key)
            .is_some_and(|current| Arc::ptr_eq(current, &entry))
        {
            in_flight.remove(&task_key);
        }
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
            terminal_outcomes: AsyncMutex::new(HashMap::new()),
            in_flight: Arc::new(Mutex::new(HashMap::new())),
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
    ) -> Result<SubagentRunResult, SubagentError> {
        let task_key = request.task_key().clone();
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }
        if let Some(outcome) = self.persistence.load_terminal(&task_key).await? {
            self.cache_terminal(task_key, &outcome).await;
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }

        let (entry, is_leader) = {
            let mut in_flight = self.in_flight.lock().expect("in-flight map poisoned");
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
            return entry
                .wait(&cancellation, &task_key.task_id)
                .await
                .map(|result| {
                    SubagentRunResult::new(
                        result.outcome,
                        match result.disposition {
                            SubagentPersistenceDisposition::PauseCommitted
                            | SubagentPersistenceDisposition::PauseReplaced
                            | SubagentPersistenceDisposition::PauseExisting => {
                                SubagentPersistenceDisposition::PauseExisting
                            }
                            SubagentPersistenceDisposition::TerminalInserted
                            | SubagentPersistenceDisposition::TerminalExisting => {
                                SubagentPersistenceDisposition::TerminalExisting
                            }
                            SubagentPersistenceDisposition::ObserverCancelled => {
                                SubagentPersistenceDisposition::ObserverCancelled
                            }
                        },
                    )
                });
        }
        let reservation = LeaderReservation {
            in_flight: self.in_flight.clone(),
            task_key: task_key.clone(),
            entry: entry.clone(),
            finished: false,
        };
        // A preceding caller may have committed a terminal result between the
        // initial cache check and this reservation. Do not reopen that task
        // after its in-flight entry has been removed.
        if let Some(outcome) = self.terminal_outcomes.lock().await.get(&task_key).cloned() {
            reservation.finish(Ok(SubagentRunResult::new(
                outcome.clone(),
                SubagentPersistenceDisposition::TerminalExisting,
            )));
            return Ok(SubagentRunResult::new(
                outcome,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }

        let result = self
            .run_reserved(request, task_key.clone(), cancellation)
            .await;
        reservation.finish(result.clone());
        result
    }

    async fn run_reserved(
        &self,
        mut request: SubagentRequest<C, H>,
        task_key: SubagentTaskKey,
        cancellation: CancellationToken,
    ) -> Result<SubagentRunResult, SubagentError> {
        let task_id = request.task_id().to_owned();

        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(task_key, SubagentOutcome::cancelled(task_id), None)
                .await;
        }

        if request.resume().is_none() {
            request.set_resume(self.persistence.load(&task_key).await?);
        }
        // Keep the exact pause seen at the durable read boundary. The
        // persistence seam uses it as a scoped compare-and-swap expectation
        // if execution pauses again, so a resumed pause can advance while a
        // duplicate continuation returns the newer durable winner.
        let expected_pause = request.resume().cloned();
        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(
                    task_key,
                    SubagentOutcome::cancelled(task_id),
                    expected_pause,
                )
                .await;
        }

        let mut prepared = self.planner.prepare(request).await?;
        if prepared.task_id != task_id {
            return Err(SubagentError::TaskIdMismatch {
                expected: task_id,
                actual: prepared.task_id,
            });
        }
        if cancellation.is_cancelled() {
            return self
                .persist_cancelled(
                    task_key,
                    SubagentOutcome::cancelled(prepared.task_id),
                    expected_pause,
                )
                .await;
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
        self.persist(task_key, outcome, expected_pause, &cancellation)
            .await
    }

    async fn persist_cancelled(
        &self,
        task_key: SubagentTaskKey,
        mut outcome: SubagentOutcome,
        expected_pause: Option<super::SubagentResume>,
    ) -> Result<SubagentRunResult, SubagentError> {
        outcome.status = SubagentStatus::Cancelled;
        // Once cancellation has won, this is the one terminal action. Do not
        // race it with the already-latched token or a caller could observe an
        // indeterminate terminal write.
        match self
            .persistence
            .record_terminal(&task_key, &outcome, expected_pause.as_ref())
            .await?
        {
            SubagentTerminalPersistenceDisposition::Inserted => {
                self.cache_terminal(task_key, &outcome).await;
                Ok(SubagentRunResult::new(
                    outcome,
                    SubagentPersistenceDisposition::TerminalInserted,
                ))
            }
            SubagentTerminalPersistenceDisposition::Existing => {
                self.load_terminal_winner(task_key).await
            }
            SubagentTerminalPersistenceDisposition::PauseExisting => {
                self.load_pause_winner(task_key).await
            }
        }
    }

    async fn persist(
        &self,
        task_key: SubagentTaskKey,
        outcome: SubagentOutcome,
        expected_pause: Option<super::SubagentResume>,
        cancellation: &CancellationToken,
    ) -> Result<SubagentRunResult, SubagentError> {
        if matches!(&outcome.status, SubagentStatus::Cancelled) {
            return self
                .persist_cancelled(task_key, outcome, expected_pause)
                .await;
        }
        let disposition = match &outcome.status {
            SubagentStatus::AwaitingInput(_) => {
                match self
                    .commit_or_cancel(
                        self.persistence.save_pause(PersistedSubagentPause {
                            key: task_key.clone(),
                            outcome: outcome.clone(),
                            replaces: expected_pause.clone(),
                        }),
                        cancellation,
                    )
                    .await?
                {
                    Some(SubagentPausePersistenceDisposition::Inserted) => {
                        SubagentPersistenceDisposition::PauseCommitted
                    }
                    Some(SubagentPausePersistenceDisposition::Replaced) => {
                        SubagentPersistenceDisposition::PauseReplaced
                    }
                    Some(SubagentPausePersistenceDisposition::Existing) => {
                        return self.load_pause_winner(task_key).await;
                    }
                    Some(SubagentPausePersistenceDisposition::TerminalExisting) => {
                        return self.load_terminal_winner(task_key).await;
                    }
                    None => {
                        return self
                            .persist_cancelled(task_key, outcome, expected_pause)
                            .await;
                    }
                }
            }
            SubagentStatus::Completed | SubagentStatus::Incomplete(_) => {
                match self
                    .commit_or_cancel(
                        async {
                            self.persistence
                                .record_terminal(&task_key, &outcome, expected_pause.as_ref())
                                .await
                        },
                        cancellation,
                    )
                    .await?
                {
                    Some(SubagentTerminalPersistenceDisposition::Inserted) => {
                        SubagentPersistenceDisposition::TerminalInserted
                    }
                    Some(SubagentTerminalPersistenceDisposition::Existing) => {
                        SubagentPersistenceDisposition::TerminalExisting
                    }
                    Some(SubagentTerminalPersistenceDisposition::PauseExisting) => {
                        return self.load_pause_winner(task_key).await;
                    }
                    None => {
                        return self
                            .persist_cancelled(task_key, outcome, expected_pause)
                            .await;
                    }
                }
            }
            SubagentStatus::Cancelled => unreachable!("handled before persistence race"),
        };
        if matches!(
            &outcome.status,
            SubagentStatus::Completed | SubagentStatus::Incomplete(_)
        ) {
            let terminal = if matches!(
                disposition,
                SubagentPersistenceDisposition::TerminalExisting
            ) {
                self.persistence
                    .load_terminal(&task_key)
                    .await?
                    .ok_or_else(|| {
                        SubagentError::Persistence(
                            "terminal insert lost without durable outcome".into(),
                        )
                    })?
            } else {
                outcome.clone()
            };
            self.cache_terminal(task_key, &terminal).await;
            return Ok(SubagentRunResult::new(terminal, disposition));
        }
        Ok(SubagentRunResult::new(outcome, disposition))
    }

    async fn load_pause_winner(
        &self,
        task_key: SubagentTaskKey,
    ) -> Result<SubagentRunResult, SubagentError> {
        // A terminal can win after the pause CAS reports contention. It is
        // authoritative over every prior pause and must never be reopened.
        if let Some(terminal) = self.persistence.load_terminal(&task_key).await? {
            self.cache_terminal(task_key, &terminal).await;
            return Ok(SubagentRunResult::new(
                terminal,
                SubagentPersistenceDisposition::TerminalExisting,
            ));
        }
        let Some(paused) = self.persistence.load_pause(&task_key).await? else {
            // A different driver can consume the pause into a terminal between
            // the first terminal read and this pause read. Recheck the
            // authoritative terminal before classifying that interleaving as a
            // persistence failure.
            if let Some(terminal) = self.persistence.load_terminal(&task_key).await? {
                self.cache_terminal(task_key, &terminal).await;
                return Ok(SubagentRunResult::new(
                    terminal,
                    SubagentPersistenceDisposition::TerminalExisting,
                ));
            }
            return Err(SubagentError::Persistence(
                "pause compare-and-swap lost without a durable pause or terminal outcome".into(),
            ));
        };
        if !matches!(paused.status, SubagentStatus::AwaitingInput(_)) {
            return Err(SubagentError::Persistence(
                "durable pause record did not contain an awaiting-input outcome".into(),
            ));
        }
        Ok(SubagentRunResult::new(
            paused,
            SubagentPersistenceDisposition::PauseExisting,
        ))
    }

    async fn load_terminal_winner(
        &self,
        task_key: SubagentTaskKey,
    ) -> Result<SubagentRunResult, SubagentError> {
        let terminal = self
            .persistence
            .load_terminal(&task_key)
            .await?
            .ok_or_else(|| {
                SubagentError::Persistence(
                    "terminal compare-and-swap lost without a durable terminal outcome".into(),
                )
            })?;
        self.cache_terminal(task_key, &terminal).await;
        Ok(SubagentRunResult::new(
            terminal,
            SubagentPersistenceDisposition::TerminalExisting,
        ))
    }

    async fn commit_or_cancel<F, T>(
        &self,
        operation: F,
        cancellation: &CancellationToken,
    ) -> Result<Option<T>, SubagentError>
    where
        F: Future<Output = Result<T, SubagentError>>,
    {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Ok(None),
            result = operation => {
                Ok(Some(result?))
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
