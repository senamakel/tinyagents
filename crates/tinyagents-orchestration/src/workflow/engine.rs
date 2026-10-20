//! Workflow execution engine: orchestrates phase scheduling, task spawning,
//! and result collection.
//!
//! [`WorkflowEngine`] drives the scheduler, manages child task creation and
//! claim-based assignment, handles phase state persistence and transitions,
//! and enforces bounded concurrency. Legacy hosts supply [`WorkflowExecutor`];
//! authoritative hosts select [`AsyncWorkflowExecutor`] and await the explicit
//! async persistence entrypoints.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tinyagents_graph::GraphEventSink;
use tinyagents_graph::parallel::{FailurePolicy, ParallelOptions, map_reduce};
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{
    RunLedgerStore, WorkflowLeaseClaim, WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert,
};
use tinyagents_storage_kit::Blocking;

use super::child_step::run_child_step_authoritative;
use super::persistence::{
    AsyncWorkflowChildRegistration, AsyncWorkflowExecutor, AsyncWorkflowStore,
    WorkflowPersistenceError, WorkflowPersistenceResult,
};
use super::state::{
    PhaseStatus, all_phases_completed, init_phase_states, next_runnable_phase, phase_prompt,
    reset_running_phases, set_phase_reason, set_phase_status, synthesize_summary, upstream_outputs,
};
use super::{WorkflowDefinition, WorkflowPhase};
use crate::subagent::AgentStepConfig;

/// Error returned by a host child executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrchestrationError(pub String);

impl fmt::Display for OrchestrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for OrchestrationError {}

impl From<anyhow::Error> for OrchestrationError {
    fn from(error: anyhow::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<tinyagents_harness::TinyAgentsError> for OrchestrationError {
    fn from(error: tinyagents_harness::TinyAgentsError) -> Self {
        Self(error.to_string())
    }
}

/// Durable workflow rows supplied by the session owner.
pub trait WorkflowStore: Send + Sync {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError>;
    fn upsert(&self, row: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError>;
    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError>;
    fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError>;
    fn renew(&self, id: &str, owner: &str, lease_for: Duration)
    -> Result<bool, OrchestrationError>;
}

/// Synchronous [`WorkflowStore`] over a bound run-ledger handle.
///
/// Every call reaches the ledger's async operations through a [`Blocking`]
/// bridge thread, so it works from inside or outside a tokio runtime. The
/// handle is whatever backend the host opened (SQLite, MongoDB, memory); this
/// adapter opens no file and has no fallback. Prefer
/// [`LedgerWorkflowStore`](super::LedgerWorkflowStore) with
/// `WorkflowEngine::new_async`, which keeps typed errors and needs no bridge.
#[derive(Clone)]
pub struct SessionWorkflowStore {
    ledger: Arc<dyn RunLedgerStore>,
    bridge: Blocking,
}

impl std::fmt::Debug for SessionWorkflowStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionWorkflowStore")
            .finish_non_exhaustive()
    }
}

impl SessionWorkflowStore {
    /// Adapt `ledger`, starting a bridge thread for the synchronous calls.
    ///
    /// # Errors
    ///
    /// The bridge thread cannot start.
    pub fn new(ledger: Arc<dyn RunLedgerStore>) -> Result<Self, OrchestrationError> {
        Ok(Self::with_bridge(
            ledger,
            Blocking::new().map_err(|error| OrchestrationError(error.to_string()))?,
        ))
    }

    /// Adapt `ledger`, sharing an existing bridge.
    pub fn with_bridge(ledger: Arc<dyn RunLedgerStore>, bridge: Blocking) -> Self {
        Self { ledger, bridge }
    }

    fn run<T: Send + 'static>(
        &self,
        work: impl std::future::Future<Output = Result<T, tinyagents_session::run_ledger::LedgerError>>
        + Send
        + 'static,
    ) -> Result<T, OrchestrationError> {
        self.bridge
            .run(work)
            .map_err(|error| OrchestrationError(error.to_string()))?
            .map_err(|error| OrchestrationError(error.to_string()))
    }
}

impl WorkflowStore for SessionWorkflowStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        let (ledger, id) = (Arc::clone(&self.ledger), id.to_owned());
        self.run(async move { ledger.get_workflow_run(&id).await })
    }

    fn upsert(&self, row: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        let ledger = Arc::clone(&self.ledger);
        self.run(async move { ledger.upsert_workflow_run(row).await })
    }

    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        let lease = chrono::Duration::from_std(lease_for)
            .map_err(|error| OrchestrationError(error.to_string()))?;
        let (ledger, id, owner) = (Arc::clone(&self.ledger), id.to_owned(), owner.to_owned());
        self.run(async move { ledger.try_claim_workflow_run(&id, &owner, lease).await })
    }

    fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        let lease = chrono::Duration::from_std(lease_for)
            .map_err(|error| OrchestrationError(error.to_string()))?;
        let (ledger, owner) = (Arc::clone(&self.ledger), owner.to_owned());
        self.run(async move {
            ledger
                .compare_and_swap_workflow_run(row, expected_revision, &owner, lease)
                .await
        })
    }

    fn renew(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<bool, OrchestrationError> {
        let lease = chrono::Duration::from_std(lease_for)
            .map_err(|error| OrchestrationError(error.to_string()))?;
        let (ledger, id, owner) = (Arc::clone(&self.ledger), id.to_owned(), owner.to_owned());
        self.run(async move { ledger.renew_workflow_run_lease(&id, &owner, lease).await })
    }
}

/// One host-authorized child invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowChildRequest {
    pub run_id: String,
    pub phase: String,
    pub agent_id: String,
    pub index_in_phase: usize,
    pub prompt: String,
}

/// One child's terminal result. `output` is retained verbatim in phase state.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowChildResult {
    pub child_id: String,
    pub output: Value,
}

/// Called by a host immediately after it has created a real child.  Registering
/// before waiting makes the child visible to a concurrent cancellation request
/// even when the worker is still in flight.
pub trait WorkflowChildRegistration: Send + Sync {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError>;
}

/// Host-owned execution and cancellation mechanism.
#[async_trait]
pub trait WorkflowExecutor: Send + Sync {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
        registration: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError>;

    async fn cancel_children(&self, child_ids: &[String]);
}

/// Generic durable workflow engine. It never creates tasks: hosts decide where
/// work runs and which authorization context is in force via [`WorkflowExecutor`].
pub struct WorkflowEngine<S, E> {
    store: Arc<S>,
    executor: Arc<E>,
    async_store: Option<Arc<dyn AsyncWorkflowStore>>,
    async_executor: Option<Arc<dyn AsyncWorkflowExecutor>>,
    event_sink: Option<Arc<dyn GraphEventSink>>,
    event_seq: AtomicU64,
    lease_for: Duration,
    step_config: AgentStepConfig,
}

const WORKFLOW_LEASE: Duration = Duration::from_secs(10 * 60);

struct PersistRequest {
    phase_states: Value,
    child_run_ids: Vec<String>,
    status: WorkflowRunStatus,
    summary: Option<String>,
    terminal: bool,
}

struct PhaseRegistration<S: WorkflowStore> {
    store: Arc<S>,
    owner: String,
    run: parking_lot::Mutex<WorkflowRun>,
    phase_states: Value,
    lease_for: Duration,
    async_store: Option<Arc<dyn AsyncWorkflowStore>>,
    operation: tokio::sync::Mutex<()>,
    failure: parking_lot::Mutex<Option<Arc<WorkflowPersistenceError>>>,
    failure_notify: tokio::sync::Notify,
    attempted: parking_lot::Mutex<Vec<String>>,
}

impl<S: WorkflowStore> PhaseRegistration<S> {
    fn current(&self) -> WorkflowRun {
        self.run.lock().clone()
    }
}

impl<S: WorkflowStore + 'static> WorkflowChildRegistration for PhaseRegistration<S> {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError> {
        if self.async_store.is_some() {
            return Err(OrchestrationError(
                "authoritative registration requires register_async".into(),
            ));
        }
        let mut run = self.run.lock();
        if run.child_run_ids.iter().any(|known| known == &child_id) {
            return Ok(());
        }
        let mut children = run.child_run_ids.clone();
        children.push(child_id);
        let Some(updated) = self.store.compare_and_swap(
            WorkflowRunUpsert {
                id: run.id.clone(),
                definition_id: run.definition_id.clone(),
                parent_thread_id: run.parent_thread_id.clone(),
                input: run.input.clone(),
                phase_states: self.phase_states.clone(),
                child_run_ids: children,
                status: WorkflowRunStatus::Running,
                summary: None,
                started_at: Some(run.started_at),
                completed_at: None,
            },
            run.revision,
            &self.owner,
            self.lease_for,
        )?
        else {
            return Err(OrchestrationError(
                "workflow lease lost while registering child".into(),
            ));
        };
        *run = updated;
        Ok(())
    }
}

#[async_trait]
impl<S: WorkflowStore + 'static> AsyncWorkflowChildRegistration for PhaseRegistration<S> {
    fn register(&self, child_id: String) -> WorkflowPersistenceResult<()> {
        if self.async_store.is_none() {
            return Ok(WorkflowChildRegistration::register(self, child_id)?);
        }
        self.attempted.lock().push(child_id);
        let error = Arc::new(WorkflowPersistenceError::AsyncRegistrationRequired);
        *self.failure.lock() = Some(error.clone());
        self.failure_notify.notify_one();
        Err(WorkflowPersistenceError::Registration(error))
    }

    async fn register_async(&self, child_id: String) -> WorkflowPersistenceResult<()> {
        let Some(store) = &self.async_store else {
            return Ok(WorkflowChildRegistration::register(self, child_id)?);
        };
        let _operation = self.operation.lock().await;
        let run = self.current();
        if run.child_run_ids.contains(&child_id) {
            return Ok(());
        }
        self.attempted.lock().push(child_id.clone());
        let mut children = run.child_run_ids.clone();
        children.push(child_id);
        let result = store
            .compare_and_swap(
                WorkflowRunUpsert {
                    id: run.id.clone(),
                    definition_id: run.definition_id.clone(),
                    parent_thread_id: run.parent_thread_id.clone(),
                    input: run.input.clone(),
                    phase_states: self.phase_states.clone(),
                    child_run_ids: children,
                    status: WorkflowRunStatus::Running,
                    summary: None,
                    started_at: Some(run.started_at),
                    completed_at: None,
                },
                run.revision,
                &self.owner,
                self.lease_for,
            )
            .await;
        match result {
            Ok(Some(updated)) => {
                *self.run.lock() = updated;
                Ok(())
            }
            other => {
                let error = Arc::new(match other {
                    Err(error) => error,
                    Ok(None) => WorkflowPersistenceError::LeaseLost,
                    Ok(Some(_)) => unreachable!(),
                });
                *self.failure.lock() = Some(error.clone());
                self.failure_notify.notify_one();
                Err(WorkflowPersistenceError::Registration(error))
            }
        }
    }
}

impl<S, E> WorkflowEngine<S, E>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    pub fn new(store: Arc<S>, executor: Arc<E>) -> Self {
        Self {
            store,
            executor,
            async_store: None,
            async_executor: None,
            event_sink: None,
            event_seq: AtomicU64::new(0),
            lease_for: WORKFLOW_LEASE,
            step_config: AgentStepConfig::default(),
        }
    }

    /// Attach an optional host sink for ordinary graph lifecycle tracing.
    pub fn with_event_sink(mut self, sink: Arc<dyn GraphEventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Apply driver policy to every agent child: spawn admission (scoped per
    /// workflow run, targeted at the agent id), timeout and result policy.
    /// Retry, token budget and role have no effect here: the executor reports
    /// no transient failures, usage or tools.
    ///
    /// Without this call the config is inert and children run exactly as
    /// before. A child refused by the spawn policy or timed out fails its
    /// phase with that reason.
    pub fn with_step_config(mut self, config: AgentStepConfig) -> Self {
        self.step_config = config;
        self
    }

    /// Override the driver lease for deterministic tests or hosts with a
    /// deliberately shorter failure-detection window.
    pub fn with_lease_duration(mut self, lease_for: Duration) -> Self {
        self.lease_for = lease_for.max(Duration::from_millis(3));
        self
    }

    /// Initialise a durable run before the host schedules [`Self::drive`].
    pub fn initialise(
        &self,
        id: String,
        definition: &WorkflowDefinition,
        input: Value,
        parent_thread_id: Option<String>,
    ) -> Result<WorkflowRun, OrchestrationError> {
        self.store.upsert(WorkflowRunUpsert {
            id,
            definition_id: definition.id.clone(),
            parent_thread_id,
            input,
            phase_states: init_phase_states(definition),
            child_run_ids: Vec::new(),
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        })
    }

    /// Initialise using the authoritative asynchronous store when selected.
    pub async fn initialise_async(
        &self,
        id: String,
        definition: &WorkflowDefinition,
        input: Value,
        parent_thread_id: Option<String>,
    ) -> WorkflowPersistenceResult<WorkflowRun> {
        let row = WorkflowRunUpsert {
            id,
            definition_id: definition.id.clone(),
            parent_thread_id,
            input,
            phase_states: init_phase_states(definition),
            child_run_ids: Vec::new(),
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        };
        match &self.async_store {
            Some(store) => store.upsert(row).await,
            None => Ok(self.store.upsert(row)?),
        }
    }

    /// Drive using the typed authoritative contract.
    pub async fn drive_async(
        &self,
        run_id: &str,
        definition: &WorkflowDefinition,
        cancel: CancellationToken,
    ) -> WorkflowPersistenceResult<()> {
        self.drive_inner(run_id, definition, cancel).await
    }

    /// Drive a run to a terminal state. Completed phases are never executed
    /// again, so a host may safely call this after process restart or resume.
    pub async fn drive(
        &self,
        run_id: &str,
        definition: &WorkflowDefinition,
        cancel: CancellationToken,
    ) -> Result<(), OrchestrationError> {
        if self.async_store.is_some() {
            return Err(OrchestrationError(
                "authoritative workflow requires drive_async".into(),
            ));
        }
        self.drive_inner(run_id, definition, cancel)
            .await
            .map_err(WorkflowPersistenceError::into_legacy)
    }

    async fn load(&self, id: &str) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        match &self.async_store {
            Some(store) => store.load(id).await,
            None => Ok(self.store.load(id)?),
        }
    }
    async fn claim(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<WorkflowLeaseClaim> {
        match &self.async_store {
            Some(store) => store.claim(id, owner, lease).await,
            None => Ok(self.store.claim(id, owner, lease)?),
        }
    }
    async fn renew(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<bool> {
        match &self.async_store {
            Some(store) => store.renew(id, owner, lease).await,
            None => Ok(self.store.renew(id, owner, lease)?),
        }
    }
    async fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        revision: u64,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        match &self.async_store {
            Some(store) => store.compare_and_swap(row, revision, owner, lease).await,
            None => Ok(self.store.compare_and_swap(row, revision, owner, lease)?),
        }
    }
    async fn cancel_children(&self, ids: &[String]) {
        match &self.async_executor {
            Some(executor) => {
                let mut seen = std::collections::HashSet::new();
                let unique = ids
                    .iter()
                    .filter(|id| seen.insert((*id).clone()))
                    .cloned()
                    .collect::<Vec<_>>();
                executor.cancel_children(&unique).await;
            }
            None => self.executor.cancel_children(ids).await,
        }
    }

    async fn drive_inner(
        &self,
        run_id: &str,
        definition: &WorkflowDefinition,
        cancel: CancellationToken,
    ) -> WorkflowPersistenceResult<()> {
        // A driver lease is acquired before looking for runnable work.  This
        // is deliberately separate from the in-process cancellation token:
        // resume can race in another process, and only the durable lease
        // prevents both drivers from spawning the same phase.
        let owner = uuid::Uuid::new_v4().to_string();
        let mut run = match self.claim(run_id, &owner, self.lease_for).await? {
            WorkflowLeaseClaim::Acquired(run) => run,
            WorkflowLeaseClaim::Busy(_) => return Ok(()),
            WorkflowLeaseClaim::Missing => {
                return Err(OrchestrationError(format!(
                    "workflow run {run_id} vanished before start"
                ))
                .into());
            }
        };
        // A crashed owner can leave the durable phase marked `running`. That
        // marker is intentionally not runnable, so reclaiming the lease must
        // turn it back into retryable work before scheduling. The claim above
        // fences every prior owner; this CAS is the new owner's durable
        // recovery transition rather than a read/modify/write race.
        if run.phase_states.as_object().is_some_and(|phases| {
            phases
                .values()
                .any(|phase| phase.get("status").and_then(Value::as_str) == Some("running"))
        }) {
            // The previous owner may have crashed after registering remote
            // children. Fence side effects before making its phase runnable.
            self.cancel_children(&run.child_run_ids).await;
            let mut phase_states = run.phase_states.clone();
            reset_running_phases(
                &mut phase_states,
                "workflow owner expired; phase will retry after lease takeover",
            );
            run = self
                .persist(
                    &run,
                    PersistRequest {
                        phase_states,
                        child_run_ids: Vec::new(),
                        status: WorkflowRunStatus::Running,
                        summary: None,
                        terminal: false,
                    },
                    &owner,
                )
                .await?;
        }
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunStarted {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
            },
        );
        // Registered children from an interrupted attempt are historical, not
        // part of the retry's spawn budget.
        let mut total_spawned = 0;

        loop {
            if cancel.is_cancelled() {
                self.cancel_children(&run.child_run_ids).await;
                let mut phase_states = run.phase_states.clone();
                reset_running_phases(
                    &mut phase_states,
                    "workflow interrupted; phase will retry on resume",
                );
                if let Err(error) = self
                    .persist(
                        &run,
                        PersistRequest {
                            phase_states,
                            child_run_ids: run.child_run_ids.clone(),
                            status: WorkflowRunStatus::Interrupted,
                            summary: None,
                            terminal: false,
                        },
                        &owner,
                    )
                    .await
                {
                    if self.async_store.is_some() {
                        return Err(error);
                    }
                    if self.owner_lost(run_id, &owner).await.unwrap_or(false)
                        || self
                            .emit_recorded_terminal(run_id, total_spawned as usize)
                            .await
                            .unwrap_or(false)
                    {
                        return Ok(());
                    }
                    self.finish_failed(run_id, error.to_string());
                    return Err(error);
                }
                self.finish_cancelled(run_id);
                return Ok(());
            }
            let Some(phase) = next_runnable_phase(definition, &run.phase_states).cloned() else {
                if all_phases_completed(definition, &run.phase_states) {
                    if let Err(error) = self
                        .persist(
                            &run,
                            PersistRequest {
                                phase_states: run.phase_states.clone(),
                                child_run_ids: run.child_run_ids.clone(),
                                status: WorkflowRunStatus::Completed,
                                summary: synthesize_summary(definition, &run.phase_states),
                                terminal: true,
                            },
                            &owner,
                        )
                        .await
                    {
                        if self.async_store.is_some() {
                            return Err(error);
                        }
                        if self.owner_lost(run_id, &owner).await.unwrap_or(false) {
                            return Ok(());
                        }
                        self.finish_failed(run_id, error.to_string());
                        return Err(error);
                    }
                    self.finish_completed(run_id, total_spawned as usize);
                } else {
                    let reason = "no runnable phase (dependency deadlock)".to_owned();
                    if let Err(error) = self
                        .persist(
                            &run,
                            PersistRequest {
                                phase_states: run.phase_states.clone(),
                                child_run_ids: run.child_run_ids.clone(),
                                status: WorkflowRunStatus::Failed,
                                summary: Some(reason.clone()),
                                terminal: true,
                            },
                            &owner,
                        )
                        .await
                    {
                        if self.async_store.is_some() {
                            return Err(error);
                        }
                        if self.owner_lost(run_id, &owner).await.unwrap_or(false) {
                            return Ok(());
                        }
                        self.finish_failed(run_id, error.to_string());
                        return Err(error);
                    }
                    self.finish_failed(run_id, reason);
                }
                return Ok(());
            };
            self.emit(
                run_id,
                tinyagents_graph::GraphEvent::NodeStarted {
                    node: tinyagents_harness::ids::NodeId::new("run_phase"),
                    step: total_spawned as usize + 1,
                },
            );
            let phase_result = self
                .run_phase(
                    &run,
                    definition,
                    &phase,
                    total_spawned,
                    cancel.clone(),
                    &owner,
                )
                .await;
            let (updated, spawned) = match phase_result {
                Ok(result) => result,
                Err(error) => {
                    // A host stop/resume fences this owner with a revision CAS.
                    // Do not turn that intentional hand-off into a stale
                    // failure event or overwrite the newer durable state.
                    if self.async_store.is_some() {
                        return Err(error);
                    }
                    if self.owner_lost(run_id, &owner).await.unwrap_or(false)
                        || self
                            .emit_recorded_terminal(run_id, total_spawned as usize)
                            .await
                            .unwrap_or(false)
                    {
                        return Ok(());
                    }
                    self.finish_failed(run_id, error.to_string());
                    return Err(error);
                }
            };
            run = updated;
            self.emit(
                run_id,
                tinyagents_graph::GraphEvent::NodeCompleted {
                    node: tinyagents_harness::ids::NodeId::new("run_phase"),
                    step: total_spawned as usize + 1,
                },
            );
            total_spawned += spawned;
            if run.status != WorkflowRunStatus::Running {
                match run.status {
                    WorkflowRunStatus::Completed => {
                        self.finish_completed(run_id, total_spawned as usize)
                    }
                    WorkflowRunStatus::Interrupted | WorkflowRunStatus::Cancelled => {
                        self.finish_cancelled(run_id)
                    }
                    WorkflowRunStatus::Failed => self.finish_failed(
                        run_id,
                        run.summary
                            .clone()
                            .unwrap_or_else(|| "workflow phase failed".to_owned()),
                    ),
                    WorkflowRunStatus::Pending | WorkflowRunStatus::Running => {}
                }
                return Ok(());
            }
        }
    }

    async fn run_phase(
        &self,
        run: &WorkflowRun,
        definition: &WorkflowDefinition,
        phase: &WorkflowPhase,
        total_spawned: u32,
        cancel: CancellationToken,
        owner: &str,
    ) -> WorkflowPersistenceResult<(WorkflowRun, u32)> {
        let mut phase_states = run.phase_states.clone();
        let mut child_ids = run.child_run_ids.clone();
        set_phase_status(&mut phase_states, &phase.name, PhaseStatus::Running, None);
        let running = self
            .persist(
                run,
                PersistRequest {
                    phase_states: phase_states.clone(),
                    child_run_ids: child_ids.clone(),
                    status: WorkflowRunStatus::Running,
                    summary: None,
                    terminal: false,
                },
                owner,
            )
            .await?;

        let budget = definition.max_children.saturating_sub(total_spawned) as usize;
        if budget == 0 {
            return self
                .fail_phase(
                    &running,
                    &mut phase_states,
                    child_ids,
                    phase,
                    format!(
                        "max_children cap ({}) reached before phase '{}' completed",
                        definition.max_children, phase.name
                    ),
                    owner,
                )
                .await;
        }
        let capacity = phase.agent_ids.len().min(budget);
        let capped = capacity != phase.agent_ids.len();
        let upstream = upstream_outputs(phase, &phase_states);
        let requests = phase.agent_ids[..capacity]
            .iter()
            .enumerate()
            .map(|(index_in_phase, agent_id)| WorkflowChildRequest {
                run_id: run.id.clone(),
                phase: phase.name.clone(),
                agent_id: agent_id.clone(),
                index_in_phase,
                prompt: phase_prompt(&run.input, phase, index_in_phase, &upstream),
            })
            .collect::<Vec<_>>();
        let registration = Arc::new(PhaseRegistration {
            store: self.store.clone(),
            owner: owner.to_owned(),
            run: parking_lot::Mutex::new(running.clone()),
            phase_states: phase_states.clone(),
            lease_for: self.lease_for,
            async_store: self.async_store.clone(),
            operation: tokio::sync::Mutex::new(()),
            failure: parking_lot::Mutex::new(None),
            failure_notify: tokio::sync::Notify::new(),
            attempted: parking_lot::Mutex::new(Vec::new()),
        });
        let executor = self.executor.clone();
        let async_executor = self.async_executor.clone();
        // Policy/parallel helpers carry textual ordinary child reasons.
        // Retain authoritative sources and wake the phase even while an
        // unrelated worker never finishes.
        let typed_failure = Arc::new(parking_lot::Mutex::new(None));
        let worker_failure = typed_failure.clone();
        let typed_notify = Arc::new(tokio::sync::Notify::new());
        let worker_notify = typed_notify.clone();
        let step_config = self.step_config.clone();
        let worker_cancel = cancel.clone();
        let worker_registration = registration.clone();
        let outcomes = map_reduce(
            requests,
            ParallelOptions::default()
                .with_max_concurrency(definition.default_concurrency as usize)
                .with_failure_policy(FailurePolicy::CollectAll)
                .with_cancellation(cancel.clone()),
            move |_index, request| {
                let executor = executor.clone();
                let async_executor = async_executor.clone();
                let typed_failure = worker_failure.clone();
                let typed_notify = worker_notify.clone();
                let cancel = worker_cancel.clone();
                let registration = worker_registration.clone();
                let step_config = step_config.clone();
                async move {
                    let async_registration = async_executor
                        .as_ref()
                        .map(|_| registration.clone() as Arc<dyn AsyncWorkflowChildRegistration>);
                    run_child_step_authoritative(
                        &step_config,
                        executor,
                        async_executor,
                        request,
                        cancel,
                        registration,
                        async_registration,
                    )
                    .await
                    .map_err(|error| {
                        let message = error.to_string();
                        if !matches!(error, WorkflowPersistenceError::Legacy(_)) {
                            *typed_failure.lock() = Some(error);
                            typed_notify.notify_one();
                        }
                        tinyagents_harness::TinyAgentsError::Graph(message)
                    })
                }
            },
        );
        tokio::pin!(outcomes);
        let renew_every = self.lease_for.div_f32(3.0).max(Duration::from_millis(1));
        let mut heartbeat = tokio::time::interval(renew_every);
        // Ignore interval's eager first tick: `claim` has just installed this
        // lease, so renewals begin only while a child can be in flight.
        heartbeat.tick().await;
        let outcomes = loop {
            tokio::select! {
                outcomes = &mut outcomes => break outcomes,
                _ = registration.failure_notify.notified() => {
                    let error = registration.failure.lock().take().expect("registration notified failure");
                    cancel.cancel();
                    let mut children = registration.current().child_run_ids;
                    children.extend(registration.attempted.lock().iter().cloned());
                    self.cancel_children(&children).await;
                    return Err(WorkflowPersistenceError::Registration(error));
                }
                _ = typed_notify.notified() => {
                    let error = typed_failure.lock().take().expect("child notified failure");
                    cancel.cancel();
                    let mut children = registration.current().child_run_ids;
                    children.extend(registration.attempted.lock().iter().cloned());
                    self.cancel_children(&children).await;
                    return Err(error);
                }

                _ = heartbeat.tick() => {
                    let renewed = self.renew(&run.id, owner, self.lease_for).await;
                    if !matches!(renewed, Ok(true)) {
                        cancel.cancel();
                        let mut children = registration.current().child_run_ids;
                        children.extend(registration.attempted.lock().iter().cloned());
                        self.cancel_children(&children).await;
                        renewed?;
                        return Err(WorkflowPersistenceError::LeaseRenewalLost);
                    }
                }
            }
        };
        let registration_error = registration.failure.lock().take();
        let child_error = typed_failure.lock().take();
        if let Some(error) = registration_error
            .map(WorkflowPersistenceError::Registration)
            .or(child_error)
        {
            let mut children = registration.current().child_run_ids;
            children.extend(registration.attempted.lock().iter().cloned());
            self.cancel_children(&children).await;
            return Err(error);
        }
        let outcomes = match outcomes {
            Ok(outcomes) => outcomes,
            Err(tinyagents_harness::TinyAgentsError::Cancelled) => {
                let children = registration.current().child_run_ids;
                let mut to_cancel = children.clone();
                to_cancel.extend(registration.attempted.lock().iter().cloned());
                self.cancel_children(&to_cancel).await;
                reset_running_phases(
                    &mut phase_states,
                    "workflow interrupted; phase will retry on resume",
                );
                let updated = self
                    .persist(
                        &registration.current(),
                        PersistRequest {
                            phase_states,
                            child_run_ids: children,
                            status: WorkflowRunStatus::Interrupted,
                            summary: None,
                            terminal: false,
                        },
                        owner,
                    )
                    .await?;
                return Ok((updated, 0));
            }
            Err(error) => return Err(OrchestrationError(error.to_string()).into()),
        };
        child_ids = registration.current().child_run_ids;
        let mut outputs = Vec::new();
        let mut failure = None;
        let mut spawned = 0_u32;
        for outcome in outcomes.outcomes {
            match outcome.result {
                Ok((result, annotations)) => {
                    spawned += 1;
                    // The executor registered the real id before it could
                    // await completion. Keep older executors harmlessly
                    // compatible by accepting an already-present id only.
                    if !child_ids.iter().any(|id| id == &result.child_id) {
                        if self.async_store.is_some() {
                            self.cancel_children(std::slice::from_ref(&result.child_id))
                                .await;
                            return Err(WorkflowPersistenceError::UnregisteredChild(
                                result.child_id,
                            ));
                        }
                        child_ids.push(result.child_id.clone());
                    }
                    let mut entry = json!({
                        // Preserve the persisted/RPC v1 projection while the
                        // v2 metadata remains lossless for engine consumers.
                        "agentId": phase.agent_ids[outcome.index],
                        "output": render_compat_output(&result.output),
                        "metadata": { "version": 2, "rawOutput": result.output },
                    });
                    if let Some(annotations) = annotations {
                        entry["metadata"]["resultPolicy"] = annotations;
                    }
                    outputs.push(entry);
                }
                Err(error) if failure.is_none() => failure = Some(error),
                Err(_) => {}
            }
        }
        if cancel.is_cancelled() {
            let children = registration.current().child_run_ids;
            let mut to_cancel = children.clone();
            to_cancel.extend(registration.attempted.lock().iter().cloned());
            self.cancel_children(&to_cancel).await;
            reset_running_phases(
                &mut phase_states,
                "workflow interrupted; phase will retry on resume",
            );
            let updated = self
                .persist(
                    &registration.current(),
                    PersistRequest {
                        phase_states,
                        child_run_ids: children,
                        status: WorkflowRunStatus::Interrupted,
                        summary: None,
                        terminal: false,
                    },
                    owner,
                )
                .await?;
            return Ok((updated, 0));
        }
        if let Some(reason) = failure.or_else(|| {
            capped.then(|| {
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                )
            })
        }) {
            return self
                .fail_phase(
                    &registration.current(),
                    &mut phase_states,
                    child_ids,
                    phase,
                    reason,
                    owner,
                )
                .await;
        }
        set_phase_status(
            &mut phase_states,
            &phase.name,
            PhaseStatus::Completed,
            Some(Value::Array(outputs)),
        );
        let updated = self
            .persist(
                &registration.current(),
                PersistRequest {
                    phase_states,
                    child_run_ids: child_ids,
                    status: WorkflowRunStatus::Running,
                    summary: None,
                    terminal: false,
                },
                owner,
            )
            .await?;
        Ok((updated, spawned))
    }

    async fn fail_phase(
        &self,
        run: &WorkflowRun,
        phase_states: &mut Value,
        child_ids: Vec<String>,
        phase: &WorkflowPhase,
        reason: String,
        owner: &str,
    ) -> WorkflowPersistenceResult<(WorkflowRun, u32)> {
        set_phase_status(
            phase_states,
            &phase.name,
            PhaseStatus::Failed,
            Some(json!([])),
        );
        set_phase_reason(phase_states, &phase.name, &reason);
        let updated = self
            .persist(
                run,
                PersistRequest {
                    phase_states: phase_states.clone(),
                    child_run_ids: child_ids,
                    status: WorkflowRunStatus::Failed,
                    summary: Some(reason),
                    terminal: true,
                },
                owner,
            )
            .await?;
        Ok((updated, 0))
    }

    async fn persist(
        &self,
        run: &WorkflowRun,
        request: PersistRequest,
        owner: &str,
    ) -> WorkflowPersistenceResult<WorkflowRun> {
        self.compare_and_swap(
            WorkflowRunUpsert {
                id: run.id.clone(),
                definition_id: run.definition_id.clone(),
                parent_thread_id: run.parent_thread_id.clone(),
                input: run.input.clone(),
                phase_states: request.phase_states,
                child_run_ids: request.child_run_ids,
                status: request.status,
                summary: request.summary,
                started_at: Some(run.started_at),
                completed_at: request.terminal.then(Utc::now),
            },
            run.revision,
            owner,
            self.lease_for,
        )
        .await?
        .ok_or_else(|| WorkflowPersistenceError::LeaseLost)
    }

    fn emit(&self, run_id: &str, event: tinyagents_graph::GraphEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(tinyagents_graph::GraphEventEnvelope {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                task_id: None,
                ns: Vec::new(),
                seq: self.event_seq.fetch_add(1, Ordering::Relaxed),
                event,
            });
        }
    }

    fn finish_completed(&self, run_id: &str, steps: usize) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunCompleted {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                steps,
            },
        );
        self.flush_terminal_events();
    }

    fn finish_failed(&self, run_id: &str, error: String) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunFailed {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
                error,
            },
        );
        self.flush_terminal_events();
    }

    fn finish_cancelled(&self, run_id: &str) {
        self.emit(
            run_id,
            tinyagents_graph::GraphEvent::RunCancelled {
                run_id: tinyagents_harness::ids::RunId::new(run_id),
            },
        );
        self.flush_terminal_events();
    }

    fn flush_terminal_events(&self) {
        if let Some(sink) = &self.event_sink {
            sink.flush();
        }
    }

    /// A lifecycle hand-off or lease takeover has fenced this driver. It must
    /// not manufacture a terminal graph event for the replacement owner.
    async fn owner_lost(&self, run_id: &str, owner: &str) -> WorkflowPersistenceResult<bool> {
        Ok(self.load(run_id).await?.is_some_and(|current| {
            current.lease_owner.as_deref() != Some(owner)
                || current
                    .lease_expires_at
                    .is_none_or(|expires| expires <= Utc::now())
        }))
    }

    async fn emit_recorded_terminal(
        &self,
        run_id: &str,
        steps: usize,
    ) -> WorkflowPersistenceResult<bool> {
        let Some(current) = self.load(run_id).await? else {
            return Ok(false);
        };
        if !current.status.is_terminal() {
            return Ok(false);
        }
        match current.status {
            WorkflowRunStatus::Completed => self.finish_completed(run_id, steps),
            WorkflowRunStatus::Interrupted | WorkflowRunStatus::Cancelled => {
                self.finish_cancelled(run_id)
            }
            WorkflowRunStatus::Failed => self.finish_failed(
                run_id,
                current
                    .summary
                    .unwrap_or_else(|| "workflow phase failed".to_owned()),
            ),
            WorkflowRunStatus::Pending | WorkflowRunStatus::Running => return Ok(false),
        }
        Ok(true)
    }
}

pub(super) fn render_compat_output(output: &Value) -> String {
    match output {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(output)
            .unwrap_or_else(|_| "<unserializable workflow output>".to_owned()),
    }
}

/// Placeholder legacy types for an explicitly authoritative engine.
/// These callbacks always refuse; they never open a database.
pub struct AuthoritativeWorkflow;

impl WorkflowStore for AuthoritativeWorkflow {
    fn load(&self, _: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        Err(OrchestrationError("async store required".into()))
    }
    fn upsert(&self, _: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        Err(OrchestrationError("async store required".into()))
    }
    fn claim(
        &self,
        _: &str,
        _: &str,
        _: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        Err(OrchestrationError("async store required".into()))
    }
    fn compare_and_swap(
        &self,
        _: WorkflowRunUpsert,
        _: u64,
        _: &str,
        _: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        Err(OrchestrationError("async store required".into()))
    }
    fn renew(&self, _: &str, _: &str, _: Duration) -> Result<bool, OrchestrationError> {
        Err(OrchestrationError("async store required".into()))
    }
}
#[async_trait]
impl WorkflowExecutor for AuthoritativeWorkflow {
    async fn execute(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        _: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError> {
        Err(OrchestrationError("async executor required".into()))
    }
    async fn cancel_children(&self, _: &[String]) {}
}
impl WorkflowEngine<AuthoritativeWorkflow, AuthoritativeWorkflow> {
    /// Select only the supplied asynchronous authority and child executor.
    /// No legacy persistence callback or workspace fallback is installed.
    pub fn new_async(
        store: Arc<dyn AsyncWorkflowStore>,
        executor: Arc<dyn AsyncWorkflowExecutor>,
    ) -> Self {
        let mut engine = Self::new(
            Arc::new(AuthoritativeWorkflow),
            Arc::new(AuthoritativeWorkflow),
        );
        engine.async_store = Some(store);
        engine.async_executor = Some(executor);
        engine
    }
}
