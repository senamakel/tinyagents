use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tinyagents_graph::GraphEventSink;
use tinyagents_graph::parallel::{FailurePolicy, ParallelOptions, map_reduce};
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{
    WorkflowLeaseClaim, WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert,
    compare_and_swap_workflow_run, get_workflow_run, try_claim_workflow_run, upsert_workflow_run,
};

use super::state::{
    PhaseStatus, all_phases_completed, init_phase_states, next_runnable_phase, phase_prompt,
    set_phase_reason, set_phase_status, synthesize_summary, upstream_outputs,
};
use super::{WorkflowDefinition, WorkflowPhase};

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
}

/// `tinyagents-session` run-ledger adapter with a caller-selected workspace.
#[derive(Debug, Clone)]
pub struct SessionWorkflowStore {
    workspace_dir: PathBuf,
}

impl SessionWorkflowStore {
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }

    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }
}

impl WorkflowStore for SessionWorkflowStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        get_workflow_run(&self.workspace_dir, id).map_err(OrchestrationError::from)
    }

    fn upsert(&self, row: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        upsert_workflow_run(&self.workspace_dir, row).map_err(OrchestrationError::from)
    }

    fn claim(
        &self,
        id: &str,
        owner: &str,
        lease_for: Duration,
    ) -> Result<WorkflowLeaseClaim, OrchestrationError> {
        try_claim_workflow_run(
            &self.workspace_dir,
            id,
            owner,
            chrono::Duration::from_std(lease_for)
                .map_err(|error| OrchestrationError(error.to_string()))?,
        )
        .map_err(OrchestrationError::from)
    }

    fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: Duration,
    ) -> Result<Option<WorkflowRun>, OrchestrationError> {
        compare_and_swap_workflow_run(
            &self.workspace_dir,
            row,
            expected_revision,
            owner,
            chrono::Duration::from_std(lease_for)
                .map_err(|error| OrchestrationError(error.to_string()))?,
        )
        .map_err(OrchestrationError::from)
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
    event_sink: Option<Arc<dyn GraphEventSink>>,
}

const WORKFLOW_LEASE: Duration = Duration::from_secs(10 * 60);

struct PhaseRegistration<S: WorkflowStore> {
    store: Arc<S>,
    owner: String,
    run: parking_lot::Mutex<WorkflowRun>,
    phase_states: Value,
}

impl<S: WorkflowStore> PhaseRegistration<S> {
    fn current(&self) -> WorkflowRun {
        self.run.lock().clone()
    }
}

impl<S: WorkflowStore + 'static> WorkflowChildRegistration for PhaseRegistration<S> {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError> {
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
            WORKFLOW_LEASE,
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

impl<S, E> WorkflowEngine<S, E>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    pub fn new(store: Arc<S>, executor: Arc<E>) -> Self {
        Self {
            store,
            executor,
            event_sink: None,
        }
    }

    /// Attach an optional host sink for ordinary graph lifecycle tracing.
    pub fn with_event_sink(mut self, sink: Arc<dyn GraphEventSink>) -> Self {
        self.event_sink = Some(sink);
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

    /// Drive a run to a terminal state. Completed phases are never executed
    /// again, so a host may safely call this after process restart or resume.
    pub async fn drive(
        &self,
        run_id: &str,
        definition: &WorkflowDefinition,
        cancel: CancellationToken,
    ) -> Result<(), OrchestrationError> {
        // A driver lease is acquired before looking for runnable work.  This
        // is deliberately separate from the in-process cancellation token:
        // resume can race in another process, and only the durable lease
        // prevents both drivers from spawning the same phase.
        let owner = uuid::Uuid::new_v4().to_string();
        let mut run = match self.store.claim(run_id, &owner, WORKFLOW_LEASE)? {
            WorkflowLeaseClaim::Acquired(run) => run,
            WorkflowLeaseClaim::Busy(_) => return Ok(()),
            WorkflowLeaseClaim::Missing => {
                return Err(OrchestrationError(format!(
                    "workflow run {run_id} vanished before start"
                )));
            }
        };
        self.emit(tinyagents_graph::GraphEvent::RunStarted {
            run_id: tinyagents_harness::ids::RunId::new(run_id),
        });
        let mut total_spawned = run.child_run_ids.len() as u32;

        loop {
            if cancel.is_cancelled() {
                self.executor.cancel_children(&run.child_run_ids).await;
                self.persist(
                    &run,
                    run.phase_states.clone(),
                    run.child_run_ids.clone(),
                    WorkflowRunStatus::Interrupted,
                    None,
                    false,
                    &owner,
                )?;
                self.emit(tinyagents_graph::GraphEvent::RunCompleted {
                    run_id: tinyagents_harness::ids::RunId::new(run_id),
                    steps: 0,
                });
                return Ok(());
            }
            let Some(phase) = next_runnable_phase(definition, &run.phase_states).cloned() else {
                if all_phases_completed(definition, &run.phase_states) {
                    self.persist(
                        &run,
                        run.phase_states.clone(),
                        run.child_run_ids.clone(),
                        WorkflowRunStatus::Completed,
                        synthesize_summary(definition, &run.phase_states),
                        true,
                        &owner,
                    )?;
                } else {
                    self.persist(
                        &run,
                        run.phase_states.clone(),
                        run.child_run_ids.clone(),
                        WorkflowRunStatus::Failed,
                        Some("no runnable phase (dependency deadlock)".to_owned()),
                        true,
                        &owner,
                    )?;
                }
                self.emit(tinyagents_graph::GraphEvent::RunCompleted {
                    run_id: tinyagents_harness::ids::RunId::new(run_id),
                    steps: 0,
                });
                return Ok(());
            };
            self.emit(tinyagents_graph::GraphEvent::NodeStarted {
                node: tinyagents_harness::ids::NodeId::new("run_phase"),
                step: total_spawned as usize + 1,
            });
            let (updated, spawned) = self
                .run_phase(
                    &run,
                    definition,
                    &phase,
                    total_spawned,
                    cancel.clone(),
                    &owner,
                )
                .await?;
            run = updated;
            self.emit(tinyagents_graph::GraphEvent::NodeCompleted {
                node: tinyagents_harness::ids::NodeId::new("run_phase"),
                step: total_spawned as usize + 1,
            });
            total_spawned += spawned;
            if run.status != WorkflowRunStatus::Running {
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
    ) -> Result<(WorkflowRun, u32), OrchestrationError> {
        let mut phase_states = run.phase_states.clone();
        let mut child_ids = run.child_run_ids.clone();
        set_phase_status(&mut phase_states, &phase.name, PhaseStatus::Running, None);
        let running = self.persist(
            run,
            phase_states.clone(),
            child_ids.clone(),
            WorkflowRunStatus::Running,
            None,
            false,
            owner,
        )?;

        let budget = definition.max_children.saturating_sub(total_spawned) as usize;
        if budget == 0 {
            return self.fail_phase(
                &running,
                &mut phase_states,
                child_ids,
                phase,
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                ),
                owner,
            );
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
        });
        let executor = self.executor.clone();
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
                let cancel = worker_cancel.clone();
                let registration = worker_registration.clone();
                async move {
                    executor
                        .execute(request, cancel, registration)
                        .await
                        .map_err(|error| {
                            tinyagents_harness::TinyAgentsError::Graph(error.to_string())
                        })
                }
            },
        )
        .await;
        let outcomes = match outcomes {
            Ok(outcomes) => outcomes,
            Err(tinyagents_harness::TinyAgentsError::Cancelled) => {
                let children = registration.current().child_run_ids;
                self.executor.cancel_children(&children).await;
                let updated = self.persist(
                    &registration.current(),
                    phase_states,
                    children,
                    WorkflowRunStatus::Interrupted,
                    None,
                    false,
                    owner,
                )?;
                return Ok((updated, 0));
            }
            Err(error) => return Err(OrchestrationError(error.to_string())),
        };
        child_ids = registration.current().child_run_ids;
        let mut outputs = Vec::new();
        let mut failure = None;
        let mut spawned = 0_u32;
        for outcome in outcomes.outcomes {
            match outcome.result {
                Ok(result) => {
                    spawned += 1;
                    // The executor registered the real id before it could
                    // await completion. Keep older executors harmlessly
                    // compatible by accepting an already-present id only.
                    if !child_ids.iter().any(|id| id == &result.child_id) {
                        child_ids.push(result.child_id.clone());
                    }
                    outputs.push(
                        json!({ "orchestrationId": result.child_id, "output": result.output }),
                    );
                }
                Err(error) if failure.is_none() => failure = Some(error),
                Err(_) => {}
            }
        }
        if cancel.is_cancelled() {
            let children = registration.current().child_run_ids;
            self.executor.cancel_children(&children).await;
            let updated = self.persist(
                &registration.current(),
                phase_states,
                children,
                WorkflowRunStatus::Interrupted,
                None,
                false,
                owner,
            )?;
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
            return self.fail_phase(
                &registration.current(),
                &mut phase_states,
                child_ids,
                phase,
                reason,
                owner,
            );
        }
        set_phase_status(
            &mut phase_states,
            &phase.name,
            PhaseStatus::Completed,
            Some(Value::Array(outputs)),
        );
        let updated = self.persist(
            &registration.current(),
            phase_states,
            child_ids,
            WorkflowRunStatus::Running,
            None,
            false,
            owner,
        )?;
        Ok((updated, spawned))
    }

    fn fail_phase(
        &self,
        run: &WorkflowRun,
        phase_states: &mut Value,
        child_ids: Vec<String>,
        phase: &WorkflowPhase,
        reason: String,
        owner: &str,
    ) -> Result<(WorkflowRun, u32), OrchestrationError> {
        set_phase_status(
            phase_states,
            &phase.name,
            PhaseStatus::Failed,
            Some(json!([])),
        );
        set_phase_reason(phase_states, &phase.name, &reason);
        let updated = self.persist(
            run,
            phase_states.clone(),
            child_ids,
            WorkflowRunStatus::Failed,
            Some(reason),
            true,
            owner,
        )?;
        Ok((updated, 0))
    }

    fn persist(
        &self,
        run: &WorkflowRun,
        phase_states: Value,
        child_run_ids: Vec<String>,
        status: WorkflowRunStatus,
        summary: Option<String>,
        terminal: bool,
        owner: &str,
    ) -> Result<WorkflowRun, OrchestrationError> {
        self.store
            .compare_and_swap(
                WorkflowRunUpsert {
                    id: run.id.clone(),
                    definition_id: run.definition_id.clone(),
                    parent_thread_id: run.parent_thread_id.clone(),
                    input: run.input.clone(),
                    phase_states,
                    child_run_ids,
                    status,
                    summary,
                    started_at: Some(run.started_at),
                    completed_at: terminal.then(Utc::now),
                },
                run.revision,
                owner,
                WORKFLOW_LEASE,
            )?
            .ok_or_else(|| {
                OrchestrationError("workflow lease lost before durable state transition".to_owned())
            })
    }

    fn emit(&self, event: tinyagents_graph::GraphEvent) {
        if let Some(sink) = &self.event_sink {
            sink.emit(event);
        }
    }
}
