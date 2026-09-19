use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tinyagents_graph::GraphEventSink;
use tinyagents_graph::parallel::{FailurePolicy, ParallelOptions, map_reduce};
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{
    WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert, get_workflow_run, upsert_workflow_run,
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

/// Host-owned execution and cancellation mechanism.
#[async_trait]
pub trait WorkflowExecutor: Send + Sync {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
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
        // The graph package owns the scheduling primitive used within every
        // phase. This bounded dispatcher preserves input order while avoiding a
        // second task registry or map/reduce implementation here.
        let _sink = &self.event_sink;
        let mut total_spawned = self
            .store
            .load(run_id)?
            .map(|run| run.child_run_ids.len() as u32)
            .ok_or_else(|| {
                OrchestrationError(format!("workflow run {run_id} vanished before start"))
            })?;

        loop {
            let run = self.store.load(run_id)?.ok_or_else(|| {
                OrchestrationError(format!("workflow run {run_id} vanished mid-loop"))
            })?;
            if cancel.is_cancelled() {
                self.executor.cancel_children(&run.child_run_ids).await;
                self.persist(
                    &run,
                    run.phase_states.clone(),
                    run.child_run_ids.clone(),
                    WorkflowRunStatus::Interrupted,
                    None,
                    false,
                )?;
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
                    )?;
                } else {
                    self.persist(
                        &run,
                        run.phase_states.clone(),
                        run.child_run_ids.clone(),
                        WorkflowRunStatus::Failed,
                        Some("no runnable phase (dependency deadlock)".to_owned()),
                        true,
                    )?;
                }
                return Ok(());
            };
            let spawned = self
                .run_phase(&run, definition, &phase, total_spawned, cancel.clone())
                .await?;
            total_spawned += spawned;
            if self
                .store
                .load(run_id)?
                .is_some_and(|current| current.status != WorkflowRunStatus::Running)
            {
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
    ) -> Result<u32, OrchestrationError> {
        let mut phase_states = run.phase_states.clone();
        let mut child_ids = run.child_run_ids.clone();
        set_phase_status(&mut phase_states, &phase.name, PhaseStatus::Running, None);
        self.persist(
            run,
            phase_states.clone(),
            child_ids.clone(),
            WorkflowRunStatus::Running,
            None,
            false,
        )?;

        let budget = definition.max_children.saturating_sub(total_spawned) as usize;
        if budget == 0 {
            return self.fail_phase(
                run,
                &mut phase_states,
                child_ids,
                phase,
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                ),
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
        let executor = self.executor.clone();
        let worker_cancel = cancel.clone();
        let outcomes = map_reduce(
            requests,
            ParallelOptions::default()
                .with_max_concurrency(definition.default_concurrency as usize)
                .with_failure_policy(FailurePolicy::CollectAll)
                .with_cancellation(cancel.clone()),
            move |_index, request| {
                let executor = executor.clone();
                let cancel = worker_cancel.clone();
                async move {
                    executor.execute(request, cancel).await.map_err(|error| {
                        tinyagents_harness::TinyAgentsError::Graph(error.to_string())
                    })
                }
            },
        )
        .await;
        let outcomes = match outcomes {
            Ok(outcomes) => outcomes,
            Err(tinyagents_harness::TinyAgentsError::Cancelled) => {
                self.executor.cancel_children(&child_ids).await;
                self.persist(
                    run,
                    phase_states,
                    child_ids,
                    WorkflowRunStatus::Interrupted,
                    None,
                    false,
                )?;
                return Ok(0);
            }
            Err(error) => return Err(OrchestrationError(error.to_string())),
        };
        let mut outputs = Vec::new();
        let mut failure = None;
        let mut spawned = 0_u32;
        for outcome in outcomes.outcomes {
            match outcome.result {
                Ok(result) => {
                    spawned += 1;
                    child_ids.push(result.child_id.clone());
                    outputs.push(
                        json!({ "orchestrationId": result.child_id, "output": result.output }),
                    );
                }
                Err(error) if failure.is_none() => failure = Some(error),
                Err(_) => {}
            }
        }
        if cancel.is_cancelled() {
            self.executor.cancel_children(&child_ids).await;
            self.persist(
                run,
                phase_states,
                child_ids,
                WorkflowRunStatus::Interrupted,
                None,
                false,
            )?;
            return Ok(0);
        }
        if let Some(reason) = failure.or_else(|| {
            capped.then(|| {
                format!(
                    "max_children cap ({}) reached before phase '{}' completed",
                    definition.max_children, phase.name
                )
            })
        }) {
            return self.fail_phase(run, &mut phase_states, child_ids, phase, reason);
        }
        set_phase_status(
            &mut phase_states,
            &phase.name,
            PhaseStatus::Completed,
            Some(Value::Array(outputs)),
        );
        self.persist(
            run,
            phase_states,
            child_ids,
            WorkflowRunStatus::Running,
            None,
            false,
        )?;
        Ok(spawned)
    }

    fn fail_phase(
        &self,
        run: &WorkflowRun,
        phase_states: &mut Value,
        child_ids: Vec<String>,
        phase: &WorkflowPhase,
        reason: String,
    ) -> Result<u32, OrchestrationError> {
        set_phase_status(
            phase_states,
            &phase.name,
            PhaseStatus::Failed,
            Some(json!([])),
        );
        set_phase_reason(phase_states, &phase.name, &reason);
        self.persist(
            run,
            phase_states.clone(),
            child_ids,
            WorkflowRunStatus::Failed,
            Some(reason),
            true,
        )?;
        Ok(0)
    }

    fn persist(
        &self,
        run: &WorkflowRun,
        phase_states: Value,
        child_run_ids: Vec<String>,
        status: WorkflowRunStatus,
        summary: Option<String>,
        terminal: bool,
    ) -> Result<WorkflowRun, OrchestrationError> {
        self.store.upsert(WorkflowRunUpsert {
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
        })
    }
}
