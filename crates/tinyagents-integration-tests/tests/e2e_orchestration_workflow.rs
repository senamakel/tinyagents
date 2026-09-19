//! Public end-to-end coverage for durable host-neutral workflows.
//!
//! This exercises the orchestration crate exactly as an embedding host does:
//! a session-backed store persists the workflow, while a host executor creates
//! children and returns structured output for downstream phases.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::CancellationToken;
use tinyagents_orchestration::workflow::{
    OrchestrationError, SessionWorkflowStore, WorkflowChildRegistration, WorkflowChildRequest,
    WorkflowChildResult, WorkflowDefinition, WorkflowEngine, WorkflowExecutor, WorkflowPhase,
    WorkflowStore,
};
use tinyagents_session::run_ledger::WorkflowRunStatus;

#[derive(Default)]
struct RecordingExecutor;

#[async_trait]
impl WorkflowExecutor for RecordingExecutor {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        _cancel: CancellationToken,
        registration: Arc<dyn WorkflowChildRegistration>,
    ) -> Result<WorkflowChildResult, OrchestrationError> {
        let child_id = format!(
            "{}-{}-{}",
            request.run_id, request.phase, request.index_in_phase
        );
        registration.register(child_id.clone())?;
        Ok(WorkflowChildResult {
            child_id,
            output: json!({"agent": request.agent_id, "prompt": request.prompt}),
        })
    }

    async fn cancel_children(&self, _child_ids: &[String]) {}
}

fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        id: "release-notes".into(),
        name: "Release notes".into(),
        description: "Plan and write a release note.".into(),
        phases: vec![
            WorkflowPhase {
                name: "plan".into(),
                description: "Identify the important changes.".into(),
                agent_ids: vec!["planner".into()],
                depends_on: vec![],
            },
            WorkflowPhase {
                name: "write".into(),
                description: "Write the notes using the plan.".into(),
                agent_ids: vec!["writer".into()],
                depends_on: vec!["plan".into()],
            },
        ],
        default_concurrency: 1,
        max_children: 2,
        extensions: BTreeMap::new(),
    }
}

#[tokio::test]
async fn workflow_persists_outputs_and_threads_them_to_dependent_phases() {
    let workspace = tempfile::tempdir().expect("temporary session workspace");
    let store = Arc::new(SessionWorkflowStore::new(workspace.path()));
    let engine = WorkflowEngine::new(store.clone(), Arc::new(RecordingExecutor));
    let definition = definition();

    engine
        .initialise(
            "run-release-notes".into(),
            &definition,
            json!({"release": "2.1.2"}),
            Some("thread-release".into()),
        )
        .expect("initialise durable workflow");
    engine
        .drive("run-release-notes", &definition, CancellationToken::new())
        .await
        .expect("drive workflow to completion");

    let run = store
        .load("run-release-notes")
        .expect("load durable workflow")
        .expect("workflow run exists");
    assert_eq!(run.status, WorkflowRunStatus::Completed);
    assert_eq!(run.child_run_ids.len(), 2);
    assert!(
        run.summary
            .as_deref()
            .is_some_and(|summary| summary.contains("writer"))
    );

    let write_output: &Value = &run.phase_states["write"]["outputs"][0]["metadata"]["rawOutput"];
    assert_eq!(write_output["agent"], "writer");
    assert!(
        write_output["prompt"]
            .as_str()
            .is_some_and(|prompt| prompt.contains("planner"))
    );
}
