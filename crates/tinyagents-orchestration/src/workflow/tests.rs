use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use chrono::Utc;
use parking_lot::Mutex;
use serde_json::json;
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{WorkflowRun, WorkflowRunStatus, WorkflowRunUpsert};

use super::state::set_phase_status;
use super::*;

fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        id: "test".into(),
        name: "Test".into(),
        description: "test workflow".into(),
        phases: vec![
            WorkflowPhase {
                name: "plan".into(),
                description: "plan".into(),
                agent_ids: vec!["planner".into()],
                depends_on: vec![],
            },
            WorkflowPhase {
                name: "research".into(),
                description: "research".into(),
                agent_ids: vec!["researcher".into(), "researcher".into()],
                depends_on: vec!["plan".into()],
            },
            WorkflowPhase {
                name: "synthesize".into(),
                description: "synthesize".into(),
                agent_ids: vec!["writer".into()],
                depends_on: vec!["research".into()],
            },
        ],
        default_concurrency: 2,
        max_children: 8,
        extensions: BTreeMap::new(),
    }
}

#[derive(Default)]
struct MemoryStore(Mutex<HashMap<String, WorkflowRun>>);

impl WorkflowStore for MemoryStore {
    fn load(&self, id: &str) -> Result<Option<WorkflowRun>, OrchestrationError> {
        Ok(self.0.lock().get(id).cloned())
    }

    fn upsert(&self, update: WorkflowRunUpsert) -> Result<WorkflowRun, OrchestrationError> {
        let now = Utc::now();
        let prior = self.0.lock().get(&update.id).cloned();
        let row = WorkflowRun {
            id: update.id.clone(),
            definition_id: update.definition_id,
            parent_thread_id: update.parent_thread_id,
            input: update.input,
            phase_states: update.phase_states,
            child_run_ids: update.child_run_ids,
            status: update.status,
            summary: update
                .summary
                .or_else(|| prior.as_ref().and_then(|row| row.summary.clone())),
            started_at: update
                .started_at
                .unwrap_or_else(|| prior.as_ref().map(|row| row.started_at).unwrap_or(now)),
            updated_at: now,
            completed_at: update
                .completed_at
                .or_else(|| prior.and_then(|row| row.completed_at)),
        };
        self.0.lock().insert(row.id.clone(), row.clone());
        Ok(row)
    }
}

#[derive(Default)]
struct FakeExecutor {
    calls: Mutex<Vec<WorkflowChildRequest>>,
    active: AtomicUsize,
    peak: AtomicUsize,
    fail_agent: Mutex<Option<String>>,
    cancelled: AtomicUsize,
}

#[async_trait]
impl WorkflowExecutor for FakeExecutor {
    async fn execute(
        &self,
        request: WorkflowChildRequest,
        cancel: CancellationToken,
    ) -> Result<WorkflowChildResult, OrchestrationError> {
        if cancel.is_cancelled() {
            return Err(OrchestrationError("cancelled".into()));
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.calls.lock().push(request.clone());
        tokio::task::yield_now().await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self.fail_agent.lock().as_deref() == Some(request.agent_id.as_str()) {
            return Err(OrchestrationError("child failure".into()));
        }
        Ok(WorkflowChildResult {
            child_id: format!("{}-{}", request.phase, request.index_in_phase),
            output: json!(format!("{} output", request.phase)),
        })
    }

    async fn cancel_children(&self, _child_ids: &[String]) {
        self.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

fn engine() -> (
    Arc<MemoryStore>,
    Arc<FakeExecutor>,
    WorkflowEngine<MemoryStore, FakeExecutor>,
) {
    let store = Arc::new(MemoryStore::default());
    let executor = Arc::new(FakeExecutor::default());
    let engine = WorkflowEngine::new(store.clone(), executor.clone());
    (store, executor, engine)
}

#[test]
fn structural_validation_covers_invalid_definitions() {
    let mut empty = definition();
    empty.phases.clear();
    assert_eq!(validate_structure(&empty), vec![DefinitionError::NoPhases]);
    let mut bad = definition();
    bad.default_concurrency = 0;
    bad.max_children = 0;
    bad.phases[1].name = "plan".into();
    bad.phases[2].depends_on = vec!["missing".into()];
    let errors = validate_structure(&bad);
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::DuplicatePhase { .. }))
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::UnknownDependency { .. }))
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error, DefinitionError::InvalidConcurrency { .. }))
    );
    let mut cyclic = definition();
    cyclic.phases[0].depends_on = vec!["synthesize".into()];
    assert!(validate_structure(&cyclic).contains(&DefinitionError::CyclicDependency));
}

#[test]
fn scheduler_topology_exposes_dispatch_run_and_done() {
    let topology = scheduler_graph().expect("topology");
    let nodes = topology
        .nodes
        .iter()
        .map(|node| node.id.as_str())
        .collect::<Vec<_>>();
    assert!(nodes.contains(&"dispatch") && nodes.contains(&"run_phase") && nodes.contains(&"done"));
}

#[tokio::test]
async fn engine_runs_in_deterministic_dependency_order_and_threads_context() {
    let (store, executor, engine) = engine();
    let def = definition();
    engine
        .initialise("run".into(), &def, json!({"question":"q"}), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    let run = store.load("run").unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Completed);
    let calls = executor.calls.lock();
    assert_eq!(
        calls
            .iter()
            .map(|call| call.phase.as_str())
            .collect::<Vec<_>>(),
        vec!["plan", "research", "research", "synthesize"]
    );
    assert!(
        calls
            .last()
            .unwrap()
            .prompt
            .contains("Context from prior phases")
    );
    assert!(run.summary.unwrap().contains("synthesize output"));
}

#[tokio::test]
async fn engine_respects_concurrency_global_cap_and_partial_failure() {
    let (store, executor, engine) = engine();
    let mut def = definition();
    def.default_concurrency = 1;
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(executor.peak.load(Ordering::SeqCst) <= 1);
    let mut cap = definition();
    cap.max_children = 2;
    engine
        .initialise("cap".into(), &cap, json!("q"), None)
        .unwrap();
    engine
        .drive("cap", &cap, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        store.load("cap").unwrap().unwrap().status,
        WorkflowRunStatus::Failed
    );
    *executor.fail_agent.lock() = Some("planner".into());
    let failing = definition();
    engine
        .initialise("failed".into(), &failing, json!("q"), None)
        .unwrap();
    engine
        .drive("failed", &failing, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        store.load("failed").unwrap().unwrap().status,
        WorkflowRunStatus::Failed
    );
}

#[tokio::test]
async fn cancellation_and_resume_do_not_repeat_completed_phases() {
    let (store, executor, engine) = engine();
    let def = definition();
    engine
        .initialise("run".into(), &def, json!("q"), None)
        .unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    engine.drive("run", &def, cancel).await.unwrap();
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Interrupted
    );
    let mut states = init_phase_states(&def);
    set_phase_status(
        &mut states,
        "plan",
        PhaseStatus::Completed,
        Some(json!([{ "output": "already" }])),
    );
    let run = store.load("run").unwrap().unwrap();
    store
        .upsert(WorkflowRunUpsert {
            id: run.id,
            definition_id: run.definition_id,
            parent_thread_id: run.parent_thread_id,
            input: run.input,
            phase_states: states,
            child_run_ids: vec!["old".into()],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: Some(run.started_at),
            completed_at: None,
        })
        .unwrap();
    engine
        .drive("run", &def, CancellationToken::new())
        .await
        .unwrap();
    assert!(
        !executor
            .calls
            .lock()
            .iter()
            .any(|call| call.phase == "plan")
    );
    assert_eq!(
        store.load("run").unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
}
