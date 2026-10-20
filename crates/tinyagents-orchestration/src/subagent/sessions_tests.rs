use super::*;
use crate::subagent::SubAgentJobsTool;
use crate::subagent::SubagentRequest;
use std::sync::Arc;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::ids::new_call_id;
use tinyagents_harness::tool::ToolDispatch;
use tinyagents_storage_kit::{DocumentStore, Scope, StorageBackend};
use tinyagents_tasks::TaskStore;
use tinystoragedrivers_testkit::MemoryBackend;

fn docs_of(storage: &dyn StorageBackend, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn selector(task_key: &str) -> SubagentSessionSelector {
    SubagentSessionSelector {
        parent_session: "parent".into(),
        parent_thread_id: Some("thread".into()),
        agent_id: "researcher".into(),
        model: Some("model-a".into()),
        sandbox_mode: "read_only".into(),
        action_root: None,
        task_key: task_key.into(),
    }
}

fn lifecycle_key() -> SubagentTaskKey {
    SubagentTaskKey {
        root_run_id: "root".into(),
        parent_run_id: "parent-run".into(),
        thread_id: Some("thread".into()),
        task_id: "task-1".into(),
    }
}

fn upsert(force_fresh: bool, task_id: &str) -> SubagentSessionUpsert {
    SubagentSessionUpsert {
        selector: selector("analysis"),
        lifecycle_key: lifecycle_key(),
        display_name: Some("Researcher".into()),
        task_title: "Analyze the repository".into(),
        worker_thread_id: Some("worker-thread".into()),
        task_id: task_id.into(),
        force_fresh,
    }
}

#[test]
fn legacy_session_rows_decode_without_a_lifecycle_key() {
    let mut value = serde_json::to_value(DurableSubagentSession {
        subagent_session_id: "legacy".into(),
        parent_session: "parent".into(),
        parent_thread_id: Some("thread".into()),
        worker_thread_id: None,
        agent_id: "researcher".into(),
        display_name: None,
        model: None,
        sandbox_mode: "read_only".into(),
        action_root: None,
        task_key: "analysis".into(),
        lifecycle_key: Some(lifecycle_key()),
        task_title: "Analyze".into(),
        current_task_id: Some("legacy-task".into()),
        status: DurableSubagentStatus::Idle,
        reusable: true,
        latest_history: None,
        latest_error: None,
        created_at: "2026-01-01T00:00:00Z".into(),
        updated_at: "2026-01-01T00:00:00Z".into(),
        last_used_at: "2026-01-01T00:00:00Z".into(),
    })
    .unwrap();
    value.as_object_mut().unwrap().remove("lifecycleKey");
    let decoded: DurableSubagentSession = serde_json::from_value(value).unwrap();
    assert!(decoded.lifecycle_key.is_none());
    assert_eq!(decoded.current_task_id.as_deref(), Some("legacy-task"));
}

#[tokio::test]
async fn session_selector_requires_matching_scope_and_reusable_status() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let selector = selector("analysis");
    let (session, _) = store.upsert_running(upsert(false, "task-1")).await.unwrap();
    assert!(session.matches_selector(&selector));

    let mut other_task = selector.clone();
    other_task.task_key = "different".into();
    assert!(!session.matches_selector(&other_task));
}

#[tokio::test]
async fn driver_store_reuses_matching_session_and_force_fresh_creates_another() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let (first, first_decision) = store.upsert_running(upsert(false, "task-1")).await.unwrap();
    assert_eq!(first_decision, ReuseDecision::SpawnedNew);

    let (reused, decision) = store.upsert_running(upsert(false, "task-2")).await.unwrap();
    assert_eq!(decision, ReuseDecision::ReusedRunning);
    assert_eq!(reused.subagent_session_id, first.subagent_session_id);
    assert_eq!(reused.current_task_id.as_deref(), Some("task-2"));

    let (fresh, decision) = store.upsert_running(upsert(true, "task-3")).await.unwrap();
    assert_eq!(decision, ReuseDecision::ForcedFresh);
    assert_ne!(fresh.subagent_session_id, first.subagent_session_id);
}

#[tokio::test]
async fn closing_a_session_removes_it_from_reuse_and_preserves_it_in_listing() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let (session, _) = store.upsert_running(upsert(false, "task-1")).await.unwrap();
    assert!(store.close(&session.subagent_session_id).await.unwrap());
    assert!(!store.close("missing").await.unwrap());

    let closed = store
        .get(&session.subagent_session_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(closed.status, DurableSubagentStatus::Closed);
    assert!(!closed.reusable);
    assert!(find_reusable_session([closed], &selector("analysis")).is_none());
}

#[tokio::test]
async fn stale_completion_cannot_overwrite_a_reassigned_or_closed_session() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let (session, _) = store.upsert_running(upsert(false, "task-1")).await.unwrap();
    store.upsert_running(upsert(false, "task-2")).await.unwrap();
    assert!(
        !store
            .mark_finished(
                &session.subagent_session_id,
                "task-1",
                DurableSubagentStatus::Idle,
                Vec::new()
            )
            .await
            .unwrap()
    );
    assert!(store.close(&session.subagent_session_id).await.unwrap());
    assert!(
        !store
            .mark_failed(
                &session.subagent_session_id,
                "task-2",
                "late failure".into()
            )
            .await
            .unwrap()
    );
    let current = store
        .get(&session.subagent_session_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current.status, DurableSubagentStatus::Closed);
    assert_eq!(current.current_task_id.as_deref(), Some("task-2"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_first_upserts_claim_one_durable_session_for_a_selector() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let left_store = store.clone();
    let right_store = store.clone();
    let (left, right) = tokio::join!(
        tokio::spawn(async move { left_store.upsert_running(upsert(false, "task-left")).await }),
        tokio::spawn(async move {
            right_store
                .upsert_running(upsert(false, "task-right"))
                .await
        }),
    );
    let (left, _) = left.unwrap().unwrap();
    let (right, _) = right.unwrap().unwrap();
    assert_eq!(left.subagent_session_id, right.subagent_session_id);
    let sessions = store.list_for_parent("parent", None).await.unwrap();
    assert_eq!(sessions.len(), 1);
}

#[tokio::test]
async fn jobs_tool_overlays_scoped_sessions_and_awaits_user_envelopes() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let parent = RunContext::new(RunConfig::new("parent"), ());
    let owner_key = parent.instance_id().to_string();
    let mut request = upsert(false, "task-waiting");
    request.selector.parent_session = owner_key.clone();
    let (session, _) = store.upsert_running(request).await.unwrap();
    assert!(
        store
            .mark_finished(
                &session.subagent_session_id,
                "task-waiting",
                DurableSubagentStatus::AwaitingUser,
                Vec::new()
            )
            .await
            .unwrap()
    );
    let task_store = Arc::new(tinyagents_tasks::InMemoryTaskStore::new());
    let parent_run = tinyagents_harness::ids::RunId::new(owner_key);
    task_store
        .insert(
            tinyagents_tasks::OrchestrationTaskSpec::new(
                tinyagents_harness::ids::TaskId::new("task-record"),
                tinyagents_tasks::OrchestrationTaskKind::SubAgent {
                    agent: "researcher".into(),
                },
            )
            .with_lineage(parent_run.clone(), parent_run.clone()),
        )
        .unwrap();
    let jobs = SubAgentJobsTool::new(crate::subagent::SubAgentJobRegistry::default())
        .with_sessions(Arc::new(store.clone()), |owner| owner.to_string())
        .with_task_store(task_store, move |_| parent_run.clone());

    let listed = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let list: serde_json::Value = serde_json::from_str(&listed.output()).unwrap();
    assert_eq!(list[0]["kind"], "session");
    assert!(
        list.as_array()
            .unwrap()
            .iter()
            .any(|row| row["kind"] == "task")
    );
    let awaited = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"action":"await","job_id":session.subagent_session_id}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let envelope: serde_json::Value = serde_json::from_str(&awaited.output()).unwrap();
    assert_eq!(envelope["status"], "awaiting_user");
    let rendered = envelope["awaiting_user_envelope"].as_str().unwrap();
    let pause = crate::subagent::SubagentAwaitingUser::parse(rendered).unwrap();
    assert_eq!(pause.task_id, session.current_task_id.clone().unwrap());
    assert_eq!(pause.agent_id, "researcher");
    let collected = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"action":"await"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let collected: serde_json::Value = serde_json::from_str(&collected.output()).unwrap();
    assert_eq!(collected["collected_tombstones"], serde_json::json!([]));
    assert_eq!(
        collected["awaiting_user"][0]["session_id"],
        session.subagent_session_id
    );
    assert!(
        collected["awaiting_user"][0]["awaiting_user_envelope"]
            .as_str()
            .unwrap()
            .contains("[SUBAGENT_AWAITING_USER]")
    );
    let closed = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"action":"close","job_id":session.subagent_session_id}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    assert!(closed.output().contains("true"));
}

#[tokio::test]
async fn jobs_wait_recovers_terminal_results_from_the_durable_task_store() {
    use tinyagents_tasks::{InMemoryTaskStore, OrchestrationTaskKind, OrchestrationTaskResult};

    let parent = RunContext::new(RunConfig::new("parent"), ());
    let owner = tinyagents_harness::ids::RunId::new("owner");
    let task_store = Arc::new(InMemoryTaskStore::new());
    let mut completed = tinyagents_tasks::OrchestrationTaskSpec::new(
        "task-completed",
        OrchestrationTaskKind::SubAgent {
            agent: "researcher".into(),
        },
    )
    .with_lineage(owner.clone(), owner.clone())
    .with_metadata("parentSession", owner.to_string())
    .with_metadata("subagentSessionId", "session-completed");
    completed.thread_id = Some(tinyagents_harness::ids::ThreadId::new("thread-a"));
    task_store.insert(completed).unwrap();
    task_store
        .mark_running(&tinyagents_harness::ids::TaskId::new("task-completed"))
        .unwrap();
    task_store
        .complete(
            &tinyagents_harness::ids::TaskId::new("task-completed"),
            OrchestrationTaskResult::text("durable answer"),
        )
        .unwrap();

    let mut awaiting = tinyagents_tasks::OrchestrationTaskSpec::new(
        "task-awaiting",
        OrchestrationTaskKind::SubAgent {
            agent: "planner".into(),
        },
    )
    .with_lineage(owner.clone(), owner.clone())
    .with_metadata("parentSession", owner.to_string())
    .with_metadata("subagentSessionId", "session-awaiting");
    awaiting.thread_id = Some(tinyagents_harness::ids::ThreadId::new("thread-a"));
    task_store.insert(awaiting).unwrap();
    task_store
        .mark_running(&tinyagents_harness::ids::TaskId::new("task-awaiting"))
        .unwrap();
    task_store
        .mark_awaiting_with_question(
            &tinyagents_harness::ids::TaskId::new("task-awaiting"),
            "Which environment?".into(),
        )
        .unwrap();

    let mut failed = tinyagents_tasks::OrchestrationTaskSpec::new(
        "task-failed",
        OrchestrationTaskKind::SubAgent {
            agent: "reviewer".into(),
        },
    )
    .with_lineage(owner.clone(), owner.clone())
    .with_metadata("parentSession", owner.to_string())
    .with_metadata("subagentSessionId", "session-failed");
    failed.thread_id = Some(tinyagents_harness::ids::ThreadId::new("thread-a"));
    task_store.insert(failed).unwrap();
    task_store
        .mark_running(&tinyagents_harness::ids::TaskId::new("task-failed"))
        .unwrap();
    task_store
        .fail(
            &tinyagents_harness::ids::TaskId::new("task-failed"),
            "durable failure".into(),
        )
        .unwrap();

    let jobs = SubAgentJobsTool::new(crate::subagent::SubAgentJobRegistry::default())
        .with_fixed_action(crate::subagent::SubAgentJobAction::Await)
        .with_identifier_fields("task_id", "subagent_session_id")
        .with_task_store(task_store.clone(), move |_| owner.clone())
        .with_task_thread_scope(|_| Some("thread-a".into()));

    let completed_by_task = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"task_id":"task-completed"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let completed_by_session: serde_json::Value =
        serde_json::from_str(&completed_by_task.output()).unwrap();
    assert_eq!(completed_by_session["status"], "completed");
    assert_eq!(completed_by_session["output"], "durable answer");
    assert_eq!(completed_by_session["agent"], "researcher");
    assert_eq!(completed_by_session["session_id"], "session-completed");
    assert_eq!(completed_by_session["extensions"]["iterations"], 0);

    let awaiting = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"subagent_session_id":"session-awaiting"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let awaiting: serde_json::Value = serde_json::from_str(&awaiting.output()).unwrap();
    assert_eq!(awaiting["status"], "awaiting_user");
    assert_eq!(awaiting["output"], "Which environment?");
    assert_eq!(awaiting["agent"], "planner");

    let failed = ToolDispatch::<(), ()>::execute(
        &jobs,
        &(),
        new_call_id(),
        serde_json::json!({"task_id":"task-failed"}),
        tinytools::ToolCallOptions::default(),
        &parent,
    )
    .await
    .unwrap();
    let failed: serde_json::Value = serde_json::from_str(&failed.output()).unwrap();
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"], "durable failure");
    assert_eq!(failed["agent"], "reviewer");

    let foreign_jobs = SubAgentJobsTool::new(crate::subagent::SubAgentJobRegistry::default())
        .with_fixed_action(crate::subagent::SubAgentJobAction::Await)
        .with_identifier_field("task_id")
        .with_task_store(task_store.clone(), |_| {
            tinyagents_harness::ids::RunId::new("foreign-owner")
        })
        .with_task_thread_scope(|_| Some("thread-a".into()));
    assert!(
        ToolDispatch::<(), ()>::execute(
            &foreign_jobs,
            &(),
            new_call_id(),
            serde_json::json!({"task_id":"task-completed"}),
            tinytools::ToolCallOptions::default(),
            &parent,
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn continuation_by_session_id_keeps_the_original_lifecycle_key() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let (session, _) = store.upsert_running(upsert(false, "task-1")).await.unwrap();
    let fresh = RunContext::new(RunConfig::new("fresh-parent").with_thread("thread"), ());

    let request = SubagentRequest::continue_with_key(
        session.subagent_session_id.as_str(),
        fresh,
        (),
        "continue work",
        None,
        Some(&store),
    )
    .await
    .unwrap();
    let parts = request.into_parts();

    assert_eq!(parts.task_key, lifecycle_key());
    assert_eq!(parts.input, "continue work");
}

#[tokio::test]
async fn continuation_by_unknown_session_id_is_rejected() {
    let store = DriverSubagentSessionStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap();
    let fresh = RunContext::new(RunConfig::new("fresh-parent").with_thread("thread"), ());

    let error = SubagentRequest::continue_with_key(
        "missing-session",
        fresh,
        (),
        "continue work",
        None,
        Some(&store),
    )
    .await
    .err()
    .unwrap();
    assert!(error.to_string().contains("unknown subagent session"));
}

#[test]
fn task_helpers_bound_length_and_normalize_display_text() {
    assert_eq!(normalize_task_key("  Fix  the BUG! "), "fix-the-bug");
    assert_eq!(normalize_task_key("  "), "untitled-task");
    assert!(normalize_task_key(&"a".repeat(160)).len() <= 96);
    assert_eq!(task_title_from_prompt("  one\n two "), "one two");
    assert_eq!(reuse_decision(None, true), ReuseDecision::ForcedFresh);
}
