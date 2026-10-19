use super::*;
use crate::run_ledger::LedgerError;
use chrono::Duration;
use serde_json::json;
use std::sync::Arc;
use tinystoragedrivers_core::{
    DocumentStoreExt, ErrorKind, MemoryStorage, Query, Scope, ScopedStorage, StorageBackend,
    Versioned,
};

async fn collection_snapshot(
    storage: &ScopedStorage,
    collection: &str,
) -> Vec<Versioned<serde_json::Value>> {
    let mut rows = storage
        .documents()
        .query_all(collection, &Query::default())
        .await
        .unwrap();
    rows.sort_by(|left, right| left.id.cmp(&right.id));
    rows
}

/// Explicit handles share a backend while retaining separate tenant namespaces.
pub(super) fn ledgers() -> (DriverRunLedger, DriverRunLedger) {
    let backend = MemoryStorage::new();
    (
        DriverRunLedger::new(
            backend
                .for_scope(&Scope::new("profile-a").unwrap())
                .unwrap(),
        ),
        DriverRunLedger::new(
            backend
                .for_scope(&Scope::new("profile-b").unwrap())
                .unwrap(),
        ),
    )
}

#[tokio::test]
async fn file_workflow_records_round_trip_and_stay_in_their_scope() {
    let dir = tempfile::tempdir().unwrap();
    let backend = tinystoragedrivers_file::FileStorage::open(dir.path()).unwrap();
    let a = DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-a").unwrap())
            .unwrap(),
    );
    let b = DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-b").unwrap())
            .unwrap(),
    );

    let written = a.upsert_workflow_run(workflow("workflow")).await.unwrap();
    assert_eq!(a.get_workflow_run("workflow").await.unwrap(), Some(written));
    assert!(b.get_workflow_run("workflow").await.unwrap().is_none());
    assert_eq!(
        a.list_workflow_runs(&WorkflowRunListRequest::default())
            .await
            .unwrap()
            .count,
        1
    );
    assert_eq!(
        b.list_workflow_runs(&WorkflowRunListRequest::default())
            .await
            .unwrap()
            .count,
        0
    );
}

/// FileStorage supports atomic lease primitives even though its document store
/// cannot execute validated multi-document batches. Exercise the complete
/// workflow lease path, including a race and a revision-fenced transition.
#[tokio::test]
async fn file_workflow_lease_race_and_revision_fencing() {
    use crate::run_ledger::WorkflowLeaseClaim;

    let dir = tempfile::tempdir().unwrap();
    let backend = tinystoragedrivers_file::FileStorage::open(dir.path()).unwrap();
    let ledger = Arc::new(DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-a").unwrap())
            .unwrap(),
    ));
    let created = ledger
        .upsert_workflow_run(workflow("leased"))
        .await
        .unwrap();

    let start = Arc::new(tokio::sync::Barrier::new(3));
    let mut claimers = tokio::task::JoinSet::new();
    for owner in ["first", "second", "third"] {
        let ledger = ledger.clone();
        let start = start.clone();
        claimers.spawn(async move {
            start.wait().await;
            ledger
                .try_claim_workflow_run("leased", owner, Duration::minutes(1))
                .await
                .unwrap()
        });
    }
    let mut acquired = Vec::new();
    let mut busy = Vec::new();
    while let Some(result) = claimers.join_next().await {
        match result.unwrap() {
            WorkflowLeaseClaim::Acquired(run) => acquired.push(run),
            WorkflowLeaseClaim::Busy(run) => busy.push(run),
            WorkflowLeaseClaim::Missing => panic!("the seeded workflow remains present"),
        }
    }
    assert_eq!(acquired.len(), 1, "one live workflow lease owner wins");
    assert_eq!(busy.len(), 2, "every losing live claimant reports Busy");
    let owner = acquired[0].lease_owner.as_deref().unwrap();
    assert!(
        busy.iter()
            .all(|run| run.lease_owner.as_deref() == Some(owner))
    );
    assert!(
        ledger
            .renew_workflow_run_lease("leased", owner, Duration::minutes(1))
            .await
            .unwrap()
    );

    let current = ledger.get_workflow_run("leased").await.unwrap().unwrap();
    let mut transition = workflow("leased");
    transition.summary = Some("committed by lease owner".into());
    let committed = ledger
        .compare_and_swap_workflow_run(
            transition.clone(),
            current.revision,
            owner,
            Duration::minutes(1),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        committed.summary.as_deref(),
        Some("committed by lease owner")
    );
    assert!(
        ledger
            .compare_and_swap_workflow_run(
                transition,
                current.revision,
                owner,
                Duration::minutes(1)
            )
            .await
            .unwrap()
            .is_none(),
        "a stale revision cannot overwrite the committed workflow"
    );

    let not_owner = if owner == "first" { "second" } else { "first" };
    assert!(
        !ledger
            .renew_workflow_run_lease("leased", not_owner, Duration::minutes(1))
            .await
            .unwrap()
    );
    assert_eq!(created.id, committed.id);
}

/// Unsupported validated-batch operations must fail before changing any rows.
/// This pins FileStorage's capability boundary while still exercising the
/// single-record telemetry path that agents depend on.
#[tokio::test]
async fn file_ledger_rejects_atomic_agent_operations_without_partial_writes() {
    use crate::run_ledger::RunEventAppend;
    use tinystoragedrivers_core::Capability;

    let dir = tempfile::tempdir().unwrap();
    let backend = tinystoragedrivers_file::FileStorage::open(dir.path()).unwrap();
    let storage = backend
        .for_scope(&Scope::new("profile-a").unwrap())
        .unwrap();
    let ledger = DriverRunLedger::new(storage.clone());
    let telemetry = ledger
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "atomic-agent".into(),
            input_tokens: Some(17),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(telemetry.input_tokens, 17);

    let agents_before = collection_snapshot(&storage, "ledger.agent_runs").await;
    let telemetry_before = collection_snapshot(&storage, "ledger.telemetry").await;
    let error = ledger
        .upsert_agent_run(crate::run_ledger::AgentRunUpsert {
            id: "atomic-agent".into(),
            kind: crate::run_ledger::AgentRunKind::Subagent,
            parent_run_id: None,
            parent_thread_id: None,
            agent_id: None,
            status: crate::run_ledger::AgentRunStatus::Running,
            prompt_ref: None,
            worker_thread_id: None,
            checkpoint_path: None,
            checkpoint: None,
            summary: None,
            error: None,
            metadata: json!({}),
            started_at: None,
            completed_at: None,
        })
        .await
        .unwrap_err();
    assert!(
        matches!(error, LedgerError::Driver(ref native) if native.kind() == ErrorKind::Unsupported(Capability::ValidatedBatch))
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.agent_runs").await,
        agents_before,
        "failed atomic agent upsert leaves every agent record unchanged"
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.telemetry").await,
        telemetry_before,
        "failed atomic agent upsert leaves every telemetry record unchanged"
    );

    let event_heads_before = collection_snapshot(&storage, "ledger.event_heads").await;
    let events_before = collection_snapshot(&storage, "ledger.event_outbox").await;
    let error = ledger
        .append_run_event(RunEventAppend {
            run_id: "atomic-agent".into(),
            event_type: "created".into(),
            payload: json!({}),
        })
        .await
        .unwrap_err();
    assert!(
        matches!(error, LedgerError::Driver(ref native) if native.kind() == ErrorKind::Unsupported(Capability::ValidatedBatch))
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.event_heads").await,
        event_heads_before,
        "failed event append leaves every event head unchanged"
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.event_outbox").await,
        events_before,
        "failed event append leaves every event record unchanged"
    );

    ledger
        .upsert_agent_team_task(crate::run_ledger::AgentTeamTaskUpsert {
            id: "atomic-task".into(),
            team_id: "atomic-team".into(),
            title: "must remain unclaimed".into(),
            objective: None,
            status: crate::run_ledger::AgentTeamTaskStatus::Todo,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        })
        .await
        .unwrap();
    let tasks_before = collection_snapshot(&storage, "ledger.tasks").await;
    let members_before = collection_snapshot(&storage, "ledger.members").await;
    let error = ledger
        .claim_agent_team_task("atomic-team", "atomic-task", "member", "token")
        .await
        .unwrap_err();
    assert!(
        matches!(error, LedgerError::Driver(ref native) if native.kind() == ErrorKind::Unsupported(Capability::ValidatedBatch))
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.tasks").await,
        tasks_before,
        "failed atomic claim leaves every task record unchanged"
    );
    assert_eq!(
        collection_snapshot(&storage, "ledger.members").await,
        members_before,
        "failed atomic claim leaves every member record unchanged"
    );
}

/// A known workflow input whose first SQL revision is zero.
fn workflow(id: &str) -> WorkflowRunUpsert {
    WorkflowRunUpsert {
        id: id.into(),
        definition_id: "definition".into(),
        parent_thread_id: Some("thread".into()),
        input: json!({"value":1}),
        phase_states: json!({}),
        child_run_ids: vec![],
        status: WorkflowRunStatus::Running,
        summary: Some("original".into()),
        started_at: Some(now() - Duration::days(1)),
        completed_at: None,
    }
}

/// Workflow upserts preserve SQL merge/revision rules and isolate the configured namespaces.
#[tokio::test]
async fn scoped_workflow_upserts_preserve_revision_and_partial_fields() {
    let (a, b) = ledgers();
    let first = a.upsert_workflow_run(workflow("workflow")).await.unwrap();
    assert_eq!(first.revision, 0);
    let mut edit = workflow("workflow");
    edit.summary = None;
    edit.parent_thread_id = None;
    edit.started_at = None;
    edit.input = json!({"value":2});
    let changed = a.upsert_workflow_run(edit).await.unwrap();
    assert_eq!(changed.revision, 1);
    assert_eq!(changed.started_at, first.started_at);
    assert_eq!(changed.summary, first.summary);
    assert_eq!(changed.parent_thread_id, first.parent_thread_id);
    assert_eq!(
        a.get_workflow_run("workflow").await.unwrap().unwrap(),
        changed
    );
    assert!(b.get_workflow_run("workflow").await.unwrap().is_none());
}

/// Counts represent all matches and blank filters/offset/zero limit follow classic SQL semantics.
#[tokio::test]
async fn scoped_workflow_queries_preserve_filter_and_page_semantics() {
    let (a, b) = ledgers();
    for id in ["one", "two", "three"] {
        a.upsert_workflow_run(workflow(id)).await.unwrap();
    }
    let request = WorkflowRunListRequest {
        status: Some("   ".into()),
        limit: Some(1),
        offset: Some(1),
        ..Default::default()
    };
    let page = a.list_workflow_runs(&request).await.unwrap();
    assert_eq!(page.count, 3);
    assert_eq!(page.runs.len(), 1);
    assert_eq!(b.list_workflow_runs(&request).await.unwrap().count, 0);
    let padded = WorkflowRunListRequest {
        status: Some(" running ".into()),
        ..Default::default()
    };
    assert_eq!(a.list_workflow_runs(&padded).await.unwrap().count, 0);
    let empty = WorkflowRunListRequest {
        limit: Some(0),
        ..Default::default()
    };
    let page = a.list_workflow_runs(&empty).await.unwrap();
    assert_eq!(page.count, 3);
    assert!(page.runs.is_empty());
    let invalid = WorkflowRunListRequest {
        offset: Some(u64::MAX),
        ..Default::default()
    };
    assert!(
        matches!(a.list_workflow_runs(&invalid).await.unwrap_err(), LedgerError::Driver(ref native) if native.kind() == ErrorKind::InvalidInput)
    );
}

/// Repeated telemetry deltas preserve counters and do not cross a scoped handle.
#[tokio::test]
async fn scoped_telemetry_upserts_merge_and_isolate_records() {
    let (a, b) = ledgers();
    let first = a
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            input_tokens: Some(42),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(first.output_tokens, 0);
    let changed = a
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            model: Some("model".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(changed.input_tokens, 42);
    assert_eq!(changed.model.as_deref(), Some("model"));
    let separate = b
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            error: Some("failure".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(separate.input_tokens, 0);
    assert!(separate.model.is_none());
}

/// Team/member/task edits preserve classic identity, optional fields and ordering across scopes.
#[tokio::test]
async fn scoped_team_records_preserve_merge_identity_order_and_unique_names() {
    let (a, b) = ledgers();
    let old = now() - Duration::days(1);
    let input = AgentTeamUpsert {
        id: "team".into(),
        parent_thread_id: Some("thread".into()),
        lead_agent_id: "lead".into(),
        status: AgentTeamStatus::Active,
        summary: Some("original".into()),
        created_at: Some(old),
        closed_at: None,
    };
    let first = a.upsert_agent_team(input.clone()).await.unwrap();
    let changed = a
        .upsert_agent_team(AgentTeamUpsert {
            parent_thread_id: None,
            summary: None,
            created_at: Some(now()),
            ..input
        })
        .await
        .unwrap();
    assert_eq!(changed.created_at, first.created_at);
    assert_eq!(changed.summary, first.summary);
    assert_eq!(a.get_agent_team("team").await.unwrap(), Some(changed));
    assert!(b.get_agent_team("team").await.unwrap().is_none());
    assert_eq!(
        a.list_agent_teams(&AgentTeamListRequest::default())
            .await
            .unwrap()
            .count,
        1
    );
    let member = AgentTeamMemberUpsert {
        id: "member".into(),
        team_id: "team".into(),
        name: "name".into(),
        agent_id: Some("agent".into()),
        member_status: AgentTeamMemberStatus::Active,
        current_task_id: Some("task".into()),
        worker_thread_id: Some("worker".into()),
        run_id: Some("run".into()),
        created_at: Some(old),
    };
    let first = a.upsert_agent_team_member(member.clone()).await.unwrap();
    let changed = a
        .upsert_agent_team_member(AgentTeamMemberUpsert {
            team_id: "other".into(),
            agent_id: None,
            current_task_id: None,
            worker_thread_id: None,
            run_id: None,
            created_at: None,
            ..member.clone()
        })
        .await
        .unwrap();
    assert_eq!(changed.team_id, "team");
    assert_eq!(changed.created_at, first.created_at);
    assert_eq!(changed.agent_id, first.agent_id);
    assert_eq!(changed.current_task_id, first.current_task_id);
    assert_eq!(
        a.get_agent_team_member("member").await.unwrap(),
        Some(changed)
    );
    assert!(
        a.upsert_agent_team_member(AgentTeamMemberUpsert {
            id: "duplicate".into(),
            ..member.clone()
        })
        .await
        .is_err()
    );
    assert_eq!(a.list_agent_team_members("team").await.unwrap().len(), 1);
    assert!(b.list_agent_team_members("team").await.unwrap().is_empty());
    b.upsert_agent_team_member(member).await.unwrap();
    let task = AgentTeamTaskUpsert {
        id: "task".into(),
        team_id: "team".into(),
        title: "title".into(),
        objective: Some("objective".into()),
        status: AgentTeamTaskStatus::Todo,
        owner_member_id: Some("member".into()),
        depends_on: vec!["dependency".into()],
        gate_status: Some("failed".into()),
        gate_reason: Some("reason".into()),
        evidence: vec!["evidence".into()],
        source_run_id: Some("run".into()),
        order_index: 4,
        created_at: Some(old),
    };
    let first = a.upsert_agent_team_task(task.clone()).await.unwrap();
    let changed = a
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            team_id: "other".into(),
            objective: None,
            owner_member_id: None,
            gate_status: None,
            gate_reason: None,
            source_run_id: None,
            depends_on: vec![],
            evidence: vec![],
            created_at: None,
            ..task.clone()
        })
        .await
        .unwrap();
    assert_eq!(changed.team_id, "team");
    assert_eq!(changed.created_at, first.created_at);
    assert_eq!(changed.objective, first.objective);
    assert_eq!(changed.gate_status, "pending");
    assert!(changed.depends_on.is_empty());
    assert!(changed.evidence.is_empty());
    assert_eq!(a.get_agent_team_task("task").await.unwrap(), Some(changed));
    a.upsert_agent_team_task(AgentTeamTaskUpsert {
        id: "earlier".into(),
        order_index: -1,
        ..task
    })
    .await
    .unwrap();
    let tasks = a.list_agent_team_tasks("team").await.unwrap();
    assert_eq!(
        tasks
            .iter()
            .map(|task| task.id.as_str())
            .collect::<Vec<_>>(),
        ["earlier", "task"]
    );
    assert!(b.list_agent_team_tasks("team").await.unwrap().is_empty());
}

/// SQL-style offsets and counts span native driver pages instead of truncating at page one.
#[tokio::test]
async fn scoped_workflow_paging_follows_native_cursors() {
    let (ledger, _) = ledgers();
    for number in 0..300 {
        ledger
            .upsert_workflow_run(workflow(&format!("workflow-{number}")))
            .await
            .unwrap();
    }
    let all = ledger
        .list_workflow_runs(&WorkflowRunListRequest {
            limit: Some(500),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(all.count, 300);
    assert_eq!(all.runs.len(), 300);
    let tail = ledger
        .list_workflow_runs(&WorkflowRunListRequest {
            limit: Some(50),
            offset: Some(275),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(tail.count, 300);
    assert_eq!(tail.runs, all.runs[275..]);
}

/// A native SQLite reopen persists documents without using the classic ledger schema.
#[tokio::test]
async fn scoped_workflow_documents_survive_sqlite_reopen() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("ports.db");
    let scope = Scope::new("profile-a").unwrap();
    let first = {
        let backend = tinystoragedrivers_sqlite::SqliteStorage::open(&path).unwrap();
        let ledger = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
        ledger
            .upsert_workflow_run(workflow("workflow"))
            .await
            .unwrap()
    };
    let backend = tinystoragedrivers_sqlite::SqliteStorage::open(&path).unwrap();
    let ledger = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    assert_eq!(
        ledger.get_workflow_run("workflow").await.unwrap(),
        Some(first)
    );
    let other = DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-b").unwrap())
            .unwrap(),
    );
    assert!(other.get_workflow_run("workflow").await.unwrap().is_none());
    let sql = tinystoragedrivers_sqlite::rusqlite::Connection::open_with_flags(
        path,
        tinystoragedrivers_sqlite::rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    assert_eq!(sql.query_row("SELECT COUNT(*) FROM sqlite_schema WHERE type='table' AND name IN ('agent_runs', 'sessions', 'workflow_runs')", [], |row| row.get::<_, i64>(0)).unwrap(), 0);
}

/// Unknown schema bodies are reported as typed corruption and never replaced with default rows.
#[tokio::test]
async fn scoped_workflow_rejects_malformed_document_body() {
    let (ledger, _) = ledgers();
    ledger.declare().await.unwrap();
    ledger
        .scoped
        .documents()
        .put(
            WORKFLOWS,
            &storage::key(&["workflow"]).unwrap(),
            json!({"unexpected": "shape"}),
            tinystoragedrivers_core::Precondition::Absent,
        )
        .await
        .unwrap();
    assert!(
        matches!(ledger.get_workflow_run("workflow").await.unwrap_err(), LedgerError::Driver(ref error) if error.kind() == ErrorKind::Serialization)
    );
    assert!(
        matches!(ledger.upsert_workflow_run(workflow("workflow")).await.unwrap_err(), LedgerError::Driver(ref error) if error.kind() == ErrorKind::Serialization)
    );
}

/// Independent same-scope handles admit one owner and fence a prior owner's commits after takeover.
#[tokio::test]
async fn scoped_workflow_claims_fence_stale_owners_and_lifecycle_revisions() {
    let (a, _) = ledgers();
    a.upsert_workflow_run(workflow("workflow")).await.unwrap();
    let b = DriverRunLedger::new(a.scoped.clone());
    let (one, two) = tokio::join!(
        a.try_claim_workflow_run("workflow", "one", Duration::hours(1)),
        b.try_claim_workflow_run("workflow", "two", Duration::hours(1))
    );
    let one = one.unwrap();
    let two = two.unwrap();
    assert_eq!(
        usize::from(matches!(one, WorkflowLeaseClaim::Acquired(_)))
            + usize::from(matches!(two, WorkflowLeaseClaim::Acquired(_))),
        1
    );
    let owner = a
        .get_workflow_run("workflow")
        .await
        .unwrap()
        .unwrap()
        .lease_owner
        .unwrap();
    let expired = a
        .try_claim_workflow_run("workflow", &owner, Duration::seconds(-1))
        .await
        .unwrap();
    assert!(matches!(expired, WorkflowLeaseClaim::Acquired(_)));
    let WorkflowLeaseClaim::Acquired(taken) = b
        .try_claim_workflow_run("workflow", "replacement", Duration::hours(1))
        .await
        .unwrap()
    else {
        panic!("expired owner should be replaced");
    };
    assert!(
        !a.renew_workflow_run_lease("workflow", &owner, Duration::hours(1))
            .await
            .unwrap()
    );
    assert!(
        a.compare_and_swap_workflow_run(
            workflow("workflow"),
            taken.revision,
            &owner,
            Duration::hours(1)
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        b.renew_workflow_run_lease("workflow", "replacement", Duration::hours(1))
            .await
            .unwrap()
    );
    assert_eq!(
        b.get_workflow_run("workflow")
            .await
            .unwrap()
            .unwrap()
            .revision,
        taken.revision
    );
    let mut cancelled = workflow("workflow");
    cancelled.status = WorkflowRunStatus::Cancelled;
    let stopped = a
        .compare_and_swap_workflow_run_lifecycle(cancelled, taken.revision)
        .await
        .unwrap()
        .unwrap();
    assert!(stopped.lease_owner.is_none());
    assert!(stopped.lease_expires_at.is_none());
    assert_eq!(stopped.revision, taken.revision + 1);
    assert!(
        b.compare_and_swap_workflow_run(
            workflow("workflow"),
            taken.revision,
            "replacement",
            Duration::hours(1)
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        a.compare_and_swap_workflow_run_lifecycle(workflow("workflow"), taken.revision)
            .await
            .unwrap()
            .is_none()
    );
}

/// A live driver's terminal transition clears its lease and missing workflows cannot be created by CAS.
#[tokio::test]
async fn scoped_workflow_terminal_cas_preserves_started_at_and_clears_lease() {
    let (ledger, _) = ledgers();
    assert!(matches!(
        ledger
            .try_claim_workflow_run("missing", "owner", Duration::hours(1))
            .await
            .unwrap(),
        WorkflowLeaseClaim::Missing
    ));
    assert!(
        !ledger
            .renew_workflow_run_lease("missing", "owner", Duration::hours(1))
            .await
            .unwrap()
    );
    let first = ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    let WorkflowLeaseClaim::Acquired(claimed) = ledger
        .try_claim_workflow_run("workflow", "owner", Duration::hours(1))
        .await
        .unwrap()
    else {
        panic!("expected claim");
    };
    let mut input = workflow("workflow");
    input.status = WorkflowRunStatus::Completed;
    input.summary = None;
    input.parent_thread_id = None;
    let finished = ledger
        .compare_and_swap_workflow_run(input, claimed.revision, "owner", Duration::hours(1))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(finished.started_at, first.started_at);
    assert_eq!(finished.summary, first.summary);
    assert_eq!(finished.parent_thread_id, first.parent_thread_id);
    assert!(finished.lease_owner.is_none());
    assert!(finished.lease_expires_at.is_none());
}

/// Renewal requires strictly future expiry; takeover is admitted at exact equality.
#[tokio::test]
async fn scoped_workflow_expiry_boundary_uses_exact_supplied_timestamp() {
    let (ledger, _) = ledgers();
    let now = now();
    ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    // "old" holds a lease that lapses exactly at `now`.
    ledger
        .leases_at(now - Duration::hours(1))
        .acquire(
            LEDGER_OWNER,
            &storage::workflow_lease_key("workflow").unwrap(),
            "old",
            3_600_000,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(
        !ledger
            .renew_workflow_at("workflow", "old", Duration::hours(1), now)
            .await
            .unwrap()
    );
    let WorkflowLeaseClaim::Acquired(claimed) = ledger
        .claim_workflow_at("workflow", "new", Duration::hours(1), now)
        .await
        .unwrap()
    else {
        panic!("expiry equality permits takeover");
    };
    assert_eq!(claimed.lease_owner.as_deref(), Some("new"));
    assert_eq!(claimed.lease_expires_at, Some(now + Duration::hours(1)));
}

/// Separate runtime threads racing independent handles cannot both acquire a live lease.
#[tokio::test]
async fn scoped_workflow_concurrent_native_claim_has_one_owner() {
    let (ledger, _) = ledgers();
    ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = ["one", "two"]
        .into_iter()
        .map(|owner| {
            let handle = DriverRunLedger::new(ledger.scoped.clone());
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                barrier.wait();
                runtime
                    .block_on(handle.try_claim_workflow_run("workflow", owner, Duration::hours(1)))
                    .unwrap()
            })
        })
        .collect();
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, WorkflowLeaseClaim::Acquired(_)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, WorkflowLeaseClaim::Busy(_)))
            .count(),
        1
    );
}

/// Tool effects retain idempotent start, partial settle, unknown-call backfill and tenant isolation.
#[tokio::test]
async fn scoped_tool_effects_preserve_restarts_settle_and_unresolved_filter() {
    use crate::run_ledger::tool_effects::ToolEffectStatus;
    let (a, b) = ledgers();
    let input = ToolEffectStart {
        run_id: "run".into(),
        call_id: "call".into(),
        tool: "tool".into(),
        idempotency_key: Some("key".into()),
        effect_summary: Some("summary".into()),
    };
    let started = a.record_tool_started(input.clone()).await.unwrap();
    assert_eq!(started.status, ToolEffectStatus::Started);
    assert!(started.settled_at.is_none());
    assert_eq!(
        a.list_unresolved_tool_effects("run").await.unwrap(),
        std::slice::from_ref(&started)
    );
    assert!(
        b.list_unresolved_tool_effects("run")
            .await
            .unwrap()
            .is_empty()
    );
    let settled = a
        .settle_tool_effect(ToolEffectSettle {
            run_id: "run".into(),
            call_id: "call".into(),
            status: ToolEffectStatus::Completed,
            effect_summary: None,
        })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(settled.started_at, started.started_at);
    assert_eq!(settled.effect_summary, started.effect_summary);
    assert_eq!(settled.tool, "tool");
    assert_eq!(settled.idempotency_key.as_deref(), Some("key"));
    assert!(
        a.list_unresolved_tool_effects("run")
            .await
            .unwrap()
            .is_empty()
    );
    let restarted = a
        .record_tool_started(ToolEffectStart {
            tool: "replacement".into(),
            idempotency_key: None,
            effect_summary: None,
            ..input
        })
        .await
        .unwrap();
    assert_eq!(restarted.status, ToolEffectStatus::Started);
    assert!(restarted.settled_at.is_none());
    assert!(restarted.effect_summary.is_none());
    assert!(restarted.idempotency_key.is_none());
    let unknown = a
        .mark_tool_effect_interrupted("run", "unknown")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(unknown.status, ToolEffectStatus::Interrupted);
    assert_eq!(unknown.tool, "");
    assert_eq!(unknown.settled_at, Some(unknown.started_at));
    assert_eq!(
        a.list_unresolved_tool_effects("run").await.unwrap(),
        [restarted]
    );
}

/// Team and history operations that need atomic batches write only through them: a refused
/// batch (here: an unavailable dependency check) leaves every scoped record untouched.
#[tokio::test]
async fn scoped_ledger_collections_start_empty() {
    let (ledger, _) = ledgers();
    let before = ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    assert_eq!(
        ledger
            .claim_agent_team_task("team", "task", "member", "claim")
            .await
            .unwrap(),
        ClaimOutcome::UnknownTask
    );
    assert_eq!(
        ledger
            .complete_agent_team_task("team", "task", "member", &[], true)
            .await
            .unwrap(),
        CompletionOutcome::UnknownTask
    );
    assert!(
        ledger
            .shutdown_agent_team_member("team", "member")
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(ledger.apply_retention(now()).await.unwrap().total(), 0);
    assert_eq!(
        ledger.get_workflow_run("workflow").await.unwrap(),
        Some(before)
    );
    for collection in [
        storage::AGENTS,
        TELEMETRY,
        TEAMS,
        MEMBERS,
        TASKS,
        storage::EFFECTS,
    ] {
        assert_eq!(
            ledger
                .scoped
                .documents()
                .count(collection, &Filter::default())
                .await
                .unwrap(),
            0
        );
    }
}

/// A supplied native fence remains active through the adapter and refusal retains its native error kind.
#[tokio::test]
async fn scoped_workflow_writes_preserve_native_owner_fence_refusal() {
    use tinystoragedrivers_core::{CollectionSpec, Fence, Precondition};
    let backend = MemoryStorage::new();
    let owner_scope = Scope::new("owners").unwrap();
    let owner = backend.for_scope(&owner_scope).unwrap();
    owner
        .documents()
        .ensure_collection(&CollectionSpec::new("leases"))
        .await
        .unwrap();
    owner
        .documents()
        .put(
            "leases",
            "profile",
            json!({"epoch":1}),
            Precondition::Absent,
        )
        .await
        .unwrap();
    let scope = Scope::new("profile-a").unwrap();
    let fence = Fence::epoch(owner_scope, "leases", "profile", "epoch", 1);
    let ledger = DriverRunLedger::new(backend.for_scope_fenced(&scope, &fence).unwrap());
    let first = ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    owner
        .documents()
        .put("leases", "profile", json!({"epoch":2}), Precondition::None)
        .await
        .unwrap();
    assert!(
        matches!(ledger.upsert_workflow_run(workflow("workflow")).await.unwrap_err(),
        LedgerError::Driver(ref error) if error.kind() == ErrorKind::Fenced)
    );
    assert!(
        matches!(ledger.try_claim_workflow_run("workflow", "worker", Duration::hours(1)).await.unwrap_err(),
        LedgerError::Driver(ref error) if error.kind() == ErrorKind::Fenced)
    );
    let inspect = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    assert_eq!(
        inspect.get_workflow_run("workflow").await.unwrap(),
        Some(first)
    );
}

/// Single-record member lifecycle and task release preserve identity filters and run/evidence history.
#[tokio::test]
async fn scoped_member_lifecycle_and_task_release_preserve_history() {
    let (ledger, _) = ledgers();
    ledger
        .upsert_agent_team_member(AgentTeamMemberUpsert {
            id: "member".into(),
            team_id: "team".into(),
            name: "name".into(),
            agent_id: None,
            member_status: AgentTeamMemberStatus::Pending,
            current_task_id: None,
            worker_thread_id: None,
            run_id: None,
            created_at: None,
        })
        .await
        .unwrap();
    assert!(
        ledger
            .mark_agent_team_member_running("wrong-team", "member", "task", "worker", "run")
            .await
            .unwrap()
            .is_none()
    );
    let running = ledger
        .mark_agent_team_member_running("team", "member", "task", "worker", "run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(running.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(running.current_task_id.as_deref(), Some("task"));
    assert!(
        ledger
            .mark_agent_team_member_idle("wrong-team", "member")
            .await
            .unwrap()
            .is_none()
    );
    let idle = ledger
        .mark_agent_team_member_idle("team", "member")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(idle.member_status, AgentTeamMemberStatus::Idle);
    assert!(idle.current_task_id.is_none());
    assert_eq!(idle.run_id, running.run_id);
    assert_eq!(idle.worker_thread_id, running.worker_thread_id);
    ledger
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            id: "task".into(),
            team_id: "team".into(),
            title: "title".into(),
            objective: None,
            status: AgentTeamTaskStatus::InProgress,
            owner_member_id: Some("owner".into()),
            depends_on: vec!["dependency".into()],
            gate_status: Some("failed".into()),
            gate_reason: Some("reason".into()),
            evidence: vec!["evidence".into()],
            source_run_id: Some("source".into()),
            order_index: 0,
            created_at: None,
        })
        .await
        .unwrap();
    ledger
        .merge(TASKS, "task", |previous: Option<AgentTeamTask>| {
            let mut row = previous.unwrap();
            row.claimed_by_member_id = Some("member".into());
            row.claim_token = Some("claim".into());
            row
        })
        .await
        .unwrap();
    assert!(
        !ledger
            .release_agent_team_task("wrong-team", "task")
            .await
            .unwrap()
    );
    assert!(
        ledger
            .release_agent_team_task("team", "task")
            .await
            .unwrap()
    );
    assert!(
        !ledger
            .release_agent_team_task("team", "task")
            .await
            .unwrap()
    );
    let released = ledger.get_agent_team_task("task").await.unwrap().unwrap();
    assert_eq!(released.status, AgentTeamTaskStatus::Todo);
    assert!(released.claimed_by_member_id.is_none());
    assert!(released.claim_token.is_none());
    assert_eq!(released.gate_status, "pending");
    assert!(released.gate_reason.is_none());
    assert_eq!(released.evidence, ["evidence"]);
    assert_eq!(released.owner_member_id.as_deref(), Some("owner"));
    assert_eq!(released.depends_on, ["dependency"]);
    assert_eq!(released.source_run_id.as_deref(), Some("source"));
}

/// Fenced cleanup from an older worker cannot idle or release successor state.
#[tokio::test]
async fn stale_team_execution_cleanup_preserves_successor_claim_and_run() {
    let (ledger, _) = ledgers();
    ledger
        .upsert_agent_team_member(AgentTeamMemberUpsert {
            id: "member".into(),
            team_id: "team".into(),
            name: "name".into(),
            agent_id: None,
            member_status: AgentTeamMemberStatus::Pending,
            current_task_id: None,
            worker_thread_id: None,
            run_id: None,
            created_at: None,
        })
        .await
        .unwrap();
    for (task, thread, run) in [
        ("task", "thread-old", "run-old"),
        ("task", "thread-new", "run-new"),
    ] {
        ledger
            .mark_agent_team_member_running("team", "member", task, thread, run)
            .await
            .unwrap()
            .unwrap();
    }
    assert!(
        ledger
            .mark_agent_team_member_idle_if_current(
                "team",
                "member",
                "task",
                "thread-old",
                "run-old"
            )
            .await
            .unwrap()
            .is_none()
    );
    let member = ledger
        .get_agent_team_member("member")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(member.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(member.worker_thread_id.as_deref(), Some("thread-new"));
    assert_eq!(member.run_id.as_deref(), Some("run-new"));

    ledger
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            id: "task".into(),
            team_id: "team".into(),
            title: "title".into(),
            objective: None,
            status: AgentTeamTaskStatus::InProgress,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        })
        .await
        .unwrap();
    ledger
        .merge(TASKS, "task", |previous: Option<AgentTeamTask>| {
            let mut row = previous.unwrap();
            row.claimed_by_member_id = Some("member".into());
            row.claim_token = Some("successor-token".into());
            row
        })
        .await
        .unwrap();
    assert!(
        !ledger
            .release_agent_team_task_if_claimed("team", "task", "member", "stale-token")
            .await
            .unwrap()
    );
    assert!(matches!(
        ledger
            .complete_agent_team_task_if_claimed(
                "team",
                "task",
                "member",
                "stale-token",
                &["stale evidence".into()],
                false,
            )
            .await
            .unwrap(),
        CompletionOutcome::NotClaimed
    ));
    let task = ledger.get_agent_team_task("task").await.unwrap().unwrap();
    assert_eq!(task.status, AgentTeamTaskStatus::InProgress);
    assert_eq!(task.claim_token.as_deref(), Some("successor-token"));
    assert!(task.evidence.is_empty());
}

/// Nonfinite JSON numeric fields cannot turn a readable rollup into a persisted null.
#[tokio::test]
async fn scoped_telemetry_rejects_lossy_nonfinite_cost_without_writes() {
    let (ledger, _) = ledgers();
    ledger
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            cost_usd: Some(1.0),
            ..Default::default()
        })
        .await
        .unwrap();
    let before = ledger
        .scoped
        .documents()
        .get(TELEMETRY, &storage::key(&["run"]).unwrap())
        .await
        .unwrap();
    for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(ledger.upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(), cost_usd: Some(cost), ..Default::default()
        }).await.unwrap_err(), LedgerError::Driver(ref error) if error.kind() == ErrorKind::Serialization));
        assert_eq!(
            ledger
                .scoped
                .documents()
                .get(TELEMETRY, &storage::key(&["run"]).unwrap())
                .await
                .unwrap(),
            before
        );
    }
}

/// Revision exhaustion fails before writing instead of escaping the classic signed SQLite range.
#[tokio::test]
async fn scoped_workflow_revision_exhaustion_preserves_committed_record() {
    let (ledger, _) = ledgers();
    ledger
        .upsert_workflow_run(workflow("workflow"))
        .await
        .unwrap();
    let before = ledger
        .merge(WORKFLOWS, "workflow", |previous: Option<WorkflowRun>| {
            let mut row = previous.unwrap();
            row.revision = i64::MAX as u64;
            row
        })
        .await
        .unwrap();
    assert!(
        matches!(ledger.upsert_workflow_run(workflow("workflow")).await.unwrap_err(),
        LedgerError::Driver(ref error) if error.kind() == ErrorKind::Serialization)
    );
    assert_eq!(
        ledger.get_workflow_run("workflow").await.unwrap(),
        Some(before)
    );
}
