use super::*;
use crate::testkit::MemoryStorage;
use tinystoragedrivers_api::StorageBackend;

fn ledger_on(backend: &dyn StorageBackend, scope: &str) -> DriverRunLedger {
    let scope = Scope::new(scope).unwrap();
    DriverRunLedger::new(ScopedStorage::new(
        backend.for_scope(&scope).unwrap(),
        scope,
    ))
}

async fn member(ledger: &DriverRunLedger, id: &str) {
    ledger
        .upsert_agent_team_member(AgentTeamMemberUpsert {
            id: id.into(),
            team_id: "team".into(),
            name: id.into(),
            agent_id: None,
            member_status: AgentTeamMemberStatus::Idle,
            current_task_id: Some(format!("{id}-task")),
            worker_thread_id: None,
            run_id: None,
            created_at: None,
        })
        .await
        .unwrap();
}

async fn task(ledger: &DriverRunLedger, id: &str, deps: &[&str], owner: Option<&str>) {
    ledger
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            id: id.into(),
            team_id: "team".into(),
            title: id.into(),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: owner.map(str::to_owned),
            depends_on: deps.iter().map(|dep| (*dep).to_owned()).collect(),
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        })
        .await
        .unwrap();
}

async fn claim_and_complete_lifecycle(backend: &dyn StorageBackend) {
    let ledger = ledger_on(backend, "team-a");
    member(&ledger, "alice").await;
    member(&ledger, "bob").await;
    task(&ledger, "base", &[], None).await;
    task(&ledger, "child", &["base"], None).await;

    // Unknown task and a task of another team are both `UnknownTask`.
    assert_eq!(
        ledger
            .claim_agent_team_task("team", "missing", "alice", "t0")
            .await
            .unwrap(),
        ClaimOutcome::UnknownTask
    );
    assert_eq!(
        ledger
            .claim_agent_team_task("other-team", "base", "alice", "t0")
            .await
            .unwrap(),
        ClaimOutcome::UnknownTask
    );

    // A dependency that is not done blocks the claim and names the dependency.
    assert_eq!(
        ledger
            .claim_agent_team_task("team", "child", "alice", "t1")
            .await
            .unwrap(),
        ClaimOutcome::Blocked {
            unmet: vec!["base".into()]
        }
    );

    let ClaimOutcome::Claimed(claimed) = ledger
        .claim_agent_team_task("team", "base", "alice", "t2")
        .await
        .unwrap()
    else {
        panic!("claim should succeed");
    };
    assert_eq!(claimed.status, AgentTeamTaskStatus::InProgress);
    assert_eq!(claimed.claimed_by_member_id.as_deref(), Some("alice"));
    assert_eq!(claimed.claim_token.as_deref(), Some("t2"));
    assert_eq!(
        ledger
            .claim_agent_team_task("team", "base", "bob", "t3")
            .await
            .unwrap(),
        ClaimOutcome::AlreadyClaimed
    );

    // Only the claimant may complete; a stranger is refused without a write.
    assert_eq!(
        ledger
            .complete_agent_team_task("team", "base", "bob", &[], false)
            .await
            .unwrap(),
        CompletionOutcome::NotClaimed
    );
    // Evidence is required, and a failed gate still keeps what was submitted.
    let CompletionOutcome::GateFailed { reasons } = ledger
        .complete_agent_team_task("team", "base", "alice", &[], true)
        .await
        .unwrap()
    else {
        panic!("gate should fail without evidence");
    };
    assert_eq!(reasons, ["completion requires at least one evidence link"]);
    let failed = ledger.get_agent_team_task("base").await.unwrap().unwrap();
    assert_eq!(failed.gate_status, "failed");
    assert_eq!(failed.status, AgentTeamTaskStatus::InProgress);

    let CompletionOutcome::Completed(done) = ledger
        .complete_agent_team_task(
            "team",
            "base",
            "alice",
            &["pr#1".into(), "pr#1".into()],
            true,
        )
        .await
        .unwrap()
    else {
        panic!("completion should pass");
    };
    assert_eq!(done.status, AgentTeamTaskStatus::Done);
    assert_eq!(done.gate_status, "passed");
    assert_eq!(done.gate_reason, None);
    assert_eq!(done.evidence, ["pr#1"]);
    // A second completion by the same member cannot flip a done task again.
    assert_eq!(
        ledger
            .complete_agent_team_task("team", "base", "alice", &[], false)
            .await
            .unwrap(),
        CompletionOutcome::NotClaimed
    );
    // A terminal task is not claimable, whatever its claim column says.
    assert_eq!(
        ledger
            .claim_agent_team_task("team", "base", "bob", "t4")
            .await
            .unwrap(),
        ClaimOutcome::AlreadyClaimed
    );
    // With its dependency done the child is now claimable.
    assert!(matches!(
        ledger
            .claim_agent_team_task("team", "child", "bob", "t5")
            .await
            .unwrap(),
        ClaimOutcome::Claimed(_)
    ));
}

#[tokio::test]
async fn memory_team_claim_and_completion_follow_the_classic_lifecycle() {
    claim_and_complete_lifecycle(&MemoryStorage::new()).await;
}

#[tokio::test]
async fn reconstructed_memory_team_claim_and_completion_follow_the_classic_lifecycle() {
    let backend = MemoryStorage::new();
    claim_and_complete_lifecycle(&backend).await;
}

#[tokio::test]
async fn completion_gate_reports_regressed_dependencies_and_foreign_owners() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "gate");
    member(&ledger, "alice").await;
    task(&ledger, "base", &[], None).await;
    task(&ledger, "owned", &["base"], Some("bob")).await;
    // Claim `base`, finish it, claim `owned`, then regress `base` by hand.
    ledger
        .claim_agent_team_task("team", "base", "alice", "t")
        .await
        .unwrap();
    ledger
        .complete_agent_team_task("team", "base", "alice", &[], false)
        .await
        .unwrap();
    ledger
        .claim_agent_team_task("team", "owned", "alice", "t")
        .await
        .unwrap();
    ledger
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            id: "base".into(),
            team_id: "team".into(),
            title: "base".into(),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
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
    let CompletionOutcome::GateFailed { reasons } = ledger
        .complete_agent_team_task("team", "owned", "alice", &[], false)
        .await
        .unwrap()
    else {
        panic!("gate should fail");
    };
    assert_eq!(
        reasons,
        [
            "dependency base is not done",
            "task is owned by bob but claimed by alice"
        ]
    );
    let stored = ledger.get_agent_team_task("owned").await.unwrap().unwrap();
    assert_eq!(
        stored.gate_reason.as_deref(),
        Some("dependency base is not done; task is owned by bob but claimed by alice")
    );
}

#[tokio::test]
async fn concurrent_claims_grant_exactly_one_member() {
    let backend = MemoryStorage::new();
    let ledger = Arc::new(ledger_on(&backend, "race"));
    task(&ledger, "contested", &[], None).await;
    let mut set = tokio::task::JoinSet::new();
    for n in 0..8 {
        let ledger = ledger.clone();
        set.spawn(async move {
            ledger
                .claim_agent_team_task("team", "contested", &format!("m{n}"), &format!("t{n}"))
                .await
                .unwrap()
        });
    }
    let mut claimed = 0;
    while let Some(outcome) = set.join_next().await {
        match outcome.unwrap() {
            ClaimOutcome::Claimed(_) => claimed += 1,
            ClaimOutcome::AlreadyClaimed => {}
            other => panic!("unexpected outcome {other:?}"),
        }
    }
    assert_eq!(claimed, 1);
}

#[tokio::test]
async fn shutdown_releases_held_tasks_and_stops_the_member() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "shutdown");
    member(&ledger, "alice").await;
    member(&ledger, "bob").await;
    for id in ["one", "two", "three"] {
        task(&ledger, id, &[], None).await;
    }
    for (id, who) in [("one", "alice"), ("two", "alice"), ("three", "bob")] {
        ledger
            .claim_agent_team_task("team", id, who, "tok")
            .await
            .unwrap();
    }
    assert!(
        ledger
            .shutdown_agent_team_member("team", "ghost")
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        ledger
            .shutdown_agent_team_member("other-team", "alice")
            .await
            .unwrap()
            .is_none()
    );
    let (stopped, mut released) = ledger
        .shutdown_agent_team_member("team", "alice")
        .await
        .unwrap()
        .unwrap();
    released.sort();
    assert_eq!(released, ["one", "two"]);
    assert_eq!(stopped.member_status, AgentTeamMemberStatus::Stopped);
    assert_eq!(stopped.current_task_id, None);
    for id in ["one", "two"] {
        let reopened = ledger.get_agent_team_task(id).await.unwrap().unwrap();
        assert_eq!(reopened.status, AgentTeamTaskStatus::Todo);
        assert_eq!(reopened.claimed_by_member_id, None);
        assert_eq!(reopened.claim_token, None);
    }
    // Bob's claim is untouched and a released task can be taken again.
    let held = ledger.get_agent_team_task("three").await.unwrap().unwrap();
    assert_eq!(held.claimed_by_member_id.as_deref(), Some("bob"));
    assert!(matches!(
        ledger
            .claim_agent_team_task("team", "one", "bob", "again")
            .await
            .unwrap(),
        ClaimOutcome::Claimed(_)
    ));
}

#[tokio::test]
async fn team_operations_never_cross_scopes() {
    let backend = MemoryStorage::new();
    let a = ledger_on(&backend, "scope-a");
    let b = ledger_on(&backend, "scope-b");
    task(&a, "task", &[], None).await;
    assert_eq!(
        b.claim_agent_team_task("team", "task", "m", "t")
            .await
            .unwrap(),
        ClaimOutcome::UnknownTask
    );
    assert!(matches!(
        a.claim_agent_team_task("team", "task", "m", "t")
            .await
            .unwrap(),
        ClaimOutcome::Claimed(_)
    ));
}
