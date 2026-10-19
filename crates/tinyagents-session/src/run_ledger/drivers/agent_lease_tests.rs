use super::agent_tests::run;
use super::*;
use tinystoragedrivers_core::{MemoryStorage, StorageBackend};

#[tokio::test]
async fn agent_lease_takeover_fences_old_heartbeat_finish_and_expiry_reap() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("shared").unwrap();
    let first = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let second = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let now = chrono::DateTime::from_timestamp(1_000, 0).unwrap();
    let initial = match first
        .try_claim_agent_run(run("leased"), "owner-a", now, chrono::Duration::seconds(5))
        .await
        .unwrap()
    {
        AgentRunLeaseClaim::Acquired { token, .. } => token,
        other => panic!("expected first lease, got {other:?}"),
    };
    assert!(matches!(
        second
            .try_claim_agent_run(
                run("leased"),
                "owner-b",
                now + chrono::Duration::seconds(1),
                chrono::Duration::seconds(5),
            )
            .await
            .unwrap(),
        AgentRunLeaseClaim::Busy(_)
    ));
    let takeover = match second
        .try_claim_agent_run(
            run("leased"),
            "owner-b",
            now + chrono::Duration::seconds(6),
            chrono::Duration::seconds(5),
        )
        .await
        .unwrap()
    {
        AgentRunLeaseClaim::Acquired { token, run } => {
            assert_eq!(run.lease_epoch, initial.epoch + 1);
            token
        }
        other => panic!("expected expired lease takeover, got {other:?}"),
    };
    assert!(
        !first
            .renew_agent_run_lease(
                &initial,
                now + chrono::Duration::seconds(7),
                chrono::Duration::seconds(5),
            )
            .await
            .unwrap()
    );
    assert_eq!(
        second
            .interrupt_expired_agents(now + chrono::Duration::seconds(7))
            .await
            .unwrap(),
        0
    );
    assert!(
        first
            .finish_agent_run(
                AgentRunUpsert {
                    status: AgentRunStatus::Completed,
                    completed_at: Some(now + chrono::Duration::seconds(7)),
                    ..run("leased")
                },
                &initial,
                now + chrono::Duration::seconds(7),
            )
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        second
            .interrupt_expired_agents(now + chrono::Duration::seconds(12))
            .await
            .unwrap(),
        1
    );
    assert!(
        second
            .finish_agent_run(
                AgentRunUpsert {
                    status: AgentRunStatus::Completed,
                    completed_at: Some(now + chrono::Duration::seconds(13)),
                    ..run("leased")
                },
                &takeover,
                now + chrono::Duration::seconds(13),
            )
            .await
            .unwrap()
            .is_none()
    );
    assert!(second.upsert_agent_run(run("leased")).await.is_err());
}

#[tokio::test]
async fn ordinary_agent_upsert_cannot_overwrite_a_finished_leased_run() {
    let backend = MemoryStorage::new();
    let ledger = DriverRunLedger::new(backend.for_scope(&Scope::new("shared").unwrap()).unwrap());
    let now = chrono::DateTime::from_timestamp(1_000, 0).unwrap();
    let token = match ledger
        .try_claim_agent_run(run("finished"), "owner", now, chrono::Duration::seconds(10))
        .await
        .unwrap()
    {
        AgentRunLeaseClaim::Acquired { token, .. } => token,
        other => panic!("expected lease, got {other:?}"),
    };
    assert!(
        ledger
            .finish_agent_run(
                AgentRunUpsert {
                    status: AgentRunStatus::Completed,
                    completed_at: Some(now + chrono::Duration::seconds(1)),
                    ..run("finished")
                },
                &token,
                now + chrono::Duration::seconds(1),
            )
            .await
            .unwrap()
            .is_some()
    );
    assert!(ledger.upsert_agent_run(run("finished")).await.is_err());
}

#[tokio::test]
async fn unfenced_status_transitions_cannot_revive_reaped_or_finished_leases() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("shared").unwrap();
    let ledger = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let now = chrono::DateTime::from_timestamp(1_000, 0).unwrap();

    assert!(matches!(
        ledger
            .try_claim_agent_run(run("reaped"), "owner", now, chrono::Duration::seconds(1))
            .await
            .unwrap(),
        AgentRunLeaseClaim::Acquired { .. }
    ));
    assert_eq!(
        ledger
            .interrupt_expired_agents(now + chrono::Duration::seconds(2))
            .await
            .unwrap(),
        1
    );
    assert!(
        ledger
            .transition_agent_run_status("reaped", AgentRunStatus::Running, None, None)
            .await
            .is_err()
    );

    let token = match ledger
        .try_claim_agent_run(run("finished"), "owner", now, chrono::Duration::seconds(10))
        .await
        .unwrap()
    {
        AgentRunLeaseClaim::Acquired { token, .. } => token,
        other => panic!("expected lease, got {other:?}"),
    };
    assert!(
        ledger
            .finish_agent_run(
                AgentRunUpsert {
                    status: AgentRunStatus::Completed,
                    completed_at: Some(now + chrono::Duration::seconds(1)),
                    ..run("finished")
                },
                &token,
                now + chrono::Duration::seconds(1),
            )
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        ledger
            .transition_agent_run_status("finished", AgentRunStatus::Running, None, None)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn shared_agent_reaper_preserves_legacy_unleased_runs() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("shared").unwrap();
    let owner = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let reaper = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    owner
        .upsert_agent_run(run("legacy-unleased"))
        .await
        .unwrap();

    assert_eq!(
        reaper
            .interrupt_expired_agents(chrono::DateTime::from_timestamp(4_000_000_000, 0).unwrap())
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        owner
            .get_agent_run("legacy-unleased")
            .await
            .unwrap()
            .unwrap()
            .status,
        AgentRunStatus::Running
    );
}
