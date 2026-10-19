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
