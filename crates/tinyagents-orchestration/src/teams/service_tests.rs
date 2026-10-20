use super::*;

use crate::teams::NewMember;
use tempfile::TempDir;
use tinyagents_session::run_ledger::{AgentTeamMemberStatus, AgentTeamTaskStatus};
use tinystoragedrivers_testkit::MemoryBackend;

fn service(_dir: &TempDir) -> TeamService<SessionTeamLedger> {
    let ledger_backend = MemoryBackend::new();
    service_with_backend(&ledger_backend)
}

fn service_with_backend(ledger_backend: &MemoryBackend) -> TeamService<SessionTeamLedger> {
    TeamService::new(
        SessionTeamLedger::new(Arc::new(crate::test_support::ledger_in(ledger_backend))).unwrap(),
    )
}

struct FixedExecutor(Result<Vec<String>, &'static str>);

#[async_trait]
impl TeamMemberExecutor for FixedExecutor {
    async fn execute(&self, _: &AgentTeamMember, _: &AgentTeamTask) -> Result<Vec<String>> {
        self.0.clone().map_err(anyhow::Error::msg)
    }
}

struct RecordingExecutor(Arc<std::sync::Mutex<Option<AgentTeamMember>>>);

#[async_trait]
impl TeamMemberExecutor for RecordingExecutor {
    async fn execute(&self, member: &AgentTeamMember, _: &AgentTeamTask) -> Result<Vec<String>> {
        *self.0.lock().unwrap() = Some(member.clone());
        Ok(vec!["done".into()])
    }
}

struct HangingExecutor(Arc<tokio::sync::Notify>);

#[async_trait]
impl TeamMemberExecutor for HangingExecutor {
    async fn execute(&self, _: &AgentTeamMember, _: &AgentTeamTask) -> Result<Vec<String>> {
        self.0.notify_one();
        std::future::pending().await
    }
}

struct CountingExecutor(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait]
impl TeamMemberExecutor for CountingExecutor {
    async fn execute(&self, _: &AgentTeamMember, _: &AgentTeamTask) -> Result<Vec<String>> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(vec!["unexpected second execution".into()])
    }
}

struct PausedExecutor {
    entered: Arc<tokio::sync::Notify>,
    resume: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl TeamMemberExecutor for PausedExecutor {
    async fn execute(&self, _: &AgentTeamMember, _: &AgentTeamTask) -> Result<Vec<String>> {
        self.entered.notify_one();
        self.resume.notified().await;
        Ok(vec!["stale completion".into()])
    }
}

#[tokio::test]
async fn start_member_runs_and_completes_claimed_task() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();

    let executed_member = Arc::new(std::sync::Mutex::new(None));
    let outcome = service
        .start_member(
            &team.team.id,
            &task.id,
            &member.id,
            &RecordingExecutor(executed_member.clone()),
            true,
        )
        .await
        .unwrap();

    assert!(matches!(outcome, CompletionOutcome::Completed(_)));
    let executed_member = executed_member.lock().unwrap().clone().unwrap();
    assert_eq!(executed_member.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(
        executed_member.current_task_id.as_deref(),
        Some(task.id.as_str())
    );
    assert!(executed_member.worker_thread_id.is_some());
    assert!(executed_member.run_id.is_some());
    assert_eq!(
        service.ledger().list_tasks(&team.team.id).unwrap()[0].status,
        AgentTeamTaskStatus::Done
    );
    assert_eq!(
        service.ledger().list_members(&team.team.id).unwrap()[0].member_status,
        AgentTeamMemberStatus::Idle
    );
}

#[tokio::test]
async fn a_member_cannot_start_a_second_task_while_its_first_task_is_running() {
    // Independent handles prove the admission fence is in the shared backend,
    // rather than a mutex local to one TeamService instance.
    let ledger_backend = MemoryBackend::new();
    let first_service = service_with_backend(&ledger_backend);
    let second_service = service_with_backend(&ledger_backend);
    let team = first_service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let first_task = first_service
        .assign_task(&team.team.id, "first", None, Some(&member.id), &[])
        .unwrap();
    let second_task = first_service
        .assign_task(&team.team.id, "second", None, Some(&member.id), &[])
        .unwrap();

    let first_entered = Arc::new(tokio::sync::Notify::new());
    let first_entered_for_worker = Arc::clone(&first_entered);
    let first_worker_service = first_service.clone();
    let first_team_id = team.team.id.clone();
    let first_task_id = first_task.id.clone();
    let member_id = member.id.clone();
    let first_worker = tokio::spawn(async move {
        first_worker_service
            .start_member(
                &first_team_id,
                &first_task_id,
                &member_id,
                &HangingExecutor(first_entered_for_worker),
                false,
            )
            .await
    });
    first_entered.notified().await;

    let executions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut loser_callback_called = false;
    let second = second_service
        .start_member_with_started(
            &team.team.id,
            &second_task.id,
            &member.id,
            &CountingExecutor(Arc::clone(&executions)),
            false,
            |_| loser_callback_called = true,
        )
        .await;

    let error = second.expect_err("an active member must reject a second task");
    assert!(matches!(
        error.downcast_ref::<TeamMemberStartError>(),
        Some(TeamMemberStartError::AlreadyActive { member_id, current_task_id })
            if member_id == &member.id && current_task_id.as_deref() == Some(first_task.id.as_str())
    ));
    assert!(
        !loser_callback_called,
        "denied admission must not report Started"
    );
    assert_eq!(executions.load(std::sync::atomic::Ordering::SeqCst), 0);
    let after = first_service.get_team(&team.team.id).unwrap().unwrap();
    assert_eq!(
        after.members[0].member_status,
        AgentTeamMemberStatus::Active
    );
    assert_eq!(
        after.members[0].current_task_id.as_deref(),
        Some(first_task.id.as_str())
    );
    assert!(
        after.members[0]
            .worker_thread_id
            .as_deref()
            .is_some_and(|id| id.starts_with("team-worker-"))
    );
    assert!(
        after.members[0]
            .run_id
            .as_deref()
            .is_some_and(|id| id.starts_with("team-run-"))
    );
    let first_after = after
        .tasks
        .iter()
        .find(|task| task.id == first_task.id)
        .unwrap();
    let second_after = after
        .tasks
        .iter()
        .find(|task| task.id == second_task.id)
        .unwrap();
    assert_eq!(first_after.status, AgentTeamTaskStatus::InProgress);
    assert!(first_after.claim_token.is_some());
    assert_eq!(second_after.status, AgentTeamTaskStatus::Todo);
    assert!(second_after.claim_token.is_none());

    first_worker.abort();
    assert!(first_worker.await.unwrap_err().is_cancelled());
}

#[tokio::test]
async fn started_callback_fires_after_claim_before_executor_finishes() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let entered = Arc::new(tokio::sync::Notify::new());
    let executor = HangingExecutor(entered.clone());
    let worker_service = service.clone();
    let team_id = team.team.id.clone();
    let task_id = task.id.clone();
    let member_id = member.id.clone();
    let worker = tokio::spawn(async move {
        worker_service
            .start_member_with_started(
                &team_id,
                &task_id,
                &member_id,
                &executor,
                false,
                |started| {
                    let _ = started_tx.send(started);
                },
            )
            .await
    });

    let started = started_rx.await.unwrap();
    assert!(!started.run_id.is_empty());
    assert_eq!(started.task.id, task.id);
    assert_eq!(started.member.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(
        started.member.run_id.as_deref(),
        Some(started.run_id.as_str())
    );
    assert!(
        started
            .member
            .worker_thread_id
            .as_deref()
            .is_some_and(|id| id.starts_with("team-worker-"))
    );
    let durable = service.get_team(&team.team.id).unwrap().unwrap();
    assert_eq!(
        durable.members[0].run_id.as_deref(),
        Some(started.run_id.as_str())
    );
    assert_eq!(durable.tasks[0].status, AgentTeamTaskStatus::InProgress);
    entered.notified().await;
    assert!(!worker.is_finished());
    worker.abort();
    let _ = worker.await;
    let released = service.get_team(&team.team.id).unwrap().unwrap();
    assert_eq!(
        released.members[0].member_status,
        AgentTeamMemberStatus::Idle
    );
    assert_eq!(released.tasks[0].status, AgentTeamTaskStatus::Todo);
}

#[tokio::test]
async fn start_member_exposes_claim_conflicts_as_typed_errors() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    service
        .claim_task(&team.team.id, &task.id, &member.id, "existing-claim")
        .unwrap();

    let error = service
        .start_member_with_started(
            &team.team.id,
            &task.id,
            &member.id,
            &FixedExecutor(Ok(vec![])),
            false,
            |_| panic!("claim conflict must not notify the host as started"),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<TeamMemberStartError>(),
        Some(TeamMemberStartError::AlreadyClaimed { task_id }) if task_id == &task.id
    ));
}

#[tokio::test]
async fn started_callback_unwind_releases_its_claim_and_idles_the_member() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    let worker_service = service.clone();
    let team_id = team.team.id.clone();
    let task_id = task.id.clone();
    let member_id = member.id.clone();
    let worker = tokio::spawn(async move {
        worker_service
            .start_member_with_started(
                &team_id,
                &task_id,
                &member_id,
                &FixedExecutor(Ok(vec![])),
                false,
                |_| panic!("observer panic"),
            )
            .await
    });
    assert!(worker.await.unwrap_err().is_panic());

    let after = service.get_team(&team.team.id).unwrap().unwrap();
    assert_eq!(after.members[0].member_status, AgentTeamMemberStatus::Idle);
    assert_eq!(after.tasks[0].status, AgentTeamTaskStatus::Todo);
}

#[tokio::test]
async fn start_member_releases_task_and_idles_member_on_executor_failure() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();

    let error = service
        .start_member(
            &team.team.id,
            &task.id,
            &member.id,
            &FixedExecutor(Err("worker stopped")),
            true,
        )
        .await
        .unwrap_err();

    assert!(format!("{error:#}").contains("worker stopped"));
    assert_eq!(
        service.ledger().list_tasks(&team.team.id).unwrap()[0].status,
        AgentTeamTaskStatus::Todo
    );
    assert_eq!(
        service.ledger().list_members(&team.team.id).unwrap()[0].member_status,
        AgentTeamMemberStatus::Idle
    );
}

#[tokio::test]
async fn cancelling_start_member_releases_task_and_idles_member() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    let entered = Arc::new(tokio::sync::Notify::new());
    let executor = Arc::new(HangingExecutor(entered.clone()));
    let worker_service = service.clone();
    let team_id = team.team.id.clone();
    let task_id = task.id.clone();
    let member_id = member.id.clone();
    let worker = tokio::spawn(async move {
        worker_service
            .start_member(&team_id, &task_id, &member_id, executor.as_ref(), true)
            .await
    });
    entered.notified().await;
    worker.abort();
    let _ = worker.await;

    assert_eq!(
        service.ledger().list_tasks(&team.team.id).unwrap()[0].status,
        AgentTeamTaskStatus::Todo
    );
    assert_eq!(
        service.ledger().list_members(&team.team.id).unwrap()[0].member_status,
        AgentTeamMemberStatus::Idle
    );
}

#[tokio::test]
async fn stale_executor_cannot_complete_successor_claim_for_same_member() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    let entered = Arc::new(tokio::sync::Notify::new());
    let resume = Arc::new(tokio::sync::Notify::new());
    let executor = Arc::new(PausedExecutor {
        entered: entered.clone(),
        resume: resume.clone(),
    });
    let worker_service = service.clone();
    let (team_id, task_id, member_id) = (team.team.id.clone(), task.id.clone(), member.id.clone());
    let worker = tokio::spawn(async move {
        worker_service
            .start_member(&team_id, &task_id, &member_id, executor.as_ref(), false)
            .await
    });
    entered.notified().await;

    service.shutdown_member(&team.team.id, &member.id).unwrap();
    let ClaimOutcome::Claimed(_) = service
        .ledger()
        .claim_task(&team.team.id, &task.id, &member.id, "successor-token")
        .unwrap()
    else {
        panic!("successor should reclaim the released task");
    };
    service
        .ledger()
        .mark_member_running(
            &team.team.id,
            &member.id,
            &task.id,
            "successor-thread",
            "successor-run",
        )
        .unwrap()
        .unwrap();

    resume.notify_one();
    assert!(matches!(
        worker.await.unwrap().unwrap(),
        CompletionOutcome::NotClaimed
    ));
    let current_task = service
        .ledger()
        .list_tasks(&team.team.id)
        .unwrap()
        .remove(0);
    assert_eq!(current_task.status, AgentTeamTaskStatus::InProgress);
    assert_eq!(current_task.claim_token.as_deref(), Some("successor-token"));
    assert!(current_task.evidence.is_empty());
    let current_member = service
        .ledger()
        .list_members(&team.team.id)
        .unwrap()
        .remove(0);
    assert_eq!(current_member.member_status, AgentTeamMemberStatus::Active);
    assert_eq!(current_member.run_id.as_deref(), Some("successor-run"));
}

#[test]
fn stale_claim_cleanup_cannot_release_a_successor_claim() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "alice".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    let member = &team.members[0];
    let task = service
        .assign_task(&team.team.id, "write", None, Some(&member.id), &[])
        .unwrap();
    let ClaimOutcome::Claimed(first_claim) = service
        .claim_task(&team.team.id, &task.id, &member.id, "first-generation")
        .unwrap()
    else {
        panic!("first generation should claim the task");
    };
    assert!(
        service
            .ledger()
            .release_task(&team.team.id, &task.id, &member.id, "first-generation")
            .unwrap()
    );
    let ClaimOutcome::Claimed(second_claim) = service
        .claim_task(&team.team.id, &task.id, &member.id, "successor-generation")
        .unwrap()
    else {
        panic!("successor should claim the released task");
    };

    assert!(
        !service
            .ledger()
            .release_task(&team.team.id, &task.id, &member.id, "first-generation")
            .unwrap()
    );
    let current = service
        .ledger()
        .list_tasks(&team.team.id)
        .unwrap()
        .remove(0);
    assert_eq!(current.claim_token.as_deref(), Some("successor-generation"));
    assert_eq!(current.claim_token, second_claim.claim_token);
    assert_ne!(first_claim.claim_token, second_claim.claim_token);
}
