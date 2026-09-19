use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use chrono::Utc;
use tempfile::TempDir;
use tinyagents_session::run_ledger::{
    AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember, AgentTeamMemberStatus,
    AgentTeamMemberUpsert, AgentTeamStatus, AgentTeamTask, AgentTeamTaskUpsert, AgentTeamUpsert,
    ClaimOutcome, CompletionOutcome, RunEvent, RunEventAppend, RunEventListRequest,
};

use super::*;

fn service(dir: &TempDir) -> TeamService<SessionTeamLedger> {
    TeamService::new(SessionTeamLedger::new(dir.path()))
}

fn solo_team(service: &TeamService<SessionTeamLedger>) -> (String, String) {
    let view = service
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
    (view.team.id, view.members[0].id.clone())
}

fn team_error(error: anyhow::Error) -> TeamError {
    error.downcast::<TeamError>().unwrap()
}

#[test]
fn rejects_duplicate_members_and_unknown_dependencies() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let error = service
        .create_team(
            "lead",
            None,
            None,
            &[
                NewMember {
                    name: "alice".into(),
                    agent_id: None,
                },
                NewMember {
                    name: "alice".into(),
                    agent_id: None,
                },
            ],
        )
        .unwrap_err();
    assert_eq!(
        team_error(error),
        TeamError::DuplicateMemberName {
            name: "alice".into()
        }
    );

    let (team_id, _) = solo_team(&service);
    let error = service
        .assign_task(&team_id, "task", None, None, &["missing".into()])
        .unwrap_err();
    assert_eq!(
        team_error(error),
        TeamError::UnknownDependency {
            depends_on: "missing".into()
        }
    );
}

#[test]
fn task_claim_completion_and_quality_gate_are_durable() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let (team_id, member_id) = solo_team(&service);
    let task = service
        .assign_task(&team_id, "ship", None, None, &[])
        .unwrap();
    assert!(matches!(
        service
            .claim_task(&team_id, &task.id, &member_id, "claim-1")
            .unwrap(),
        ClaimOutcome::Claimed(_)
    ));
    assert!(matches!(
        service
            .complete_task(&team_id, &task.id, &member_id, &[], true)
            .unwrap(),
        CompletionOutcome::GateFailed { .. }
    ));
    let after_failed_gate = service.get_team(&team_id).unwrap().unwrap();
    let durable_task = after_failed_gate
        .tasks
        .iter()
        .find(|candidate| candidate.id == task.id)
        .unwrap();
    assert_eq!(
        durable_task.status,
        tinyagents_session::run_ledger::AgentTeamTaskStatus::InProgress
    );
    assert_eq!(durable_task.gate_status, "failed");
    assert!(matches!(
        service
            .complete_task(&team_id, &task.id, &member_id, &["proof".into()], true)
            .unwrap(),
        CompletionOutcome::Completed(_)
    ));
}

#[test]
fn racing_claims_have_one_winner_and_one_already_claimed_loser() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let view = service
        .create_team(
            "lead",
            None,
            None,
            &[
                NewMember {
                    name: "alice".into(),
                    agent_id: None,
                },
                NewMember {
                    name: "bob".into(),
                    agent_id: None,
                },
            ],
        )
        .unwrap();
    let task = service
        .assign_task(&view.team.id, "race", None, None, &[])
        .unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for (member, token) in [
        (view.members[0].id.clone(), "alice-token"),
        (view.members[1].id.clone(), "bob-token"),
    ] {
        let service = service.clone();
        let team_id = view.team.id.clone();
        let task_id = task.id.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            service
                .claim_task(&team_id, &task_id, &member, token)
                .unwrap()
        }));
    }
    let outcomes: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, ClaimOutcome::Claimed(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, ClaimOutcome::AlreadyClaimed))
            .count(),
        1
    );
}

#[test]
fn completion_rejects_non_claimants_and_owner_mismatches() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let view = service
        .create_team(
            "lead",
            None,
            None,
            &[
                NewMember {
                    name: "alice".into(),
                    agent_id: None,
                },
                NewMember {
                    name: "bob".into(),
                    agent_id: None,
                },
            ],
        )
        .unwrap();
    let alice = &view.members[0].id;
    let bob = &view.members[1].id;

    let claimed_by_alice = service
        .assign_task(&view.team.id, "alice work", None, None, &[])
        .unwrap();
    service
        .claim_task(&view.team.id, &claimed_by_alice.id, alice, "alice-token")
        .unwrap();
    assert!(matches!(
        service
            .complete_task(
                &view.team.id,
                &claimed_by_alice.id,
                bob,
                &["proof".into()],
                false
            )
            .unwrap(),
        CompletionOutcome::NotClaimed
    ));

    let owned_by_alice = service
        .assign_task(&view.team.id, "owned work", None, Some(alice), &[])
        .unwrap();
    service
        .claim_task(&view.team.id, &owned_by_alice.id, bob, "bob-token")
        .unwrap();
    let outcome = service
        .complete_task(
            &view.team.id,
            &owned_by_alice.id,
            bob,
            &["proof".into()],
            false,
        )
        .unwrap();
    assert!(matches!(
        outcome,
        CompletionOutcome::GateFailed { ref reasons }
            if reasons.iter().any(|reason| reason.contains("owned by"))
    ));
}

#[test]
fn messages_remain_ordered_and_member_shutdown_releases_work() {
    let dir = TempDir::new().unwrap();
    let service = service(&dir);
    let view = service
        .create_team(
            "lead",
            None,
            None,
            &[
                NewMember {
                    name: "alice".into(),
                    agent_id: None,
                },
                NewMember {
                    name: "bob".into(),
                    agent_id: None,
                },
            ],
        )
        .unwrap();
    let team_id = view.team.id;
    let alice = view.members[0].id.clone();
    let bob = view.members[1].id.clone();
    service
        .message_member(&team_id, Some(&alice), Some(&bob), "first", None)
        .unwrap();
    service
        .message_member(&team_id, None, Some(&alice), "second", None)
        .unwrap();
    let messages = service.list_messages(&team_id, None).unwrap();
    assert_eq!(
        messages
            .iter()
            .map(|event| event.payload["content"].as_str())
            .collect::<Vec<_>>(),
        vec![Some("first"), Some("second")]
    );
    assert_eq!(messages[1].payload["from"], LEAD_SENDER);

    let task = service
        .assign_task(&team_id, "ship", None, None, &[])
        .unwrap();
    service
        .claim_task(&team_id, &task.id, &alice, "claim-1")
        .unwrap();
    let shutdown = service.shutdown_member(&team_id, &alice).unwrap();
    assert_eq!(shutdown.released_task_ids, vec![task.id]);
    assert_eq!(
        shutdown.member.member_status,
        AgentTeamMemberStatus::Stopped
    );
}

#[test]
fn fake_ledger_exercises_member_validation_without_session_storage() {
    let ledger = FakeLedger::default();
    let service = TeamService::new(ledger.clone());
    ledger.teams.lock().unwrap().push(team("team-1"));
    let error = service
        .claim_task("team-1", "task-1", "unknown", "token")
        .unwrap_err();
    assert_eq!(
        team_error(error),
        TeamError::UnknownMember {
            member_id: "unknown".into()
        }
    );
}

fn team(id: &str) -> AgentTeam {
    AgentTeam {
        id: id.into(),
        parent_thread_id: None,
        lead_agent_id: "lead".into(),
        status: AgentTeamStatus::Active,
        summary: None,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        closed_at: None,
    }
}

#[derive(Clone, Default)]
struct FakeLedger {
    teams: Arc<Mutex<Vec<AgentTeam>>>,
}

impl TeamLedger for FakeLedger {
    fn upsert_team(&self, upsert: AgentTeamUpsert) -> anyhow::Result<AgentTeam> {
        Ok(team(&upsert.id))
    }
    fn get_team(&self, id: &str) -> anyhow::Result<Option<AgentTeam>> {
        Ok(self
            .teams
            .lock()
            .unwrap()
            .iter()
            .find(|team| team.id == id)
            .cloned())
    }
    fn list_teams(&self, _: &AgentTeamListRequest) -> anyhow::Result<AgentTeamListResponse> {
        Ok(AgentTeamListResponse {
            teams: self.teams.lock().unwrap().clone(),
            count: self.teams.lock().unwrap().len(),
        })
    }
    fn upsert_member(&self, _: AgentTeamMemberUpsert) -> anyhow::Result<AgentTeamMember> {
        unreachable!()
    }
    fn list_members(&self, _: &str) -> anyhow::Result<Vec<AgentTeamMember>> {
        Ok(vec![])
    }
    fn list_tasks(&self, _: &str) -> anyhow::Result<Vec<AgentTeamTask>> {
        Ok(vec![])
    }
    fn upsert_task(&self, _: AgentTeamTaskUpsert) -> anyhow::Result<AgentTeamTask> {
        unreachable!()
    }
    fn claim_task(&self, _: &str, _: &str, _: &str, _: &str) -> anyhow::Result<ClaimOutcome> {
        unreachable!()
    }
    fn complete_task(
        &self,
        _: &str,
        _: &str,
        _: &str,
        _: &[String],
        _: bool,
    ) -> anyhow::Result<CompletionOutcome> {
        unreachable!()
    }
    fn shutdown_member(
        &self,
        _: &str,
        _: &str,
    ) -> anyhow::Result<Option<(AgentTeamMember, Vec<String>)>> {
        Ok(None)
    }
    fn append_event(&self, _: RunEventAppend) -> anyhow::Result<RunEvent> {
        unreachable!()
    }
    fn list_events(&self, _: &RunEventListRequest) -> anyhow::Result<Vec<RunEvent>> {
        Ok(vec![])
    }
}
