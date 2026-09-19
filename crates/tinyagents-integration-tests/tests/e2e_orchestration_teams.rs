//! Public end-to-end coverage for durable, dependency-aware agent teams.
//!
//! The scenario intentionally uses a fresh session-backed ledger after the
//! first worker has progressed: an embedding host can therefore rely on the
//! durable state and event stream across a restart without a model provider.

use tinyagents_orchestration::teams::{
    MESSAGE_DELIVERED_EVENT, NewMember, SessionTeamLedger, TEAM_MESSAGE_EVENT, TeamService,
    deliver_pending_messages, drain_run_events,
};
use tinyagents_session::run_ledger::{
    AgentTeamMemberStatus, AgentTeamStatus, AgentTeamTaskStatus, ClaimOutcome, CompletionOutcome,
};

#[test]
fn durable_team_coordinates_dependencies_messages_shutdown_and_restart() {
    let workspace = tempfile::tempdir().expect("temporary session workspace");
    let ledger = SessionTeamLedger::new(workspace.path());
    let service = TeamService::new(ledger.clone());

    let created = service
        .create_team(
            "release-lead",
            Some("thread-release"),
            Some("prepare and publish release notes"),
            &[
                NewMember {
                    name: "writer".into(),
                    agent_id: Some("writer-agent".into()),
                },
                NewMember {
                    name: "reviewer".into(),
                    agent_id: Some("reviewer-agent".into()),
                },
            ],
        )
        .expect("create team");
    let team_id = created.team.id;
    let writer_id = created.members[0].id.clone();
    let reviewer_id = created.members[1].id.clone();
    assert_eq!(created.team.status, AgentTeamStatus::Active);
    assert_eq!(created.members.len(), 2);

    let draft = service
        .assign_task(
            &team_id,
            "draft release notes",
            Some("summarize customer-facing changes"),
            Some(&writer_id),
            &[],
        )
        .expect("assign initial task");
    let publish = service
        .assign_task(
            &team_id,
            "publish release notes",
            Some("review and publish the approved draft"),
            None,
            std::slice::from_ref(&draft.id),
        )
        .expect("assign dependent task");

    assert!(matches!(
        service
            .claim_task(&team_id, &publish.id, &reviewer_id, "review-claim")
            .expect("attempt dependent claim"),
        ClaimOutcome::Blocked { ref unmet } if unmet == &vec![draft.id.clone()]
    ));

    service
        .message_member(
            &team_id,
            Some(&writer_id),
            Some(&reviewer_id),
            "The draft includes the migration note.",
            Some("direct"),
        )
        .expect("send direct message");
    service
        .message_member(
            &team_id,
            None,
            None,
            "Use the approved release template.",
            Some("team"),
        )
        .expect("broadcast message");

    let writer_delivery =
        deliver_pending_messages(&ledger, &team_id, &writer_id).expect("deliver writer messages");
    assert_eq!(
        writer_delivery.messages,
        ["Use the approved release template."]
    );
    assert!(writer_delivery.up_to_sequence.is_some());
    let reviewer_delivery = deliver_pending_messages(&ledger, &team_id, &reviewer_id)
        .expect("deliver reviewer messages");
    assert_eq!(
        reviewer_delivery.messages,
        [
            "The draft includes the migration note.",
            "Use the approved release template."
        ]
    );
    assert!(reviewer_delivery.up_to_sequence.is_some());

    assert!(matches!(
        service
            .claim_task(&team_id, &draft.id, &writer_id, "draft-claim")
            .expect("claim initial task"),
        ClaimOutcome::Claimed(_)
    ));
    assert!(matches!(
        service
            .complete_task(
                &team_id,
                &draft.id,
                &writer_id,
                &["artifact://release-notes-draft".into()],
                true,
            )
            .expect("complete initial task"),
        CompletionOutcome::Completed(task)
            if task.status == AgentTeamTaskStatus::Done
                && task.gate_status == "passed"
                && task.evidence == ["artifact://release-notes-draft"]
    ));

    assert!(matches!(
        service
            .claim_task(&team_id, &publish.id, &reviewer_id, "publish-claim")
            .expect("claim unblocked task"),
        ClaimOutcome::Claimed(_)
    ));
    let shutdown = service
        .shutdown_member(&team_id, &reviewer_id)
        .expect("shut down active reviewer");
    assert_eq!(
        shutdown.member.member_status,
        AgentTeamMemberStatus::Stopped
    );
    assert_eq!(
        shutdown.released_task_ids.as_slice(),
        std::slice::from_ref(&publish.id)
    );

    let after_shutdown = service
        .get_team(&team_id)
        .expect("load team after shutdown")
        .expect("team still exists");
    let released = after_shutdown
        .tasks
        .iter()
        .find(|task| task.id == publish.id)
        .expect("released task");
    assert_eq!(released.status, AgentTeamTaskStatus::Todo);
    assert_eq!(released.claimed_by_member_id, None);

    // A new service/ledger pair models a host restart. It must see the same
    // durable task release, member lifecycle, and append-only message history.
    let reopened_ledger = SessionTeamLedger::new(workspace.path());
    let reopened = TeamService::new(reopened_ledger.clone());
    assert!(matches!(
        reopened
            .claim_task(&team_id, &publish.id, &writer_id, "recovery-claim")
            .expect("claim released task after restart"),
        ClaimOutcome::Claimed(_)
    ));
    assert!(matches!(
        reopened
            .complete_task(
                &team_id,
                &publish.id,
                &writer_id,
                &["https://example.invalid/releases/2.1.2".into()],
                true,
            )
            .expect("complete recovered task"),
        CompletionOutcome::Completed(task) if task.status == AgentTeamTaskStatus::Done
    ));
    let closed = reopened
        .close_team(&team_id, Some("release notes published"))
        .expect("close completed team");
    assert_eq!(closed.status, AgentTeamStatus::Closed);
    assert_eq!(closed.summary.as_deref(), Some("release notes published"));

    let reloaded = reopened
        .get_team(&team_id)
        .expect("reload team")
        .expect("team persists after restart");
    assert_eq!(reloaded.team.status, AgentTeamStatus::Closed);
    assert_eq!(
        reloaded.members[1].member_status,
        AgentTeamMemberStatus::Stopped
    );
    assert!(
        reloaded
            .tasks
            .iter()
            .all(|task| task.status == AgentTeamTaskStatus::Done)
    );

    let events = drain_run_events(&reopened_ledger, &team_id).expect("drain durable events");
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence)
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == TEAM_MESSAGE_EVENT)
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == MESSAGE_DELIVERED_EVENT)
            .count(),
        2
    );
}
