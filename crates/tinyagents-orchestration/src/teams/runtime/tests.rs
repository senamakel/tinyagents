use tempfile::TempDir;

use super::*;
use crate::teams::{NewMember, SessionTeamLedger, TeamService};

fn team(dir: &TempDir) -> (SessionTeamLedger, String, String) {
    let ledger = SessionTeamLedger::new(dir.path());
    let service = TeamService::new(ledger.clone());
    let team = service
        .create_team(
            "lead",
            None,
            None,
            &[NewMember {
                name: "member".into(),
                agent_id: None,
            }],
        )
        .unwrap();
    (ledger, team.team.id, team.members[0].id.clone())
}

#[test]
fn delivery_selects_direct_and_broadcast_messages_once() {
    let dir = TempDir::new().unwrap();
    let (ledger, team_id, member_id) = team(&dir);
    let service = TeamService::new(ledger.clone());
    service
        .message_member(&team_id, None, Some(&member_id), "direct", None)
        .unwrap();
    service
        .message_member(&team_id, None, None, "broadcast", None)
        .unwrap();
    let first = deliver_pending_messages(&ledger, &team_id, &member_id).unwrap();
    assert_eq!(first.messages, ["direct", "broadcast"]);
    assert!(first.up_to_sequence.is_some());
    assert!(
        deliver_pending_messages(&ledger, &team_id, &member_id)
            .unwrap()
            .messages
            .is_empty()
    );
}

#[test]
fn delivery_pages_past_the_session_ledger_cap() {
    let dir = TempDir::new().unwrap();
    let (ledger, team_id, member_id) = team(&dir);
    for index in 0..=EVENT_PAGE_SIZE {
        ledger
            .append_event(tinyagents_session::run_ledger::RunEventAppend {
                run_id: team_id.clone(),
                event_type: "noise".into(),
                payload: serde_json::json!({ "index": index }),
            })
            .unwrap();
    }
    let service = TeamService::new(ledger.clone());
    service
        .message_member(&team_id, None, Some(&member_id), "late", None)
        .unwrap();
    assert_eq!(
        deliver_pending_messages(&ledger, &team_id, &member_id)
            .unwrap()
            .messages,
        ["late"]
    );
}

#[test]
fn prompt_and_truncation_preserve_text_boundaries() {
    let dir = TempDir::new().unwrap();
    let (ledger, team_id, _) = team(&dir);
    let service = TeamService::new(ledger);
    let task = service
        .assign_task(&team_id, "Ship", Some("  Build it  "), None, &[])
        .unwrap();
    let prompt = build_member_prompt(&task, &["coordinate".into()]);
    assert!(prompt.contains("Build it"));
    assert!(prompt.contains("coordinate"));
    assert_eq!(truncate_chars("aébc", 2), "aé…");
}
