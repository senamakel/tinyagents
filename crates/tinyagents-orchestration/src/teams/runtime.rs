//! Host-neutral mechanics used when a durable team member starts work.
//!
//! Hosts decide when and how to run a worker. This module only reads the
//! durable event log, advances delivery watermarks, and produces the prompt
//! that a worker should receive at its start boundary.

use anyhow::Result;
use serde_json::json;
use tinyagents_session::run_ledger::{
    AgentTeamTask, RunEvent, RunEventAppend, RunEventListRequest,
};

use super::TeamLedger;

/// Event type used for durable lead and teammate messages.
pub const TEAM_MESSAGE_EVENT: &str = "team_message";
/// Event type used to record that a member consumed messages through a sequence.
pub const MESSAGE_DELIVERED_EVENT: &str = "team_message_delivered";
/// Maximum page accepted by the session run ledger.
pub const EVENT_PAGE_SIZE: u32 = 1_000;

/// The result of reading one member's undelivered team messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveredMessages {
    /// Message bodies in durable sequence order.
    pub messages: Vec<String>,
    /// The highest delivered message sequence, when a watermark was written.
    pub up_to_sequence: Option<u64>,
}

/// Compose the stable worker prompt from the claimed task and delivered messages.
pub fn build_member_prompt(task: &AgentTeamTask, messages: &[String]) -> String {
    let mut prompt = format!("You are a teammate on an agent team. Task: {}", task.title);
    if let Some(objective) = task
        .objective
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        prompt.push_str("\n\nObjective:\n");
        prompt.push_str(objective.trim());
    }
    if !messages.is_empty() {
        prompt.push_str("\n\nMessages from your lead / teammates:\n");
        for message in messages {
            prompt.push_str("- ");
            prompt.push_str(message);
            prompt.push('\n');
        }
    }
    prompt.push_str("\n\nComplete the task and report what you did.");
    prompt
}

/// Drain every durable event in ascending sequence order.
///
/// The session ledger caps each request, so callers must use this rather than
/// assuming that one event-list request represents a complete team history.
pub fn drain_run_events<L: TeamLedger>(ledger: &L, team_id: &str) -> Result<Vec<RunEvent>> {
    let mut events = Vec::new();
    let mut after_sequence = None;
    loop {
        let page = ledger.list_events(&RunEventListRequest {
            run_id: team_id.to_string(),
            after_sequence,
            limit: Some(EVENT_PAGE_SIZE),
        })?;
        let exhausted = page.len() < EVENT_PAGE_SIZE as usize;
        after_sequence = page.last().map(|event| event.sequence);
        events.extend(page);
        if exhausted || after_sequence.is_none() {
            return Ok(events);
        }
    }
}

/// Read messages for a member, then durably advance that member's watermark.
///
/// Direct messages and broadcasts are selected; messages for other members
/// stay untouched. The watermark is append-only, which makes repeated calls
/// idempotent while retaining the full event history for audit and replay.
pub fn deliver_pending_messages<L: TeamLedger>(
    ledger: &L,
    team_id: &str,
    member_id: &str,
) -> Result<DeliveredMessages> {
    let events = drain_run_events(ledger, team_id)?;
    let watermark = events
        .iter()
        .filter(|event| event.event_type == MESSAGE_DELIVERED_EVENT)
        .filter(|event| {
            event
                .payload
                .get("memberId")
                .and_then(|value| value.as_str())
                == Some(member_id)
        })
        .filter_map(|event| {
            event
                .payload
                .get("upToSeq")
                .and_then(|value| value.as_u64())
        })
        .max()
        .unwrap_or_default();

    let mut up_to_sequence = watermark;
    let mut messages = Vec::new();
    for event in &events {
        if event.event_type != TEAM_MESSAGE_EVENT || event.sequence <= watermark {
            continue;
        }
        let recipient = event.payload.get("to").and_then(|value| value.as_str());
        if recipient.is_none() || recipient == Some(member_id) {
            if let Some(content) = event
                .payload
                .get("content")
                .and_then(|value| value.as_str())
            {
                messages.push(content.to_string());
            }
            up_to_sequence = up_to_sequence.max(event.sequence);
        }
    }

    let delivered_up_to = (!messages.is_empty()).then_some(up_to_sequence);
    if let Some(up_to_sequence) = delivered_up_to {
        ledger.append_event(RunEventAppend {
            run_id: team_id.to_string(),
            event_type: MESSAGE_DELIVERED_EVENT.to_string(),
            payload: json!({ "memberId": member_id, "upToSeq": up_to_sequence }),
        })?;
    }
    Ok(DeliveredMessages {
        messages,
        up_to_sequence: delivered_up_to,
    })
}

/// Truncate text at a Unicode character boundary and append an ellipsis.
pub fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut output: String = value.chars().take(max_chars).collect();
    output.push('…');
    output
}

#[cfg(test)]
mod tests;
