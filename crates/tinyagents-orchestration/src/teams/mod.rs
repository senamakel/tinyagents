//! Durable, dependency-aware agent-team composition.

mod graph;
mod runtime;
mod service;
mod types;

pub use graph::{MemberOutcome, member_graph_topology, run_member_graph};
pub use runtime::{
    DeliveredMessages, EVENT_PAGE_SIZE, MESSAGE_DELIVERED_EVENT, TEAM_MESSAGE_EVENT,
    build_member_prompt, deliver_pending_messages, drain_run_events, truncate_chars,
};
pub use service::{SessionTeamLedger, TeamLedger, TeamService, claimable_task};
pub use types::{LEAD_SENDER, MemberShutdown, NewMember, TeamError, TeamView};

#[cfg(test)]
mod tests;
