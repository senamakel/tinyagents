//! Durable, dependency-aware agent-team composition.

mod graph;
mod service;
mod types;

pub use graph::{MemberOutcome, member_graph_topology, run_member_graph};
pub use service::{SessionTeamLedger, TeamLedger, TeamService, claimable_task};
pub use types::{LEAD_SENDER, MemberShutdown, NewMember, TeamError, TeamView};

#[cfg(test)]
mod tests;
