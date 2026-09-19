use async_trait::async_trait;
use serde::Serialize;
use tinyagents_session::run_ledger::{AgentTeam, AgentTeamMember, AgentTeamTask};

use crate::OrchestrationError;

/// Sentinel sender for a lead or user message rather than a member row.
pub const LEAD_SENDER: &str = "lead";

/// One member supplied when a team is created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewMember {
    pub name: String,
    pub agent_id: Option<String>,
}

/// A durable team and the member/task rows needed to render it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamView {
    pub team: AgentTeam,
    pub members: Vec<AgentTeamMember>,
    pub tasks: Vec<AgentTeamTask>,
}

/// Result of stopping a member and releasing its active claims.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberShutdown {
    pub member: AgentTeamMember,
    pub released_task_ids: Vec<String>,
}

/// Coordination validation errors that are independent of a host or storage backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "detail")]
pub enum TeamError {
    DuplicateMemberName { name: String },
    UnknownMember { member_id: String },
    SelfDependency { task_id: String },
    CyclicDependency,
    UnknownDependency { depends_on: String },
}

impl std::fmt::Display for TeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateMemberName { name } => write!(f, "duplicate member name: {name}"),
            Self::UnknownMember { member_id } => write!(f, "unknown member: {member_id}"),
            Self::SelfDependency { task_id } => write!(f, "task {task_id} cannot depend on itself"),
            Self::CyclicDependency => write!(f, "dependency cycle detected"),
            Self::UnknownDependency { depends_on } => {
                write!(f, "unknown dependency: {depends_on}")
            }
        }
    }
}

impl std::error::Error for TeamError {}

/// A host-authorized unit of live team work. The orchestration crate never
/// chooses the model, tools, workspace, or policy used by this request.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamWorkRequest {
    pub team_id: String,
    pub member_id: String,
    pub task: AgentTeamTask,
}

/// Host-produced terminal worker result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeamWorkResult {
    pub output: String,
}

/// Host seam for executing already-authorized team work.
#[async_trait]
pub trait TeamWorker: Send + Sync {
    async fn run(&self, request: TeamWorkRequest) -> Result<TeamWorkResult, OrchestrationError>;
}
