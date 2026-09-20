//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! [`SubAgent`], [`SubAgentTool`], and [`SubAgentSession`] compose harness
//! agents as direct child runs. The `Subagent*` lifecycle traits and driver
//! separately coordinate resume loading, preparation, execution, and one
//! mutually exclusive persistence action. Hosts still resolve policy, agent
//! definitions, prompts, tool allowlists, and persistence implementations.
//!
//! Dependency direction remains `orchestration -> {harness, runtime}`. Lower
//! TinyAgents layers must not depend on this module.

mod driver;
mod executor;
mod invocation;
mod persistence;
mod planner;
mod types;

pub use driver::{SubagentCapabilities, SubagentDriver};
pub use executor::SubagentExecutor;
pub use invocation::{
    ChildDataPolicy, SubAgent, SubAgentJob, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry,
    SubAgentJobStatus, SubAgentJobsTool, SubAgentMessageTool, SubAgentSession, SubAgentTool,
    register_subagent_job_tools,
};
pub use persistence::SubagentPersistence;
pub use planner::SubagentPlanner;
pub use types::{
    ArtifactReference, PersistedSubagentPause, PreparedSubagent, SubagentError, SubagentExecution,
    SubagentIncomplete, SubagentOutcome, SubagentPause, SubagentPausePersistenceDisposition,
    SubagentPersistenceDisposition, SubagentRequest, SubagentRequestParts, SubagentResume,
    SubagentRunResult, SubagentStatus, SubagentTaskKey, SubagentTerminalPersistenceDisposition,
};

#[cfg(test)]
mod test;
