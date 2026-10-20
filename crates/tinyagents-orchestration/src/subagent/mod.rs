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

mod admission;
mod awaiting_user;
mod completion;
mod detached;
mod document_persistence;
mod driver;
mod executor;
mod invocation;
mod outcome_status_map;
mod persistence;
mod planner;
mod policy;
mod result_policy;
mod role;
mod sessions;
mod step;
mod summarization;
mod types;

pub use admission::{SpawnAdmission, SpawnPolicy, SpawnRejection, SpawnReservation};
pub use awaiting_user::SubagentAwaitingUser;
pub use detached::{
    CancelDetachedOutcome, CancelledSubagentMetadata, DETACHED_LEDGER_TIMEOUT_MS,
    DetachedCompletionTarget, DetachedSubagentOwners, DetachedSubagentStatus, FinishedOutcome,
    SpawnedSubagent, SteerAccess, SteerError, SteerReceipt, SteerRoute, SubagentIdentity,
    SubagentResumeRef, SubagentSnapshot, WaitAllItem, WaitError, WaitOutcome,
    cancel_detached_or_durable, cancel_detached_or_durable_with_owners, cancel_for_thread,
    distinct_parent_threads, list_subagent_records, orphaned_subagent_reason, queue_lane_name,
    record_agent_id, record_cancelled, record_detached_completion, record_parent_session,
    record_spawned, record_status, record_subagent_session_id, record_to_wait_outcome,
    resolve_detached_or_durable, resolve_detached_or_durable_with_owners, resume_ref_for_task,
    resume_ref_from_record, snapshot_for_owner, spawn_status_watcher,
    spawn_status_watcher_with_completions, steer_detached, steer_detached_with_request_id,
    steering_command_for_lane, subagent_record_for_task, task_id_for_session,
    task_id_for_session_in_records, task_status_label, wait_all_detached,
    wait_all_detached_with_owners, wait_detached, wait_detached_or_durable,
    wait_detached_or_durable_with_owners, wait_error_from_registry,
};
pub use document_persistence::DocumentSubagentPersistence;
pub use driver::{SubagentCapabilities, SubagentDriver, SubagentInFlight};
pub use executor::SubagentExecutor;
pub use invocation::{
    ChildDataPolicy, DELEGATION_EXECUTE_PROMPT, DELEGATION_MODE_FIELD, DELEGATION_PLAN_PROMPT,
    DELEGATION_REVIEW_PROMPT, DelegationEnvelope, DelegationRequest, DelegationTool,
    DriverSubAgentTool, MultiTargetDelegationTool, ReviewApprovalPolicy, SUBAGENT_MODE_FIELD,
    SubAgent, SubAgentJob, SubAgentJobAction, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry,
    SubAgentJobStatus, SubAgentJobsTool, SubAgentMessageTool, SubAgentMode, SubAgentSession,
    SubAgentTool, SubagentDriverToolHost, parse_approval, register_subagent_job_tools,
    review_stage_output,
};
pub use persistence::SubagentPersistence;
pub use planner::SubagentPlanner;
pub use policy::{SubAgentBudget, SubAgentPolicy};
pub use result_policy::{
    AppliedResult, ArtifactStore, ResultOverflow, ResultPolicy, frame_unexecuted_outcome,
    is_safe_task_id, is_unexecuted_tool_call_stub, truncate_head_tail,
};
pub use role::{
    FLEET_TOOLS, FleetToolSet, SUBAGENT_JOBS_TOOL, SUBAGENT_MESSAGE_TOOL, SubagentRole,
    is_delegation_tool, restrict_tools, restrict_tools_with_scope, subagent_framing,
};
pub use sessions::{
    DriverSubagentSessionStore, DurableSubagentSession, DurableSubagentSessionSummary,
    DurableSubagentStatus, ReuseDecision, SubagentSessionSelector, SubagentSessionStore,
    SubagentSessionUpsert, find_reusable_session, normalize_task_key, reuse_decision,
    task_title_from_prompt,
};
pub use step::{
    AgentStepConfig, AgentStepError, AgentStepIdentity, AgentStepResult, StepContext, StepSuccess,
    StepWorkError, run_agent_step,
};
pub use summarization::{CapSummary, CapSummaryModel, summarize_at_cap};
#[allow(deprecated)]
pub use types::SubagentStatus;
pub use types::{
    ArtifactReference, AttemptContextFactory, IncompleteKind, PersistedSubagentPause,
    PreparedSubagent, SubagentContinuationTarget, SubagentError, SubagentExecution,
    SubagentIncomplete, SubagentOutcome, SubagentOutcomeKind, SubagentPause,
    SubagentPausePersistenceDisposition, SubagentPersistenceDisposition, SubagentRequest,
    SubagentRequestParts, SubagentResume, SubagentRunResult, SubagentTaskKey,
    SubagentTerminalPersistenceDisposition,
};

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

#[cfg(test)]
#[path = "driver_completion_tests.rs"]
mod driver_completion_test;

#[cfg(test)]
#[path = "driver_policy_tests.rs"]
mod driver_policy_test;

#[cfg(test)]
#[path = "mod_outcome_kind_tests.rs"]
mod outcome_kind_test;
