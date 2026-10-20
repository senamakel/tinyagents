//! Storage DTOs of the run ledger.
//!
//! The wire types in [`super::super::types`] are camelCase with `DateTime`
//! fields and stay that way for hosts. What the ledger stores is a separate,
//! bare snake_case body with epoch-millisecond times (`*_ms`), so every stored
//! record has the one shape the v2 schema fixes. Each DTO converts to and from
//! its wire type; a time loses its sub-millisecond part on the way in.
//!
//! The times stay body fields rather than record metadata: they are set by the
//! writer (an import keeps the classic time, a caller may supply `started_at`),
//! not by the driver's clock, and retention and paging order by them.

use super::super::tool_effects::{ToolEffectRow, ToolEffectStatus};
use super::super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use tinystoragedrivers_types::StorageError;

/// A wire type and the body the ledger stores for it.
pub(super) trait Stored: Sized {
    /// The storage DTO: snake_case, `*_ms` times, no wrappers.
    type Dto: Serialize + DeserializeOwned + Send + Sync;

    fn to_dto(&self) -> Self::Dto;

    /// # Errors
    /// A stored time outside the range `chrono` can represent.
    fn from_dto(dto: Self::Dto) -> Result<Self, StorageError>;
}

/// Mark storage-only types whose serde shape already is the stored body.
macro_rules! stored_as_is {
    ($($type:ty),+ $(,)?) => {$(
        impl $crate::run_ledger::drivers::dto::Stored for $type {
            type Dto = $type;
            fn to_dto(&self) -> Self::Dto {
                self.clone()
            }
            fn from_dto(dto: Self::Dto) -> ::std::result::Result<Self, tinystoragedrivers_types::StorageError> {
                Ok(dto)
            }
        }
    )+};
}
pub(super) use stored_as_is;

pub(super) fn to_ms(time: DateTime<Utc>) -> i64 {
    time.timestamp_millis()
}

pub(super) fn from_ms(ms: i64) -> Result<DateTime<Utc>, StorageError> {
    DateTime::from_timestamp_millis(ms)
        .ok_or_else(|| StorageError::serialization("stored time is out of range"))
}

pub(super) fn from_ms_opt(ms: Option<i64>) -> Result<Option<DateTime<Utc>>, StorageError> {
    ms.map(from_ms).transpose()
}

/// `ledger.agent_runs`. The joined `telemetry` is not stored: it lives in
/// `ledger.telemetry` and is attached on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct AgentRunDto {
    pub(super) id: String,
    pub(super) kind: AgentRunKind,
    pub(super) parent_run_id: Option<String>,
    pub(super) parent_thread_id: Option<String>,
    pub(super) agent_id: Option<String>,
    pub(super) status: AgentRunStatus,
    pub(super) prompt_ref: Option<String>,
    pub(super) worker_thread_id: Option<String>,
    pub(super) checkpoint_path: Option<String>,
    pub(super) checkpoint: Option<Value>,
    pub(super) summary: Option<String>,
    pub(super) error: Option<String>,
    pub(super) metadata: Value,
    pub(super) started_ms: i64,
    pub(super) updated_ms: i64,
    pub(super) completed_ms: Option<i64>,
}

impl Stored for AgentRun {
    type Dto = AgentRunDto;

    fn to_dto(&self) -> AgentRunDto {
        AgentRunDto {
            id: self.id.clone(),
            kind: self.kind,
            parent_run_id: self.parent_run_id.clone(),
            parent_thread_id: self.parent_thread_id.clone(),
            agent_id: self.agent_id.clone(),
            status: self.status,
            prompt_ref: self.prompt_ref.clone(),
            worker_thread_id: self.worker_thread_id.clone(),
            checkpoint_path: self.checkpoint_path.clone(),
            checkpoint: self.checkpoint.clone(),
            summary: self.summary.clone(),
            error: self.error.clone(),
            metadata: self.metadata.clone(),
            started_ms: to_ms(self.started_at),
            updated_ms: to_ms(self.updated_at),
            completed_ms: self.completed_at.map(to_ms),
        }
    }

    fn from_dto(dto: AgentRunDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            kind: dto.kind,
            parent_run_id: dto.parent_run_id,
            parent_thread_id: dto.parent_thread_id,
            agent_id: dto.agent_id,
            status: dto.status,
            prompt_ref: dto.prompt_ref,
            worker_thread_id: dto.worker_thread_id,
            checkpoint_path: dto.checkpoint_path,
            checkpoint: dto.checkpoint,
            summary: dto.summary,
            error: dto.error,
            metadata: dto.metadata,
            telemetry: None,
            started_at: from_ms(dto.started_ms)?,
            updated_at: from_ms(dto.updated_ms)?,
            completed_at: from_ms_opt(dto.completed_ms)?,
        })
    }
}

/// `ledger.workflow_runs`. The driver lease is not stored here: it is the
/// `tsd.leases` record `Key::of(["ledger", "workflow", <run id>])` and is
/// attached on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct WorkflowRunDto {
    pub(super) id: String,
    pub(super) definition_id: String,
    pub(super) parent_thread_id: Option<String>,
    pub(super) input: Value,
    pub(super) phase_states: Value,
    pub(super) child_run_ids: Vec<String>,
    pub(super) status: WorkflowRunStatus,
    pub(super) summary: Option<String>,
    pub(super) started_ms: i64,
    pub(super) updated_ms: i64,
    pub(super) completed_ms: Option<i64>,
    pub(super) revision: u64,
}

impl Stored for WorkflowRun {
    type Dto = WorkflowRunDto;

    fn to_dto(&self) -> WorkflowRunDto {
        WorkflowRunDto {
            id: self.id.clone(),
            definition_id: self.definition_id.clone(),
            parent_thread_id: self.parent_thread_id.clone(),
            input: self.input.clone(),
            phase_states: self.phase_states.clone(),
            child_run_ids: self.child_run_ids.clone(),
            status: self.status,
            summary: self.summary.clone(),
            started_ms: to_ms(self.started_at),
            updated_ms: to_ms(self.updated_at),
            completed_ms: self.completed_at.map(to_ms),
            revision: self.revision,
        }
    }

    fn from_dto(dto: WorkflowRunDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            definition_id: dto.definition_id,
            parent_thread_id: dto.parent_thread_id,
            input: dto.input,
            phase_states: dto.phase_states,
            child_run_ids: dto.child_run_ids,
            status: dto.status,
            summary: dto.summary,
            started_at: from_ms(dto.started_ms)?,
            updated_at: from_ms(dto.updated_ms)?,
            completed_at: from_ms_opt(dto.completed_ms)?,
            revision: dto.revision,
            lease_owner: None,
            lease_expires_at: None,
        })
    }
}

/// `ledger.event_outbox`. The same body, minus `run_id` and `sequence`, is the
/// entry of the `ledger.events/<run>` stream (see [`StreamEvent`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RunEventDto {
    pub(super) run_id: String,
    pub(super) sequence: u64,
    pub(super) event_type: String,
    pub(super) payload: Value,
    pub(super) ts_ms: i64,
}

impl Stored for RunEvent {
    type Dto = RunEventDto;

    fn to_dto(&self) -> RunEventDto {
        RunEventDto {
            run_id: self.run_id.clone(),
            sequence: self.sequence,
            event_type: self.event_type.clone(),
            payload: self.payload.clone(),
            ts_ms: to_ms(self.timestamp),
        }
    }

    fn from_dto(dto: RunEventDto) -> Result<Self, StorageError> {
        Ok(Self {
            run_id: dto.run_id,
            sequence: dto.sequence,
            event_type: dto.event_type,
            payload: dto.payload,
            timestamp: from_ms(dto.ts_ms)?,
        })
    }
}

/// An entry of the `ledger.events/<run>` stream: the stream name is the run,
/// so the entry does not repeat it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StreamEvent {
    pub(super) sequence: u64,
    pub(super) event_type: String,
    pub(super) payload: Value,
    pub(super) ts_ms: i64,
}

impl StreamEvent {
    pub(super) fn of(event: &RunEvent) -> Self {
        Self {
            sequence: event.sequence,
            event_type: event.event_type.clone(),
            payload: event.payload.clone(),
            ts_ms: to_ms(event.timestamp),
        }
    }

    pub(super) fn into_event(self, run_id: &str) -> Result<RunEvent, StorageError> {
        Ok(RunEvent {
            run_id: run_id.to_owned(),
            sequence: self.sequence,
            event_type: self.event_type,
            payload: self.payload,
            timestamp: from_ms(self.ts_ms)?,
        })
    }
}

/// `ledger.telemetry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RunTelemetryDto {
    pub(super) run_id: String,
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) cached_input_tokens: u64,
    pub(super) cost_usd: f64,
    pub(super) elapsed_ms: Option<u64>,
    pub(super) tool_count: u64,
    pub(super) model: Option<String>,
    pub(super) provider: Option<String>,
    pub(super) error: Option<String>,
    pub(super) updated_ms: Option<i64>,
}

impl Stored for RunTelemetry {
    type Dto = RunTelemetryDto;

    fn to_dto(&self) -> RunTelemetryDto {
        RunTelemetryDto {
            run_id: self.run_id.clone(),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cached_input_tokens: self.cached_input_tokens,
            cost_usd: self.cost_usd,
            elapsed_ms: self.elapsed_ms,
            tool_count: self.tool_count,
            model: self.model.clone(),
            provider: self.provider.clone(),
            error: self.error.clone(),
            updated_ms: self.updated_at.map(to_ms),
        }
    }

    fn from_dto(dto: RunTelemetryDto) -> Result<Self, StorageError> {
        Ok(Self {
            run_id: dto.run_id,
            input_tokens: dto.input_tokens,
            output_tokens: dto.output_tokens,
            cached_input_tokens: dto.cached_input_tokens,
            cost_usd: dto.cost_usd,
            elapsed_ms: dto.elapsed_ms,
            tool_count: dto.tool_count,
            model: dto.model,
            provider: dto.provider,
            error: dto.error,
            updated_at: from_ms_opt(dto.updated_ms)?,
        })
    }
}

/// `ledger.teams`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TeamDto {
    pub(super) id: String,
    pub(super) parent_thread_id: Option<String>,
    pub(super) lead_agent_id: String,
    pub(super) status: AgentTeamStatus,
    pub(super) summary: Option<String>,
    pub(super) created_ms: i64,
    pub(super) updated_ms: i64,
    pub(super) closed_ms: Option<i64>,
}

impl Stored for AgentTeam {
    type Dto = TeamDto;

    fn to_dto(&self) -> TeamDto {
        TeamDto {
            id: self.id.clone(),
            parent_thread_id: self.parent_thread_id.clone(),
            lead_agent_id: self.lead_agent_id.clone(),
            status: self.status,
            summary: self.summary.clone(),
            created_ms: to_ms(self.created_at),
            updated_ms: to_ms(self.updated_at),
            closed_ms: self.closed_at.map(to_ms),
        }
    }

    fn from_dto(dto: TeamDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            parent_thread_id: dto.parent_thread_id,
            lead_agent_id: dto.lead_agent_id,
            status: dto.status,
            summary: dto.summary,
            created_at: from_ms(dto.created_ms)?,
            updated_at: from_ms(dto.updated_ms)?,
            closed_at: from_ms_opt(dto.closed_ms)?,
        })
    }
}

/// `ledger.members`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MemberDto {
    pub(super) id: String,
    pub(super) team_id: String,
    pub(super) name: String,
    pub(super) agent_id: Option<String>,
    pub(super) member_status: AgentTeamMemberStatus,
    pub(super) current_task_id: Option<String>,
    pub(super) worker_thread_id: Option<String>,
    pub(super) run_id: Option<String>,
    pub(super) created_ms: i64,
    pub(super) updated_ms: i64,
}

impl Stored for AgentTeamMember {
    type Dto = MemberDto;

    fn to_dto(&self) -> MemberDto {
        MemberDto {
            id: self.id.clone(),
            team_id: self.team_id.clone(),
            name: self.name.clone(),
            agent_id: self.agent_id.clone(),
            member_status: self.member_status,
            current_task_id: self.current_task_id.clone(),
            worker_thread_id: self.worker_thread_id.clone(),
            run_id: self.run_id.clone(),
            created_ms: to_ms(self.created_at),
            updated_ms: to_ms(self.updated_at),
        }
    }

    fn from_dto(dto: MemberDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            team_id: dto.team_id,
            name: dto.name,
            agent_id: dto.agent_id,
            member_status: dto.member_status,
            current_task_id: dto.current_task_id,
            worker_thread_id: dto.worker_thread_id,
            run_id: dto.run_id,
            created_at: from_ms(dto.created_ms)?,
            updated_at: from_ms(dto.updated_ms)?,
        })
    }
}

/// `ledger.tasks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TaskDto {
    pub(super) id: String,
    pub(super) team_id: String,
    pub(super) title: String,
    pub(super) objective: Option<String>,
    pub(super) status: AgentTeamTaskStatus,
    pub(super) owner_member_id: Option<String>,
    pub(super) claimed_by_member_id: Option<String>,
    pub(super) claim_token: Option<String>,
    pub(super) depends_on: Vec<String>,
    pub(super) gate_status: String,
    pub(super) gate_reason: Option<String>,
    pub(super) evidence: Vec<String>,
    pub(super) source_run_id: Option<String>,
    pub(super) order_index: i64,
    pub(super) created_ms: i64,
    pub(super) updated_ms: i64,
}

impl Stored for AgentTeamTask {
    type Dto = TaskDto;

    fn to_dto(&self) -> TaskDto {
        TaskDto {
            id: self.id.clone(),
            team_id: self.team_id.clone(),
            title: self.title.clone(),
            objective: self.objective.clone(),
            status: self.status,
            owner_member_id: self.owner_member_id.clone(),
            claimed_by_member_id: self.claimed_by_member_id.clone(),
            claim_token: self.claim_token.clone(),
            depends_on: self.depends_on.clone(),
            gate_status: self.gate_status.clone(),
            gate_reason: self.gate_reason.clone(),
            evidence: self.evidence.clone(),
            source_run_id: self.source_run_id.clone(),
            order_index: self.order_index,
            created_ms: to_ms(self.created_at),
            updated_ms: to_ms(self.updated_at),
        }
    }

    fn from_dto(dto: TaskDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            team_id: dto.team_id,
            title: dto.title,
            objective: dto.objective,
            status: dto.status,
            owner_member_id: dto.owner_member_id,
            claimed_by_member_id: dto.claimed_by_member_id,
            claim_token: dto.claim_token,
            depends_on: dto.depends_on,
            gate_status: dto.gate_status,
            gate_reason: dto.gate_reason,
            evidence: dto.evidence,
            source_run_id: dto.source_run_id,
            order_index: dto.order_index,
            created_at: from_ms(dto.created_ms)?,
            updated_at: from_ms(dto.updated_ms)?,
        })
    }
}

/// `ledger.tool_effects`. The status stays a string so an unknown persisted
/// value keeps the conservative parse fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct EffectDto {
    pub(super) run_id: String,
    pub(super) call_id: String,
    pub(super) tool: String,
    pub(super) status: String,
    pub(super) idempotency_key: Option<String>,
    pub(super) effect_summary: Option<String>,
    pub(super) started_ms: i64,
    pub(super) settled_ms: Option<i64>,
}

impl Stored for ToolEffectRow {
    type Dto = EffectDto;

    fn to_dto(&self) -> EffectDto {
        EffectDto {
            run_id: self.run_id.clone(),
            call_id: self.call_id.clone(),
            tool: self.tool.clone(),
            status: self.status.as_str().into(),
            idempotency_key: self.idempotency_key.clone(),
            effect_summary: self.effect_summary.clone(),
            started_ms: to_ms(self.started_at),
            settled_ms: self.settled_at.map(to_ms),
        }
    }

    fn from_dto(dto: EffectDto) -> Result<Self, StorageError> {
        Ok(Self {
            run_id: dto.run_id,
            call_id: dto.call_id,
            tool: dto.tool,
            status: ToolEffectStatus::parse(&dto.status),
            idempotency_key: dto.idempotency_key,
            effect_summary: dto.effect_summary,
            started_at: from_ms(dto.started_ms)?,
            settled_at: from_ms_opt(dto.settled_ms)?,
        })
    }
}
