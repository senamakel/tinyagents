//! Storage DTOs of the session history: the same rows as [`crate::types`] with
//! epoch-millisecond times (`*_ms`) instead of `DateTime`, and the full-text row.
//!
//! The classic form also stored microsecond sort keys (`startedUs`, `createdUs`)
//! beside each record; the one `*_ms` field replaces both.

use super::dto::{Stored, from_ms, from_ms_opt, stored_as_is, to_ms};
use crate::types::{SessionMessage, SessionRecord, SessionStatus, SessionToolCall};
use serde::{Deserialize, Serialize};
use tinystoragedrivers_types::StorageError;

/// `history.sessions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SessionDto {
    pub(super) id: String,
    pub(super) agent_definition_id: String,
    pub(super) agent_definition_name: String,
    pub(super) session_key: String,
    pub(super) parent_session_id: Option<String>,
    pub(super) thread_id: Option<String>,
    pub(super) source_channel: Option<String>,
    pub(super) status: SessionStatus,
    pub(super) model: Option<String>,
    pub(super) turn_count: u32,
    pub(super) input_tokens: u64,
    pub(super) output_tokens: u64,
    pub(super) cached_input_tokens: u64,
    pub(super) cost_usd: f64,
    pub(super) transcript_path: Option<String>,
    pub(super) started_ms: i64,
    pub(super) ended_ms: Option<i64>,
}

impl Stored for SessionRecord {
    type Dto = SessionDto;

    fn to_dto(&self) -> SessionDto {
        SessionDto {
            id: self.id.clone(),
            agent_definition_id: self.agent_definition_id.clone(),
            agent_definition_name: self.agent_definition_name.clone(),
            session_key: self.session_key.clone(),
            parent_session_id: self.parent_session_id.clone(),
            thread_id: self.thread_id.clone(),
            source_channel: self.source_channel.clone(),
            status: self.status,
            model: self.model.clone(),
            turn_count: self.turn_count,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cached_input_tokens: self.cached_input_tokens,
            cost_usd: self.cost_usd,
            transcript_path: self.transcript_path.clone(),
            started_ms: to_ms(self.started_at),
            ended_ms: self.ended_at.map(to_ms),
        }
    }

    fn from_dto(dto: SessionDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            agent_definition_id: dto.agent_definition_id,
            agent_definition_name: dto.agent_definition_name,
            session_key: dto.session_key,
            parent_session_id: dto.parent_session_id,
            thread_id: dto.thread_id,
            source_channel: dto.source_channel,
            status: dto.status,
            model: dto.model,
            turn_count: dto.turn_count,
            input_tokens: dto.input_tokens,
            output_tokens: dto.output_tokens,
            cached_input_tokens: dto.cached_input_tokens,
            cost_usd: dto.cost_usd,
            transcript_path: dto.transcript_path,
            started_at: from_ms(dto.started_ms)?,
            ended_at: from_ms_opt(dto.ended_ms)?,
        })
    }
}

/// `history.messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct MessageDto {
    pub(super) id: i64,
    pub(super) session_id: String,
    pub(super) role: String,
    pub(super) content: String,
    pub(super) reasoning_content: Option<String>,
    pub(super) model: Option<String>,
    pub(super) input_tokens: Option<u64>,
    pub(super) output_tokens: Option<u64>,
    pub(super) cost_usd: Option<f64>,
    pub(super) created_ms: i64,
}

impl Stored for SessionMessage {
    type Dto = MessageDto;

    fn to_dto(&self) -> MessageDto {
        MessageDto {
            id: self.id,
            session_id: self.session_id.clone(),
            role: self.role.clone(),
            content: self.content.clone(),
            reasoning_content: self.reasoning_content.clone(),
            model: self.model.clone(),
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cost_usd: self.cost_usd,
            created_ms: to_ms(self.created_at),
        }
    }

    fn from_dto(dto: MessageDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            session_id: dto.session_id,
            role: dto.role,
            content: dto.content,
            reasoning_content: dto.reasoning_content,
            model: dto.model,
            input_tokens: dto.input_tokens,
            output_tokens: dto.output_tokens,
            cost_usd: dto.cost_usd,
            created_at: from_ms(dto.created_ms)?,
        })
    }
}

/// `history.tool_calls`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolCallDto {
    pub(super) id: i64,
    pub(super) session_id: String,
    pub(super) message_id: Option<i64>,
    pub(super) tool_name: String,
    pub(super) tool_input: Option<String>,
    pub(super) tool_output: Option<String>,
    pub(super) status: String,
    pub(super) duration_ms: Option<i64>,
    pub(super) created_ms: i64,
}

impl Stored for SessionToolCall {
    type Dto = ToolCallDto;

    fn to_dto(&self) -> ToolCallDto {
        ToolCallDto {
            id: self.id,
            session_id: self.session_id.clone(),
            message_id: self.message_id,
            tool_name: self.tool_name.clone(),
            tool_input: self.tool_input.clone(),
            tool_output: self.tool_output.clone(),
            status: self.status.clone(),
            duration_ms: self.duration_ms,
            created_ms: to_ms(self.created_at),
        }
    }

    fn from_dto(dto: ToolCallDto) -> Result<Self, StorageError> {
        Ok(Self {
            id: dto.id,
            session_id: dto.session_id,
            message_id: dto.message_id,
            tool_name: dto.tool_name,
            tool_input: dto.tool_input,
            tool_output: dto.tool_output,
            status: dto.status,
            duration_ms: dto.duration_ms,
            created_at: from_ms(dto.created_ms)?,
        })
    }
}

/// `history.search`: one full-text row, a session's agent name, a message
/// snippet or a tool name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct FtsRow {
    pub(super) session_id: String,
    pub(super) agent_name: String,
    pub(super) content: String,
    pub(super) tool_name: String,
}

stored_as_is!(FtsRow);
