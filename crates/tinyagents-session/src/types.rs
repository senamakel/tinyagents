//! Serde record types for the session-history tables: sessions, messages,
//! tool calls, and the search request/response shapes `super::ops` accepts
//! and returns.
//!
//! These mirror the columns created by migration 0 (and later) in
//! `super::migrations`. Unlike `run_ledger`'s types, there is no separate
//! `*Upsert` type here: `super::ops::record_session_start` and friends take
//! individual arguments rather than a struct, since each call maps to a
//! single INSERT with a fixed, small column set.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Lifecycle status of a [`SessionRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// The session is actively producing turns.
    Running,
    /// Finished successfully. Terminal.
    Completed,
    /// Finished with an error. Terminal.
    Failed,
    /// Interrupted by a process restart (see `super::ops::mark_interrupted`).
    /// Terminal.
    Interrupted,
}

impl SessionStatus {
    /// Renders the status as the string stored in the `status` column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }

    /// Parses a stored `status` string, defaulting to [`SessionStatus::Running`]
    /// for any unrecognized value.
    pub fn parse(s: &str) -> Self {
        match s {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "interrupted" => Self::Interrupted,
            _ => Self::Running,
        }
    }
}

/// A persisted agent session: one row per conversation/run, with lineage,
/// cost, and lifecycle status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    /// Identifier of the agent definition that drove this session.
    pub agent_definition_id: String,
    /// Display name of the agent definition, indexed for full-text search.
    pub agent_definition_name: String,
    /// Caller-owned key used to look up or resume this session.
    pub session_key: String,
    /// The session that spawned this one, if this is a sub-session.
    pub parent_session_id: Option<String>,
    /// Caller-owned logical thread identifier, if this session is
    /// thread-scoped.
    pub thread_id: Option<String>,
    /// Caller-owned channel/surface this session originated from (e.g. a
    /// chat UI, an API integration).
    pub source_channel: Option<String>,
    pub status: SessionStatus,
    /// Model used for the most recent turn.
    pub model: Option<String>,
    pub turn_count: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cost_usd: f64,
    /// Path to the session's durable transcript file, if one was recorded.
    pub transcript_path: Option<String>,
    pub started_at: DateTime<Utc>,
    /// When the session reached a terminal status, if it has.
    pub ended_at: Option<DateTime<Utc>>,
}

/// A single persisted message within a [`SessionRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub id: i64,
    pub session_id: String,
    pub role: String,
    pub content: String,
    /// Hidden model reasoning, when the provider exposed it. Kept separate
    /// from visible `content` so search/UI consumers never mistake it for an
    /// assistant reply.
    pub reasoning_content: Option<String>,
    /// Model that produced this message, when it is an assistant turn.
    pub model: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost_usd: Option<f64>,
    pub created_at: DateTime<Utc>,
}

/// A single persisted tool call within a [`SessionRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToolCall {
    pub id: i64,
    pub session_id: String,
    /// The [`SessionMessage`] this call was made from, if known.
    pub message_id: Option<i64>,
    pub tool_name: String,
    pub tool_input: Option<String>,
    /// Tool output, bounded and possibly truncated — see
    /// `super::ops::MAX_TOOL_OUTPUT_BYTES`.
    pub tool_output: Option<String>,
    pub status: String,
    pub duration_ms: Option<i64>,
    pub created_at: DateTime<Utc>,
}

/// Filter/pagination parameters for `super::ops::search_sessions`.
///
/// `query` is plain text, not raw FTS5 syntax — see
/// `super::ops::fts_match_query` for how it is escaped before reaching
/// `MATCH`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSearchParams {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub source_channel: Option<String>,
    #[serde(default)]
    pub parent_session_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub thread_id: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

/// Response shape for `super::ops::search_sessions` and
/// `super::ops::list_sessions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSearchResult {
    pub sessions: Vec<SessionRecord>,
    /// Total number of sessions matching the filter, independent of
    /// pagination (i.e. not `sessions.len()`).
    pub total: u64,
}
