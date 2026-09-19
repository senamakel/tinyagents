//! Type definitions for deferred tool calls (A2).
//!
//! A tool call leaves the agent loop *without* a result in three ways: the
//! tool's declared policy requires human approval
//! (`ToolPolicy.access.approval_required`), the tool (or a `before_tool`
//! middleware) raised [`TinyAgentsError::ApprovalRequired`] /
//! [`TinyAgentsError::CallDeferred`], or the tool was registered schema-only
//! through [`ToolRegistry::register_external`]. The loop then finishes the
//! rest of the batch and exits with [`DeferredToolRequests`], which the host
//! resolves into [`DeferredToolResults`] and hands back to resume.
//!
//! [`TinyAgentsError::ApprovalRequired`]: crate::error::TinyAgentsError::ApprovalRequired
//! [`TinyAgentsError::CallDeferred`]: crate::error::TinyAgentsError::CallDeferred
//! [`ToolRegistry::register_external`]: crate::tool::ToolRegistry::register_external

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::ids::CallId;
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

/// Every tool call one assistant turn left pending, keyed by the provider's
/// `tool_call_id`.
///
/// The transcript the loop returns alongside this (`AgentRun::messages`)
/// still ends with the assistant row that requested these calls; the
/// non-deferred siblings in the same batch already have their tool-result
/// rows appended, so only the ids listed here are unanswered.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DeferredToolRequests {
    /// Calls the *host* must execute (an external/schema-only tool, or a tool
    /// that raised `CallDeferred`). Resolve each with a
    /// [`DeferredCallResult`].
    #[serde(default)]
    pub calls: Vec<ToolCall>,
    /// Calls that need a human decision before the harness runs them.
    /// Resolve each with an [`ApprovalDecision`].
    #[serde(default)]
    pub approvals: Vec<ToolCall>,
    /// Host-only metadata attached at deferral time (the `metadata` payload
    /// of `ApprovalRequired`/`CallDeferred`, or the tool's declared policy
    /// display fields for a policy-driven approval). Never shown to the model.
    #[serde(default)]
    pub metadata: BTreeMap<CallId, Value>,
}

/// A human decision on one call listed in [`DeferredToolRequests::approvals`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Run the tool now with the arguments the model supplied.
    Approve,
    /// Run the tool now with these edited arguments instead of the model's.
    ApproveWithArgs(Value),
    /// Do not run the tool; the model sees `message` as a tool-error result.
    Deny {
        /// Explanation handed to the model as the tool result.
        message: String,
    },
}

/// The host-supplied outcome for one call listed in
/// [`DeferredToolRequests::calls`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum DeferredCallResult {
    /// The host ran the tool and this is what it produced.
    Result(ToolResult),
    /// Ask the model to try again (folded into [`ToolResult::retry`]).
    Retry(String),
    /// A permanent failure (folded into [`ToolResult::failed`]).
    Failed(String),
}

/// Resolutions for a [`DeferredToolRequests`] batch, keyed by call id.
///
/// Partial resolution is allowed at the type level;
/// [`DeferredToolRequests::remaining`] reports what is still missing and the
/// loop refuses to resume until every pending id is covered.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeferredToolResults {
    /// Decisions for the calls in [`DeferredToolRequests::approvals`].
    #[serde(default)]
    pub approvals: BTreeMap<CallId, ApprovalDecision>,
    /// Outcomes for the calls in [`DeferredToolRequests::calls`].
    #[serde(default)]
    pub calls: BTreeMap<CallId, DeferredCallResult>,
}

/// Resolves deferred tool calls *inline*, so the loop never has to surface
/// [`DeferredToolRequests`] to its caller.
///
/// Register one with
/// [`AgentHarness::with_deferred_tool_handler`][crate::runtime::AgentHarness::with_deferred_tool_handler].
/// The handler must resolve every pending id: an incomplete
/// [`DeferredToolResults`] fails the run with
/// [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation].
/// A desktop host's approval dialog (park the call on a oneshot, wait for the
/// user) is the canonical implementation.
#[async_trait]
pub trait DeferredToolHandler: Send + Sync {
    /// Resolves every call in `requests`.
    async fn handle(&self, requests: &DeferredToolRequests) -> Result<DeferredToolResults>;
}
