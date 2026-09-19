//! Deferred tool calls: the typed, resumable "the loop paused on a tool"
//! output (A2). See [`types`] for the vocabulary and the agent-loop docs for
//! how the loop produces and consumes it.

mod types;

pub use types::*;

use serde_json::Value;

use crate::ids::CallId;
use tinytools::ToolResult;

impl DeferredToolRequests {
    /// `true` when no call is pending.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.approvals.is_empty()
    }

    /// Every pending call id, approvals first, in deferral order.
    pub fn call_ids(&self) -> Vec<CallId> {
        self.approvals
            .iter()
            .chain(self.calls.iter())
            .map(|call| CallId::new(call.id.clone()))
            .collect()
    }

    /// The pending call ids `results` does not resolve, in deferral order.
    ///
    /// A decision in *either* map resolves an id: a host that ran an
    /// approval-gated tool itself answers it through `calls`, and one that
    /// prefers to let the harness run an external tool answers through
    /// `approvals`.
    pub fn remaining(&self, results: &DeferredToolResults) -> Vec<CallId> {
        self.call_ids()
            .into_iter()
            .filter(|id| !results.resolves(id))
            .collect()
    }

    /// Builds a [`DeferredToolResults`] that approves every pending approval.
    /// External `calls` are left unresolved (the host must still supply
    /// them). Mirrors Pydantic AI's `build_results(approve_all=True)`.
    pub fn approve_all(&self) -> DeferredToolResults {
        let mut results = DeferredToolResults::default();
        for call in &self.approvals {
            results
                .approvals
                .insert(CallId::new(call.id.clone()), ApprovalDecision::Approve);
        }
        results
    }
}

impl DeferredToolResults {
    /// An empty resolution set; add decisions with the builder methods.
    pub fn new() -> Self {
        Self::default()
    }

    /// Approves `call_id` with the model's original arguments.
    #[must_use]
    pub fn approve(mut self, call_id: impl Into<String>) -> Self {
        self.approvals
            .insert(CallId::new(call_id), ApprovalDecision::Approve);
        self
    }

    /// Approves `call_id` with edited arguments.
    #[must_use]
    pub fn approve_with_args(mut self, call_id: impl Into<String>, arguments: Value) -> Self {
        self.approvals.insert(
            CallId::new(call_id),
            ApprovalDecision::ApproveWithArgs(arguments),
        );
        self
    }

    /// Denies `call_id`; the model sees `message` as a tool-error result.
    #[must_use]
    pub fn deny(mut self, call_id: impl Into<String>, message: impl Into<String>) -> Self {
        self.approvals.insert(
            CallId::new(call_id),
            ApprovalDecision::Deny {
                message: message.into(),
            },
        );
        self
    }

    /// Supplies the host-produced result for an externally executed call.
    #[must_use]
    pub fn respond(mut self, call_id: impl Into<String>, result: ToolResult) -> Self {
        self.calls
            .insert(CallId::new(call_id), DeferredCallResult::Result(result));
        self
    }

    /// Whether `call_id` has a decision in either map.
    pub fn resolves(&self, call_id: &CallId) -> bool {
        self.approvals.contains_key(call_id) || self.calls.contains_key(call_id)
    }
}

impl DeferredCallResult {
    /// The [`ToolResult`] the model sees for this outcome.
    pub fn into_tool_result(self) -> ToolResult {
        match self {
            DeferredCallResult::Result(result) => result,
            DeferredCallResult::Retry(message) => ToolResult::retry(message),
            DeferredCallResult::Failed(message) => ToolResult::failed(message),
        }
    }
}

#[cfg(test)]
mod test;
