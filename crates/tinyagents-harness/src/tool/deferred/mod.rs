//! Deferred tool calls: the typed, resumable "the loop paused on a tool"
//! output (A2). See [`types`] for the vocabulary and the agent-loop docs for
//! how the loop produces and consumes it.

mod types;

pub use types::*;

use crate::ids::CallId;

impl DeferredToolRequests {
    /// `true` when no call is pending.
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty() && self.approvals.is_empty()
    }

    /// Every pending call id, approvals first, in deferral order.
    pub fn call_ids(&self) -> Vec<CallId> {
        Vec::new()
    }

    /// The pending call ids `results` does not resolve, in deferral order.
    pub fn remaining(&self, _results: &DeferredToolResults) -> Vec<CallId> {
        Vec::new()
    }
}

#[cfg(test)]
mod test;
