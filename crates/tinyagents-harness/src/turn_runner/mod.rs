//! Channel-independent turn lifecycle policy for hosts.
//!
//! [`TurnRunner`] owns ingress ordering, per-scope session serialization,
//! cancellation generations, deadline/idle handling, reply persistence, and
//! exactly-once terminal delivery. Hosts provide the side effects through the
//! ports in this module; channel protocols and their wire events stay host
//! owned.

mod types;
mod work_tracker;

pub use types::{
    EventSink, FailureClassifier, IngressDecision, IngressPolicy, IngressStage, LiveTurn,
    QueueDisposition, ReplyStore, RequeuePayload, ScopeKey, Summarizer, TerminalEvent, TurnHost,
    TurnRequest, TurnResult, TurnRunner, TurnRunnerConfig, TurnRunnerError, TurnSession,
    TurnSignal, deterministic_reply_id,
};
pub use work_tracker::WorkTracker;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
