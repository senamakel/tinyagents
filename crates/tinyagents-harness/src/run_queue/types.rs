//! Public types for the active-run queue.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tinyinference_llm::message::Message;

/// The shared queue the agent loop drains: a [`RunQueue`][super::RunQueue]
/// of transcript-ready [`Message`]s.
///
/// Attach one to a run with
/// [`RunContext::with_run_queue`][crate::context::RunContext::with_run_queue]
/// and keep a clone to push into from outside the run. `Steer` and
/// `Followup` items are appended to the transcript verbatim, so push them as
/// [`Message::user`] (or [`Message::system`]) — the host chooses the role.
pub type RunQueueHandle = Arc<super::RunQueue<Message>>;

/// A queue lane consumed by the agent runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueLane {
    /// Inject at the next safe iteration boundary as an instruction.
    Steer,
    /// Dispatch as a fresh turn after the active run completes.
    Followup,
    /// Inject at the next safe boundary as additional context.
    Collect,
}

impl QueueLane {
    /// Returns a stable, snake_case name for this lane, suitable for logging
    /// and event labels (e.g. `"followup"`).
    pub fn as_str(self) -> &'static str {
        match self {
            QueueLane::Steer => "steer",
            QueueLane::Followup => "followup",
            QueueLane::Collect => "collect",
        }
    }
}

/// How many queued items the agent loop takes from a lane at one safe
/// boundary. Mirrors pi's `QueueMode` (`"one-at-a-time" | "all"`).
///
/// Set on [`crate::runtime::RunPolicy::queue_mode`]; consulted by
/// [`RunQueue::take`][crate::run_queue::RunQueue::take].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueMode {
    /// Apply only the oldest queued item per boundary; the rest wait for the
    /// next one. Gives the model a chance to react to each instruction
    /// separately.
    OneAtATime,
    /// Apply every item queued in the lane at the boundary, in FIFO order.
    /// The default.
    #[default]
    All,
}

/// Snapshot of the queue depth per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct QueueStatus {
    /// Number of pending steer items.
    pub steers: usize,
    /// Number of pending follow-up items.
    pub followups: usize,
    /// Number of pending collected-context items.
    pub collects: usize,
    /// Total number of pending items across all lanes.
    pub total: usize,
}
