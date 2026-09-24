//! Generic multi-lane queue for messages arriving during an active run.
//!
//! Hosts decide which incoming events should be queued and retain ownership of
//! the queued payload. TinyAgents owns the reusable FIFO mechanics for the
//! three lanes an agent runtime can consume at safe iteration boundaries:
//! immediate steering, deferred follow-up work, and collected context.
//!
//! # On the agent loop path (A4)
//!
//! Attach a [`RunQueueHandle`] (an `Arc<RunQueue<Message>>`) to a run with
//! [`crate::context::RunContext::with_run_queue`] and the built-in
//! [`crate::agent_loop`] drains it at its safe turn boundaries, taking
//! [`QueueMode::All`] or [`QueueMode::OneAtATime`] items per boundary as
//! [`crate::runtime::RunPolicy::queue_mode`] says:
//!
//! - [`QueueLane::Steer`] — appended to the transcript right after a tool
//!   batch's results (never mid-batch), and at a natural finish before any
//!   follow-up. A steer that arrives after the model's final answer still
//!   gets one more turn.
//! - [`QueueLane::Followup`] — appended only when the model has finished and
//!   no steer is pending; the loop runs another turn instead of returning.
//! - [`QueueLane::Collect`] — never enters the transcript; drained once at
//!   run end onto [`crate::middleware::AgentRun::collected`].
//!
//! A middleware stop, limit stop, pause, or deferral is terminal: whatever is
//! still queued stays queued for the host. Every application emits
//! [`crate::events::AgentEvent::QueuedMessageApplied`]. The existing
//! [`crate::steering::SteeringHandle`] control channel (pause/resume/cancel/
//! inject) is unchanged and independent — `RunQueue` is content injection,
//! not run control.
//!
//! `RunQueue<T>` itself stays generic: hosts may keep using it with any `T`
//! for their own bookkeeping; only a `RunQueue<Message>` is loop-consumable.

mod types;

use tokio::sync::Mutex;

pub use types::{QueueLane, QueueMode, QueueStatus, RunQueueHandle};

/// Thread-safe FIFO queue split into steer, follow-up, and collect lanes.
#[derive(Debug)]
pub struct RunQueue<T> {
    inner: Mutex<RunQueueInner<T>>,
}

/// The three lanes behind [`RunQueue`]'s lock, each an append-only FIFO until
/// drained.
#[derive(Debug)]
struct RunQueueInner<T> {
    /// Immediate steering messages, consumed as soon as the loop reaches a
    /// safe iteration boundary.
    steers: Vec<T>,
    /// Deferred follow-up work, consumed once the current run settles.
    followups: Vec<T>,
    /// Collected context (e.g. observations) accumulated for later use.
    collects: Vec<T>,
}

impl<T> RunQueue<T> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(RunQueueInner {
                steers: Vec::new(),
                followups: Vec::new(),
                collects: Vec::new(),
            }),
        }
    }

    /// Appends `item` to `lane`.
    pub async fn push(&self, lane: QueueLane, item: T) {
        let mut inner = self.inner.lock().await;
        match lane {
            QueueLane::Steer => inner.steers.push(item),
            QueueLane::Followup => inner.followups.push(item),
            QueueLane::Collect => inner.collects.push(item),
        }
    }

    /// Drains one lane in FIFO order.
    pub async fn drain(&self, lane: QueueLane) -> Vec<T> {
        let mut inner = self.inner.lock().await;
        match lane {
            QueueLane::Steer => std::mem::take(&mut inner.steers),
            QueueLane::Followup => std::mem::take(&mut inner.followups),
            QueueLane::Collect => std::mem::take(&mut inner.collects),
        }
    }

    /// Takes items from `lane` in FIFO order according to `mode`: the oldest
    /// item only under [`QueueMode::OneAtATime`], or the whole lane under
    /// [`QueueMode::All`]. Returns an empty vec when the lane is empty.
    pub async fn take(&self, lane: QueueLane, mode: QueueMode) -> Vec<T> {
        match mode {
            QueueMode::All => self.drain(lane).await,
            QueueMode::OneAtATime => {
                let mut inner = self.inner.lock().await;
                let items = match lane {
                    QueueLane::Steer => &mut inner.steers,
                    QueueLane::Followup => &mut inner.followups,
                    QueueLane::Collect => &mut inner.collects,
                };
                if items.is_empty() {
                    Vec::new()
                } else {
                    vec![items.remove(0)]
                }
            }
        }
    }

    /// Returns the current queue depth per lane.
    pub async fn status(&self) -> QueueStatus {
        let inner = self.inner.lock().await;
        let steers = inner.steers.len();
        let followups = inner.followups.len();
        let collects = inner.collects.len();
        QueueStatus {
            steers,
            followups,
            collects,
            total: steers + followups + collects,
        }
    }

    /// Clears every lane and returns the number of dropped items.
    pub async fn clear(&self) -> usize {
        let mut inner = self.inner.lock().await;
        let total = inner.steers.len() + inner.followups.len() + inner.collects.len();
        inner.steers.clear();
        inner.followups.clear();
        inner.collects.clear();
        total
    }

    /// Returns a snapshot of every queued item, tagged with its lane, in
    /// lane/delivery order: [`QueueLane::Steer`], then
    /// [`QueueLane::Followup`], then [`QueueLane::Collect`], each lane
    /// preserving its own FIFO order. Non-destructive — the queue is
    /// unchanged, so a host can inspect pending work (e.g. to render it, or
    /// to decide whether [`Self::remove_where`] applies) without racing the
    /// agent loop's own drain.
    pub async fn snapshot(&self) -> Vec<(QueueLane, T)>
    where
        T: Clone,
    {
        let inner = self.inner.lock().await;
        let mut items =
            Vec::with_capacity(inner.steers.len() + inner.followups.len() + inner.collects.len());
        items.extend(
            inner
                .steers
                .iter()
                .cloned()
                .map(|item| (QueueLane::Steer, item)),
        );
        items.extend(
            inner
                .followups
                .iter()
                .cloned()
                .map(|item| (QueueLane::Followup, item)),
        );
        items.extend(
            inner
                .collects
                .iter()
                .cloned()
                .map(|item| (QueueLane::Collect, item)),
        );
        items
    }

    /// Removes every queued item across all lanes for which `pred` returns
    /// `true`, preserving the relative order of the items that remain in
    /// each lane. Returns the number of items removed.
    ///
    /// Use to retract a specific queued item (e.g. one the host decided not
    /// to apply after all) without clearing the rest of the queue.
    pub async fn remove_where(&self, pred: impl Fn(&T) -> bool) -> usize {
        let mut inner = self.inner.lock().await;
        let before = inner.steers.len() + inner.followups.len() + inner.collects.len();
        inner.steers.retain(|item| !pred(item));
        inner.followups.retain(|item| !pred(item));
        inner.collects.retain(|item| !pred(item));
        let after = inner.steers.len() + inner.followups.len() + inner.collects.len();
        before - after
    }
}

impl<T> Default for RunQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
