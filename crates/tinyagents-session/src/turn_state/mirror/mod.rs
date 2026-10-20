//! In-process cursor keeping the authoritative [`TurnState`] in sync with an
//! agent loop and writing it through to a [`TurnStates`] store.
//!
//! The mirror owns the snapshot, the flush policy (iteration / tool boundaries,
//! never per streaming delta), the size caps on persisted tool I/O and prose,
//! the interrupted-turn finalization, and the transcript / tool-timeline
//! bookkeeping helpers. A host translates its own progress events into calls on
//! these: mutate [`TurnStateMirror::state`], call the `push_*` helpers, then
//! [`TurnStateMirror::flush`] at boundaries.
//!
//! On a completed turn the host sets [`TurnStateMirror::turn_completed`] so
//! [`TurnStateMirror::finish`] leaves the snapshot alone; if the bridge exits
//! without ever seeing completion (for example because the agent loop errored),
//! `finish` flags the snapshot [`crate::turn_state::types::TurnLifecycle::Interrupted`] and carries the
//! partial streamed answer into the session transcript.

pub mod caps;
mod lifecycle;
mod push;

use std::sync::Arc;

use crate::port::TurnStates;
use crate::transcript::TranscriptLocator;
use crate::turn_state::types::TurnState;

pub use caps::MAX_PERSISTED_TRANSCRIPT_ITEM;

/// In-process cursor that keeps the authoritative [`TurnState`] in sync
/// with the agent loop and writes it through to a [`TurnStates`] store.
pub struct TurnStateMirror {
    pub(super) store: Arc<dyn TurnStates>,
    /// Where an interrupted turn's partial answer is appended, when set.
    pub(super) transcripts: Option<Arc<dyn TranscriptLocator>>,
    /// The agent whose root transcript receives the partial, when scoped.
    pub(super) agent_id: Option<String>,
    /// The in-memory snapshot; hosts mutate it directly, then [`Self::flush`].
    pub state: TurnState,
    /// Set to `true` once the host observes turn completion so
    /// [`Self::finish`] knows to keep the snapshot rather than mark it
    /// interrupted.
    pub turn_completed: bool,
    /// Monotonic ordering key for transcript items. Round alone can't order
    /// narration vs thinking vs tool calls *within* one iteration, so every
    /// transcript push stamps and increments this.
    pub(super) next_seq: u32,
    /// Separate monotonic ordering key for `ToolTimelineEntry::seq` — the flat
    /// timeline is an independent projection from the interleaved transcript, so
    /// it gets its own space (sharing `next_seq` would leave gaps in the
    /// transcript's contiguous ordering).
    pub(super) next_tool_seq: u64,
}

impl TurnStateMirror {
    /// Build a mirror primed with a `Started` snapshot and immediately
    /// flush so a crash before the first agent event still leaves a
    /// recoverable record.
    pub fn new(
        store: Arc<dyn TurnStates>,
        thread_id: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let state = TurnState::started(thread_id, request_id, 0, now);
        let mut mirror = Self {
            store,
            transcripts: None,
            agent_id: None,
            state,
            turn_completed: false,
            next_seq: 0,
            next_tool_seq: 0,
        };
        mirror.flush();
        mirror
    }

    /// Carries an interrupted turn's partial answer into the root transcript
    /// `transcripts` resolves for the thread, scoped to `agent_id` when given.
    #[must_use]
    pub fn with_transcripts(
        mut self,
        transcripts: Arc<dyn TranscriptLocator>,
        agent_id: Option<String>,
    ) -> Self {
        self.transcripts = Some(transcripts);
        self.agent_id = agent_id;
        self
    }

    /// The current in-memory snapshot.
    pub fn snapshot(&self) -> &TurnState {
        &self.state
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
