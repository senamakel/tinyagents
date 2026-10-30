//! An in-memory [`TranscriptHistory`] double.
//!
//! The driver-backed history ([`crate::port::DriverTranscriptHistory`]) is the
//! production implementation of the trait; this is a second, independent
//! one so [`TranscriptHistory`]'s conformance suite
//! ([`super::conformance::transcript_history_conformance`]) actually certifies
//! the *contract*, not one backend's accidental behavior, and so a test that
//! only needs the trait's semantics does not have to touch a filesystem.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::transcript::{
    DisplayRecord, DisplaySessionTranscript, SessionTranscript, TranscriptHistory,
    TranscriptMessage, TranscriptMeta, TranscriptPartial, TranscriptRead, TranscriptTurn,
    display_message_of, partial_display_message,
};

/// A [`TranscriptHistory`] backed by a `Vec<TranscriptMessage>` behind a
/// [`Mutex`], with no filesystem I/O at all.
///
/// Mirrors the driver-backed history's observable contract:
///
/// * [`TranscriptRead::read_session`] returns `Ok(None)` until the first
///   write (matching "the file does not exist yet"), and `Some` — including an
///   empty message list — after that.
/// * [`TranscriptHistory::append_turn`] replaces the logical set with
///   `turn.next` regardless of `turn.prev`: the file backend's own
///   extension-vs-compaction diff is an on-disk byte-thriftiness optimization,
///   not part of the *logical* contract — [`TranscriptHistory::messages`]
///   after `append_turn` always equals `turn.next` either way.
pub struct InMemoryTranscriptHistory {
    path: PathBuf,
    state: Mutex<InMemoryTranscriptState>,
    generation_gate: Arc<Mutex<()>>,
}

/// The complete logical transcript state. Keeping it behind one mutex makes a
/// turn update observable as one transition, just like the file backend.
struct InMemoryTranscriptState {
    meta: TranscriptMeta,
    messages: Vec<TranscriptMessage>,
    tools: Option<serde_json::Value>,
    /// `false` until the first write, mirroring a file that does not exist yet.
    written: bool,
    /// Display-only partials of interrupted turns, with their request ids
    /// and how many logical rows preceded them when recorded (where the
    /// display read places them). Never part of [`TranscriptHistory::messages`].
    partials: Vec<(TranscriptPartial, Option<String>, usize)>,
    sealed: bool,
}

/// Rows as the file backend returns them: a legacy string row is lifted into
/// its typed form on read, so the in-memory logical view does the same on
/// write and the two backends agree.
fn normalized_rows(rows: &[TranscriptMessage]) -> Vec<TranscriptMessage> {
    rows.iter()
        .cloned()
        .map(TranscriptMessage::normalized)
        .collect()
}

impl InMemoryTranscriptHistory {
    /// Creates a fresh, empty history. `label` only affects the diagnostic
    /// [`TranscriptRead::path`] value (`memory://{label}`); it is never used
    /// for lookup.
    pub fn new(label: impl Into<String>, seed_meta: TranscriptMeta) -> Self {
        Self::new_with_gate(label, seed_meta, Arc::new(Mutex::new(())))
    }

    pub(crate) fn new_with_gate(
        label: impl Into<String>,
        seed_meta: TranscriptMeta,
        generation_gate: Arc<Mutex<()>>,
    ) -> Self {
        Self {
            path: PathBuf::from(format!("memory://{}", label.into())),
            state: Mutex::new(InMemoryTranscriptState {
                meta: seed_meta,
                messages: Vec::new(),
                tools: None,
                written: false,
                partials: Vec::new(),
                sealed: false,
            }),
            generation_gate,
        }
    }

    /// Returns the transcript's session ID, if set.
    pub fn session_id(&self) -> Option<String> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .meta
            .session_id
            .clone()
    }

    /// Records the display-only `partial` of an interrupted turn.
    pub fn record_partial(&self, partial: TranscriptPartial, request_id: Option<String>) -> bool {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.record_partial_under_gate(partial, request_id)
    }

    /// Records a partial while the caller already holds the shared generation
    /// gate. This keeps locator selection and the mutation atomic.
    pub(crate) fn record_partial_under_gate(
        &self,
        partial: TranscriptPartial,
        request_id: Option<String>,
    ) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.sealed || partial.content.is_empty() {
            return false;
        }
        let position = state.messages.len();
        state.partials.push((partial, request_id, position));
        state.written = true;
        true
    }

    /// The display-only partials recorded so far, oldest first, with their
    /// request ids.
    pub fn partials(&self) -> Vec<(TranscriptPartial, Option<String>)> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .partials
            .iter()
            .map(|(partial, request_id, _)| (partial.clone(), request_id.clone()))
            .collect()
    }

    pub(crate) fn seal(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).sealed = true;
    }

    pub(crate) fn unseal(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).sealed = false;
    }

    /// Replaces discovery metadata until the first write, when an existence
    /// probe created the in-memory stand-in before the real session was bound.
    pub(crate) fn set_seed_if_unwritten(&self, seed_meta: TranscriptMeta) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.written {
            state.meta = seed_meta;
        }
    }
}

impl TranscriptRead for InMemoryTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.written {
            return Ok(None);
        }
        Ok(Some(SessionTranscript {
            meta: state.meta.clone(),
            messages: state.messages.clone(),
            tools: state.tools.clone(),
        }))
    }

    /// The logical rows with the recorded interrupted partials placed where
    /// they were written, as the file and driver display reads show them.
    /// There is no append log here, so earlier compactions are not replayed.
    fn read_display(&self) -> anyhow::Result<Option<DisplaySessionTranscript>> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.written {
            return Ok(None);
        }
        let mut partials: Vec<_> = state.partials.iter().collect();
        partials.sort_by_key(|(_, _, position)| *position);
        let mut partials = partials.into_iter().peekable();
        let mut records = Vec::new();
        for index in 0..=state.messages.len() {
            while let Some((partial, request_id, _)) =
                partials.next_if(|(_, _, position)| (*position).min(state.messages.len()) == index)
            {
                records.push(DisplayRecord::Message(Box::new(partial_display_message(
                    partial,
                    request_id.as_deref(),
                ))));
            }
            if let Some(message) = state.messages.get(index) {
                records.push(DisplayRecord::Message(Box::new(display_message_of(
                    message,
                ))));
            }
        }
        Ok(Some(DisplaySessionTranscript {
            meta: state.meta.clone(),
            records,
        }))
    }
}

impl TranscriptHistory for InMemoryTranscriptHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        state.messages = normalized_rows(turn.next);
        state.meta = turn.meta.clone();
        // `None` means this logical turn does not replace the last durable
        // snapshot, matching the file writer which emits no tools record.
        if let Some(tools) = turn.tools {
            state.tools = Some(tools.clone());
        }
        state.written = true;
        Ok(())
    }

    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let request_id = turn.request_id.map(str::to_string);
        // One lock for both halves, so the turn and its partial land as one
        // transition, as the trait asks.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        state.messages = normalized_rows(turn.next);
        state.meta = turn.meta.clone();
        if let Some(tools) = turn.tools {
            state.tools = Some(tools.clone());
        }
        if let Some(partial) = partial.filter(|partial| !partial.content.is_empty()) {
            let position = state.messages.len();
            state.partials.push((partial.clone(), request_id, position));
        }
        state.written = true;
        Ok(())
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .messages
            .clone())
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        state.messages.push(message.normalized());
        state.written = true;
        Ok(())
    }

    fn adopt_if_empty(&self, messages: &[TranscriptMessage]) -> anyhow::Result<bool> {
        if messages.is_empty() {
            return Ok(false);
        }
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        if state.written {
            return Ok(false);
        }
        state.messages = normalized_rows(messages);
        state.written = true;
        Ok(true)
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        state.messages = normalized_rows(messages);
        state.written = true;
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        let _gate = self
            .generation_gate
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        anyhow::ensure!(!state.sealed, "transcript generation is sealed");
        if !state.written {
            return Ok(());
        }
        state.messages.clear();
        state.partials.clear();
        Ok(())
    }
}
