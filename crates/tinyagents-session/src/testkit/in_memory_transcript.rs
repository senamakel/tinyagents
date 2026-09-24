//! An in-memory [`TranscriptHistory`] double.
//!
//! [`FileTranscriptHistory`](crate::transcript::FileTranscriptHistory) is the
//! only production implementation of the trait; this is a second, independent
//! one so [`TranscriptHistory`]'s conformance suite
//! ([`super::conformance::transcript_history_conformance`]) actually certifies
//! the *contract*, not one backend's accidental behavior, and so a test that
//! only needs the trait's semantics does not have to touch a filesystem.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::transcript::{
    SessionTranscript, TranscriptHistory, TranscriptMessage, TranscriptMeta, TranscriptRead,
    TranscriptTurn,
};

/// A [`TranscriptHistory`] backed by a `Vec<TranscriptMessage>` behind a
/// [`Mutex`], with no filesystem I/O at all.
///
/// Mirrors [`FileTranscriptHistory`](crate::transcript::FileTranscriptHistory)'s
/// observable contract:
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
}

/// The complete logical transcript state. Keeping it behind one mutex makes a
/// turn update observable as one transition, just like the file backend.
struct InMemoryTranscriptState {
    meta: TranscriptMeta,
    messages: Vec<TranscriptMessage>,
    tools: Option<serde_json::Value>,
    /// `false` until the first write, mirroring a file that does not exist yet.
    written: bool,
}

impl InMemoryTranscriptHistory {
    /// Creates a fresh, empty history. `label` only affects the diagnostic
    /// [`TranscriptRead::path`] value (`memory://{label}`); it is never used
    /// for lookup.
    pub fn new(label: impl Into<String>, seed_meta: TranscriptMeta) -> Self {
        Self {
            path: PathBuf::from(format!("memory://{}", label.into())),
            state: Mutex::new(InMemoryTranscriptState {
                meta: seed_meta,
                messages: Vec::new(),
                tools: None,
                written: false,
            }),
        }
    }

    fn mark_written(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).written = true;
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
}

impl TranscriptHistory for InMemoryTranscriptHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.messages = turn.next.to_vec();
        state.meta = turn.meta.clone();
        // `None` means this logical turn does not replace the last durable
        // snapshot, matching the file writer which emits no tools record.
        if let Some(tools) = turn.tools {
            state.tools = Some(tools.clone());
        }
        state.written = true;
        Ok(())
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self.state.lock().unwrap_or_else(|e| e.into_inner()).messages.clone())
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .messages
            .push(message);
        self.mark_written();
        Ok(())
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).messages = messages.to_vec();
        self.mark_written();
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.written {
            return Ok(());
        }
        state.messages.clear();
        Ok(())
    }
}
