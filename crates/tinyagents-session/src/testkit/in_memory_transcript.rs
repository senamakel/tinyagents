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
    meta: Mutex<TranscriptMeta>,
    messages: Mutex<Vec<TranscriptMessage>>,
    tools: Mutex<Option<serde_json::Value>>,
    /// `None` until the first write, mirroring a file that does not exist yet.
    written: Mutex<bool>,
}

impl InMemoryTranscriptHistory {
    /// Creates a fresh, empty history. `label` only affects the diagnostic
    /// [`TranscriptRead::path`] value (`memory://{label}`); it is never used
    /// for lookup.
    pub fn new(label: impl Into<String>, seed_meta: TranscriptMeta) -> Self {
        Self {
            path: PathBuf::from(format!("memory://{}", label.into())),
            meta: Mutex::new(seed_meta),
            messages: Mutex::new(Vec::new()),
            tools: Mutex::new(None),
            written: Mutex::new(false),
        }
    }

    fn mark_written(&self) {
        *self.written.lock().unwrap_or_else(|e| e.into_inner()) = true;
    }
}

impl TranscriptRead for InMemoryTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !*self.written.lock().unwrap_or_else(|e| e.into_inner()) {
            return Ok(None);
        }
        Ok(Some(SessionTranscript {
            meta: self.meta.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            messages: self
                .messages
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            tools: self.tools.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }))
    }
}

impl TranscriptHistory for InMemoryTranscriptHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        *self.messages.lock().unwrap_or_else(|e| e.into_inner()) = turn.next.to_vec();
        *self.meta.lock().unwrap_or_else(|e| e.into_inner()) = turn.meta.clone();
        if let Some(tools) = turn.tools {
            *self.tools.lock().unwrap_or_else(|e| e.into_inner()) = Some(tools.clone());
        }
        self.mark_written();
        Ok(())
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self
            .messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone())
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(message);
        self.mark_written();
        Ok(())
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        *self.messages.lock().unwrap_or_else(|e| e.into_inner()) = messages.to_vec();
        self.mark_written();
        Ok(())
    }

    fn clear(&self) -> anyhow::Result<()> {
        if !*self.written.lock().unwrap_or_else(|e| e.into_inner()) {
            return Ok(());
        }
        self.messages
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Ok(())
    }
}
