//! Test fixtures: classic `session_raw/*.jsonl` files read into an in-memory
//! locator. The file reader is used here only to seed view fixtures.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use super::{
    DisplayRecord, DisplaySessionTranscript, SessionRef, SessionTranscript, ThreadTranscripts,
    TranscriptHistory, TranscriptLocator, TranscriptMeta, TranscriptRead, TranscriptTurn,
    read_transcript_display,
};
use crate::port::SessionStoreProvider;

/// An empty driver-backed locator (the production store, over memory).
pub(crate) fn driver_locator() -> Arc<dyn TranscriptLocator> {
    crate::testkit::in_memory_transcript()
}

/// A fixture locator holding parsed transcript rows from `<workspace>`.
pub(crate) fn imported_locator(workspace: &Path) -> Arc<dyn TranscriptLocator> {
    fn visit(directory: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path, files)?;
            } else if path
                .extension()
                .is_some_and(|extension| extension == "jsonl")
            {
                files.push(path);
            }
        }
        Ok(())
    }

    let locator = crate::port::InMemorySessionStores::new()
        .for_agent("fixture")
        .transcripts;
    let raw = workspace.join("session_raw");
    let mut files = Vec::new();
    let mut displays = HashMap::new();
    if raw.exists() {
        visit(&raw, &mut files).expect("walk transcript fixtures");
    }
    files.sort();
    for path in files {
        let transcript = read_transcript_display(&path).expect("read transcript fixture");
        let Some(stem) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".jsonl"))
        else {
            continue;
        };
        displays.insert(stem.to_owned(), transcript.clone());
        let history = locator
            .open_stem(stem, transcript.meta.clone())
            .expect("open fixture transcript");
        let mut previous = Vec::new();
        for record in transcript.records {
            let (next, request_id) = match record {
                DisplayRecord::Message(message) if message.interrupted => continue,
                DisplayRecord::Message(message) => {
                    let mut next = previous.clone();
                    next.push(message.message);
                    (next, message.request_id)
                }
                DisplayRecord::Compaction(marker) => (
                    marker
                        .replacement
                        .into_iter()
                        .map(|message| message.message)
                        .collect(),
                    marker.request_id,
                ),
            };
            history
                .append_turn(TranscriptTurn {
                    prev: &previous,
                    next: &next,
                    meta: &transcript.meta,
                    turn_usage: None,
                    request_id: request_id.as_deref(),
                    tools: None,
                })
                .expect("seed fixture transcript record");
            previous = next;
        }
    }
    Arc::new(DisplayLocator {
        inner: locator,
        displays,
    })
}

/// [`thread_spend_in`](super::spend::thread_spend_in) over the fixtures
/// imported from `workspace`.
pub(crate) fn thread_spend(workspace: &Path, thread_id: &str) -> super::spend::ThreadSpend {
    super::spend::thread_spend_in(imported_locator(workspace).as_ref(), thread_id)
}

struct DisplayRead {
    inner: Arc<dyn TranscriptRead>,
    display: Option<DisplaySessionTranscript>,
}

impl TranscriptRead for DisplayRead {
    fn path(&self) -> &Path {
        self.inner.path()
    }

    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        self.inner.read_session()
    }

    fn read_display(&self) -> anyhow::Result<Option<DisplaySessionTranscript>> {
        match &self.display {
            Some(display) => Ok(Some(display.clone())),
            None => self.inner.read_display(),
        }
    }
}

struct DisplayLocator {
    inner: Arc<dyn TranscriptLocator>,
    displays: HashMap<String, DisplaySessionTranscript>,
}

impl DisplayLocator {
    fn wrap(&self, transcript: Arc<dyn TranscriptRead>) -> Arc<dyn TranscriptRead> {
        let stem = super::history::stem_of_handle(transcript.path());
        Arc::new(DisplayRead {
            inner: transcript,
            display: self.displays.get(&stem).cloned(),
        })
    }

    fn wrap_listing(&self, mut listed: ThreadTranscripts) -> ThreadTranscripts {
        for item in listed.roots.iter_mut().chain(&mut listed.subagents) {
            item.transcript = self.wrap(Arc::clone(&item.transcript));
        }
        listed
    }
}

impl TranscriptLocator for DisplayLocator {
    fn destination_key(&self) -> Option<String> {
        self.inner.destination_key()
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.inner
            .latest_for_agent(agent_name)
            .map(|transcript| self.wrap(transcript))
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.inner
            .root_for_thread(thread_id)
            .map(|transcript| self.wrap(transcript))
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        self.inner
            .root_for_thread_scoped(thread_id, agent_id)
            .map(|transcript| self.wrap(transcript))
    }

    fn thread_transcripts(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<ThreadTranscripts> {
        self.inner
            .thread_transcripts(thread_id, agent_id)
            .map(|listed| self.wrap_listing(listed))
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.inner.open_stem(stem, seed)
    }

    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        self.inner
            .read_session_transcript(session)
            .map(|transcript| self.wrap(transcript))
    }
}
