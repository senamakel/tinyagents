//! Lossless durable transcript histories.
//!
//! This module deliberately exposes [`TranscriptMessage`] rather than a
//! provider or harness message. A transcript is an on-disk compatibility
//! boundary: it must preserve native tool calls, malformed raw arguments,
//! usage, thinking content, provider extensions and caller-owned metadata.
//! Converting it through a narrower runtime message here would make a later
//! replay silently lossy. Hosts perform any runtime conversion explicitly at
//! their own boundary.
//!
//! A history handle is bound to a transcript file. Thread and agent discovery
//! belong to [`TranscriptLocator`], because one thread can have several
//! transcript stems (for example a root agent and sub-agents).
//!
//! Every mutation is append-only: a reduced logical context is represented by
//! a `{"kind":"compaction","replacement":[…]}` record, never a destructive
//! rewrite. [`TranscriptHistory::clear`] is therefore an empty compaction.
//!

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::transcript::types::TranscriptMessage;

use crate::transcript::{
    SessionAdoption, SessionRef, SessionTranscript, TranscriptMeta, TurnUsage,
    adopt_legacy_session_transcripts, find_latest_transcript, find_root_transcript_for_thread,
    find_root_transcript_for_thread_scoped, read_transcript, resolve_keyed_transcript_path,
    session_stem,
};

/// Upper bound on the compaction generations one session may accumulate.
///
/// Generation resolution probes `{stem}`, `{stem}.g1`, `{stem}.g2` … on disk
/// rather than consulting an index, so it needs a stop condition that holds
/// even if something in the directory is unexpected. A conversation that
/// compacts more than this many times has other problems.
const MAX_GENERATIONS: u32 = 4096;

/// One turn's worth of transcript write, borrowed.
///
/// The fields mirror the transcript writer's turn-append argument list one-for-one and
/// in order, so [`TranscriptHistory::append_turn`]'s forwarding is visually
/// checkable against the format's own signature. Nothing is transformed on the
/// way through; that is the entire correctness claim of this seam and
/// `append_turn_is_byte_identical_to_the_free_function` in the tests pins it.
///
/// `prev` is a field rather than handle state on purpose: the turn path tracks
/// the previously-persisted logical set in memory on `Agent`
/// (`persisted_transcript_messages`) precisely so it never has to re-read a
/// growing file, and a disk re-read is not a faithful substitute — see
/// `FileTranscriptHistory::write_logical_set_locked`.
pub struct TranscriptTurn<'a> {
    /// Logical message set already persisted, for the extension-vs-compaction diff.
    pub prev: &'a [TranscriptMessage],
    /// Logical message set after this turn.
    pub next: &'a [TranscriptMessage],
    /// `_meta` header to append after this turn's lines.
    pub meta: &'a TranscriptMeta,
    /// Usage + provenance attributed to the turn's last assistant row.
    pub turn_usage: Option<&'a TurnUsage>,
    /// Caller-provided request id, stamped on every line of the turn.
    pub request_id: Option<&'a str>,
    /// Tool declarations this ordinary turn was sent with. `None` records
    /// nothing and leaves the previous record in force (for exact-tool turns).
    pub tools: Option<&'a serde_json::Value>,
}

/// Display-only content produced before a turn stopped without a final answer.
///
/// This deliberately uses transcript-neutral fields. It is never added to a
/// model-context replay: the file writer records it as an interrupted message
/// line for the display projection only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptPartial {
    /// Visible assistant text accumulated before the interruption.
    pub content: String,
    /// Optional provider reasoning text associated with the partial.
    pub reasoning_content: Option<String>,
    /// Optional one-based engine iteration associated with the partial.
    pub iteration: Option<u32>,
}

impl TranscriptPartial {
    /// Creates a display-only partial with no provider-specific metadata.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            reasoning_content: None,
            iteration: None,
        }
    }
}

/// The seam a host turn path holds as `Arc<dyn TranscriptHistory>`.
///
/// `append_turn` is deliberately **sync**: `persist_session_transcript` is a
/// sync `&mut self` method and the whole write chain under it is sync, so an
/// async method here would ripple `.await` through the turn loop for no gain.
pub trait TranscriptHistory: TranscriptRead {
    /// Appends one turn, forwarding every argument to the format owner.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()>;

    /// Appends the logical turn and its optional display-only partial as one
    /// history operation.
    ///
    /// Implementors that cannot make the combined mutation atomic must reject
    /// a partial rather than persist either half. Existing implementors which
    /// only support logical turns remain source-compatible through this default.
    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        if partial.is_some() {
            anyhow::bail!("transcript history does not support atomic display partials");
        }
        self.append_turn(turn)
    }

    /// Returns the lossless model-context replay of this transcript.
    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>>;

    /// Appends one durable message while preserving all of its fields.
    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()>;

    /// Replaces the logical model context by appending a compaction record.
    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()>;

    /// Clears the logical model context by appending an empty compaction.
    fn clear(&self) -> anyhow::Result<()>;
}

/// The read half of a bound transcript — the seam the turn path's two resume
/// reads hold.
///
/// Split out of [`TranscriptHistory`] rather than added as one more method on it,
/// for a reason that is not stylistic: a *discovered* transcript can still be a
/// legacy `.md` file (see [`FileTranscriptHistory::opened_at`]), and
/// `append_transcript_turn` writes JSONL. Handing discovery results out as
/// `Arc<dyn TranscriptRead>` makes it impossible to `append_turn` into
/// one by construction, instead of by convention.
///
/// Sync for the same reason [`TranscriptHistory::append_turn`] is: both callers are
/// sync `&mut self` methods on `Agent`.
pub trait TranscriptRead: Send + Sync {
    /// The transcript file this handle is bound to.
    ///
    /// The turn path still needs the concrete path after the read:
    /// `maybe_shadow_read_session_store` takes `&Path`, and the dual-write
    /// mirror derives its record key from `file_stem()`.
    fn path(&self) -> &Path;

    /// The model-context replay of this transcript, `_meta` included, or
    /// `Ok(None)` when the file does not exist.
    ///
    /// Exactly [`read_transcript`], so compaction records have already replaced
    /// the accumulator and `interrupted: true` partials are already skipped —
    /// §3.1's "single most important constraint". Returning the whole
    /// [`SessionTranscript`] rather than messages alone is what lets the shadow
    /// read keep working through this seam.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>>;
}

/// Resolves transcripts by the two keys the turn path actually has, and binds
/// this session's own write handle.
///
/// One injected object covers the whole turn path: both resume reads and the
/// first-write bind. A host holds it as `Option<Arc<dyn TranscriptLocator>>`
/// and falls back to [`FileTranscriptLocator`] built from the *current*
/// `workspace_dir` — lazily, never frozen at build time,
/// because tests reassign `agent.workspace_dir` after `build()` and a
/// build-time locator would silently keep pointing at the old directory.
pub trait TranscriptLocator: Send + Sync {
    /// The durable destination this locator addresses, when it can name one.
    ///
    /// Two locators with equal, `Some` keys resolve every lookup and every
    /// bind to the same place, so a caller comparing bindings may treat them
    /// as interchangeable however they were allocated. `None` means "cannot
    /// say", and such a locator only ever matches itself.
    ///
    /// This exists because the guidance above tells a host to build the
    /// locator lazily from the *current* `workspace_dir` and never freeze it,
    /// which necessarily yields a fresh `Arc` per call. A caller that
    /// identified a locator by allocation would reject the very hosts that
    /// followed that instruction, so it identifies one by this key instead.
    fn destination_key(&self) -> Option<String> {
        None
    }

    /// Newest transcript for `agent_name` in this session's raw subtree,
    /// including the legacy `session_raw/DDMMYYYY/` + `.md` fallback.
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// Newest **root** transcript whose `_meta.thread_id` matches.
    ///
    /// Root-only on purpose: several transcripts share one thread id (every
    /// sub-agent spawned within it does), so a stem-keyed lookup would be
    /// ambiguous.
    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>>;

    /// [`Self::root_for_thread`], additionally scoped to `agent_id` when
    /// given — see
    /// [`transcript::find_root_transcript_for_thread_scoped`](crate::transcript::find_root_transcript_for_thread_scoped)
    /// for why. Defaults to the unscoped lookup so an implementor that never
    /// serves several distinct agents over the same `thread_id` (this test
    /// double, notably) does not have to know about agent scoping at all.
    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        let _ = agent_id;
        self.root_for_thread(thread_id)
    }

    /// Binds (creating on first write) this session's own write handle for
    /// `stem`, with `seed` used only when no file exists yet.
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>>;

    /// The newest generation of `session` that exists, or `session` itself when
    /// none has been written yet.
    ///
    /// A compaction seals a generation and opens the next
    /// ([`Self::begin_generation`]), so the head is the one a resume must load
    /// and append to. The default walks the successor chain through
    /// [`Self::session_exists`]; an implementor with an index may override it.
    fn head_generation(&self, session: &SessionRef) -> SessionRef {
        let mut head = session.clone();
        if !self.session_exists(&head) {
            return head;
        }
        while head.generation < MAX_GENERATIONS {
            let next = head.next_generation();
            if !self.session_exists(&next) {
                break;
            }
            head = next;
        }
        head
    }

    /// Whether `session` has a transcript on disk.
    fn session_exists(&self, session: &SessionRef) -> bool {
        self.read_session_transcript(session).is_some()
    }

    /// Every generation of `session` that exists, oldest first.
    ///
    /// A compaction seals a generation and opens the next, so a long
    /// conversation is a chain rather than one file. The model reads only the
    /// head ([`Self::head_generation`]); a host rendering or exporting the
    /// conversation wants the whole chain. Empty when nothing is written yet.
    fn session_chain(&self, session: &SessionRef) -> Vec<SessionRef> {
        let mut chain = Vec::new();
        let mut generation = session.first_generation();
        while generation.generation <= MAX_GENERATIONS && self.session_exists(&generation) {
            chain.push(generation.clone());
            generation = generation.next_generation();
        }
        chain
    }

    /// Reads `session`'s transcript, or `None` when it has none yet.
    ///
    /// Unlike [`Self::root_for_thread`] this is an exact lookup, not a
    /// newest-wins scan: one session resolves to one file, in every process and
    /// on every launch.
    ///
    /// Defaults to opening the stem the session names through
    /// [`Self::open_stem`] and reading it back. [`Self::open_stem`] alone is
    /// not sufficient — it binds a handle regardless of whether anything has
    /// ever been written there, so this default has to perform the read and
    /// report `None` unless the transcript actually exists, rather than
    /// reporting a handle for a file that was never created. An implementor
    /// with a cheaper existence check (a path probe, an index) should still
    /// override this.
    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        let stem = session_stem(session);
        let handle = self
            .open_stem(&stem, seed_meta_for_discovered(&stem))
            .ok()?;
        match handle.read_session() {
            Ok(Some(_)) => Some(handle as Arc<dyn TranscriptRead>),
            _ => None,
        }
    }

    /// Binds `session`'s own transcript for reading **and** appending.
    ///
    /// This is the method that closes the bug the whole session identity exists
    /// for: resume reads and the subsequent append address the same file, so a
    /// restart extends the conversation instead of re-materialising it into a
    /// fresh stem and orphaning the original.
    fn open_session(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.open_stem(&session_stem(session), seed)
    }

    /// Folds any pre-identity transcripts of `thread_id` into `session`, once.
    ///
    /// A conversation written before session identity existed is spread across
    /// one or more timestamped stems, of which resume only ever loaded the
    /// newest — so its opening turns became unreachable to the model. This
    /// recovers them the first time the session is resumed. Returns `Ok(None)`
    /// when the session already has a transcript or the thread has no legacy
    /// roots, which makes repeat calls harmless.
    ///
    /// Defaults to doing nothing, for locators that are not file-backed.
    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        let _ = (session, thread_id, seed);
        Ok(None)
    }

    /// Seals `session` and binds its successor generation.
    ///
    /// Called when a turn's logical message set is no longer an extension of
    /// what is persisted — a compaction. Rewriting the sealed file in place
    /// would destroy the replaced turns; instead generation `n` is left
    /// byte-for-byte as it was and generation `n+1` takes the compacted set as
    /// its opening write, recording `n` as its parent. The conversation stays
    /// fully recoverable by walking the chain even though the model only sees
    /// the head.
    ///
    /// The returned handle is bound but empty: the caller writes the retained
    /// set through the ordinary turn path (`prev: &[]`), so usage, request ids
    /// and display partials are recorded exactly as on any other turn.
    ///
    /// Bounded by [`MAX_GENERATIONS`] — the same limit [`Self::head_generation`]
    /// and [`Self::session_chain`] stop probing at. Enforcing it here, at the
    /// only place a new generation is minted, is what keeps those two bounded
    /// scans complete: without it a chain could grow past what they are
    /// willing to walk, leaving its newest generation undiscoverable by resume
    /// and its head silently stuck on a stale, capped-off generation that the
    /// ordinary append path would then go on writing into.
    ///
    /// Defaults to sealing through [`Self::open_session`] and the trait's own
    /// existence check, which is enough for most implementors; a
    /// file-backed locator overrides it only to reuse an already-resolved
    /// path. Kept non-defaulted before this comment existed as a required
    /// method would have broken every external implementor the moment this
    /// method was added — this default is what restores that compatibility.
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        anyhow::ensure!(
            successor.generation <= MAX_GENERATIONS,
            "session {} has reached the {MAX_GENERATIONS}-generation compaction limit; \
             refusing to create generation {}",
            session.session_id(),
            successor.generation
        );
        anyhow::ensure!(
            !self.session_exists(&successor),
            "session generation {} already exists; refusing to overwrite a sealed transcript",
            successor.session_id()
        );

        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        let handle = self.open_session(&successor, meta)?;
        Ok((successor, handle))
    }

    /// Forks `session`'s head generation for edit or regenerate, without
    /// erasing history.
    ///
    /// This is [`Self::begin_generation`]'s compaction move, aimed at a
    /// different caller: instead of a summarizer replacing old turns with a
    /// digest, a host wants to edit a past message or regenerate the last
    /// answer. Both need the exact same guarantee compaction already
    /// provides — the current generation is sealed **untouched** on disk
    /// (nothing is written to it; that is the whole point of never rewriting
    /// a sealed file) and the next generation records it as parent — so this
    /// is built on the same primitive rather than a second, parallel one.
    ///
    /// Reads the current [`Self::head_generation`]'s messages, resolves
    /// `cut` against them, seals that head and opens its successor via
    /// [`Self::begin_generation`], and writes the retained prefix into the
    /// successor with [`TranscriptHistory::replace`] — the identical call a
    /// compaction makes to persist its own replacement set. Because the
    /// successor's parent is the sealed head exactly as `begin_generation`
    /// records it, [`Self::session_chain`] walks both generations, so the
    /// full pre-truncation history stays recoverable even though the model
    /// now reads only the truncated head.
    ///
    /// Returns the new generation's [`SessionRef`], its bound handle (already
    /// carrying the truncated messages), and the truncated messages
    /// themselves for the caller's own use (e.g. re-driving the model on the
    /// retained context).
    ///
    /// Fails if `session` has no transcript yet, or if `cut` is
    /// [`TruncateCut::BeforeMessageId`] naming an id absent from the head
    /// generation — silently falling back to some other cut point would risk
    /// truncating the wrong turn.
    fn truncate_into_next_generation(
        &self,
        session: &SessionRef,
        cut: TruncateCut,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(
        SessionRef,
        Arc<dyn TranscriptHistory>,
        Vec<TranscriptMessage>,
    )> {
        let head = self.head_generation(session);
        let head_read = self.read_session_transcript(&head).ok_or_else(|| {
            anyhow::anyhow!(
                "session {} has no transcript to truncate",
                head.session_id()
            )
        })?;
        let transcript = head_read.read_session()?.ok_or_else(|| {
            anyhow::anyhow!(
                "session {} has no transcript to truncate",
                head.session_id()
            )
        })?;
        let keep = cut.resolve(&transcript.messages)?;
        let truncated = transcript.messages[..keep].to_vec();

        let (successor, handle) = self.begin_generation(&head, seed)?;
        // Same call a compaction makes to persist its own replacement set —
        // see `a_compaction_seals_a_generation_and_leaves_it_untouched`.
        handle.replace(&truncated)?;
        Ok((successor, handle, truncated))
    }
}

/// Where to cut a session's head-generation messages when forking it with
/// [`TranscriptLocator::truncate_into_next_generation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruncateCut {
    /// Keep messages `[0, index)`; drop the message at `index` and everything
    /// after it. Clamped to the message count, so an out-of-range index keeps
    /// every message.
    BeforeIndex(usize),
    /// Keep everything before the message carrying this id. The id must name
    /// a message in the head generation — [`TranscriptMessage::id`] is only
    /// ever set by a host that assigns stable ids, so this is the most
    /// robust key to cut on when the caller has one; unlike an index, it
    /// cannot point at the wrong turn after an earlier truncation shifted
    /// everything else.
    BeforeMessageId(String),
    /// Drop the trailing assistant turn: everything strictly after the last
    /// `role == "user"` message, matching the `role == "assistant"` cutpoint
    /// convention already used across this crate's writer (e.g.
    /// `writer::append_transcript_turn`'s `last_assistant_idx`). Used for
    /// "regenerate the last answer." When there is no user message at all,
    /// every message is dropped.
    LastAssistantTurn,
}

impl TruncateCut {
    /// Resolves this cut to a keep-count (`messages[..keep]` survives)
    /// against the head generation's `messages`.
    fn resolve(&self, messages: &[TranscriptMessage]) -> anyhow::Result<usize> {
        match self {
            TruncateCut::BeforeIndex(index) => Ok((*index).min(messages.len())),
            TruncateCut::BeforeMessageId(id) => messages
                .iter()
                .position(|message| message.id.as_deref() == Some(id.as_str()))
                .ok_or_else(|| anyhow::anyhow!("no message with id `{id}` in the head generation")),
            TruncateCut::LastAssistantTurn => Ok(messages
                .iter()
                .rposition(|message| message.role == "user")
                .map(|index| index + 1)
                .unwrap_or(0)),
        }
    }
}

/// The default [`TranscriptLocator`]: real files under
/// `{workspace_dir}/session_raw`.
///
/// Thin by design — each method wraps exactly one `transcript::` free function
/// and changes nothing about it, so swapping the turn path onto the locator is
/// behaviour-preserving.
pub struct FileTranscriptLocator {
    workspace_dir: PathBuf,
}

impl FileTranscriptLocator {
    /// Builds a locator rooted at `workspace_dir` (i.e. it resolves
    /// `{workspace_dir}/session_raw/...`).
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }
}

impl TranscriptLocator for FileTranscriptLocator {
    /// The workspace root every lookup and bind resolves under, which is
    /// this locator's only field and therefore its whole identity.
    fn destination_key(&self) -> Option<String> {
        Some(self.workspace_dir.to_string_lossy().into_owned())
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_latest_transcript(&self.workspace_dir, agent_name)?;
        tracing::debug!(
            "[transcript-history] locator latest_for_agent agent={agent_name} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(agent_name),
        )))
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        let path = find_root_transcript_for_thread(&self.workspace_dir, thread_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread thread={thread_id} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        // Same cross-dir, newest-wins scan as `root_for_thread`, additionally
        // filtered on `_meta.agent_id` so one runtime agent's resume cannot
        // pick up a different agent's transcript for a caller-reused
        // `thread_id` — see `find_root_transcript_for_thread_scoped`.
        let path =
            find_root_transcript_for_thread_scoped(&self.workspace_dir, thread_id, agent_id)?;
        tracing::debug!(
            "[transcript-history] locator root_for_thread_scoped thread={thread_id} \
             agent_id={agent_id:?} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(thread_id),
        )))
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(Arc::new(FileTranscriptHistory::new(
            &self.workspace_dir,
            stem,
            seed,
        )?))
    }

    fn session_exists(&self, session: &SessionRef) -> bool {
        // A direct path probe, not a read: `head_generation` calls this once
        // per generation and only needs to know whether the file is there.
        // `is_file()` rather than `exists()`: a directory, FIFO or other
        // non-regular entry occupying the canonical path must not be
        // reported as an existing generation — reads/appends against it
        // would fail (or, for a directory, silently target the wrong thing)
        // downstream, and `head_generation`'s chain walk would stop at a
        // phantom "generation" that was never actually written.
        resolve_keyed_transcript_path(&self.workspace_dir, &session_stem(session))
            .is_ok_and(|path| path.is_file())
    }

    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        let stem = session_stem(session);
        let path = resolve_keyed_transcript_path(&self.workspace_dir, &stem).ok()?;
        if !path.is_file() {
            return None;
        }
        tracing::debug!(
            "[transcript-history] locator read_session session={stem} path={}",
            path.display()
        );
        Some(Arc::new(FileTranscriptHistory::opened_at(
            path,
            seed_meta_for_discovered(&stem),
        )))
    }

    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        adopt_legacy_session_transcripts(&self.workspace_dir, session, thread_id, seed)
    }

    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        anyhow::ensure!(
            successor.generation <= MAX_GENERATIONS,
            "session {} has reached the {MAX_GENERATIONS}-generation compaction limit; \
             refusing to create generation {}",
            session.session_id(),
            successor.generation
        );
        let stem = session_stem(&successor);
        let path = resolve_keyed_transcript_path(&self.workspace_dir, &stem)?;
        anyhow::ensure!(
            !path.exists(),
            "session generation {stem} already exists; refusing to overwrite a sealed transcript"
        );

        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        tracing::info!(
            "[transcript-history] sealed session={} and opened generation {} at {}",
            session.session_id(),
            successor.generation,
            path.display()
        );
        Ok((
            successor,
            Arc::new(FileTranscriptHistory::new(
                &self.workspace_dir,
                &stem,
                meta,
            )?),
        ))
    }
}

/// A placeholder `_meta` for a handle bound to an already-existing transcript.
///
/// `seed_meta` is consulted only when the file is **absent**, and a discovered
/// path exists by definition, so this value is never written. It exists because
/// [`FileTranscriptHistory`] is one type serving both roles; giving read-only
/// handles a `None` meta would mean an `Option` field every write path then has
/// to unwrap for no benefit.
fn seed_meta_for_discovered(agent_name: &str) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: agent_name.to_string(),
        agent_id: None,
        agent_type: None,
        dispatcher: String::new(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// A lossless history backed by one `session_raw/{stem}.jsonl` transcript.
///
/// Construct with [`FileTranscriptHistory::new`] (workspace-rooted, i.e.
/// `{workspace}/session_raw/`). The `seed_meta` is used only when the
/// transcript file does not exist yet; for an existing file the authoritative
/// cumulative `_meta` is read back from disk so turn counts and token rollups
/// keep accumulating rather than resetting.
pub struct FileTranscriptHistory {
    /// Fully-resolved transcript file, fixed at construction.
    ///
    /// Resolved eagerly rather than derived per call from a `(workspace, stem)`
    /// pair: the old shape hardcoded `{workspace}/session_raw/`, which is the
    /// **wrong directory** for a canonical session and would have silently
    /// cross-written into the canonical session's transcripts the moment this
    /// handle was wired into the turn path.
    path: PathBuf,
    /// `_meta` used for the very first write, before a file exists.
    seed_meta: TranscriptMeta,
}

impl FileTranscriptHistory {
    /// Binds a history handle to `{workspace_dir}/session_raw/{stem}.jsonl`.
    ///
    pub fn new(
        workspace_dir: impl AsRef<Path>,
        stem: &str,
        seed_meta: TranscriptMeta,
    ) -> anyhow::Result<Self> {
        let path = resolve_keyed_transcript_path(workspace_dir.as_ref(), stem)?;
        tracing::debug!(
            "[transcript-history] bound stem={stem} path={}",
            path.display()
        );
        Ok(Self { path, seed_meta })
    }

    /// Binds a handle to an **already-discovered** transcript file, verbatim.
    ///
    /// Deliberately does **not** go through `resolve_keyed_transcript_path*`,
    /// which the two stem constructors above use. That helper `create_dir_all`s
    /// its parent and forces a `.jsonl` extension — both wrong for a discovered
    /// path: `find_latest_transcript` can still return a legacy `.md`
    /// file (`read_transcript` routes by extension), and re-resolving would
    /// mangle it into a sibling `.jsonl` that does not exist while creating
    /// stray directories on a pure read.
    ///
    /// Hand the result out as `Arc<dyn TranscriptRead>`, not
    /// `Arc<dyn TranscriptHistory>` — see [`TranscriptRead`]'s doc.
    pub fn opened_at(path: PathBuf, seed_meta: TranscriptMeta) -> Self {
        tracing::debug!(
            "[transcript-history] opened discovered path={}",
            path.display()
        );
        Self { path, seed_meta }
    }

    /// This handle's transcript file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the current transcript, or `None` when no file exists yet.
    ///
    /// A missing transcript is the normal first-turn state, not an error.
    fn read(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            return Ok(None);
        }
        read_transcript(&self.path).map(Some)
    }

    /// The logical (model-context) message set currently on disk.
    ///
    /// Routes through [`read_transcript`], so compaction records have already
    /// replaced the accumulator and `interrupted: true` partials are skipped.
    fn persisted(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self.read()?.map(|t| t.messages).unwrap_or_default())
    }

    /// The `_meta` to write: the file's own cumulative meta when it exists,
    /// otherwise this handle's seed.
    ///
    /// Uses the existing durable metadata as a default when a caller does not
    /// caller-computed meta. The **turn path must never route through here** —
    /// it computes `turn_count` and the four token/cost rollups fresh each turn,
    /// and re-reading the file's `_meta` would freeze them at the previous
    /// turn's values, silently breaking `read_thread_usage_summary`.
    fn meta_for_write(&self) -> anyhow::Result<TranscriptMeta> {
        Ok(self
            .read()?
            .map(|t| t.meta)
            .unwrap_or_else(|| self.seed_meta.clone()))
    }
}

/// A process-wide, per-path mutex serializing the read-modify-write sequence
/// [`FileTranscriptHistory::append`]/`replace`/`clear` run against one file.
///
/// [`SessionRef`]'s own doc names this as a supported shape: two cores in one
/// process sharing a workspace should both see and extend one conversation.
/// Without this, two `FileTranscriptHistory` instances bound to the same
/// path (a legitimate, common way to get there — `open_session` is called
/// fresh per `Session::resume`) can each read the file's current content,
/// compute a diff against that now-stale view, and write. Whichever finishes
/// its own read first computes a `next` that does not extend what the file
/// looks like by the time it *writes* — `append_transcript_turn_with_partial`
/// then reads that mismatch as "the context was reduced" and appends a
/// **compaction record** instead of a plain tail, and a compaction's
/// replacement value is what canonical reads return going forward. The
/// other write's whole contribution becomes unreachable, even though its
/// bytes are still physically on disk as a now-superseded line — a silent
/// lost update, not a crash.
///
/// Keyed by path rather than by `Arc<Mutex<_>>` identity because the two
/// racing instances are typically *separate* `FileTranscriptHistory` values,
/// not a shared handle. Entries are [`Weak`] and swept opportunistically so
/// the registry does not grow for the lifetime of a long-running host: once
/// every in-flight critical section for a path finishes, nothing keeps that
/// path's entry alive, and the next unrelated call reclaims the slot.
fn path_lock(path: &Path) -> Arc<Mutex<()>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();
    let registry = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.retain(|_, weak| weak.strong_count() > 0);
    if let Some(existing) = locks.get(path).and_then(Weak::upgrade) {
        return existing;
    }
    let fresh = Arc::new(Mutex::new(()));
    locks.insert(path.to_path_buf(), Arc::downgrade(&fresh));
    fresh
}

impl TranscriptRead for FileTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    /// Same call the free-function readers make, on the same path, with the
    /// same return type — so there is nothing left for the round trip to lose.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        if !self.path.exists() {
            tracing::debug!(
                "[transcript-history] read_session absent path={}",
                self.path.display()
            );
            return Ok(None);
        }
        let session = read_transcript(&self.path)?;
        tracing::debug!(
            "[transcript-history] read_session messages={} path={}",
            session.messages.len(),
            self.path.display()
        );
        Ok(Some(session))
    }
}

impl FileTranscriptHistory {
    /// The actual `append_turn` write. Assumes the caller already holds
    /// [`path_lock`] for [`Self::path`] — never call this directly; every
    /// public entry point below acquires the lock once and then routes
    /// through here (and [`Self::append_turn_with_partial_locked`]) so the
    /// lock is taken exactly once per call, never nested (this crate's
    /// `Mutex` is not reentrant).
    fn append_turn_locked(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        tracing::debug!(
            "[transcript-history] append_turn prev={} next={} usage={} request_id={:?} path={}",
            turn.prev.len(),
            turn.next.len(),
            turn.turn_usage.is_some(),
            turn.request_id,
            self.path.display()
        );
        crate::transcript::writer::append_transcript_turn_with_extras(
            &self.path,
            turn.prev,
            turn.next,
            turn.meta,
            turn.turn_usage,
            turn.request_id,
            crate::transcript::writer::AppendTranscriptExtras {
                partial: None,
                tools: turn.tools,
            },
        )?;
        Ok(())
    }

    /// [`Self::append_turn_locked`]'s counterpart for the display-partial
    /// variant. Same locking contract.
    fn append_turn_with_partial_locked(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        tracing::debug!(
            "[transcript-history] append_turn_with_partial prev={} next={} partial={} path={}",
            turn.prev.len(),
            turn.next.len(),
            partial.is_some(),
            self.path.display()
        );
        crate::transcript::writer::append_transcript_turn_with_extras(
            &self.path,
            turn.prev,
            turn.next,
            turn.meta,
            turn.turn_usage,
            turn.request_id,
            crate::transcript::writer::AppendTranscriptExtras {
                partial,
                tools: turn.tools,
            },
        )?;
        Ok(())
    }

    /// Writes `next` as the new logical set, diffing against what is
    /// persisted. Assumes the caller already holds [`path_lock`] for
    /// [`Self::path`] — see [`Self::append_turn_locked`]'s doc for why.
    ///
    /// Routes through [`Self::append_turn_locked`] so every write in this
    /// module — trait-driven and turn-path alike — funnels through one call
    /// to [`append_transcript_turn`], and the extension-vs-compaction
    /// decision stays with the format owner rather than drifting here.
    ///
    /// The `self.persisted()` disk re-read is what the generic trait path has
    /// to do, and is deliberately **not** what the turn path does.
    /// [`read_transcript`] reconstructs `TranscriptMessage`s from line records: the
    /// `failure` / `failure_detail` fields have been lifted out of
    /// `extra_metadata` and turn-usage fields hoisted to top-level line fields.
    /// Feeding that back in as `prev` would make `common_prefix_len` mismatch
    /// at the first such message, so the writer would emit a full compaction
    /// record — re-appending the entire message set — on every single turn.
    fn write_logical_set_locked(&self, next: &[TranscriptMessage]) -> anyhow::Result<()> {
        let prev = self.persisted()?;
        let meta = self.meta_for_write()?;
        self.append_turn_locked(TranscriptTurn {
            prev: &prev,
            next,
            meta: &meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
    }
}

impl TranscriptHistory for FileTranscriptHistory {
    /// Pure forwarder: every argument reaches the transcript writer's turn append
    /// untouched, so the bytes this writes are identical to what the free
    /// function would have written at the call site.
    ///
    /// This — not [`TranscriptHistory::append`] — is the turn path's own
    /// write call (`Session::persist` in `tinyagents-runtime` calls
    /// [`TranscriptHistory::append_turn_with_partial`] directly), so the
    /// same [`path_lock`] serialization `append`/`replace`/`clear` need
    /// applies here too: two `FileTranscriptHistory` handles bound to the
    /// same successor generation (two compactions racing on
    /// `TranscriptLocator::begin_generation` for one session) would
    /// otherwise both see the file absent and both take the writer's
    /// create-fresh path, and whichever `fs::write` lands last would
    /// silently discard the other's retained set.
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.append_turn_locked(turn)
    }

    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.append_turn_with_partial_locked(turn, partial)
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        self.persisted()
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut next = self.persisted()?;
        next.push(message);
        self.write_logical_set_locked(&next)
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        self.write_logical_set_locked(messages)
    }

    fn clear(&self) -> anyhow::Result<()> {
        let lock = path_lock(&self.path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !self.path.exists() {
            return Ok(());
        }
        self.write_logical_set_locked(&[])
    }
}
