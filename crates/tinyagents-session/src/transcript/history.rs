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

use std::path::Path;
use std::sync::Arc;

use crate::transcript::types::TranscriptMessage;

use crate::transcript::{
    DisplaySessionTranscript, SessionAdoption, SessionRef, SessionTranscript, ThreadTranscript,
    ThreadTranscripts, TranscriptMeta, TurnUsage, session_stem,
};

/// The stem a read handle is bound to: its path's last component, without the
/// `.jsonl` extension a file handle carries. (A stem may itself contain dots,
/// so this never uses `file_stem`.)
pub(super) fn stem_of_handle(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.strip_suffix(".jsonl")
        .map(str::to_string)
        .unwrap_or(name)
}

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
/// growing log, and a store re-read is not a faithful substitute.
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
/// for a reason that is not stylistic: handing discovery results out as
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
    /// The log replayed, so compaction records have already replaced
    /// the accumulator and `interrupted: true` partials are already skipped —
    /// §3.1's "single most important constraint". Returning the whole
    /// [`SessionTranscript`] rather than messages alone is what lets the shadow
    /// read keep working through this seam.
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>>;

    /// The display projection of this transcript — every record, including
    /// pre-compaction history, compaction markers and interrupted partials —
    /// or `Ok(None)` when it does not exist.
    ///
    /// The default builds it from [`Self::read_session`], which holds only
    /// the reduced model context: an implementor that keeps more (the file
    /// log, a driver's entry log) overrides it.
    fn read_display(&self) -> anyhow::Result<Option<DisplaySessionTranscript>> {
        Ok(self.read_session()?.map(super::display_from_session))
    }
}

/// Resolves transcripts by the two keys the turn path actually has, and binds
/// this session's own write handle.
///
/// One injected object covers the whole turn path: both resume reads and the
/// first-write bind. A host holds it as `Option<Arc<dyn TranscriptLocator>>`,
/// built from the store it currently targets rather than frozen at build time,
/// so a host that rebuilds its locator per turn still resolves the same
/// destination (see [`Self::destination_key`]).
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
    /// given: one caller-supplied thread id can be handed to several
    /// independently configured agents, and each needs its own transcript.
    /// Defaults to the unscoped lookup so an implementor that never
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

    /// Every transcript backing `thread_id`'s view: its root generations in
    /// projection order and its sub-agent transcripts, or `None` when the
    /// thread has no root transcript yet. Scoped to the roots `agent_id` owns
    /// when given (sub-agents are then found by stem only).
    ///
    /// Defaults to [`Self::root_for_thread_scoped`] alone, with no
    /// sub-agents, for a locator that cannot list.
    fn thread_transcripts(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<ThreadTranscripts> {
        let root = self.root_for_thread_scoped(thread_id, agent_id)?;
        Some(ThreadTranscripts {
            roots: vec![ThreadTranscript {
                stem: stem_of_handle(root.path()),
                transcript: root,
            }],
            subagents: Vec::new(),
        })
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

    /// Appends the display-only `partial` of an interrupted turn to the
    /// newest root transcript of `thread_id` (scoped to `agent_id` when
    /// given); returns whether there was one to append to.
    ///
    /// An interrupted **first** turn has no transcript yet, so there is
    /// nothing to append to and this returns `Ok(false)`. The model-context
    /// replay never includes a partial.
    ///
    /// Defaults to `Ok(false)`, for locators that keep no display partials.
    ///
    /// # Errors
    ///
    /// When a transcript exists but cannot be appended to.
    fn append_interrupted_partial(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        let _ = (thread_id, agent_id, partial, request_id);
        Ok(false)
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
    /// Bounded by the maximum generation count — the same limit [`Self::head_generation`]
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
            "session generation limit reached"
        );
        anyhow::ensure!(
            !self.session_exists(&successor),
            "session generation already exists"
        );
        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        let handle = self.open_session(&successor, meta)?;
        Ok((successor, handle))
    }

    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        if let Some(transcript) = self.read_session_transcript(session)
            && let Some(transcript) = transcript.read_session()?
        {
            anyhow::ensure!(
                same_transcript_messages(&transcript.messages, baseline),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        } else {
            anyhow::ensure!(
                baseline.is_empty(),
                "transcript baseline is stale; reload the session before creating a generation"
            );
        }
        self.begin_generation(session, seed)
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

        let (successor, handle) =
            self.begin_generation_from_baseline(&head, seed, &transcript.messages)?;
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

pub(crate) fn same_transcript_messages(
    left: &[TranscriptMessage],
    right: &[TranscriptMessage],
) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.same_row_as(b))
}

/// A placeholder `_meta` for a handle bound to an already-existing transcript.
///
/// `seed_meta` is consulted only when the transcript is **absent**, and a
/// discovered one exists by definition, so this value is never written. One
/// history type serves both roles; giving read-only handles a `None` meta
/// would mean an `Option` field every write path then has to unwrap for no
/// benefit.
pub(super) fn seed_meta_for_discovered(agent_name: &str) -> TranscriptMeta {
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
