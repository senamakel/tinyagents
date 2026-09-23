use std::{future::Future, sync::Arc};

use tinyagents_harness::CancellationToken;
use tinyagents_session::transcript::{
    SessionRef, TranscriptHistory, TranscriptMessage, TranscriptPartial, TranscriptTurn, TurnUsage,
};
use tinyinference_llm::message::Message;

use crate::{
    CommitReceipt, DriverRequest, PrefixSnapshot, ResumeMode, ResumePreparation, RuntimeError,
    SessionDriver, SessionHooks, SessionResume, SessionStateView, SessionTerminal,
    SessionTurnOutcome, SessionTurnRequest, ToolSnapshot, TranscriptCodec, TranscriptCommitReceipt,
    TranscriptDelta, TranscriptTarget, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

/// Host-neutral mutable state for one conversation session.
pub struct Session<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    prefix: PrefixSnapshot,
    default_tools: ToolSnapshot,
    history: Vec<Message>,
    persisted: Vec<TranscriptMessage>,
    target: Option<TranscriptTarget>,
    transcript: Option<Arc<dyn TranscriptHistory>>,
    committed_turns: usize,
    /// Tool declarations this session last sent, restored from the transcript
    /// on resume and updated after every recorded turn.
    recorded_tools: Option<ToolSnapshot>,
    /// The `tools` record currently in force in the bound transcript file,
    /// used to write a new record only when the declarations change.
    recorded_tools_json: Option<serde_json::Value>,
    retain_recorded_tools: bool,
}

impl<C: Clone + Send + Sync + 'static> Session<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        driver: Arc<dyn SessionDriver<C>>,
        codec: Option<Arc<dyn TranscriptCodec<C>>>,
        hooks: Arc<dyn SessionHooks<C>>,
        prefix: PrefixSnapshot,
        default_tools: ToolSnapshot,
        target: Option<TranscriptTarget>,
    ) -> Self {
        Self {
            driver,
            codec,
            hooks,
            history: prefix.messages().to_vec(),
            prefix,
            default_tools,
            persisted: Vec::new(),
            target,
            transcript: None,
            committed_turns: 0,
            recorded_tools: None,
            recorded_tools_json: None,
            retain_recorded_tools: false,
        }
    }

    pub(crate) fn set_retain_recorded_tools(&mut self, retain: bool) {
        self.retain_recorded_tools = retain;
    }

    /// Tool declarations this session last sent (restored on resume).
    pub fn recorded_tools(&self) -> Option<&ToolSnapshot> {
        self.recorded_tools.as_ref()
    }

    /// Returns the currently committed model history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns the stable prefix currently applied to this session.
    pub fn prefix_snapshot(&self) -> &PrefixSnapshot {
        &self.prefix
    }

    /// Returns the builder compatibility default used only when a preparation
    /// supplies no per-turn tool snapshot.
    pub fn tool_snapshot(&self) -> &ToolSnapshot {
        &self.default_tools
    }

    /// Seeds an uncommitted session from an explicit, lossless host snapshot.
    ///
    /// This replaces neither the host's raw rows nor their metadata. It is the
    /// supported alternative to a host keeping a shadow history beside the
    /// runtime. Seeding after any durable transition is rejected.
    pub fn seed_history(
        &mut self,
        history: Vec<Message>,
        raw: Vec<TranscriptMessage>,
    ) -> Result<(), RuntimeError> {
        if self.committed_turns != 0 {
            return Err(RuntimeError::InvalidSessionState(
                "cannot seed history after a committed turn".into(),
            ));
        }
        self.history = self.with_prefix(history);
        self.persisted = raw;
        Ok(())
    }

    /// Loads the selected durable transcript, retaining its lossless raw rows
    /// as the base for the next append-only delta.
    pub async fn resume(
        &mut self,
        options: &TurnOptions<C>,
    ) -> Result<SessionResume, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let Some(target) = self.target.as_ref() else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        // Captured before the scanned transcript's metadata overwrites
        // `target.meta` below, so a session-bound target resumed through
        // `Thread`/`LatestForAgent` can fall back to its own pre-resume
        // metadata if the write destination turns out not to exist yet —
        // see the re-derivation block near the end of this method.
        let pre_scan_meta = target.meta.clone();
        let mut session_binding: Option<SessionRef> = None;
        let read = match options.resume {
            ResumeMode::Never => None,
            ResumeMode::LatestForAgent => target
                .locator
                .latest_for_agent(target.resume_agent.as_deref().unwrap_or(&target.stem)),
            ResumeMode::Thread => options.thread_id.as_deref().and_then(|thread| {
                target
                    .locator
                    .root_for_thread_scoped(thread, target.meta.agent_id.as_deref())
            }),
            ResumeMode::Session => {
                let Some(session) = options.session.clone().or_else(|| target.session.clone())
                else {
                    return Ok(SessionResume {
                        loaded: false,
                        history: self.history.clone(),
                    });
                };
                // The head generation, not the session the host named: a
                // compaction may have sealed that one and opened a successor,
                // and the head is the conversation the model is continuing.
                let head = target.locator.head_generation(&session);
                let read = target.locator.read_session_transcript(&head);
                if read.is_some() {
                    session_binding = Some(head);
                } else if let Some(thread) = options.thread_id.as_deref() {
                    // Nothing under this identity yet. A conversation written
                    // before session identity existed is spread over one or
                    // more timestamped stems; fold them in once so the model
                    // regains the turns the newest-wins lookup had stranded.
                    // Adoption is best effort: it recovers history that would
                    // otherwise be stranded, but failing to recover it must not
                    // fail the turn the user is waiting on.
                    if let Err(error) = target.locator.adopt_legacy(&session, thread, &target.meta)
                    {
                        tracing::warn!(
                            "[session] legacy adoption failed session={} thread={thread}: {error}",
                            session.session_id()
                        );
                    }
                    session_binding = Some(session.clone());
                }
                match read {
                    Some(read) => Some(read),
                    None => session_binding
                        .as_ref()
                        .and_then(|bound| target.locator.read_session_transcript(bound)),
                }
            }
        };
        let Some(read) = read else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let Some(transcript) = read
            .read_session()
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?
        else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let codec = self
            .codec
            .as_ref()
            .ok_or(RuntimeError::MissingDependency("TranscriptCodec"))?;
        let decoded = codec.decode_history(&transcript)?;
        // The transcript already holds the prefix it was sent with as its
        // leading system rows. A session built without a prefix of its own
        // adopts those rows, so resuming never has to re-render the prompt
        // and the prefix guard below protects the stored one.
        if self.prefix.messages().is_empty() {
            let leading: Vec<Message> = decoded
                .iter()
                .take_while(|message| matches!(message, Message::System(_)))
                .cloned()
                .collect();
            if !leading.is_empty() {
                self.prefix = PrefixSnapshot::new(leading);
            }
        }
        let history = self.with_prefix(decoded);
        self.history = history.clone();
        // Every turn already on disk counts as committed: the prefix those
        // turns were sent with is part of the conversation, in this process
        // or the one that wrote it.
        self.committed_turns = self.committed_turns.max(transcript.meta.turn_count);
        self.recorded_tools_json = transcript.tools.clone();
        self.recorded_tools = match transcript.tools.as_ref() {
            Some(value) => match ToolSnapshot::from_json(value) {
                Ok(tools) => Some(tools),
                Err(error) => {
                    tracing::warn!("[session] ignoring unreadable recorded tools: {error}");
                    None
                }
            },
            None => None,
        };
        tracing::debug!(
            "[session] resumed history={} committed_turns={} recorded_tools={}",
            history.len(),
            self.committed_turns,
            self.recorded_tools
                .as_ref()
                .map_or(0, |tools| tools.specs().len())
        );
        self.persisted = transcript.messages;
        // The discovered metadata, not the builder seed, is authoritative for
        // the subsequent append. This keeps resume-only host fields intact.
        if let Some(target) = self.target.as_mut() {
            target.meta = transcript.meta;
        }
        // A successful explicit resume always rebinds the write handle to the
        // selected transcript. Builder construction itself remains I/O-free.
        //
        // For a session resume the handle must address **the file that was just
        // read**, not the target's original stem. Binding elsewhere is what
        // used to re-materialise a resumed history into a fresh stem and
        // orphan the original, leaving two roots claiming one thread.
        if let (Some(target), Some(head)) = (self.target.as_mut(), session_binding) {
            target.rebind_session(head);
        } else if let Some(target) = self.target.as_mut()
            && let Some(session) = target.session.clone()
        {
            // `session_binding` above is set only on the `ResumeMode::Session`
            // path, so a session-bound target resumed through `Thread` or
            // `LatestForAgent` would otherwise reach the bind below still
            // naming generation 0 — even when an earlier compaction already
            // sealed it and opened a later head. That write would land in a
            // generation the design requires to stay sealed and byte-for-byte
            // unchanged. Resolving the head here, for every mode, is what
            // `persist`'s own equivalent guard (`self.transcript.is_none()`)
            // cannot substitute for: `self.transcript` is bound unconditionally
            // a few lines down, so by the time `persist` runs on this turn
            // that guard has already been satisfied.
            let head = target.locator.head_generation(&session);
            if head != session {
                target.rebind_session(head);
            }
        }
        let target = self.target.as_ref().expect("target checked above");
        self.transcript = Some(match target.session.as_ref() {
            Some(session) => target
                .locator
                .open_session(session, target.meta.clone())
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            None => target
                .locator
                .open_stem(&target.stem, target.meta.clone())
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
        });
        // For a session-bound target, `target.session`/`target.stem` always
        // name the same file (construction and `rebind_session` keep them in
        // lockstep) — the bind above is always that file, regardless of
        // resume mode. Under `ResumeMode::Session`, `read` was already that
        // same file, so `self.persisted` (set above from `transcript`,
        // i.e. from `read`) already matches what this turn will append to.
        // Under `Thread`/`LatestForAgent`, `read` can legitimately be a
        // *different* file — a newest-wins scan recovering history from
        // wherever it exists is exactly their contract — while the destination
        // this turn writes to is still the session's own, separately-tracked
        // file. Using the scan's raw rows as the append-diff baseline for a
        // write that lands elsewhere would corrupt whatever is already on
        // that other file. Re-derive the baseline from the file this turn
        // actually writes to; `self.history` (what the model sees) keeps
        // coming from the scanned `read`, which is the intended recovery
        // behavior for those modes.
        if target.session.is_some() && options.resume != ResumeMode::Session {
            // The scanned file's `_meta` (set a few lines up, from `read`)
            // is equally wrong as an append baseline when `read` was a
            // different file: without this, the destination's next `_meta`
            // record would carry over the scanned file's `agent_id`,
            // `created`, provider/model, token/cost totals and (unless the
            // head changed) session identifiers — none of which describe
            // the file actually being appended to.
            let destination = self
                .transcript
                .as_ref()
                .expect("bound above")
                .read_session()
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
            match destination {
                Some(destination_transcript) => {
                    self.recorded_tools_json = destination_transcript.tools;
                    self.persisted = destination_transcript.messages;
                    if let Some(target) = self.target.as_mut() {
                        target.meta = destination_transcript.meta;
                    }
                }
                None => {
                    // Nothing at the destination yet: fall back to this
                    // target's own pre-resume metadata rather than the
                    // scanned file's, then reapply the session binding so
                    // `session_id`/`parent_session_id` stay canonical for
                    // whatever session this target now names (`resume`'s
                    // own head-resolution above may have rebound it).
                    self.recorded_tools_json = None;
                    self.persisted = Vec::new();
                    if let Some(target) = self.target.as_mut() {
                        target.meta = pre_scan_meta;
                        if let Some(session) = target.session.clone() {
                            target.meta.session_id = Some(session.session_id());
                            target.meta.parent_session_id = session.parent_session_id();
                        }
                    }
                }
            }
        }
        Ok(SessionResume {
            loaded: true,
            history,
        })
    }

    /// Executes and commits one state transition.
    pub async fn turn(
        &mut self,
        mut request: SessionTurnRequest,
        mut options: TurnOptions<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        let mut terminal_guard = TerminalGuard::new(self.hooks.clone());
        let result = self
            .turn_inner(&mut request, &mut options, &mut terminal_guard)
            .await;
        if !terminal_guard.is_committed() {
            let terminal = match &result {
                Ok(outcome) => SessionTerminal::Completed(outcome.clone()),
                Err(RuntimeError::Cancelled) => SessionTerminal::Cancelled,
                Err(error) => SessionTerminal::Failed(error.to_string()),
            };
            terminal_guard.set(terminal);
        }
        // Terminal observation cannot revoke a durable successful commit.
        let _ = terminal_guard.finish().await;
        result
    }

    async fn turn_inner(
        &mut self,
        request: &mut SessionTurnRequest,
        options: &mut TurnOptions<C>,
        terminal_guard: &mut TerminalGuard<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let cancellation = options.cancellation.clone();
        let resume_preparation = cancelable(
            &cancellation,
            self.hooks
                .before_resume(request, options, self.state_view(false)),
        )
        .await?;
        self.apply_resume_preparation(resume_preparation)?;
        let resumed = if options.resume == ResumeMode::Never {
            false
        } else {
            self.resume(options).await?.loaded
        };
        // `resume` is synchronous after its read, so this explicit boundary
        // makes cancellation between loading and before-turn preparation
        // observable without handing work to the driver.
        if options.cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let preparation = cancelable(
            &cancellation,
            self.hooks
                .before_turn(request, options, self.state_view(resumed)),
        )
        .await?;
        let exact_tools = preparation.exact_tools;
        let (tools, prepared_prefix) = self.apply_preparation(preparation)?;
        if let Some(prefix) = prepared_prefix {
            self.apply_prefix(prefix)?;
        }
        let tools = if exact_tools {
            tools
        } else {
            self.retain_recorded(tools)?
        };
        // What this turn records as the session's tools: the set actually
        // sent, unless the host marked the turn's set as one-off.
        let record_tools = (!exact_tools).then(|| tools.clone());

        let mut input = self.history.clone();
        if input.last() != Some(&request.input) {
            input.push(request.input.clone());
        }
        let codec_options = options.transcript_options();
        let request_id = options.request_id.clone();
        let thread_id = options.thread_id.clone();
        let stream = options.stream;
        let cancellation = options.cancellation.clone();
        // `RunContext` is consumed exactly once. The host context captured in
        // `codec_options` is the one after preparation and before handoff.
        let run_context = std::mem::replace(
            &mut options.run_context,
            tinyagents_harness::context::RunContext::new(
                tinyagents_harness::context::RunConfig::new("consumed-session-context"),
                codec_options.context.clone(),
            ),
        )
        .with_cancellation(cancellation.clone());
        let driver_result = tokio::select! {
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
            result = self.driver.execute(DriverRequest { history: input, tools, run_context, stream }) => result,
        };
        let outcome = match driver_result {
            Ok(outcome) => outcome,
            Err(failure) => {
                if let Some(partial) = failure.partial {
                    if cancellation.is_cancelled() {
                        return Err(RuntimeError::Cancelled);
                    }
                    let partial_history = self.with_prefix(partial.history);
                    let raw = self.encode(&self.history, &partial_history, &codec_options)?;
                    let turn_usage = self.turn_usage(&codec_options)?;
                    let receipt = self.persist(
                        &raw,
                        request_id.as_deref(),
                        thread_id.as_deref(),
                        partial.partial.as_ref(),
                        turn_usage.as_ref(),
                        record_tools.as_ref(),
                    )?;
                    self.history = partial_history;
                    self.persisted = raw;
                    if receipt.is_some() {
                        self.committed_turns += 1;
                    }
                }
                return Err(failure.error);
            }
        };
        let candidate = self.with_prefix(outcome.history);
        let committed = SessionTurnOutcome {
            history: candidate.clone(),
            output: outcome.output,
            interrupted: outcome.interrupted,
        };
        cancelable(
            &cancellation,
            self.hooks.before_commit(&committed, &codec_options),
        )
        .await?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let raw = self.encode(&self.history, &candidate, &codec_options)?;
        let turn_usage = self.turn_usage(&codec_options)?;
        let transcript = self.persist(
            &raw,
            request_id.as_deref(),
            thread_id.as_deref(),
            None,
            turn_usage.as_ref(),
            record_tools.as_ref(),
        )?;
        self.history = committed.history.clone();
        self.persisted = raw;
        self.committed_turns += 1;
        // The receipt is constructed only after append and state replacement.
        // Its hook and the completed terminal are owned by one task: errors or
        // cancellation cannot relabel the successful durable transition, and
        // dropping the caller future cannot drop finalization mid-flight.
        let receipt = CommitReceipt {
            outcome: committed.clone(),
            options: codec_options,
            transcript,
        };
        let finalization = terminal_guard.finalize_commit(receipt);
        // This await deliberately does not observe cancellation. If this turn
        // future is dropped, dropping `JoinHandle` detaches rather than aborts
        // the owned finalization task.
        let _ = finalization.await;
        Ok(committed)
    }

    fn apply_preparation(
        &mut self,
        preparation: TurnPreparation,
    ) -> Result<(ToolSnapshot, Option<PrefixSnapshot>), RuntimeError> {
        // A returned snapshot never updates `default_tools`: it applies only
        // to the `DriverRequest` being built by this call.
        Ok((
            preparation
                .tools
                .unwrap_or_else(|| self.default_tools.clone()),
            preparation.prefix,
        ))
    }

    /// Merges back recorded declarations the host did not re-supply, when
    /// retention is on. See [`crate::SessionBuilder::retain_recorded_tools`].
    fn retain_recorded(&self, tools: ToolSnapshot) -> Result<ToolSnapshot, RuntimeError> {
        let Some(recorded) = self.recorded_tools.as_ref().filter(|_| self.retain_recorded_tools)
        else {
            return Ok(tools);
        };
        let (merged, retained) = tools.with_retained(recorded)?;
        if retained != 0 {
            tracing::info!(
                "[session] retained {retained} recorded tool declaration(s) the host did not re-supply (sending {})",
                merged.specs().len()
            );
        }
        Ok(merged)
    }

    fn apply_resume_preparation(
        &mut self,
        preparation: ResumePreparation,
    ) -> Result<(), RuntimeError> {
        if let Some(target) = preparation.transcript {
            if self.transcript.is_some() || self.committed_turns != 0 {
                if !self
                    .target
                    .as_ref()
                    .is_some_and(|bound| bound.same_binding(&target))
                {
                    return Err(RuntimeError::InvalidSessionState(
                        "cannot change a transcript target after it is bound or committed".into(),
                    ));
                }
            } else {
                self.target = Some(target);
            }
        }
        if self.target.is_some() && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        Ok(())
    }

    fn apply_prefix(&mut self, prefix: PrefixSnapshot) -> Result<(), RuntimeError> {
        if prefix == self.prefix {
            return Ok(());
        }
        if self.committed_turns != 0 {
            return Err(RuntimeError::InvalidSessionState(
                "cannot change a session prefix after a committed turn".into(),
            ));
        }
        let history = std::mem::take(&mut self.history);
        let history = history
            .strip_prefix(self.prefix.messages())
            .unwrap_or(&history)
            .to_vec();
        self.prefix = prefix;
        self.history = self.with_prefix(history);
        Ok(())
    }

    fn state_view(&self, resumed: bool) -> SessionStateView<'_> {
        SessionStateView {
            history: &self.history,
            raw_history: &self.persisted,
            prefix: &self.prefix,
            transcript_target: self.target.as_ref(),
            committed_turns: self.committed_turns,
            resumed,
            recorded_tools: self.recorded_tools.as_ref(),
        }
    }

    fn encode(
        &self,
        previous: &[Message],
        next: &[Message],
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.reconcile(&self.persisted, previous, next, options),
            None => Ok(Vec::new()),
        }
    }

    fn turn_usage(
        &self,
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Option<TurnUsage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.turn_usage(options),
            None => Ok(None),
        }
    }

    fn persist(
        &mut self,
        raw: &[TranscriptMessage],
        request_id: Option<&str>,
        thread_id: Option<&str>,
        partial: Option<&TranscriptPartial>,
        turn_usage: Option<&TurnUsage>,
        tools: Option<&ToolSnapshot>,
    ) -> Result<Option<TranscriptCommitReceipt>, RuntimeError> {
        let Some(target) = self.target.as_mut() else {
            return Ok(None);
        };
        if self.transcript.is_none() {
            // A turn can reach the first bind through a resume mode other
            // than `ResumeMode::Session` (e.g. `Never`, `LatestForAgent`,
            // `Thread`) on a session-bound target — `resume` only rebinds to
            // the head generation on its own `Session` path. Without this,
            // such a turn binds generation 0 even when a later `.g{n}`
            // exists: it appends into a generation the design requires to
            // stay sealed, and the next compaction's `begin_generation` then
            // fails outright because that later generation already exists.
            if let Some(session) = target.session.clone() {
                let head = target.locator.head_generation(&session);
                if head != session {
                    target.rebind_session(head);
                }
            }
            self.transcript = Some(match target.session.as_ref() {
                Some(session) => target
                    .locator
                    .open_session(session, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
                None => target
                    .locator
                    .open_stem(&target.stem, target.meta.clone())
                    .map_err(|error| RuntimeError::Persistence(error.to_string()))?,
            });
        }

        let previous_len = self.persisted.len();
        let next_len = raw.len();
        let common_len = previous_len.min(next_len);
        let extends = next_len >= previous_len && raw[..common_len] == self.persisted[..common_len];

        // A turn that no longer extends what is persisted is a compaction. For
        // a session-bound target that seals the current generation and opens
        // the next one rather than appending a replacement record: rewriting
        // the logical set in place would make the replaced turns unreadable
        // forever, and they are the conversation's own history.
        //
        // The successor generation and handle are kept in locals, not written
        // onto `target`/`self.transcript`, until the append into them below
        // actually succeeds. Committing them first — as this used to — left
        // `target` pointing at `.g{n+1}` even when the append failed to
        // create it: the next turn's `begin_generation` would then find no
        // file at `.g{n+1}`, mint `.g{n+2}` instead, and `head_generation`
        // would keep resolving the old sealed generation as the head,
        // orphaning both the failed generation and the one after it.
        let mut prev: &[TranscriptMessage] = &self.persisted;
        let empty: [TranscriptMessage; 0] = [];
        let mut pending_generation: Option<(SessionRef, Arc<dyn TranscriptHistory>)> = None;
        let mut meta = target.meta.clone();
        if !extends && let Some(session) = target.session.clone() {
            let (successor, handle) = target
                .locator
                .begin_generation(&session, target.meta.clone())
                .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
            // The successor starts empty, so the retained set is written
            // through the ordinary turn path below and keeps its usage,
            // request ids and display partial.
            meta.turn_count = 0;
            meta.session_id = Some(successor.session_id());
            meta.parent_session_id = successor.parent_session_id();
            pending_generation = Some((successor, handle));
            prev = &empty;
        }

        let transcript: &dyn TranscriptHistory = match pending_generation.as_ref() {
            Some((_, handle)) => handle.as_ref(),
            None => self.transcript.as_deref().expect("bound above"),
        };
        meta.turn_count += 1;
        meta.updated = chrono::Utc::now().to_rfc3339();
        // Record the declarations when they differ from the record in force
        // in the file being written — always for a fresh generation, whose
        // file starts with none.
        let tools_json = tools.map(ToolSnapshot::to_json);
        let tools_record = tools_json.as_ref().filter(|json| {
            pending_generation.is_some() || self.recorded_tools_json.as_ref() != Some(*json)
        });
        meta.thread_id = thread_id.map(str::to_owned).or(meta.thread_id);
        transcript
            .append_turn_with_partial(
                TranscriptTurn {
                    prev,
                    next: raw,
                    meta: &meta,
                    turn_usage,
                    request_id,
                    tools: tools_record,
                },
                partial,
            )
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
        // Captured before `pending_generation`/`self.transcript` are moved
        // from below — `transcript` borrows out of whichever of the two held
        // the just-appended handle.
        let path = transcript.path().to_path_buf();
        // Only now that the append into the successor generation has
        // actually succeeded does the target move onto it.
        if let Some((successor, handle)) = pending_generation {
            target.rebind_session(successor);
            self.transcript = Some(handle);
        }
        target.meta = meta;
        if let (Some(snapshot), Some(json)) = (tools, tools_json) {
            self.recorded_tools = Some(snapshot.clone());
            self.recorded_tools_json = Some(json);
        }
        let delta = if extends {
            TranscriptDelta::Append {
                previous_len,
                appended: previous_len..next_len,
            }
        } else {
            TranscriptDelta::Replace {
                previous_len,
                next_len,
            }
        };
        Ok(Some(TranscriptCommitReceipt { path, delta }))
    }

    fn with_prefix(&self, history: Vec<Message>) -> Vec<Message> {
        let prefix = self.prefix.messages();
        let overlap = (0..=prefix.len().min(history.len()))
            .rev()
            .find(|&len| prefix[prefix.len() - len..] == history[..len])
            .unwrap_or_default();
        let mut reconciled = prefix[..prefix.len() - overlap].to_vec();
        reconciled.extend(history);
        reconciled
    }
}

/// Ensures a terminal hook is scheduled once even if a caller drops a turn
/// future while it is awaiting preparation, driving, persistence, or hooks.
struct TerminalGuard<C: Clone + Send + Sync + 'static> {
    hooks: Arc<dyn SessionHooks<C>>,
    terminal: Option<SessionTerminal>,
    committed: bool,
}

impl<C: Clone + Send + Sync + 'static> TerminalGuard<C> {
    fn new(hooks: Arc<dyn SessionHooks<C>>) -> Self {
        Self {
            hooks,
            terminal: Some(SessionTerminal::Failed("session turn dropped".into())),
            committed: false,
        }
    }

    fn set(&mut self, terminal: SessionTerminal) {
        self.terminal = Some(terminal);
    }

    fn finalize_commit(&mut self, receipt: CommitReceipt<C>) -> tokio::task::JoinHandle<()> {
        let terminal = SessionTerminal::Completed(receipt.outcome.clone());
        // Removing the guard's terminal transfers exactly-once ownership to
        // the finalizer. `finish` and `Drop` then become no-ops for this turn.
        self.terminal = None;
        self.committed = true;
        let hooks = self.hooks.clone();
        tokio::spawn(async move {
            let _ = hooks.after_commit(receipt).await;
            let _ = hooks.on_terminal(terminal).await;
        })
    }

    fn is_committed(&self) -> bool {
        self.committed
    }

    async fn finish(mut self) -> Result<(), RuntimeError> {
        let Some(terminal) = self.terminal.take() else {
            return Ok(());
        };
        self.hooks.on_terminal(terminal).await
    }
}

impl<C: Clone + Send + Sync + 'static> Drop for TerminalGuard<C> {
    fn drop(&mut self) {
        let Some(terminal) = self.terminal.take() else {
            return;
        };
        let hooks = self.hooks.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = hooks.on_terminal(terminal).await;
            });
        }
    }
}

async fn cancelable<T>(
    cancellation: &CancellationToken,
    future: impl Future<Output = Result<T, RuntimeError>>,
) -> Result<T, RuntimeError> {
    if cancellation.is_cancelled() {
        return Err(RuntimeError::Cancelled);
    }
    tokio::select! {
        _ = cancellation.cancelled() => Err(RuntimeError::Cancelled),
        result = future => result,
    }
}
