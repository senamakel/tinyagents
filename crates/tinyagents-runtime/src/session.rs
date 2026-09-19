use std::{future::Future, sync::Arc};

use tinyagents_harness::CancellationToken;
use tinyagents_session::transcript::{
    TranscriptHistory, TranscriptLocator, TranscriptMeta, TranscriptPartial, TranscriptTurn,
};
use tinyinference_llm::message::Message;

use crate::{
    DriverRequest, PrefixSnapshot, ResumeMode, RuntimeError, SessionDriver, SessionHooks,
    SessionResume, SessionTerminal, SessionTurnOutcome, SessionTurnRequest, ToolSnapshot,
    TranscriptCodec, TurnOptions,
};

/// Host-neutral mutable state for one conversation session.
pub struct Session<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks>,
    prefix: PrefixSnapshot,
    tools: ToolSnapshot,
    history: Vec<Message>,
    persisted: Vec<tinyagents_session::transcript::TranscriptMessage>,
    locator: Option<Arc<dyn TranscriptLocator>>,
    stem: Option<String>,
    meta: Option<TranscriptMeta>,
    transcript: Option<Arc<dyn TranscriptHistory>>,
}

impl<C: Clone + Send + Sync + 'static> Session<C> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        driver: Arc<dyn SessionDriver<C>>,
        codec: Option<Arc<dyn TranscriptCodec<C>>>,
        hooks: Arc<dyn SessionHooks>,
        prefix: PrefixSnapshot,
        tools: ToolSnapshot,
        locator: Option<Arc<dyn TranscriptLocator>>,
        stem: Option<String>,
        meta: Option<TranscriptMeta>,
        transcript: Option<Arc<dyn TranscriptHistory>>,
    ) -> Self {
        Self {
            driver,
            codec,
            hooks,
            history: prefix.messages().to_vec(),
            prefix,
            tools,
            persisted: Vec::new(),
            locator,
            stem,
            meta,
            transcript,
        }
    }

    /// Returns the currently committed model history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Returns the unchanging history prefix captured by the builder.
    pub fn prefix_snapshot(&self) -> &PrefixSnapshot {
        &self.prefix
    }

    /// Returns the immutable model-visible tool declaration set.
    pub fn tool_snapshot(&self) -> &ToolSnapshot {
        &self.tools
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
        let Some(locator) = self.locator.as_ref() else {
            return Ok(SessionResume {
                loaded: false,
                history: self.history.clone(),
            });
        };
        let read = match options.resume {
            ResumeMode::Never => None,
            ResumeMode::LatestForAgent => self
                .stem
                .as_deref()
                .and_then(|stem| locator.latest_for_agent(stem)),
            ResumeMode::Thread => options
                .thread_id
                .as_deref()
                .and_then(|thread| locator.root_for_thread(thread)),
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
        let history = self.with_prefix(codec.decode_history(&transcript)?);
        self.history = history.clone();
        self.persisted = transcript.messages;
        Ok(SessionResume {
            loaded: true,
            history,
        })
    }

    /// Executes and durably commits one state transition.
    pub async fn turn(
        &mut self,
        mut request: SessionTurnRequest,
        options: TurnOptions<C>,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        let mut terminal_guard = TerminalGuard::new(self.hooks.clone());
        let result = self
            .turn_inner(&mut request, options, &mut terminal_guard)
            .await;
        if !terminal_guard.is_committed() {
            let terminal = match &result {
                Ok(outcome) => SessionTerminal::Completed(outcome.clone()),
                Err(RuntimeError::Cancelled) => SessionTerminal::Cancelled,
                Err(error) => SessionTerminal::Failed(error.to_string()),
            };
            terminal_guard.set(terminal);
        }
        // Terminal observation cannot revoke a successful durable commit.
        // `finish` still schedules it exactly once; hook failures are
        // deliberately observational rather than a second terminal result.
        let _ = terminal_guard.finish().await;
        result
    }

    async fn turn_inner(
        &mut self,
        request: &mut SessionTurnRequest,
        options: TurnOptions<C>,
        terminal_guard: &mut TerminalGuard,
    ) -> Result<SessionTurnOutcome, RuntimeError> {
        if options.resume != ResumeMode::Never {
            self.resume(&options).await?;
        }
        cancelable(&options.cancellation, self.hooks.before_turn(request)).await?;
        let mut input = self.history.clone();
        if input.last() != Some(&request.input) {
            input.push(request.input.clone());
        }
        // `RunContext` is intentionally consumed exactly once.  There is no
        // task-local fallback: the host context selected for this turn is what
        // reaches model, middleware, and tool execution.
        let codec_options = options.transcript_options();
        let TurnOptions {
            request_id,
            thread_id,
            stream,
            cancellation,
            run_context,
            ..
        } = options;
        let run_context = run_context.with_cancellation(cancellation.clone());
        let driver_result = tokio::select! {
            _ = cancellation.cancelled() => return Err(RuntimeError::Cancelled),
            result = self.driver.execute(DriverRequest {
                history: input,
                tools: self.tools.clone(),
                run_context,
                stream,
            }) => result,
        };
        let outcome = match driver_result {
            Ok(outcome) => outcome,
            Err(failure) => {
                if let Some(partial) = failure.partial {
                    let partial_history = self.with_prefix(partial.history);
                    let raw = self.encode(&self.history, &partial_history, &codec_options)?;
                    // `append_turn` is the only durable mutation.  Do not
                    // append display partials first: a later append failure
                    // would leave an unreportable half-commit on disk.
                    self.persist(
                        &raw,
                        request_id.as_deref(),
                        thread_id.as_deref(),
                        partial.partial.as_ref(),
                    )?;
                    self.history = partial_history;
                    self.persisted = raw;
                }
                return Err(failure.error);
            }
        };
        let candidate = self.with_prefix(outcome.history);
        // `after_turn` is a pre-commit hook.  It can reject or be cancelled
        // without any durable mutation; after `persist` returns success this
        // turn is committed and cancellation can no longer change its result.
        let committed = SessionTurnOutcome {
            history: candidate.clone(),
            output: outcome.output,
            interrupted: outcome.interrupted,
        };
        cancelable(&cancellation, self.hooks.after_turn(&committed)).await?;
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        let raw = self.encode(&self.history, &candidate, &codec_options)?;
        self.persist(&raw, request_id.as_deref(), thread_id.as_deref(), None)?;
        self.history = committed.history.clone();
        self.persisted = raw;
        // Set the truthful durable terminal before invoking an observational
        // finalizer. If the caller drops this future while it is running, the
        // guard's Drop implementation still reports the completed commit.
        terminal_guard.mark_committed(committed.clone());
        // This runs after `append_turn_with_partial` has made the logical
        // transition durable. Failure or cooperative cancellation in a host
        // finalizer is observational: it cannot relabel that committed turn.
        let hooks = self.hooks.clone();
        let finalization = committed.clone();
        let _ = tokio::spawn(async move { hooks.after_commit(&finalization).await }).await;
        Ok(committed)
    }

    fn encode(
        &self,
        previous: &[Message],
        next: &[Message],
        options: &crate::TranscriptTurnOptions<C>,
    ) -> Result<Vec<tinyagents_session::transcript::TranscriptMessage>, RuntimeError> {
        match &self.codec {
            Some(codec) => codec.reconcile(&self.persisted, previous, next, options),
            None => Ok(Vec::new()),
        }
    }

    fn persist(
        &mut self,
        raw: &[tinyagents_session::transcript::TranscriptMessage],
        request_id: Option<&str>,
        thread_id: Option<&str>,
        partial: Option<&TranscriptPartial>,
    ) -> Result<(), RuntimeError> {
        let (Some(transcript), Some(meta)) = (&self.transcript, &self.meta) else {
            return Ok(());
        };
        let mut meta = meta.clone();
        meta.turn_count += 1;
        meta.updated = chrono::Utc::now().to_rfc3339();
        meta.thread_id = thread_id.map(str::to_owned).or(meta.thread_id);
        transcript
            .append_turn_with_partial(
                TranscriptTurn {
                    prev: &self.persisted,
                    next: raw,
                    meta: &meta,
                    turn_usage: None,
                    request_id,
                },
                partial,
            )
            .map_err(|error| RuntimeError::Persistence(error.to_string()))?;
        self.meta = Some(meta);
        Ok(())
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
struct TerminalGuard {
    hooks: Arc<dyn SessionHooks>,
    terminal: Option<SessionTerminal>,
    committed: bool,
}

impl TerminalGuard {
    fn new(hooks: Arc<dyn SessionHooks>) -> Self {
        Self {
            hooks,
            terminal: Some(SessionTerminal::Failed("session turn dropped".into())),
            committed: false,
        }
    }

    fn set(&mut self, terminal: SessionTerminal) {
        self.terminal = Some(terminal);
    }

    fn mark_committed(&mut self, outcome: SessionTurnOutcome) {
        self.terminal = Some(SessionTerminal::Completed(outcome));
        self.committed = true;
    }

    fn is_committed(&self) -> bool {
        self.committed
    }

    async fn finish(mut self) -> Result<(), RuntimeError> {
        let terminal = self.terminal.take().ok_or(RuntimeError::Hook(
            "terminal guard already completed".into(),
        ))?;
        let hooks = self.hooks.clone();
        tokio::spawn(async move { hooks.on_terminal(&terminal).await })
            .await
            .map_err(|error| RuntimeError::Hook(format!("terminal hook task failed: {error}")))?
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let Some(terminal) = self.terminal.take() else {
            return;
        };
        let hooks = self.hooks.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = hooks.on_terminal(&terminal).await;
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
