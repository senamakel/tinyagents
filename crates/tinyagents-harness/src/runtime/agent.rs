//! Host-driven agent invocation.
//!
//! The lower-level `invoke*` APIs intentionally accept an already-selected
//! model. This module is the separate product-host boundary: it resolves an
//! agent definition and model, composes/screen contexts, and installs a
//! per-`RunContext` binding that the normal loop consumes. Keeping the two
//! entry points separate prevents an embedding SDK call from accidentally
//! acquiring product policy merely because a harness was also configured for a
//! hosted turn.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};

use crate::agent_loop::AgentStreamItem;
use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::host::{
    ContentOrigin, Experience, ProgressEvent, RecallRequest, ScreenOutcome, TurnContextRequest,
    TurnSummary,
};
use crate::ids::ThreadId;
use crate::middleware::AgentRun;

use super::{AgentHarness, HostRunBinding};

/// The exact host bundle that authorized the parent invocation.
///
/// A child context carries this as type-erased runtime state because
/// [`RunContext`] is intentionally independent of the application's `State`.
/// `SubAgent` downcasts it at the recursive boundary and therefore cannot
/// substitute an unhosted or differently-hosted child harness for the
/// parent's policy.
pub(crate) struct HostInvocationAuthority<State: Send + Sync> {
    pub(crate) host: crate::host::HostCapabilities<State>,
}

/// A host-owned turn request.
///
/// `agent_id` is opaque to the harness. It is resolved only through the host
/// definition registry and is threaded back to every host capability as an
/// attribution value.
#[derive(Clone, Debug)]
pub struct AgentTurnRequest {
    /// Host definition to invoke.
    pub agent_id: String,
    /// Initial transcript supplied by the host.
    pub messages: Vec<tinyinference_llm::message::Message>,
}

impl AgentTurnRequest {
    /// Creates a request for `agent_id` with the supplied conversation input.
    pub fn new(
        agent_id: impl Into<String>,
        messages: Vec<tinyinference_llm::message::Message>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            messages,
        }
    }
}

/// A caller-consumable hosted stream.
///
/// Dropping it removes the per-context host routing entry even when a caller
/// stops listening before a terminal item.
pub struct AgentStream<'a, State: Send + Sync + 'static, Ctx: Send + Sync> {
    inner: Option<Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + 'a>>>,
    cancellation: crate::CancellationToken,
    terminal_observer: std::sync::Arc<std::sync::Mutex<Option<crate::context::TerminalObserver>>>,
    terminal_observed: bool,
    marker: std::marker::PhantomData<(&'a State, Ctx)>,
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync> Stream for AgentStream<'_, State, Ctx> {
    type Item = AgentStreamItem;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `inner` is pinned independently by `Box`; this projection never moves
        // the boxed stream or any other field of `AgentStream`.
        let stream = unsafe { self.get_unchecked_mut() };
        match stream.inner.as_mut() {
            Some(inner) => match inner.as_mut().poll_next(context) {
                Poll::Ready(Some(item)) => {
                    stream.terminal_observed = matches!(
                        item,
                        AgentStreamItem::Completed(_) | AgentStreamItem::Failed { .. }
                    );
                    Poll::Ready(Some(item))
                }
                Poll::Ready(None) => {
                    stream.terminal_observed = true;
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            },
            None => Poll::Ready(None),
        }
    }
}

impl<State: Send + Sync + 'static, Ctx: Send + Sync> Drop for AgentStream<'_, State, Ctx> {
    fn drop(&mut self) {
        // Dropping the driving stream drops the loop's `TerminalRunGuard`,
        // which forwards its actual partial run to the installed observer.
        if !self.terminal_observed {
            self.cancellation.cancel();
        }
        self.inner.take();
        if let Ok(mut observer) = self.terminal_observer.lock()
            && let Some(observer) = observer.take()
        {
            observer(
                AgentRun::new(),
                false,
                Some("hosted stream cancelled before execution began".to_string()),
            );
        }
    }
}

struct PreparedAgentTurn<State: Send + Sync> {
    host: crate::host::HostCapabilities<State>,
    agent_id: String,
    thread_id: ThreadId,
    run_id: crate::ids::RunId,
    input_text: String,
    messages: Vec<tinyinference_llm::message::Message>,
    progress: Option<ProgressSender>,
}

#[derive(Clone)]
pub(crate) struct ProgressSender {
    tx: tokio::sync::mpsc::Sender<ProgressEvent>,
    nonterminal_slots: std::sync::Arc<tokio::sync::Semaphore>,
}

impl ProgressSender {
    fn send_nonterminal(&self, event: ProgressEvent) {
        let Ok(permit) = self.nonterminal_slots.clone().try_acquire_owned() else {
            return;
        };
        if self.tx.try_send(event).is_ok() {
            permit.forget();
        }
    }

    fn send_terminal(&self, event: ProgressEvent) {
        let _ = self.tx.try_send(event);
    }
}

impl<State: Send + Sync> Clone for PreparedAgentTurn<State> {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            agent_id: self.agent_id.clone(),
            thread_id: self.thread_id.clone(),
            run_id: self.run_id.clone(),
            input_text: self.input_text.clone(),
            messages: self.messages.clone(),
            progress: self.progress.clone(),
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Runs an agent through the installed host-capability bundle.
    ///
    /// Missing capabilities fail before a provider call. Optional capabilities
    /// remain genuinely optional: when absent they are neither constructed nor
    /// called.
    pub async fn invoke_agent(
        &self,
        request: AgentTurnRequest,
        context: RunContext<Ctx>,
        state: &State,
    ) -> Result<AgentRun>
    where
        State: 'static,
    {
        let host = self.host.clone().ok_or_else(|| {
            TinyAgentsError::Validation(
                "host-driven invocation requires AgentHarness::with_host_capabilities".into(),
            )
        })?;
        self.invoke_agent_with_host_capabilities(host, request, context, state)
            .await
    }

    /// Re-enters the canonical hosted entry point with the parent's exact
    /// capabilities. Used only by recursive delegation after the parent
    /// authority has authorized the child.
    pub(crate) async fn invoke_agent_with_host_capabilities(
        &self,
        host: crate::host::HostCapabilities<State>,
        request: AgentTurnRequest,
        mut context: RunContext<Ctx>,
        state: &State,
    ) -> Result<AgentRun>
    where
        State: 'static,
    {
        let prepared = self
            .prepare_agent_turn_bounded(host, request, &context)
            .await?;
        let context_id = context.instance_id();
        let agent_id = prepared.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            host: prepared.host.clone(),
        }));
        self.install_host_terminal_observer(&mut context, context_id, prepared.clone());
        self.emit_host_progress(
            context_id,
            ProgressEvent::Started {
                run: context.run_id().clone(),
                thread: context.thread_id().cloned(),
                agent: agent_id,
            },
        );

        let outcome = self
            .invoke_in_context_collecting_partial(state, context, prepared.messages.clone())
            .await;
        match outcome.error {
            None => Ok(outcome.run),
            Some(error) => Err(error),
        }
    }

    /// Collects a hosted turn through the streaming driver while preserving the
    /// parent's exact capability bundle. Recursive streaming delegation uses
    /// this rather than the unary entry point so model deltas and delta
    /// middleware remain part of the shared parent event stream.
    pub(crate) async fn invoke_agent_streaming_with_host_capabilities(
        &self,
        host: crate::host::HostCapabilities<State>,
        request: AgentTurnRequest,
        context: RunContext<Ctx>,
        state: &State,
    ) -> Result<AgentRun>
    where
        Ctx: 'static,
        State: 'static,
    {
        let stream = self
            .invoke_agent_stream_with_host_capabilities(host, request, context, state)
            .await?;
        futures::pin_mut!(stream);
        while let Some(item) = stream.next().await {
            match item {
                AgentStreamItem::Completed(run) => return Ok(*run),
                AgentStreamItem::Failed { error, .. } => return Err(TinyAgentsError::Model(error)),
                AgentStreamItem::Event(_) => {}
            }
        }
        Err(TinyAgentsError::Model(
            "hosted stream ended without a terminal result".to_string(),
        ))
    }

    /// Starts a hosted streaming turn.
    ///
    /// The returned stream is the existing event projection, so host-driven
    /// and explicit-model streams expose the same canonical event vocabulary.
    /// The run-scoped model binding is removed when its terminal item is
    /// observed; callers must drain (or drop) the stream to end the turn.
    pub async fn invoke_agent_stream<'a>(
        &'a self,
        request: AgentTurnRequest,
        context: RunContext<Ctx>,
        state: &'a State,
    ) -> Result<AgentStream<'a, State, Ctx>>
    where
        Ctx: 'static,
        State: 'static,
    {
        let host = self.host.clone().ok_or_else(|| {
            TinyAgentsError::Validation(
                "host-driven invocation requires AgentHarness::with_host_capabilities".into(),
            )
        })?;
        self.invoke_agent_stream_with_host_capabilities(host, request, context, state)
            .await
    }

    async fn invoke_agent_stream_with_host_capabilities<'a>(
        &'a self,
        host: crate::host::HostCapabilities<State>,
        request: AgentTurnRequest,
        mut context: RunContext<Ctx>,
        state: &'a State,
    ) -> Result<AgentStream<'a, State, Ctx>>
    where
        Ctx: 'static,
        State: 'static,
    {
        let prepared = self
            .prepare_agent_turn_bounded(host, request, &context)
            .await?;
        let context_id = context.instance_id();
        let agent_id = prepared.agent_id.clone();
        context.host_agent_id = Some(agent_id.clone());
        context.host_authority = Some(std::sync::Arc::new(HostInvocationAuthority {
            host: prepared.host.clone(),
        }));
        let cancellation = context.cancellation.clone();
        let terminal_observer =
            self.install_host_terminal_observer(&mut context, context_id, prepared.clone());
        self.emit_host_progress(
            context_id,
            ProgressEvent::Started {
                run: context.run_id().clone(),
                thread: context.thread_id().cloned(),
                agent: agent_id,
            },
        );
        let stream = self
            .invoke_stream_in_context(state, context, prepared.messages.clone())
            .map(|item| item);
        Ok(AgentStream {
            inner: Some(Box::pin(stream)),
            cancellation,
            terminal_observer,
            terminal_observed: false,
            marker: std::marker::PhantomData,
        })
    }

    /// Runs host-owned preparation under the same cancellation and wall-clock
    /// controls as model resolution.  Definitions, security screening, and
    /// context composition are all host I/O boundaries, not setup work that
    /// may outlive a cancelled turn.
    async fn prepare_agent_turn_bounded(
        &self,
        host: crate::host::HostCapabilities<State>,
        request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State>> {
        let cancellation = context.cancellation.clone();
        let preparation = self.prepare_agent_turn(host, request, context);
        match context.remaining_wall_clock() {
            Some(remaining) => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = tokio::time::timeout(remaining, preparation) => result.map_err(|_| TinyAgentsError::Timeout(format!(
                    "host turn preparation for run `{}` exceeded its remaining wall-clock budget",
                    context.run_id()
                )))?,
            },
            None => tokio::select! {
                biased;
                _ = cancellation.cancelled() => Err(TinyAgentsError::Cancelled),
                result = preparation => result,
            },
        }
    }

    async fn prepare_agent_turn(
        &self,
        host: crate::host::HostCapabilities<State>,
        mut request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State>> {
        if request.agent_id.trim().is_empty() {
            return Err(TinyAgentsError::Validation(
                "host-driven invocation requires a non-empty agent id".into(),
            ));
        }
        let definition = host
            .definitions
            .resolve(&request.agent_id)
            .await
            .map_err(|error| {
                let _ = error;
                tinyagents_tracing::warn!(agent_id = %request.agent_id, "[host] definition lookup failed");
                TinyAgentsError::Validation("agent definition lookup failed".to_string())
            })?
            .ok_or_else(|| {
                TinyAgentsError::Validation(format!(
                    "agent definition `{}` was not found",
                    request.agent_id
                ))
            })?;
        if !definition.is_valid() {
            return Err(TinyAgentsError::Validation(format!(
                "agent definition `{}` is invalid: {}",
                request.agent_id,
                definition
                    .diagnostics()
                    .into_iter()
                    .map(|diagnostic| diagnostic.message)
                    .collect::<Vec<_>>()
                    .join("; ")
            )));
        }

        let thread_id = context
            .thread_id()
            .cloned()
            .unwrap_or_else(|| ThreadId::from(context.run_id().as_str()));
        let input_text = screen_user_messages(&host, &mut request.messages).await?;
        let context_request =
            TurnContextRequest::new(&request.agent_id, thread_id.clone(), &input_text);
        let system = host.context.compose_system_prompt(&context_request).await?;
        let mut preamble = host.context.preamble(&context_request).await?;

        if let Some(memory) = &host.memory {
            if let Some(summary) = memory.thread_summary(&thread_id).await? {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &summary).await?,
                ));
            }
            for memory in memory
                .recall(
                    RecallRequest::new(&input_text)
                        .with_agent(&request.agent_id)
                        .with_thread(thread_id.clone()),
                )
                .await?
            {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &memory.text).await?,
                ));
            }
        }
        if let Some(experience) = &host.experience {
            for experience in experience
                .recall_for(&request.agent_id, &input_text)
                .await?
            {
                preamble.push(tinyinference_llm::message::Message::system(
                    screen_stored(&host, &experience.outcome).await?,
                ));
            }
        }
        let mut messages = Vec::with_capacity(request.messages.len() + preamble.len() + 1);
        if !system.is_empty() {
            messages.push(tinyinference_llm::message::Message::system(system));
        }
        messages.append(&mut preamble);
        messages.append(&mut request.messages);
        let progress = start_progress_dispatcher(host.progress.clone());
        self.insert_host_run(
            context.instance_id(),
            HostRunBinding {
                host: host.clone(),
                agent_id: request.agent_id.clone(),
                model_pin: definition.model,
                role: definition.role,
                allowed_tools: definition.tools.into_iter().collect(),
                progress: progress.clone(),
            },
        )?;
        Ok(PreparedAgentTurn {
            host,
            agent_id: request.agent_id,
            thread_id,
            run_id: context.run_id().clone(),
            input_text,
            messages,
            progress,
        })
    }

    pub(crate) fn host_run_binding(
        &self,
        context_id: u64,
    ) -> Result<Option<HostRunBinding<State>>> {
        self.host_runs
            .lock()
            .map_err(|_| TinyAgentsError::Validation("host run binding lock poisoned".into()))
            .map(|runs| runs.get(&context_id).cloned())
    }

    fn insert_host_run(&self, context_id: u64, binding: HostRunBinding<State>) -> Result<()> {
        self.host_runs
            .lock()
            .map_err(|_| TinyAgentsError::Validation("host run binding lock poisoned".into()))?
            .insert(context_id, binding);
        Ok(())
    }

    fn install_host_terminal_observer(
        &self,
        context: &mut RunContext<Ctx>,
        context_id: u64,
        prepared: PreparedAgentTurn<State>,
    ) -> std::sync::Arc<std::sync::Mutex<Option<crate::context::TerminalObserver>>>
    where
        State: 'static,
    {
        let runs = std::sync::Arc::clone(&self.host_runs);
        let observer = std::sync::Arc::new(std::sync::Mutex::new(Some(Box::new(
            move |run, succeeded, error: Option<String>| {
                if let Ok(mut runs) = runs.lock() {
                    runs.remove(&context_id);
                } else {
                    tinyagents_tracing::warn!(
                        "[host] host run binding lock poisoned during terminal cleanup"
                    );
                }
                spawn_host_finalizer(
                    prepared,
                    run,
                    succeeded,
                    error.map(|error| {
                        tinyagents_tracing::warn!(%error, "[host] agent run failed");
                        "agent run failed".to_string()
                    }),
                );
            },
        )
            as crate::context::TerminalObserver)));
        let for_context = std::sync::Arc::clone(&observer);
        context.set_terminal_observer(Box::new(move |run, succeeded, error| {
            if let Ok(mut observer) = for_context.lock()
                && let Some(observer) = observer.take()
            {
                observer(run, succeeded, error);
            }
        }));
        observer
    }

    /// Best-effort progress projection. A host UI must never make the turn
    /// wait or fail, so delivery is detached and dropped when no Tokio runtime
    /// is available.
    pub(crate) fn emit_host_progress(&self, context_id: u64, event: ProgressEvent) {
        let Ok(Some(binding)) = self.host_run_binding(context_id) else {
            return;
        };
        let Some(progress) = binding.progress else {
            return;
        };
        progress.send_nonterminal(event);
    }
}

fn spawn_host_finalizer<State: Send + Sync + 'static>(
    prepared: PreparedAgentTurn<State>,
    run: AgentRun,
    succeeded: bool,
    error: Option<String>,
) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(async move { finish_host_turn(prepared, run, succeeded, error).await });
    } else {
        tinyagents_tracing::warn!(
            run_id = %prepared.run_id,
            "[host] dropping terminal host bookkeeping because no Tokio runtime is available"
        );
    }
}

async fn finish_host_turn<State: Send + Sync>(
    prepared: PreparedAgentTurn<State>,
    run: AgentRun,
    succeeded: bool,
    error: Option<String>,
) {
    // The per-turn queue preserves event order while keeping a slow progress
    // consumer completely outside the agent's critical path.
    if let Some(progress) = &prepared.progress {
        if let Some(message) = error {
            progress.send_terminal(ProgressEvent::Error {
                run: prepared.run_id.clone(),
                message,
            });
        } else {
            progress.send_terminal(ProgressEvent::Finished {
                run: prepared.run_id.clone(),
                usage: Some(run.usage.usage),
            });
        }
    }
    let output = run.text().unwrap_or_default();
    let mut summary = TurnSummary::new(prepared.thread_id.clone(), &prepared.agent_id)
        .with_text(&prepared.input_text, &output)
        .with_usage(run.usage.usage);
    for tool in &run.executed_tools {
        summary.record_tool(tool);
    }
    if let Some(memory) = &prepared.host.memory {
        let item = crate::host::NewMemory::new(&output)
            .with_thread(prepared.thread_id.clone())
            .with_agent(&prepared.agent_id)
            .with_tag(if succeeded {
                "turn_success"
            } else {
                "turn_failure"
            });
        if let Err(error) = memory.remember(item).await {
            tinyagents_tracing::warn!(%error, "[host] memory sink failed after terminal turn");
        }
    }
    if let Some(learning) = &prepared.host.learning
        && let Err(error) = learning.on_turn_complete(&summary).await
    {
        tinyagents_tracing::warn!(%error, "[host] learning sink failed after terminal turn");
    }
    if let Some(store) = &prepared.host.experience {
        let mut experience = Experience::new(&prepared.agent_id, &prepared.input_text, &output);
        if succeeded {
            experience = experience.succeeded();
        }
        if let Err(error) = store.record(&experience).await {
            tinyagents_tracing::warn!(%error, "[host] experience store failed after terminal turn");
        }
    }
}

fn start_progress_dispatcher(
    sink: Option<std::sync::Arc<dyn crate::host::ProgressSink>>,
) -> Option<ProgressSender> {
    let sink = sink?;
    let handle = tokio::runtime::Handle::try_current().ok()?;
    // Progress is observational. Bound it so a slow sink cannot retain every
    // streamed token; producers use `try_send` and drop overflowed updates.
    // One slot is reserved for the single terminal outcome. Producers use
    // `try_send` for ordinary progress, so at most 128 nonterminal events can
    // fill before finalization claims the remaining slot.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<ProgressEvent>(129);
    let nonterminal_slots = std::sync::Arc::new(tokio::sync::Semaphore::new(128));
    let released_slots = nonterminal_slots.clone();
    handle.spawn(async move {
        while let Some(event) = rx.recv().await {
            if !event.is_terminal() {
                released_slots.add_permits(1);
            }
            sink.emit(event).await;
        }
    });
    Some(ProgressSender {
        tx,
        nonterminal_slots,
    })
}

async fn screen_user_messages<State: Send + Sync>(
    host: &crate::host::HostCapabilities<State>,
    messages: &mut [tinyinference_llm::message::Message],
) -> Result<String> {
    let mut visible = Vec::new();
    for message in messages {
        if let tinyinference_llm::message::Message::User(user) = message {
            for block in &mut user.content {
                if let tinyinference_llm::message::ContentBlock::Text(text)
                | tinyinference_llm::message::ContentBlock::Thinking { text, .. } = block
                {
                    let screened = host
                        .security
                        .screen_input(text, ContentOrigin::User)
                        .await?;
                    match screened {
                        ScreenOutcome::Pass => visible.push(text.clone()),
                        ScreenOutcome::Redacted(redacted) => {
                            *text = redacted;
                            visible.push(text.clone());
                        }
                        ScreenOutcome::Block { reason } => {
                            return Err(TinyAgentsError::Validation(reason));
                        }
                    }
                } else if let tinyinference_llm::message::ContentBlock::Json(value)
                | tinyinference_llm::message::ContentBlock::ProviderExtension(value) =
                    block
                {
                    let rendered = value.to_string();
                    match host
                        .security
                        .screen_input(&rendered, ContentOrigin::User)
                        .await?
                    {
                        ScreenOutcome::Pass => visible.push(rendered),
                        ScreenOutcome::Redacted(redacted) => {
                            *value = serde_json::from_str(&redacted)
                                .unwrap_or(serde_json::Value::String(redacted));
                            visible.push(value.to_string());
                        }
                        ScreenOutcome::Block { reason } => {
                            return Err(TinyAgentsError::Validation(reason));
                        }
                    }
                }
            }
        }
    }
    Ok(visible.join("\n"))
}

async fn screen_stored<State: Send + Sync>(
    host: &crate::host::HostCapabilities<State>,
    text: &str,
) -> Result<String> {
    match host
        .security
        .screen_input(text, ContentOrigin::Stored)
        .await?
    {
        ScreenOutcome::Pass => Ok(text.to_string()),
        ScreenOutcome::Redacted(text) => Ok(text),
        ScreenOutcome::Block { reason } => Err(TinyAgentsError::Validation(reason)),
    }
}

#[cfg(test)]
mod progress_dispatcher_tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;

    use super::start_progress_dispatcher;
    use crate::host::{ProgressEvent, ProgressSink};
    use crate::ids::RunId;

    struct BlockingProgressSink {
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Semaphore>,
        events: Mutex<Vec<ProgressEvent>>,
    }

    #[async_trait]
    impl ProgressSink for BlockingProgressSink {
        async fn emit(&self, event: ProgressEvent) {
            self.entered.notify_one();
            self.release
                .acquire()
                .await
                .expect("test progress sink remains open")
                .forget();
            self.events.lock().expect("progress lock").push(event);
        }
    }

    #[tokio::test]
    async fn terminal_progress_is_delivered_after_nonterminal_slots_are_saturated() {
        let sink = Arc::new(BlockingProgressSink {
            entered: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Semaphore::new(0)),
            events: Mutex::new(Vec::new()),
        });
        let sender = start_progress_dispatcher(Some(sink.clone()))
            .expect("Tokio test runtime provides a dispatcher");
        let run = RunId::new("saturated-progress");
        sender.send_nonterminal(ProgressEvent::Token {
            run: run.clone(),
            text: "first".to_string(),
        });
        sink.entered.notified().await;

        for slot in 0..128 {
            sender.send_nonterminal(ProgressEvent::Token {
                run: run.clone(),
                text: format!("queued-{slot}"),
            });
        }
        sender.send_terminal(ProgressEvent::Finished { run, usage: None });

        // The sink holds the receiver on the first item, so all 128 ordinary
        // slots are occupied. Finished therefore proves the reserved terminal
        // slot was still available after nonterminal backpressure saturated.
        sink.release.add_permits(130);
        for _ in 0..256 {
            if sink
                .events
                .lock()
                .expect("progress lock")
                .iter()
                .any(ProgressEvent::is_terminal)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            sink.events
                .lock()
                .expect("progress lock")
                .iter()
                .filter(|event| event.is_terminal())
                .count(),
            1,
            "the terminal event cannot be dropped behind 128 progress updates"
        );
    }
}
