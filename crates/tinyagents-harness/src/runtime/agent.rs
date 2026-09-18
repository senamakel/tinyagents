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
pub struct AgentStream<'a, State: Send + Sync, Ctx: Send + Sync> {
    inner: Pin<Box<dyn Stream<Item = AgentStreamItem> + Send + 'a>>,
    harness: &'a AgentHarness<State, Ctx>,
    context_id: u64,
}

impl<State: Send + Sync, Ctx: Send + Sync> Stream for AgentStream<'_, State, Ctx> {
    type Item = AgentStreamItem;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(context)
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Drop for AgentStream<'_, State, Ctx> {
    fn drop(&mut self) {
        self.harness.remove_host_binding(self.context_id);
    }
}

struct PreparedAgentTurn<State: Send + Sync> {
    host: crate::host::HostCapabilities<State>,
    agent_id: String,
    thread_id: ThreadId,
    run_id: crate::ids::RunId,
    input_text: String,
    messages: Vec<tinyinference_llm::message::Message>,
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
    ) -> Result<AgentRun> {
        let prepared = self.prepare_agent_turn(request, &context).await?;
        let context_id = context.instance_id();
        self.install_host_binding(context_id, &prepared);
        if let Some(progress) = &prepared.host.progress {
            progress
                .emit(ProgressEvent::Started {
                    run: context.run_id().clone(),
                    thread: context.thread_id().cloned(),
                    agent: prepared.agent_id.clone(),
                })
                .await;
        }

        let outcome = self
            .invoke_in_context_collecting_partial(state, context, prepared.messages.clone())
            .await;
        self.remove_host_binding(context_id);

        match outcome.error {
            None => {
                self.finish_host_turn(&prepared, &outcome.run, true).await;
                Ok(outcome.run)
            }
            Some(error) => {
                self.finish_host_turn(&prepared, &outcome.run, false).await;
                if let Some(progress) = &prepared.host.progress {
                    progress
                        .emit(ProgressEvent::Error {
                            run: prepared_run_id(&prepared, context_id),
                            message: error.to_string(),
                        })
                        .await;
                }
                Err(error)
            }
        }
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
    {
        let prepared = self.prepare_agent_turn(request, &context).await?;
        let context_id = context.instance_id();
        self.install_host_binding(context_id, &prepared);
        if let Some(progress) = &prepared.host.progress {
            progress
                .emit(ProgressEvent::Started {
                    run: context.run_id().clone(),
                    thread: context.thread_id().cloned(),
                    agent: prepared.agent_id.clone(),
                })
                .await;
        }
        let stream = self
            .invoke_stream_in_context(state, context, prepared.messages)
            .map(|item| item);
        Ok(AgentStream {
            inner: Box::pin(stream),
            harness: self,
            context_id,
        })
    }

    async fn prepare_agent_turn(
        &self,
        mut request: AgentTurnRequest,
        context: &RunContext<Ctx>,
    ) -> Result<PreparedAgentTurn<State>> {
        let host = self.host.clone().ok_or_else(|| {
            TinyAgentsError::Validation(
                "host-driven invocation requires AgentHarness::with_host_capabilities".into(),
            )
        })?;
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
                TinyAgentsError::Validation(format!("definition registry failed: {error}"))
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
        let model = host
            .models
            .resolve(
                &crate::host::ModelResolveRequest::new(&request.agent_id)
                    .with_model_pin(definition.model.clone().unwrap_or_default()),
            )
            .await?;
        let model_name = model
            .profile()
            .and_then(|profile| profile.model.clone())
            .unwrap_or_else(|| format!("host:{}", request.agent_id));

        let mut messages = Vec::with_capacity(request.messages.len() + preamble.len() + 1);
        if !system.is_empty() {
            messages.push(tinyinference_llm::message::Message::system(system));
        }
        messages.append(&mut preamble);
        messages.append(&mut request.messages);
        self.insert_host_run(
            context.instance_id(),
            HostRunBinding {
                agent_id: request.agent_id.clone(),
                resolved: tinyinference_llm::model::ResolvedModel {
                    name: model_name,
                    requested: definition.model,
                    source: tinyinference_llm::model::ModelResolutionSource::AgentDefault,
                },
                model,
            },
        )?;
        Ok(PreparedAgentTurn {
            host,
            agent_id: request.agent_id,
            thread_id,
            run_id: context.run_id().clone(),
            input_text,
            messages,
        })
    }

    async fn finish_host_turn(
        &self,
        prepared: &PreparedAgentTurn<State>,
        run: &AgentRun,
        succeeded: bool,
    ) {
        if let Some(progress) = &prepared.host.progress {
            progress
                .emit(ProgressEvent::Finished {
                    run: prepared_run_id(prepared, 0),
                    usage: None,
                })
                .await;
        }
        let output = run.text().unwrap_or_default();
        let mut summary = TurnSummary::new(prepared.thread_id.clone(), &prepared.agent_id)
            .with_text(&prepared.input_text, &output);
        for message in &run.messages {
            if let tinyinference_llm::message::Message::Assistant(message) = message {
                for tool in &message.tool_calls {
                    summary.record_tool(&tool.name);
                }
            }
        }
        if let Some(learning) = &prepared.host.learning
            && let Err(error) = learning.on_turn_complete(&summary).await
        {
            tinyagents_tracing::warn!(%error, "[host] learning sink failed after completed turn");
        }
        if let Some(store) = &prepared.host.experience {
            let mut experience = Experience::new(&prepared.agent_id, &prepared.input_text, &output);
            if succeeded {
                experience = experience.succeeded();
            }
            if let Err(error) = store.record(&experience).await {
                tinyagents_tracing::warn!(%error, "[host] experience store failed after completed turn");
            }
        }
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

    fn install_host_binding(&self, _context_id: u64, _prepared: &PreparedAgentTurn<State>) {
        // `prepare_agent_turn` installs the binding before returning so model
        // resolution cannot race with a concurrently-started child. Kept as a
        // named step to make the run lifecycle obvious at the entry point.
    }

    pub(crate) fn remove_host_binding(&self, context_id: u64) {
        if let Ok(mut runs) = self.host_runs.lock() {
            runs.remove(&context_id);
        }
    }

    /// Best-effort progress projection. A host UI must never make the turn
    /// wait or fail, so delivery is detached and dropped when no Tokio runtime
    /// is available.
    pub(crate) fn emit_host_progress(&self, context_id: u64, event: ProgressEvent) {
        let Ok(Some(_binding)) = self.host_run_binding(context_id) else {
            return;
        };
        let Some(progress) = self.host.as_ref().and_then(|host| host.progress.clone()) else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                progress.emit(event).await;
            });
        }
    }
}

async fn screen_user_messages<State: Send + Sync>(
    host: &crate::host::HostCapabilities<State>,
    messages: &mut [tinyinference_llm::message::Message],
) -> Result<String> {
    let mut visible = Vec::new();
    for message in messages {
        if let tinyinference_llm::message::Message::User(user) = message {
            for block in &mut user.content {
                if let tinyinference_llm::message::ContentBlock::Text(text) = block {
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

fn prepared_run_id<State: Send + Sync>(
    prepared: &PreparedAgentTurn<State>,
    _context_id: u64,
) -> crate::ids::RunId {
    prepared.run_id.clone()
}
