//! Content-free hooks at the provider invocation boundary.
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use tinyinference_llm::model::{
    ChatModel, DeferredHandle, DeferredStatus, ModelRequest, ModelResponse, ModelStream,
    ModelStreamItem,
};
mod types;
pub use types::{
    PhysicalDispatchGuard, PhysicalDispatchMetadata, PhysicalDispatchObserver,
    PhysicalDispatchOutcome, PhysicalDispatchStatus, PhysicalDispatchTerminal,
    PhysicalDispatchUsage,
};

/// A provider wrapper that reports each admitted invocation exactly once.
/// Streaming calls stay open until a terminal item, clean EOF, or drop.
pub struct ObservedChatModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    observer: Arc<dyn PhysicalDispatchObserver>,
}

impl<State: Send + Sync> ObservedChatModel<State> {
    /// Wrap a concrete provider. Put budget admission outside this wrapper.
    pub fn new(
        inner: Arc<dyn ChatModel<State>>,
        observer: Arc<dyn PhysicalDispatchObserver>,
    ) -> Self {
        Self { inner, observer }
    }
}

impl<State: Send + Sync> std::fmt::Debug for ObservedChatModel<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedChatModel").finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl<State: Send + Sync> ChatModel<State> for ObservedChatModel<State> {
    fn profile(&self) -> Option<&tinyinference_llm::model::ModelProfile> {
        self.inner.profile()
    }

    fn supports_input(
        &self,
        modality: tinyinference_llm::model::InputModality,
        mime: &str,
        source: tinyinference_llm::model::InputSource,
    ) -> bool {
        self.inner.supports_input(modality, mime, source)
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    fn tracks_physical_attempts(&self) -> bool {
        self.inner.tracks_physical_attempts()
    }

    fn has_pre_invocation_admission_gate(&self) -> bool {
        self.inner.has_pre_invocation_admission_gate()
    }

    async fn invoke(
        &self,
        state: &State,
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let metadata = PhysicalDispatchMetadata {
            requested_model: request.model.clone(),
            call_id: request
                .correlation
                .as_ref()
                .map(|correlation| correlation.call_id.clone()),
        };
        let guard = PhysicalDispatchGuard::new(self.observer.on_start(&metadata));
        let result = self.inner.invoke(state, request).await;
        guard.finish(dispatch_outcome(result.as_ref().ok(), result.is_ok()));
        result
    }

    async fn stream(
        &self,
        state: &State,
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        let metadata = PhysicalDispatchMetadata {
            requested_model: request.model.clone(),
            call_id: request
                .correlation
                .as_ref()
                .map(|correlation| correlation.call_id.clone()),
        };
        let guard = PhysicalDispatchGuard::new(self.observer.on_start(&metadata));
        let stream = match self.inner.stream(state, request).await {
            Ok(stream) => stream,
            Err(error) => {
                guard.finish(dispatch_outcome(None, false));
                return Err(error);
            }
        };
        let metadata = stream.metadata().clone();
        let mut stream = Box::pin(stream);
        let mut guard = Some(guard);
        let mut ended = false;
        let observed = futures::stream::poll_fn(move |cx| {
            if ended {
                return std::task::Poll::Ready(None);
            }
            let item = stream.as_mut().poll_next(cx);
            if let std::task::Poll::Ready(Some(item)) = &item {
                if super::response_facts::is_output(item)
                    && let Some(guard) = guard.as_mut()
                {
                    guard.note_output();
                }
                let outcome = match item {
                    ModelStreamItem::Completed(response) => {
                        Some(dispatch_outcome(Some(response), true))
                    }
                    ModelStreamItem::Deferred(_) => Some(dispatch_outcome(None, true)),
                    ModelStreamItem::Failed(_) | ModelStreamItem::ProviderFailed(_) => {
                        Some(dispatch_outcome(None, false))
                    }
                    _ => None,
                };
                if let Some(outcome) = outcome {
                    if let Some(guard) = guard.take() {
                        ended = true;
                        guard.finish(outcome);
                    }
                } else if let ModelStreamItem::UsageDelta(usage) = item
                    && let Some(guard) = guard.as_mut()
                {
                    guard.update_usage(PhysicalDispatchUsage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        cached_tokens: usage.cache_read_tokens,
                        reasoning_tokens: usage.reasoning_tokens,
                        cost_usd: usage
                            .charged_amount
                            .as_ref()
                            .map(|amount| amount.micros as f64 / 1_000_000.0),
                    });
                }
            }
            if matches!(item, std::task::Poll::Ready(None))
                && let Some(guard) = guard.take()
            {
                ended = true;
                guard.finish(dispatch_outcome(None, false));
            }
            item
        });
        Ok(ModelStream::new(Box::pin(observed)).with_metadata(metadata))
    }

    async fn fetch_deferred(
        &self,
        handle: &DeferredHandle,
    ) -> tinyinference_llm::Result<DeferredStatus> {
        self.inner.fetch_deferred(handle).await
    }
}

fn dispatch_outcome(response: Option<&ModelResponse>, succeeded: bool) -> PhysicalDispatchOutcome {
    let raw = response.and_then(|response| response.raw.as_ref());
    let usage = response
        .and_then(|response| response.usage.as_ref())
        .map(|usage| PhysicalDispatchUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_tokens: usage.cache_read_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            cost_usd: usage
                .charged_amount
                .as_ref()
                .map(|amount| amount.micros as f64 / 1_000_000.0)
                .or_else(|| {
                    let raw = raw?;
                    if crate::cost::has_authoritative_billing_source(raw) {
                        return None;
                    }
                    crate::cost::provider_cost_usd(raw)
                }),
        });
    let refused = response.is_some_and(|response| {
        matches!(
            response.finish_reason.as_deref(),
            Some("refusal" | "content_filter")
        ) || response
            .raw
            .as_ref()
            .and_then(|raw| raw.pointer("/choices/0/message/refusal"))
            .is_some_and(|refusal| refusal.as_str().is_some_and(|text| !text.is_empty()))
    });
    let status = match (refused, succeeded) {
        (true, _) => PhysicalDispatchStatus::Refused,
        (_, true) => PhysicalDispatchStatus::Succeeded,
        _ => PhysicalDispatchStatus::Failed,
    };
    PhysicalDispatchOutcome {
        status,
        refused,
        duration: Duration::ZERO,
        usage,
        answered_model: raw
            .and_then(|raw| raw.get("model"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        finish_reason: response.and_then(|response| response.finish_reason.clone()),
        served_by: response.and_then(super::response_facts::served_by),
        response_id: response.and_then(super::response_facts::response_id),
        first_output: None,
    }
}

#[cfg(test)]
#[path = "../physical_dispatch_tests.rs"]
mod tests;
