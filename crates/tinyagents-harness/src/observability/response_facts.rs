//! Provider-neutral facts an observer reads off one model response or stream.
//!
//! These read only identifiers and timing signals, never content:
//!
//! - [`served_by`]: the upstream provider a routing gateway sent the call to,
//!   from a top-level `provider` string on the response (OpenRouter names the
//!   provider that served the call there, for example `"DeepInfra"`). It is
//!   `None` when the response does not say, which is the case for any
//!   provider that serves calls itself.
//! - [`response_id`]: the provider's own id for the response (OpenAI-style
//!   `chatcmpl-…`, OpenRouter's `gen-…`, Anthropic's `msg_…`), so a call can
//!   be looked up with the provider afterwards. The provider adapters keep it
//!   on the assistant message for both buffered and streamed responses; a
//!   top-level `id` on `raw` is the fallback.
//! - [`is_output`]: whether a stream item is model output (text,
//!   reasoning or a tool call), the event that marks the time to first token.
use serde_json::Value;
use tinyinference_llm::model::{ModelResponse, ModelStreamItem};

/// The upstream provider that served `response`, when a gateway reports it.
pub(crate) fn served_by(response: &ModelResponse) -> Option<String> {
    response
        .raw
        .as_ref()
        .and_then(|raw| raw.get("provider"))
        .and_then(Value::as_str)
        .filter(|provider| !provider.is_empty())
        .map(str::to_owned)
}

/// The provider's id for `response`, when it reported one.
pub(crate) fn response_id(response: &ModelResponse) -> Option<String> {
    response
        .message
        .id
        .as_deref()
        .or_else(|| {
            response
                .raw
                .as_ref()
                .and_then(|raw| raw.get("id"))
                .and_then(Value::as_str)
        })
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

/// Whether `item` carries model output: text, reasoning or a tool call.
/// Stream-opened markers, usage updates and empty deltas do not count.
pub(crate) fn is_output(item: &ModelStreamItem) -> bool {
    match item {
        ModelStreamItem::MessageDelta(delta) => {
            !delta.text.is_empty() || !delta.reasoning.is_empty() || delta.tool_call.is_some()
        }
        ModelStreamItem::ToolCallDelta(_) => true,
        _ => false,
    }
}

#[cfg(test)]
#[path = "response_facts_tests.rs"]
mod tests;
