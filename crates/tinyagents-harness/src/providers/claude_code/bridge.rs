//! Internal normalized shapes used by the Claude Code stream adapter.
//!
//! These are the crate-private types `mod.rs` converts `ModelRequest` /
//! `ModelResponse` to and from before handing off to `driver.rs` and
//! `event_mapper.rs`, keeping the provider's request/response bridging
//! independent of `tinyinference_llm`'s wire types.

/// A single flattened chat turn (role + rendered text) sent to the CLI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChatMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}

impl ChatMessage {
    #[cfg(test)]
    pub(crate) fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    #[cfg(test)]
    pub(crate) fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    #[cfg(test)]
    pub(crate) fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
    #[cfg(test)]
    pub(crate) fn tool(content: impl Into<String>) -> Self {
        Self::new("tool", content)
    }
    pub(crate) fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Token/cost accounting parsed out of the CLI's terminal `result` event.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UsageInfo {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_input_tokens: u64,
    pub(crate) cache_creation_tokens: u64,
    pub(crate) reasoning_tokens: u64,
    pub(crate) charged_amount_usd: f64,
}

/// Aggregated result of one CC turn, assembled by `EventMapper` as the
/// stream is consumed.
#[derive(Clone, Debug, Default)]
pub(crate) struct ChatResponse {
    pub(crate) text: Option<String>,
    pub(crate) usage: Option<UsageInfo>,
}

/// One incremental chunk forwarded to a streaming caller while a turn is in
/// flight.
#[derive(Clone, Debug)]
pub(crate) enum ProviderDelta {
    /// Visible response text delta.
    TextDelta { delta: String },
    /// Extended-thinking / reasoning text delta.
    ThinkingDelta { delta: String },
}
