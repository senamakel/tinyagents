//! Internal normalized shapes used by the Claude Code stream adapter.

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

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UsageInfo {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_input_tokens: u64,
    pub(crate) cache_creation_tokens: u64,
    pub(crate) reasoning_tokens: u64,
    pub(crate) charged_amount_usd: f64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ChatResponse {
    pub(crate) text: Option<String>,
    pub(crate) usage: Option<UsageInfo>,
}

#[derive(Clone, Debug)]
pub(crate) enum ProviderDelta {
    TextDelta { delta: String },
    ThinkingDelta { delta: String },
}
