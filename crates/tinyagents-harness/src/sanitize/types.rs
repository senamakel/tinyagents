//! Policy type for [`super::sanitize_history`].

/// Which classes of untrusted history content [`super::sanitize_history`]
/// strips. All fields default to `true`: a host that calls
/// [`super::sanitize_history`] at all almost always wants every check, and an
/// opt-out should be a visible, deliberate `false` rather than a silent gap
/// left by a partially-filled struct literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SanitizePolicy {
    /// Strip every caller-supplied [`Message::System`][tinyinference_llm::message::Message::System]
    /// entry so it cannot override the host's own system prompt.
    pub strip_system_prompts: bool,
    /// Strip image/file content blocks whose URL is not `http(s)://` or an
    /// inline `data:` URI.
    pub strip_non_http_file_urls: bool,
    /// Remove dangling tool calls and tool results (an assistant tool call
    /// with no answering result, or a result naming an undeclared call id).
    pub strip_dangling_tool_calls: bool,
}

impl Default for SanitizePolicy {
    fn default() -> Self {
        Self {
            strip_system_prompts: true,
            strip_non_http_file_urls: true,
            strip_dangling_tool_calls: true,
        }
    }
}

impl SanitizePolicy {
    /// A policy with every check enabled. Equivalent to [`Default::default`];
    /// exists so call sites can read the intent explicitly (`SanitizePolicy::all()`
    /// vs. relying on defaults matching all-enabled).
    pub fn all() -> Self {
        Self::default()
    }

    /// A policy with every check disabled — a starting point for a host that
    /// wants only one or two of the three checks and prefers to opt in.
    pub fn none() -> Self {
        Self {
            strip_system_prompts: false,
            strip_non_http_file_urls: false,
            strip_dangling_tool_calls: false,
        }
    }
}
