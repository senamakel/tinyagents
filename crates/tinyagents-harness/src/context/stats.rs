//! Host-free transcript statistics.
//!
//! [`context_statistics`] derives deterministic, tokenizer-free counts
//! ([`ContextStatistics`]) directly from a message transcript, and
//! [`estimate_context_tokens`] estimates a token count through a
//! caller-supplied tokenizer. Neither function names a host tokenizer, memory
//! system, or product message type, so callers — prompt budgeting,
//! compaction heuristics, diagnostics — can use these without pulling host
//! policy into the harness.

use tinyinference_llm::message::{ContentBlock, Message};

/// Counts useful for prompt budgeting and diagnostics without naming a host
/// tokenizer, memory system, or product message type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextStatistics {
    /// Total messages in the transcript.
    pub messages: usize,
    /// Visible text characters across every content block.
    pub text_chars: usize,
    /// Image blocks across every role.
    pub images: usize,
    /// Tool calls requested by assistant messages.
    pub tool_calls: usize,
    /// Tool result messages.
    pub tool_results: usize,
    /// Result messages whose call id occurs in a preceding assistant request.
    pub paired_tool_results: usize,
}

/// Calculates deterministic statistics for a transcript.
pub fn context_statistics(messages: &[Message]) -> ContextStatistics {
    let mut stats = ContextStatistics {
        messages: messages.len(),
        ..ContextStatistics::default()
    };
    let mut requested = std::collections::HashSet::new();
    for message in messages {
        let content = match message {
            Message::System(message) => &message.content,
            Message::User(message) => &message.content,
            Message::Assistant(message) => {
                stats.tool_calls += message.tool_calls.len();
                requested.extend(message.tool_calls.iter().map(|call| call.id.clone()));
                &message.content
            }
            Message::Tool(message) => {
                stats.tool_results += 1;
                if requested.contains(&message.tool_call_id) {
                    stats.paired_tool_results += 1;
                }
                &message.content
            }
        };
        for block in content {
            match block {
                ContentBlock::Text(text) | ContentBlock::Thinking { text, .. } => {
                    stats.text_chars += text.chars().count();
                }
                ContentBlock::Json(value) | ContentBlock::ProviderExtension(value) => {
                    stats.text_chars += value.to_string().chars().count();
                }
                ContentBlock::Image(_) => stats.images += 1,
                ContentBlock::RedactedThinking { .. } => {}
            }
        }
    }
    stats
}

/// Estimates transcript tokens through a caller-supplied tokenizer.
///
/// The harness deliberately does not choose a tokenizer: providers vary, and
/// callers can account for their exact model dialect without importing host
/// policy into this crate.
pub fn estimate_context_tokens(messages: &[Message], tokenize: impl Fn(&str) -> usize) -> usize {
    messages
        .iter()
        .map(|message| {
            let content = match message {
                Message::System(message) => &message.content,
                Message::User(message) => &message.content,
                Message::Assistant(message) => &message.content,
                Message::Tool(message) => &message.content,
            };
            let mut visible = content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text(text) | ContentBlock::Thinking { text, .. } => {
                        Some(text.clone())
                    }
                    ContentBlock::Json(value) | ContentBlock::ProviderExtension(value) => {
                        Some(value.to_string())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if let Message::Assistant(message) = message {
                visible.extend(
                    message
                        .tool_calls
                        .iter()
                        .filter_map(|call| serde_json::to_string(call).ok()),
                );
            }
            let visible = visible.join("\n");
            tokenize(&visible)
        })
        .sum()
}
