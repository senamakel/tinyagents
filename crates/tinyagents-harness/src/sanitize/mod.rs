//! History sanitization for untrusted or externally-assembled message
//! histories.
//!
//! A host that lets a caller resume a run with a caller-supplied history (a
//! resumed session, an imported transcript, a client replaying its own
//! record of a conversation) must not trust that history the way it trusts
//! its own agent loop's output. Three shapes of untrusted input are common:
//!
//! - A caller-supplied `system` message trying to override the host's own
//!   system prompt (prompt injection via history replay).
//! - An image/file content block pointing at a non-HTTP URL (`file://`, a
//!   bare local path, or another scheme) that would make the host's own
//!   process fetch from a caller-controlled location when the block is
//!   resolved.
//! - A dangling tool call or tool result: an assistant `tool_calls` entry
//!   with no answering [`Message::Tool`], or a tool result naming a call id
//!   that was never declared. Every provider rejects these, and letting one
//!   through turns a client-side bug into a 400 deep inside a run.
//!
//! [`sanitize_history`] strips all three under an explicit [`SanitizePolicy`]
//! so a host opts into exactly the checks its trust boundary needs.

mod types;

pub use types::SanitizePolicy;

use std::collections::HashSet;
use tinyinference_llm::message::{ContentBlock, Message};

/// URL prefixes [`sanitize_history`] treats as fetchable by the host's own
/// process rather than an opaque local/foreign path. `data:` URIs are inline
/// and carry no fetch, so they are always allowed regardless of policy.
const ALLOWED_URL_PREFIXES: &[&str] = &["http://", "https://", "data:"];

/// Sanitizes `messages` in place according to `policy`.
///
/// Checks apply in this order: system-prompt stripping, then file-URL
/// stripping, then dangling tool-call repair (which must run last so it sees
/// the final message shape). Each check is independently toggleable; a
/// disabled check leaves that class of content untouched.
pub fn sanitize_history(messages: &mut Vec<Message>, policy: &SanitizePolicy) {
    if policy.strip_system_prompts {
        strip_system_prompts(messages);
    }
    if policy.strip_non_http_file_urls {
        strip_non_http_file_urls(messages);
    }
    if policy.strip_dangling_tool_calls {
        strip_dangling_tool_calls(messages);
    }
}

/// Removes every [`Message::System`] entry.
///
/// A host that injects its own authoritative system prompt at request-build
/// time never wants a caller-supplied history to carry a competing one; the
/// host's own prompt is added back separately (this function only removes,
/// it never inserts).
fn strip_system_prompts(messages: &mut Vec<Message>) {
    messages.retain(|message| !matches!(message, Message::System(_)));
}

/// Drops [`ContentBlock::Image`] blocks whose URL is not `http(s)://` or an
/// inline `data:` URI, from every message kind that carries content blocks.
/// The message itself is kept (with the remaining blocks, possibly empty) so
/// this never disturbs tool-call pairing.
fn strip_non_http_file_urls(messages: &mut [Message]) {
    for message in messages.iter_mut() {
        let content = match message {
            Message::System(m) => &mut m.content,
            Message::User(m) => &mut m.content,
            Message::Assistant(m) => &mut m.content,
            Message::Tool(m) => &mut m.content,
            Message::Custom(_) => continue,
        };
        content.retain(|block| match block {
            ContentBlock::Image(image) => ALLOWED_URL_PREFIXES
                .iter()
                .any(|prefix| image.url.starts_with(prefix)),
            _ => true,
        });
    }
}

/// Removes dangling tool-call structure: an assistant `tool_calls` entry with
/// no answering [`Message::Tool`], and a tool result whose `tool_call_id` was
/// never declared by any assistant turn. Runs over the whole message list
/// (not a single trim boundary), so it repairs history assembled out of order
/// or from multiple sources, not just a single cut point.
fn strip_dangling_tool_calls(messages: &mut Vec<Message>) {
    let answered: HashSet<&str> = messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(tool) => Some(tool.tool_call_id.as_str()),
            _ => None,
        })
        .collect();
    let declared: HashSet<String> = messages
        .iter()
        .flat_map(|message| match message {
            Message::Assistant(assistant) => assistant
                .tool_calls
                .iter()
                .map(|call| call.id.clone())
                .collect(),
            _ => Vec::new(),
        })
        .collect();

    for message in messages.iter_mut() {
        if let Message::Assistant(assistant) = message {
            assistant
                .tool_calls
                .retain(|call| answered.contains(call.id.as_str()));
        }
    }
    messages.retain(|message| match message {
        Message::Tool(tool) => declared.contains(tool.tool_call_id.as_str()),
        _ => true,
    });
}

#[cfg(test)]
mod test;
