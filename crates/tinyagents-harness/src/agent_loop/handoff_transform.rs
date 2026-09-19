//! Cross-provider handoff transform.
//!
//! A run's transcript can span more than one provider or model across its
//! lifetime: an explicit [`ModelRequest::model`][tinyinference_llm::model::ModelRequest::model]
//! override, a fallback chain, or a host-driven routing decision can each
//! hand the *next* model call a transcript whose assistant messages were
//! produced by a *different* provider. Left alone, that transcript can carry
//! content the new target cannot replay:
//!
//! * a provider-encrypted [`ContentBlock::RedactedThinking`] block, opaque
//!   outside the provider that emitted it;
//! * a signed [`ContentBlock::Thinking`] block whose signature only that
//!   provider can verify;
//! * tool-call ids shaped for the origin provider (for example an OpenAI
//!   Responses id) that the target provider rejects outright (Anthropic caps
//!   `tool_use`/`tool_result` ids at 64 characters matching
//!   `^[a-zA-Z0-9_-]{1,64}$`);
//! * an image block when the target model has no vision input.
//!
//! [`prepare_for_model`] runs as a pure, read-only-in/owned-out pass over the
//! working transcript immediately before a [`ModelRequest`] is dispatched. It
//! never touches [`Turn`][crate::run_queue]/`RunQueue` bookkeeping (a
//! different concern owned elsewhere in the loop) and never mutates its
//! input: same-origin transcripts — the overwhelming common case, a run that
//! never switches provider — are returned as
//! [`Cow::Borrowed`][std::borrow::Cow::Borrowed] with zero allocation.
//!
//! # What counts as "foreign"
//!
//! An [`AssistantMessage`] with an [`origin`][AssistantMessage::origin] is
//! foreign when that origin differs from the call's `target_origin` in
//! *any* of `provider`, `api`, or `model`. An assistant message with no
//! origin (replayed from a journal written before this field existed, or
//! authored directly by the host) is foreign only if it structurally
//! carries content the target cannot accept — the same-origin optimism a
//! stamped message gets is not extended to a message the harness cannot
//! actually verify came from the target.
//!
//! [`Message::User`] and [`Message::Tool`] carry no origin at all (only a
//! model produces an [`AssistantMessage`]), so they are always checked
//! structurally: an image block is downgraded whenever the target lacks
//! vision input, regardless of which turn produced the message.
//!
//! # Tool-call id remapping
//!
//! A foreign assistant message's non-conforming tool-call ids are rewritten
//! to a target-conforming shape (ASCII alphanumeric/`_`/`-`, truncated to
//! [`ModelProfile::max_tool_call_id_len`] when set, de-duplicated against
//! every id already in the transcript) through one id map built for the
//! whole call. Every [`Message::Tool`] whose `tool_call_id` answers a
//! remapped call is rewritten with the same mapping, so a tool result never
//! ends up orphaned from the call it answers.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use regex::Regex;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, MessageOrigin};
use tinyinference_llm::model::ModelProfile;

/// Placeholder text substituted for an image block the target model cannot
/// accept.
const IMAGE_PLACEHOLDER: &str = "[image omitted: not supported by the target model]";

/// Outcome of [`prepare_for_model`]: the (possibly unchanged) messages plus
/// how many messages were rewritten, so the caller can decide whether to
/// emit [`crate::events::AgentEvent::HandoffTransformApplied`].
pub(super) struct HandoffTransformOutcome<'a> {
    /// The transcript to send to the model. Borrowed when nothing changed.
    pub(super) messages: Cow<'a, [Message]>,
    /// Number of messages rewritten (0 when `messages` is [`Cow::Borrowed`]).
    pub(super) changes: usize,
}

/// Prepares a transcript for a call against `target`/`target_origin`,
/// rewriting only the messages a cross-provider handoff makes unsafe to
/// replay verbatim. See the module documentation for the exact rules.
///
/// Pure and allocation-free on the common path: a transcript with no
/// foreign content (including every same-origin run, which is most runs)
/// returns [`Cow::Borrowed`] over `messages`.
pub(super) fn prepare_for_model<'a>(
    messages: &'a [Message],
    target: &ModelProfile,
    target_origin: &MessageOrigin,
) -> HandoffTransformOutcome<'a> {
    let id_pattern = target
        .tool_call_id_pattern
        .as_deref()
        .and_then(|pattern| Regex::new(pattern).ok());

    // Pass 1: decide which assistant tool-call ids need remapping, and seed
    // the "already used" set with every id already in the transcript so a
    // freshly minted id can never collide with one that was already fine.
    let mut used_ids: HashSet<String> = HashSet::new();
    let mut id_map: HashMap<String, String> = HashMap::new();
    for message in messages {
        if let Message::Assistant(assistant) = message {
            let foreign = is_foreign(assistant, target, target_origin, id_pattern.as_ref());
            for call in &assistant.tool_calls {
                if foreign && !tool_call_id_conforms(&call.id, target, id_pattern.as_ref()) {
                    let new_id = mint_tool_call_id(&call.id, target, &mut used_ids);
                    id_map.insert(call.id.clone(), new_id);
                } else {
                    used_ids.insert(call.id.clone());
                }
            }
        }
    }

    if id_map.is_empty()
        && !messages.iter().any(|message| match message {
            Message::Assistant(assistant) => {
                is_foreign(assistant, target, target_origin, id_pattern.as_ref())
            }
            Message::User(user) => content_needs_image_downgrade(&user.content, target),
            Message::Tool(tool) => content_needs_image_downgrade(&tool.content, target),
            Message::System(_) | Message::Custom(_) => false,
        })
    {
        return HandoffTransformOutcome {
            messages: Cow::Borrowed(messages),
            changes: 0,
        };
    }

    let mut changes = 0usize;
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                if is_foreign(assistant, target, target_origin, id_pattern.as_ref()) {
                    out.push(Message::Assistant(transform_assistant(
                        assistant, target, &id_map,
                    )));
                    changes += 1;
                } else {
                    out.push(message.clone());
                }
            }
            Message::Tool(tool) => {
                let remapped_id = id_map.get(&tool.tool_call_id);
                let needs_image_downgrade = content_needs_image_downgrade(&tool.content, target);
                if remapped_id.is_some() || needs_image_downgrade {
                    let mut tool = tool.clone();
                    if let Some(new_id) = remapped_id {
                        tool.tool_call_id = new_id.clone();
                    }
                    if needs_image_downgrade {
                        tool.content = downgrade_images(tool.content, target);
                    }
                    out.push(Message::Tool(tool));
                    changes += 1;
                } else {
                    out.push(message.clone());
                }
            }
            Message::User(user) => {
                if content_needs_image_downgrade(&user.content, target) {
                    let mut user = user.clone();
                    user.content = downgrade_images(user.content, target);
                    out.push(Message::User(user));
                    changes += 1;
                } else {
                    out.push(message.clone());
                }
            }
            other => out.push(other.clone()),
        }
    }

    HandoffTransformOutcome {
        messages: Cow::Owned(out),
        changes,
    }
}

/// Whether `assistant` was produced by a different provider/api/model than
/// `target_origin` (or, lacking a stamped origin, structurally carries
/// content the target cannot accept).
fn is_foreign(
    assistant: &AssistantMessage,
    target: &ModelProfile,
    target_origin: &MessageOrigin,
    id_pattern: Option<&Regex>,
) -> bool {
    match &assistant.origin {
        Some(origin) => origin != target_origin,
        None => {
            assistant.content.iter().any(|block| {
                matches!(block, ContentBlock::RedactedThinking { .. })
                    || matches!(block, ContentBlock::Thinking { signature: Some(_), .. })
                    || (matches!(block, ContentBlock::Image(_)) && !target.modalities.image_in)
            }) || assistant
                .tool_calls
                .iter()
                .any(|call| !tool_call_id_conforms(&call.id, target, id_pattern))
        }
    }
}

/// Whether `content` carries an image block the target cannot accept.
fn content_needs_image_downgrade(content: &[ContentBlock], target: &ModelProfile) -> bool {
    !target.modalities.image_in
        && content
            .iter()
            .any(|block| matches!(block, ContentBlock::Image(_)))
}

/// Replaces every [`ContentBlock::Image`] with a text placeholder.
fn downgrade_images(content: Vec<ContentBlock>, target: &ModelProfile) -> Vec<ContentBlock> {
    if target.modalities.image_in {
        return content;
    }
    content
        .into_iter()
        .map(|block| match block {
            ContentBlock::Image(_) => ContentBlock::Text(IMAGE_PLACEHOLDER.to_string()),
            other => other,
        })
        .collect()
}

/// Rewrites a foreign assistant message: drops
/// [`ContentBlock::RedactedThinking`], converts a signed
/// [`ContentBlock::Thinking`] to plain text (or drops it when empty),
/// downgrades images the target cannot accept, and remaps any tool-call id
/// found in `id_map`.
fn transform_assistant(
    assistant: &AssistantMessage,
    target: &ModelProfile,
    id_map: &HashMap<String, String>,
) -> AssistantMessage {
    let content = assistant
        .content
        .iter()
        .cloned()
        .filter_map(|block| match block {
            ContentBlock::RedactedThinking { .. } => None,
            ContentBlock::Thinking {
                text,
                signature: Some(_),
            } => {
                if text.is_empty() {
                    None
                } else {
                    Some(ContentBlock::Text(text))
                }
            }
            ContentBlock::Image(_) if !target.modalities.image_in => {
                Some(ContentBlock::Text(IMAGE_PLACEHOLDER.to_string()))
            }
            other => Some(other),
        })
        .collect();
    let tool_calls = assistant
        .tool_calls
        .iter()
        .cloned()
        .map(|mut call| {
            if let Some(new_id) = id_map.get(&call.id) {
                call.id = new_id.clone();
            }
            call
        })
        .collect();
    AssistantMessage {
        id: assistant.id.clone(),
        content,
        tool_calls,
        usage: assistant.usage,
        // The message no longer verbatim-replays what the origin provider
        // produced (thinking stripped, ids remapped): it is no longer a
        // faithful record of that origin, so clear it rather than leave a
        // stale claim.
        origin: None,
    }
}

/// Whether `id` already satisfies the target's shape/length constraints. A
/// target with neither constraint accepts every id.
fn tool_call_id_conforms(id: &str, target: &ModelProfile, id_pattern: Option<&Regex>) -> bool {
    if let Some(max_len) = target.max_tool_call_id_len
        && id.chars().count() > max_len
    {
        return false;
    }
    match id_pattern {
        Some(pattern) => pattern.is_match(id),
        None => true,
    }
}

/// Mints a target-conforming replacement for a non-conforming tool-call id:
/// disallowed characters become `_`, the result is truncated to
/// [`ModelProfile::max_tool_call_id_len`] when set, and a numeric suffix is
/// added (shrinking the base further if needed) until the candidate is
/// unique against every id `used` records. The winning candidate is
/// inserted into `used` before returning so a later collision is not
/// possible.
fn mint_tool_call_id(id: &str, target: &ModelProfile, used: &mut HashSet<String>) -> String {
    let sanitized: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let sanitized = if sanitized.is_empty() {
        "tc".to_string()
    } else {
        sanitized
    };
    let max_len = target.max_tool_call_id_len.unwrap_or(usize::MAX);
    let truncated: String = sanitized.chars().take(max_len).collect();

    let mut candidate = truncated.clone();
    let mut suffix = 1u32;
    while used.contains(&candidate) {
        let suffix_str = format!("-{suffix}");
        let keep = max_len
            .saturating_sub(suffix_str.chars().count())
            .max(1);
        let base: String = truncated.chars().take(keep).collect();
        candidate = format!("{base}{suffix_str}");
        suffix += 1;
    }
    used.insert(candidate.clone());
    candidate
}

#[cfg(test)]
mod test;
