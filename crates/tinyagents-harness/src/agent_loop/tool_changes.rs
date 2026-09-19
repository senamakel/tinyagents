//! Transcript-carried tool-set change patches (gap B6,
//! `docs/runtime-comparison/plan.md`).
//!
//! A [`crate::tool::toolset::ToolSet`] chain's live tool set can legitimately
//! vary turn to turn (`ToolSet::tools` is documented as "called once per
//! turn"). Naively rebuilding the whole system prompt to describe a changed
//! loadout would bust a provider's cached prefix on every such change. These
//! two pure functions are the write side of that mechanism: [`diff_tool_set`]
//! computes the minimal delta between what a transcript has declared so far
//! and what is live now, and [`apply_tool_change_patch`] lays that delta onto
//! the transcript — either as a small mid-conversation
//! [`tinyinference_llm::message::Message::System`] patch (when the resolved
//! model's [`tinyinference_llm::model::ModelProfile::
//! mid_conversation_system_messages`] allows it) or folded into the leading
//! system message (when it does not). The read side,
//! [`tinyinference_llm::message::replay_system_state`], reconstructs the
//! effective tool set from the same transcript.

use std::collections::{BTreeMap, HashSet};

use tinyinference_llm::message::{Message, SystemMessage};
use tinyinference_llm::tool::ToolSchema;

/// Name of the [`SystemMessage::sections`] entry a tool-change patch writes
/// its human-readable summary under.
pub(super) const TOOL_CHANGES_SECTION: &str = "tool_changes";

/// Computes the [`SystemMessage`] patch needed to bring `previous` (what the
/// transcript has declared so far, via [`tinyinference_llm::message::
/// replay_system_state`] or an equivalent running tally) in line with
/// `current` (the toolset chain's live set this turn), or `None` when they
/// already agree.
///
/// Comparison is by full schema equality (name, description, parameters,
/// format): a tool whose declaration changed under the same name is reported
/// only via `tools_added` (the newer schema), never additionally via
/// `tools_removed` — [`replay_system_state`](tinyinference_llm::message::replay_system_state)'s
/// fold semantics already let a later `tools_added` entry for an existing
/// name supersede the earlier one.
pub(super) fn diff_tool_set(previous: &[ToolSchema], current: &[ToolSchema]) -> Option<SystemMessage> {
    let previous_by_name: BTreeMap<&str, &ToolSchema> =
        previous.iter().map(|schema| (schema.name.as_str(), schema)).collect();
    let current_names: HashSet<&str> = current.iter().map(|schema| schema.name.as_str()).collect();

    let mut tools_added: Vec<ToolSchema> = current
        .iter()
        .filter(|schema| previous_by_name.get(schema.name.as_str()) != Some(schema))
        .cloned()
        .collect();
    tools_added.sort_by(|left, right| left.name.cmp(&right.name));

    let mut tools_removed: Vec<String> = previous_by_name
        .keys()
        .filter(|name| !current_names.contains(*name))
        .map(|name| (*name).to_string())
        .collect();
    tools_removed.sort();

    if tools_added.is_empty() && tools_removed.is_empty() {
        return None;
    }

    let mut sections = BTreeMap::new();
    sections.insert(
        TOOL_CHANGES_SECTION.to_string(),
        Some(describe_delta(&tools_added, &tools_removed)),
    );

    Some(SystemMessage {
        content: Vec::new(),
        sections,
        tools_added,
        tools_removed,
    })
}

/// Renders a short human-readable summary of a tool-set delta, used as the
/// patch's `tool_changes` section text.
fn describe_delta(added: &[ToolSchema], removed: &[String]) -> String {
    let mut lines = Vec::new();
    if !added.is_empty() {
        let names: Vec<&str> = added.iter().map(|schema| schema.name.as_str()).collect();
        lines.push(format!("Tools now available: {}.", names.join(", ")));
    }
    if !removed.is_empty() {
        lines.push(format!("Tools no longer available: {}.", removed.join(", ")));
    }
    lines.join(" ")
}

/// Applies a tool-change `patch` to the working transcript.
///
/// When `mid_conversation` is `true` (the resolved model's
/// `ModelProfile::mid_conversation_system_messages` allows a system message
/// anywhere in the transcript, e.g. the OpenAI-compatible chat path), `patch`
/// is appended as a new tail [`Message::System`]. Because
/// `agent_loop/run_loop.rs`'s `system_end` treats only the transcript's
/// **leading** run of `Message::System` entries as the cacheable prefix, a
/// patch appended after any non-system message (the ordinary case: a run
/// always opens with at least one user turn before any tool-set change can
/// occur) lands in the non-cacheable tail and the prefix's
/// [`crate::prompt::PromptBuilder`] fingerprint is unchanged.
///
/// When `mid_conversation` is `false` (for example Anthropic, whose Messages
/// API hoists every `Message::System` in the transcript into one leading
/// `system` array regardless of position — a "mid-conversation" system
/// message would not actually land where it appears), `patch` is folded into
/// the leading system message instead: its `content` is appended, its
/// `sections` are merged key-by-key, and its `tools_added`/`tools_removed`
/// are merged into the leading message's own lists. This *does* change the
/// leading message's content and therefore the prefix fingerprint — expected,
/// since the provider has no mid-transcript system slot for the patch to
/// occupy without moving it there on the wire anyway. A transcript with no
/// leading `Message::System` gets one inserted at the front.
pub(super) fn apply_tool_change_patch(messages: &mut Vec<Message>, patch: SystemMessage, mid_conversation: bool) {
    if mid_conversation {
        messages.push(Message::System(patch));
        return;
    }
    match messages.first_mut() {
        Some(Message::System(leading)) => fold_patch(leading, patch),
        _ => messages.insert(0, Message::System(patch)),
    }
}

/// Merges `patch` into `leading` in place: content is appended, sections are
/// merged key-by-key (`None` removes), and tool deltas are merged so the
/// leading message's own `tools_added`/`tools_removed` reflect the net
/// effect of both the original declaration and the patch.
fn fold_patch(leading: &mut SystemMessage, patch: SystemMessage) {
    leading.content.extend(patch.content);
    for (name, value) in patch.sections {
        match value {
            Some(text) => {
                leading.sections.insert(name, Some(text));
            }
            None => {
                leading.sections.remove(&name);
            }
        }
    }
    for tool in patch.tools_added {
        leading.tools_removed.retain(|name| *name != tool.name);
        leading.tools_added.retain(|existing| existing.name != tool.name);
        leading.tools_added.push(tool);
    }
    for name in patch.tools_removed {
        leading.tools_added.retain(|existing| existing.name != name);
        if !leading.tools_removed.contains(&name) {
            leading.tools_removed.push(name);
        }
    }
}

#[cfg(test)]
mod test;
