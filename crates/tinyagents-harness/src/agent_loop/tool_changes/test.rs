//! Unit tests for the tool-change patch mechanism (gap B6).
//!
//! Covers: exactly-one-patch diffing, transcript round-tripping through
//! [`replay_system_state`], and that the mid-conversation insert path keeps
//! the leading-prefix [`PromptBuilder`] fingerprint stable while the folded
//! path (correctly) does not.

use serde_json::json;
use tinyinference_llm::message::{Message, replay_system_state};
use tinyinference_llm::tool::ToolSchema;

use super::*;
use crate::prompt::PromptBuilder;

fn tool(name: &str) -> ToolSchema {
    ToolSchema::new(name, format!("{name} tool"), json!({"type": "object"}))
}

#[test]
fn no_diff_when_tool_sets_match() {
    let set = vec![tool("search"), tool("read_file")];
    assert!(diff_tool_set(&set, &set).is_none());
}

#[test]
fn diff_reports_additions_and_removals() {
    let previous = vec![tool("search"), tool("read_file")];
    let current = vec![tool("search"), tool("browse")];

    let patch = diff_tool_set(&previous, &current).expect("tool set changed");
    let added: Vec<&str> = patch.tools_added.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(added, vec!["browse"]);
    assert_eq!(patch.tools_removed, vec!["read_file".to_string()]);
    assert!(patch.sections.contains_key(TOOL_CHANGES_SECTION));
    let summary = patch.sections[TOOL_CHANGES_SECTION].as_ref().unwrap();
    assert!(summary.contains("browse"));
    assert!(summary.contains("read_file"));
}

#[test]
fn diff_reports_a_changed_schema_as_an_addition_only() {
    let previous = vec![ToolSchema::new("search", "v1", json!({"type": "object"}))];
    let current = vec![ToolSchema::new("search", "v2", json!({"type": "object"}))];

    let patch = diff_tool_set(&previous, &current).expect("schema changed");
    assert_eq!(patch.tools_added.len(), 1);
    assert_eq!(patch.tools_added[0].description, "v2");
    assert!(patch.tools_removed.is_empty());
}

/// Test 1: a tool-set change mid-run results in **exactly one** patch system
/// message being appended when the profile supports mid-conversation system
/// messages.
#[test]
fn mid_conversation_patch_appends_exactly_one_system_message() {
    let mut messages = vec![
        Message::system("baseline persona"),
        Message::user("hi"),
        Message::assistant("hello"),
    ];
    let system_count_before = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .count();

    let previous = vec![tool("search")];
    let current = vec![tool("search"), tool("browse")];
    let patch = diff_tool_set(&previous, &current).expect("tool set changed");
    apply_tool_change_patch(&mut messages, patch, true);

    let system_count_after = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .count();
    assert_eq!(system_count_after, system_count_before + 1);
    // The patch landed at the tail, after the existing conversation.
    assert!(matches!(messages.last(), Some(Message::System(_))));
}

/// Test 1 (fold variant): when the profile does not support mid-conversation
/// system messages, the patch is folded into the leading system message
/// instead of appending a new one — the system-message *count* does not
/// grow, but the delta is still fully recorded.
#[test]
fn folded_patch_does_not_add_a_new_system_message() {
    let mut messages = vec![Message::system("baseline persona"), Message::user("hi")];
    let previous = vec![tool("search")];
    let current = vec![tool("search"), tool("browse")];
    let patch = diff_tool_set(&previous, &current).expect("tool set changed");
    apply_tool_change_patch(&mut messages, patch, false);

    let system_count = messages
        .iter()
        .filter(|m| matches!(m, Message::System(_)))
        .count();
    assert_eq!(system_count, 1);
    let Message::System(leading) = &messages[0] else {
        panic!("expected a leading system message");
    };
    assert_eq!(
        leading
            .tools_added
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        vec!["browse"]
    );
}

/// Test 2: `replay_system_state` on a transcript carrying multiple patches
/// reconstructs the same effective tool set that was live when each patch was
/// produced.
#[test]
fn replay_round_trips_a_sequence_of_mid_conversation_patches() {
    let mut messages = vec![Message::system("baseline persona"), Message::user("hi")];

    // Turn 1 -> 2: the live set grows from {} to {search, read_file}.
    let live_1 = vec![tool("search"), tool("read_file")];
    if let Some(patch) = diff_tool_set(&[], &live_1) {
        apply_tool_change_patch(&mut messages, patch, true);
    }
    messages.push(Message::assistant("using search"));

    // Turn 2 -> 3: read_file drops, browse appears.
    let live_2 = vec![tool("search"), tool("browse")];
    if let Some(patch) = diff_tool_set(&live_1, &live_2) {
        apply_tool_change_patch(&mut messages, patch, true);
    }
    messages.push(Message::assistant("using browse"));

    let (_, replayed_tools) = replay_system_state(&messages);
    let mut replayed_names: Vec<&str> = replayed_tools.iter().map(|t| t.name.as_str()).collect();
    replayed_names.sort();
    let mut expected_names: Vec<&str> = live_2.iter().map(|t| t.name.as_str()).collect();
    expected_names.sort();
    assert_eq!(replayed_names, expected_names);
}

/// Test 3: with a profile that supports mid-conversation system messages,
/// inserting the patch does not change the fingerprint of the unchanged
/// leading system-segment prefix.
#[test]
fn mid_conversation_patch_keeps_the_leading_prefix_fingerprint_stable() {
    let leading = vec![Message::system("baseline persona")];

    let mut before = PromptBuilder::new();
    before.push_system("system", leading.clone());
    let fingerprint_before = before.fingerprint();

    // Simulate the patched transcript: the leading system run is untouched;
    // the patch lands after the tail (a user turn, then the patch).
    let mut messages = leading.clone();
    messages.push(Message::user("hi"));
    let previous = vec![tool("search")];
    let current = vec![tool("search"), tool("browse")];
    let patch = diff_tool_set(&previous, &current).expect("tool set changed");
    apply_tool_change_patch(&mut messages, patch, true);

    // Recompute `system_end` the way `run_loop.rs` does: the leading run of
    // `Message::System` messages only.
    let system_end = messages
        .iter()
        .take_while(|m| matches!(m, Message::System(_)))
        .count();
    let mut after = PromptBuilder::new();
    after.push_system("system", messages[..system_end].to_vec());
    let fingerprint_after = after.fingerprint();

    assert_eq!(fingerprint_before, fingerprint_after);
}

/// Test 3 (companion): when the profile does *not* support mid-conversation
/// system messages, the patch is folded into the leading system message, so
/// the prefix fingerprint is expected to change — this asserts correctness
/// of the folded content, not cache stability.
#[test]
fn folded_patch_changes_the_leading_prefix_fingerprint_but_stays_correct() {
    let leading = vec![Message::system("baseline persona")];

    let mut before = PromptBuilder::new();
    before.push_system("system", leading.clone());
    let fingerprint_before = before.fingerprint();

    let mut messages = leading.clone();
    let previous = vec![tool("search")];
    let current = vec![tool("search"), tool("browse")];
    let patch = diff_tool_set(&previous, &current).expect("tool set changed");
    apply_tool_change_patch(&mut messages, patch, false);

    let mut after = PromptBuilder::new();
    after.push_system("system", messages.clone());
    let fingerprint_after = after.fingerprint();

    // The fold path is expected to recompute the fingerprint...
    assert_ne!(fingerprint_before, fingerprint_after);
    // ...but the fold itself is correct: the leading message now declares the
    // full current tool set.
    let Message::System(leading_after) = &messages[0] else {
        panic!("expected a leading system message");
    };
    assert_eq!(
        leading_after
            .tools_added
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        vec!["browse"]
    );
}
