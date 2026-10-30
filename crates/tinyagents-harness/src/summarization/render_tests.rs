use super::*;
use crate::summarization::ToolRendering;
use serde_json::json;
use tinyinference_llm::message::AssistantMessage;
use tinyinference_llm::tool::ToolCall;

fn assistant(text: Option<&str>, calls: Vec<ToolCall>) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: text
            .map(|t| ContentBlock::Text(t.into()))
            .into_iter()
            .collect(),
        tool_calls: calls,
        usage: None,
        origin: None,
    })
}

#[test]
fn full_rendering_is_the_existing_lossless_summary_format() {
    let message = Message::tool("call-1", "tool result");
    assert_eq!(
        render_message_for_summary_with(&message, ToolRendering::Full),
        Some(render_message_for_summary(&message))
    );
}

#[test]
fn marker_rendering_keeps_calls_but_omits_tool_results() {
    let message = assistant(
        Some("checking"),
        vec![ToolCall::new(
            "call-1",
            "search",
            json!({"query": "weather"}),
        )],
    );
    assert_eq!(
        render_message_for_summary_with(&message, ToolRendering::Markers).as_deref(),
        Some("assistant: checking\n[used search: {\"query\":\"weather\"}]")
    );
    assert_eq!(
        render_message_for_summary_with(
            &Message::tool("call-1", "private payload"),
            ToolRendering::Markers
        ),
        None
    );
}

#[test]
fn marker_rendering_keeps_tool_only_turns_and_bounds_arguments() {
    let message = assistant(
        None,
        vec![ToolCall::new(
            "call-1",
            "search",
            json!({"query": "é".repeat(200)}),
        )],
    );
    let rendered = render_message_for_summary_with(&message, ToolRendering::Markers)
        .expect("tool-only assistant turn is retained");
    assert!(rendered.starts_with("[used search: "));
    assert!(rendered.ends_with("…]"));
    assert_eq!(rendered.split_once(": ").unwrap().1.chars().count(), 120);
}
