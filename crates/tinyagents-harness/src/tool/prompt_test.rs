//! Tests for the prompt-guided tool-call protocol.

use super::*;

#[test]
fn parser_extracts_a_delimited_tool_call() {
    let (cleaned, calls) = parse_prompt_tool_calls_from_text(
        "Before <tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}</tool_call> after",
    );

    assert_eq!(cleaned, "Before  after");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "read_file");
    assert_eq!(calls[0].arguments, serde_json::json!({"path": "a.txt"}));
}

#[test]
fn bare_tool_call_accepts_relaxed_json_in_a_fence() {
    let call =
        parse_bare_tool_call("```json\n{'name':'search','parameters':{'query':'rust'}}\n```")
            .expect("a fenced relaxed JSON tool call");

    assert_eq!(call.name, "search");
    assert_eq!(call.arguments, serde_json::json!({"query": "rust"}));
}

#[test]
fn bare_tool_call_does_not_consume_prose() {
    assert!(parse_bare_tool_call("Try this: {\"name\":\"search\"}").is_none());
}

#[test]
fn recovery_only_runs_when_tools_were_offered() {
    assert!(should_recover(false, true, 0));
    assert!(should_recover(true, true, 0));
    assert!(!should_recover(true, true, 1));
    assert!(!should_recover(false, false, 0));
}
