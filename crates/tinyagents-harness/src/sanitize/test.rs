use super::*;
use tinyinference_llm::message::{
    AssistantMessage, ImageRef, SystemMessage, ToolMessage, UserMessage,
};
use tinyinference_llm::tool::ToolCall;

fn assistant_with_tool_call(id: &str) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text("calling".into())],
        tool_calls: vec![ToolCall {
            id: id.into(),
            name: "search".into(),
            arguments: serde_json::json!({}),
        }],
        usage: None,
    })
}

fn tool_result(id: &str) -> Message {
    Message::Tool(ToolMessage {
        tool_call_id: id.into(),
        content: vec![ContentBlock::Text("ok".into())],
        trusted_verbatim: false,
        artifact: None,
    })
}

#[test]
fn strips_system_prompts_when_enabled() {
    let mut messages = vec![Message::system("caller-injected"), Message::user("hi")];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    assert_eq!(messages.len(), 1);
    assert!(matches!(messages[0], Message::User(_)));
}

#[test]
fn leaves_system_prompts_when_disabled() {
    let mut messages = vec![Message::system("kept"), Message::user("hi")];
    sanitize_history(
        &mut messages,
        &SanitizePolicy {
            strip_system_prompts: false,
            ..SanitizePolicy::none()
        },
    );
    assert_eq!(messages.len(), 2);
}

#[test]
fn strips_non_http_image_urls_but_keeps_http_and_data() {
    let mut messages = vec![Message::User(UserMessage {
        content: vec![
            ContentBlock::Image(ImageRef {
                url: "file:///etc/passwd".into(),
                mime_type: None,
            }),
            ContentBlock::Image(ImageRef {
                url: "https://example.com/a.png".into(),
                mime_type: None,
            }),
            ContentBlock::Image(ImageRef {
                url: "data:image/png;base64,AAAA".into(),
                mime_type: None,
            }),
            ContentBlock::Text("caption".into()),
        ],
    })];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    let Message::User(user) = &messages[0] else {
        panic!("expected user message");
    };
    assert_eq!(user.content.len(), 3);
    assert!(user.content.iter().any(|b| matches!(
        b,
        ContentBlock::Image(img) if img.url == "https://example.com/a.png"
    )));
    assert!(user.content.iter().any(|b| matches!(
        b,
        ContentBlock::Image(img) if img.url.starts_with("data:")
    )));
    assert!(!user.content.iter().any(|b| matches!(
        b,
        ContentBlock::Image(img) if img.url.starts_with("file:")
    )));
}

#[test]
fn strips_dangling_tool_call_with_no_result() {
    let mut messages = vec![Message::user("go"), assistant_with_tool_call("c1")];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    let Message::Assistant(assistant) = &messages[1] else {
        panic!("expected assistant message");
    };
    assert!(assistant.tool_calls.is_empty());
}

#[test]
fn strips_dangling_tool_result_with_no_declaring_call() {
    let mut messages = vec![Message::user("go"), tool_result("orphan")];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    assert_eq!(messages.len(), 1);
}

#[test]
fn keeps_a_well_paired_tool_call_and_result() {
    let mut messages = vec![
        Message::user("go"),
        assistant_with_tool_call("c1"),
        tool_result("c1"),
    ];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    assert_eq!(messages.len(), 3);
    let Message::Assistant(assistant) = &messages[1] else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.tool_calls.len(), 1);
}

#[test]
fn disabled_dangling_check_leaves_broken_pairing_untouched() {
    let mut messages = vec![Message::user("go"), assistant_with_tool_call("c1")];
    sanitize_history(&mut messages, &SanitizePolicy::none());
    assert_eq!(messages.len(), 2);
    let Message::Assistant(assistant) = &messages[1] else {
        panic!("expected assistant message");
    };
    assert_eq!(assistant.tool_calls.len(), 1);
}

#[test]
fn custom_messages_pass_through_untouched() {
    let mut messages = vec![Message::Custom(tinyinference_llm::message::CustomMessage {
        kind: "label".into(),
        payload: serde_json::json!({"name": "checkpoint"}),
        display: None,
    })];
    sanitize_history(&mut messages, &SanitizePolicy::all());
    assert_eq!(messages.len(), 1);
}

#[test]
fn default_policy_enables_every_check() {
    let policy = SanitizePolicy::default();
    assert!(policy.strip_system_prompts);
    assert!(policy.strip_non_http_file_urls);
    assert!(policy.strip_dangling_tool_calls);
    assert_eq!(policy, SanitizePolicy::all());
}
