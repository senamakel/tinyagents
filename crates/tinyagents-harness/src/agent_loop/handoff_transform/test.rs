//! Unit tests for the cross-provider handoff transform.

use std::borrow::Cow;

use tinyinference_llm::message::{AssistantMessage, ContentBlock, ImageRef, Message, ToolMessage};
use tinyinference_llm::model::{Modalities, ModelProfile};
use tinyinference_llm::tool::ToolCall;

use super::*;

fn anthropic_target() -> (ModelProfile, MessageOrigin) {
    let profile = ModelProfile {
        provider: Some("anthropic".to_string()),
        model: Some("claude-opus-4-6".to_string()),
        modalities: Modalities {
            image_in: true,
            ..Modalities::default()
        },
        tool_call_id_pattern: Some("^[a-zA-Z0-9_-]{1,64}$".to_string()),
        max_tool_call_id_len: Some(64),
        ..ModelProfile::default()
    };
    let origin = MessageOrigin {
        provider: "anthropic".to_string(),
        api: "messages".to_string(),
        model: "claude-opus-4-6".to_string(),
    };
    (profile, origin)
}

fn openai_origin() -> MessageOrigin {
    MessageOrigin {
        provider: "openai".to_string(),
        api: "responses".to_string(),
        model: "gpt-5".to_string(),
    }
}

fn assistant(content: Vec<ContentBlock>, tool_calls: Vec<ToolCall>) -> AssistantMessage {
    AssistantMessage {
        id: None,
        content,
        tool_calls,
        usage: None,
        origin: None,
    }
}

// ---------------------------------------------------------------------------
// Same-origin: no allocation, nothing rewritten.
// ---------------------------------------------------------------------------

#[test]
fn same_origin_transcript_is_untouched_and_borrowed() {
    let (profile, target_origin) = anthropic_target();
    let mut same_origin = assistant(
        vec![
            ContentBlock::thinking("reasoning"),
            ContentBlock::Text("hi".into()),
        ],
        vec![ToolCall::new("toolu_ok", "search", serde_json::json!({}))],
    );
    same_origin.origin = Some(target_origin.clone());
    let messages = vec![
        Message::system("sys"),
        Message::user("hello"),
        Message::Assistant(same_origin),
        Message::Tool(ToolMessage {
            tool_call_id: "toolu_ok".into(),
            content: vec![ContentBlock::Text("42".into())],
            trusted_verbatim: false,
            artifact: None,
        }),
    ];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 0);
    assert!(matches!(outcome.messages, Cow::Borrowed(_)));
    assert_eq!(outcome.messages.as_ref(), messages.as_slice());
}

// ---------------------------------------------------------------------------
// Thinking: redacted dropped, signed converted to text, unsigned untouched.
// ---------------------------------------------------------------------------

#[test]
fn foreign_redacted_thinking_is_dropped_and_signed_thinking_becomes_text() {
    let (profile, target_origin) = anthropic_target();
    let mut foreign = assistant(
        vec![
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
            ContentBlock::Thinking {
                text: "signed reasoning".into(),
                signature: Some("sig-123".into()),
            },
            ContentBlock::Text("visible answer".into()),
        ],
        vec![],
    );
    foreign.origin = Some(openai_origin());
    let messages = vec![Message::Assistant(foreign)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 1);
    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    // Redacted thinking is gone entirely.
    assert!(
        !rewritten
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::RedactedThinking { .. }))
    );
    // The signed thinking became plain visible text (no signature to replay).
    assert!(rewritten.content.contains(&ContentBlock::Text(
        "signed reasoning".to_string()
    )));
    assert!(rewritten.content.contains(&ContentBlock::Text(
        "visible answer".to_string()
    )));
    // The rewritten message no longer claims the origin provider verbatim.
    assert!(rewritten.origin.is_none());
}

#[test]
fn foreign_unsigned_thinking_is_left_alone() {
    let (profile, target_origin) = anthropic_target();
    let mut foreign = assistant(
        vec![
            ContentBlock::Thinking {
                text: "unsigned reasoning".into(),
                signature: None,
            },
            ContentBlock::Text("answer".into()),
        ],
        vec![ToolCall::new("call-needs-fix!", "search", serde_json::json!({}))],
    );
    foreign.origin = Some(openai_origin());
    let messages = vec![Message::Assistant(foreign)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    assert!(rewritten.content.iter().any(|block| matches!(
        block,
        ContentBlock::Thinking { signature: None, .. }
    )));
}

#[test]
fn empty_signed_thinking_is_dropped_rather_than_becoming_an_empty_text_block() {
    let (profile, target_origin) = anthropic_target();
    let mut foreign = assistant(
        vec![
            ContentBlock::Thinking {
                text: String::new(),
                signature: Some("sig".into()),
            },
            ContentBlock::Text("answer".into()),
        ],
        vec![],
    );
    foreign.origin = Some(openai_origin());
    let messages = vec![Message::Assistant(foreign)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    assert_eq!(rewritten.content, vec![ContentBlock::Text("answer".into())]);
}

// ---------------------------------------------------------------------------
// Tool-call id normalization: assistant call id and matching tool result id.
// ---------------------------------------------------------------------------

#[test]
fn foreign_tool_call_ids_are_normalized_and_tool_results_follow() {
    let (profile, target_origin) = anthropic_target();
    let long_id = "resp_call_".to_string() + &"x".repeat(80);
    let mut foreign = assistant(
        vec![ContentBlock::Text("checking".into())],
        vec![ToolCall::new(long_id.clone(), "search", serde_json::json!({"q":"x"}))],
    );
    foreign.origin = Some(openai_origin());
    let messages = vec![
        Message::Assistant(foreign),
        Message::Tool(ToolMessage {
            tool_call_id: long_id.clone(),
            content: vec![ContentBlock::Text("result".into())],
            trusted_verbatim: false,
            artifact: None,
        }),
    ];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 2);
    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    let new_id = rewritten.tool_calls[0].id.clone();
    assert_ne!(new_id, long_id, "the id must be rewritten");
    assert!(new_id.chars().count() <= 64);
    assert!(
        new_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    );

    let Message::Tool(tool_result) = &outcome.messages[1] else {
        panic!("expected a tool message");
    };
    assert_eq!(
        tool_result.tool_call_id, new_id,
        "the tool result must follow the same remapping"
    );
}

#[test]
fn conforming_tool_call_ids_are_left_untouched_even_on_a_foreign_message() {
    let (profile, target_origin) = anthropic_target();
    let mut foreign = assistant(
        vec![ContentBlock::Text("checking".into())],
        vec![ToolCall::new("toolu_fine", "search", serde_json::json!({}))],
    );
    foreign.origin = Some(openai_origin());
    let messages = vec![Message::Assistant(foreign)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    assert_eq!(rewritten.tool_calls[0].id, "toolu_fine");
}

#[test]
fn mint_tool_call_id_deduplicates_against_used_ids() {
    let profile = ModelProfile {
        max_tool_call_id_len: Some(6),
        ..ModelProfile::default()
    };
    let mut used = std::collections::HashSet::new();
    let first = mint_tool_call_id("abc!def", &profile, &mut used);
    assert_eq!(first, "abc_de");
    let second = mint_tool_call_id("abc!def", &profile, &mut used);
    assert_ne!(second, first, "a colliding id must be disambiguated");
    assert!(second.chars().count() <= 6);
}

// ---------------------------------------------------------------------------
// Image downgrade: assistant content, and user/tool content (no origin).
// ---------------------------------------------------------------------------

fn no_vision_target() -> (ModelProfile, MessageOrigin) {
    let profile = ModelProfile {
        provider: Some("openai".to_string()),
        model: Some("gpt-5-mini".to_string()),
        ..ModelProfile::default()
    };
    let origin = MessageOrigin {
        provider: "openai".to_string(),
        api: "chat_completions".to_string(),
        model: "gpt-5-mini".to_string(),
    };
    (profile, origin)
}

#[test]
fn foreign_assistant_image_is_downgraded_when_target_lacks_vision() {
    let (profile, target_origin) = no_vision_target();
    let mut foreign = assistant(
        vec![
            ContentBlock::Image(ImageRef {
                url: "data:image/png;base64,xyz".into(),
                mime_type: Some("image/png".into()),
            }),
            ContentBlock::Text("see above".into()),
        ],
        vec![],
    );
    foreign.origin = Some(MessageOrigin {
        provider: "anthropic".into(),
        api: "messages".into(),
        model: "claude-opus-4-6".into(),
    });
    let messages = vec![Message::Assistant(foreign)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    assert!(
        !rewritten
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Image(_)))
    );
    assert!(
        rewritten
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Text(text) if text.contains("image")))
    );
}

#[test]
fn user_image_is_downgraded_regardless_of_origin_because_users_carry_none() {
    let (profile, target_origin) = no_vision_target();
    let messages = vec![Message::User(tinyinference_llm::message::UserMessage {
        content: vec![
            ContentBlock::Image(ImageRef {
                url: "https://example.com/pic.png".into(),
                mime_type: None,
            }),
            ContentBlock::Text("what is this?".into()),
        ],
    })];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 1);
    let Message::User(rewritten) = &outcome.messages[0] else {
        panic!("expected a user message");
    };
    assert!(
        !rewritten
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Image(_)))
    );
}

#[test]
fn image_is_untouched_when_the_target_supports_vision() {
    let (profile, target_origin) = anthropic_target();
    let messages = vec![Message::User(tinyinference_llm::message::UserMessage {
        content: vec![ContentBlock::Image(ImageRef {
            url: "https://example.com/pic.png".into(),
            mime_type: None,
        })],
    })];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 0);
    assert!(matches!(outcome.messages, Cow::Borrowed(_)));
}

// ---------------------------------------------------------------------------
// Legacy (no-origin) messages: foreign only when structurally unacceptable.
// ---------------------------------------------------------------------------

#[test]
fn legacy_message_with_clean_content_is_not_touched() {
    let (profile, target_origin) = anthropic_target();
    let legacy = assistant(vec![ContentBlock::Text("hello".into())], vec![]);
    assert!(legacy.origin.is_none());
    let messages = vec![Message::Assistant(legacy)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 0);
    assert!(matches!(outcome.messages, Cow::Borrowed(_)));
}

#[test]
fn legacy_message_with_redacted_thinking_is_treated_as_foreign() {
    let (profile, target_origin) = anthropic_target();
    let legacy = assistant(
        vec![
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
            ContentBlock::Text("hello".into()),
        ],
        vec![],
    );
    let messages = vec![Message::Assistant(legacy)];

    let outcome = prepare_for_model(&messages, &profile, &target_origin);

    assert_eq!(outcome.changes, 1);
    let Message::Assistant(rewritten) = &outcome.messages[0] else {
        panic!("expected an assistant message");
    };
    assert!(
        !rewritten
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::RedactedThinking { .. }))
    );
}

// ---------------------------------------------------------------------------
// `target_origin_for`
// ---------------------------------------------------------------------------

#[test]
fn target_origin_for_names_anthropic_messages_api() {
    let profile = ModelProfile {
        provider: Some("anthropic".to_string()),
        model: Some("claude-opus-4-6".to_string()),
        ..ModelProfile::default()
    };
    let origin = target_origin_for(&profile);
    assert_eq!(origin.provider, "anthropic");
    assert_eq!(origin.api, "messages");
    assert_eq!(origin.model, "claude-opus-4-6");
}

#[test]
fn target_origin_for_assumes_chat_completions_for_other_known_providers() {
    let profile = ModelProfile {
        provider: Some("ollama".to_string()),
        model: Some("llama3".to_string()),
        ..ModelProfile::default()
    };
    let origin = target_origin_for(&profile);
    assert_eq!(origin.api, "chat_completions");
}
