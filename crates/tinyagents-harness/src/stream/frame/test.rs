use serde_json::json;

use tinyinference_llm::message::{AssistantMessage, ContentBlock};
use tinyinference_llm::model::{BlockDelta, BlockKind, ModelResponse, ModelStreamItem};
use tinyinference_llm::usage::Usage;

use super::*;

fn interleaved_stream_items() -> Vec<ModelStreamItem> {
    vec![
        ModelStreamItem::Started,
        ModelStreamItem::BlockStart {
            index: 0,
            kind: BlockKind::Thinking,
        },
        ModelStreamItem::BlockDelta {
            index: 0,
            delta: BlockDelta::Thinking("plan".into()),
        },
        ModelStreamItem::BlockEnd {
            index: 0,
            block: ContentBlock::Thinking {
                text: "plan".into(),
                signature: None,
            },
        },
        ModelStreamItem::BlockStart {
            index: 1,
            kind: BlockKind::Text,
        },
        ModelStreamItem::BlockDelta {
            index: 1,
            delta: BlockDelta::Text("hel".into()),
        },
        ModelStreamItem::BlockDelta {
            index: 1,
            delta: BlockDelta::Text("lo".into()),
        },
        ModelStreamItem::BlockEnd {
            index: 1,
            block: ContentBlock::Text("hello".into()),
        },
        ModelStreamItem::BlockStart {
            index: 2,
            kind: BlockKind::ToolCall {
                id: "call-1".into(),
                name: "search".into(),
            },
        },
        ModelStreamItem::BlockDelta {
            index: 2,
            delta: BlockDelta::ToolArgs("{\"q\":".into()),
        },
        ModelStreamItem::BlockDelta {
            index: 2,
            delta: BlockDelta::ToolArgs("1}".into()),
        },
        ModelStreamItem::BlockEnd {
            index: 2,
            block: ContentBlock::Json(json!({"id": "call-1", "name": "search", "arguments": {"q": 1}})),
        },
        ModelStreamItem::UsageDelta(Usage::new(5, 7)),
        ModelStreamItem::Completed(ModelResponse {
            message: AssistantMessage {
                id: Some("msg-1".into()),
                content: vec![
                    ContentBlock::Thinking {
                        text: "plan".into(),
                        signature: None,
                    },
                    ContentBlock::Text("hello".into()),
                ],
                tool_calls: vec![tinyinference_llm::tool::ToolCall::new(
                    "call-1",
                    "search",
                    json!({"q": 1}),
                )],
                usage: Some(Usage::new(5, 7)),
            },
            usage: Some(Usage::new(5, 7)),
            finish_reason: Some("tool_use".into()),
            raw: None,
            resolved_model: None,
            continue_turn: None,
            served_from_cache: false,
            correlation: None,
            resolved_route: None,
        }),
    ]
}

#[test]
fn encode_skips_flat_compatibility_items_and_frames_block_items() {
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::BlockStart {
            index: 0,
            kind: BlockKind::Text,
        },
        ModelStreamItem::BlockDelta {
            index: 0,
            delta: BlockDelta::Text("hi".into()),
        },
        ModelStreamItem::MessageDelta(tinyinference_llm::message::MessageDelta::text("hi")),
    ];
    let frames = encode_frames(&items);
    assert_eq!(
        frames,
        vec![
            AssistantFrame::BlockStart {
                index: 0,
                kind: BlockKind::Text,
            },
            AssistantFrame::BlockDelta {
                index: 0,
                delta: BlockDelta::Text("hi".into()),
            },
        ]
    );
}

#[test]
fn encode_then_reduce_round_trips_to_the_terminal_message() {
    let items = interleaved_stream_items();
    let frames = encode_frames(&items);
    let partial = reduce_frames(&frames);

    assert!(partial.open_blocks.is_empty(), "every block closed");
    assert_eq!(
        partial.content,
        vec![
            ContentBlock::Thinking {
                text: "plan".into(),
                signature: None,
            },
            ContentBlock::Text("hello".into()),
        ]
    );
    assert_eq!(partial.tool_calls.len(), 1);
    assert_eq!(partial.tool_calls[0].id, "call-1");
    assert_eq!(partial.tool_calls[0].name, "search");
    assert_eq!(partial.tool_calls[0].arguments, json!({"q": 1}));
    assert_eq!(partial.usage, Some(Usage::new(5, 7)));

    let Some(PartialTerminal::Completed {
        message,
        stop_reason,
    }) = partial.terminal
    else {
        panic!("expected a Completed terminal");
    };
    assert_eq!(stop_reason.as_deref(), Some("tool_use"));
    assert_eq!(message.text(), "hello");
    assert_eq!(message.tool_calls.len(), 1);
}

#[test]
fn reduce_of_a_truncated_sequence_yields_a_consistent_partial() {
    // Drop everything from the tool-call block's argument deltas onward: no
    // BlockEnd, no Completed. The reducer must still expose the closed
    // thinking/text blocks and the in-progress tool-call argument string.
    let items = interleaved_stream_items();
    let frames = encode_frames(&items);
    let cut = frames
        .iter()
        .position(|frame| matches!(frame, AssistantFrame::BlockDelta { index: 2, .. }))
        .expect("a block-2 delta frame");
    let truncated = &frames[..=cut];
    let partial = reduce_frames(truncated);

    assert_eq!(
        partial.content,
        vec![
            ContentBlock::Thinking {
                text: "plan".into(),
                signature: None,
            },
            ContentBlock::Text("hello".into()),
        ],
        "closed blocks are unaffected by the truncation"
    );
    assert!(partial.tool_calls.is_empty(), "tool block never closed");
    assert_eq!(partial.terminal, None);
    assert_eq!(partial.open_blocks, vec![(2, "{\"q\":".to_string())]);
}

#[test]
fn checkpoint_is_a_full_snapshot_not_a_delta() {
    let frames = vec![
        AssistantFrame::BlockStart {
            index: 0,
            kind: BlockKind::ToolCall {
                id: "call-1".into(),
                name: "search".into(),
            },
        },
        AssistantFrame::BlockDelta {
            index: 0,
            delta: BlockDelta::ToolArgs("{\"q\":1".into()),
        },
        // A checkpoint replaces the accumulated string; a reader that only
        // sees frames from here onward must reduce to the same result as one
        // that saw every fragment.
        AssistantFrame::ToolArgsCheckpoint {
            index: 0,
            json_so_far: "{\"q\":1".into(),
        },
        AssistantFrame::BlockDelta {
            index: 0,
            delta: BlockDelta::ToolArgs("}".into()),
        },
    ];
    let full = reduce_frames(&frames);
    let truncated = reduce_frames(&frames[2..]);
    assert_eq!(full.open_blocks, vec![(0, "{\"q\":1}".to_string())]);
    assert_eq!(full.open_blocks, truncated.open_blocks);
}

#[test]
fn automatic_checkpoints_appear_every_configured_interval() {
    let mut encoder = FrameEncoder::new();
    encoder.push(&ModelStreamItem::BlockStart {
        index: 0,
        kind: BlockKind::ToolCall {
            id: "call-1".into(),
            name: "search".into(),
        },
    });
    for _ in 0..TOOL_ARGS_CHECKPOINT_INTERVAL {
        encoder.push(&ModelStreamItem::BlockDelta {
            index: 0,
            delta: BlockDelta::ToolArgs("a".into()),
        });
    }
    let frames = encoder.into_frames();
    let checkpoint = frames
        .iter()
        .find_map(|frame| match frame {
            AssistantFrame::ToolArgsCheckpoint { json_so_far, .. } => Some(json_so_far.clone()),
            _ => None,
        })
        .expect("a checkpoint frame after the configured interval");
    assert_eq!(checkpoint, "a".repeat(TOOL_ARGS_CHECKPOINT_INTERVAL));
}

#[test]
fn provider_failed_frame_carries_partial_message_and_stop_reason() {
    let items = vec![ModelStreamItem::ProviderFailed(
        tinyinference_llm::model::ProviderError {
            provider: "anthropic".into(),
            message: "overloaded".into(),
            stop_reason: Some("pause_turn".into()),
            partial_message: Some(AssistantMessage {
                id: None,
                content: vec![ContentBlock::Text("partial".into())],
                tool_calls: Vec::new(),
                usage: None,
            }),
            ..Default::default()
        },
    )];
    let frames = encode_frames(&items);
    let partial = reduce_frames(&frames);
    let Some(PartialTerminal::Failed {
        message,
        partial: partial_message,
        stop_reason,
    }) = partial.terminal
    else {
        panic!("expected a Failed terminal");
    };
    assert_eq!(message, "overloaded");
    assert_eq!(stop_reason.as_deref(), Some("pause_turn"));
    assert_eq!(
        partial_message.unwrap().content,
        vec![ContentBlock::Text("partial".into())]
    );
}
