//! Durable frame codec for assistant-message streaming.
//!
//! [`ModelStreamItem`]s are the wire-level shape a provider adapter emits;
//! they are not, on their own, durable — a consumer that reconnects mid-turn
//! (or a journal reader replaying a crashed run) needs a compact, self
//! describing record it can persist and fold back into a partial message
//! without re-running the provider stream.
//!
//! [`AssistantFrame`] is that record. [`FrameEncoder`] turns a sequence of
//! [`ModelStreamItem`]s into frames (emitting a periodic
//! [`AssistantFrame::ToolArgsCheckpoint`] for long-running tool-argument
//! streams so a reconnecting reader does not have to replay every single
//! fragment from the start of the block); [`reduce_frames`] folds frames back
//! into a [`PartialAssistantMessage`] — the harness event journal persists
//! frames as they are encoded, and a crashed/reconnecting consumer rebuilds
//! its view by reducing whatever frames it has, including a sequence
//! truncated mid-block.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use tinyinference_llm::message::{AssistantMessage, ContentBlock};
use tinyinference_llm::model::{BlockDelta, BlockKind, ModelStreamItem};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

/// Number of tool-argument fragments accumulated between automatic
/// [`AssistantFrame::ToolArgsCheckpoint`] snapshots.
///
/// A checkpoint is a full snapshot (not a delta), so a reader that only has
/// frames from the checkpoint onward — because earlier per-fragment frames
/// were compacted out of the journal — still reduces to the correct partial
/// argument string.
const TOOL_ARGS_CHECKPOINT_INTERVAL: usize = 16;

/// A compact, self-describing, serializable record of one increment of an
/// in-progress assistant message stream.
///
/// Frames are the unit the harness event journal persists for a streaming
/// model call. Reducing a sequence of frames with [`reduce_frames`]
/// reconstructs a [`PartialAssistantMessage`] without needing the original
/// provider stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum AssistantFrame {
    /// A new content block has opened at `index`.
    BlockStart {
        /// Position of the block within the assistant message.
        index: usize,
        /// The block's syntactic category.
        kind: BlockKind,
    },
    /// An incremental fragment for the open block at `index`.
    BlockDelta {
        /// Position of the block this fragment belongs to.
        index: usize,
        /// The fragment payload.
        delta: BlockDelta,
    },
    /// A full snapshot of a tool-call block's accumulated argument JSON,
    /// emitted periodically (every [`TOOL_ARGS_CHECKPOINT_INTERVAL`]
    /// fragments) so a reader with a truncated frame log can still recover a
    /// consistent partial argument string.
    ToolArgsCheckpoint {
        /// Position of the tool-call block this checkpoint snapshots.
        index: usize,
        /// The full argument JSON accumulated for this block so far.
        json_so_far: String,
    },
    /// The block at `index` has closed; `block` is its fully assembled
    /// content.
    BlockEnd {
        /// Position of the closed block.
        index: usize,
        /// The finished content block.
        block: ContentBlock,
    },
    /// A usage update.
    Usage(Usage),
    /// Terminal success: the fully merged response.
    Completed {
        /// The complete assistant message.
        message: AssistantMessage,
        /// The provider's reported stop/finish reason, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
    },
    /// Terminal failure. Carries whatever partial message had accumulated
    /// before the failure, mirroring
    /// [`tinyinference_llm::model::ProviderError::partial_message`].
    Failed {
        /// Human-readable failure message.
        message: String,
        /// The assistant message accumulated before the failure, when any
        /// content had arrived.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partial: Option<AssistantMessage>,
        /// The stop/finish reason reported before the failure, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stop_reason: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// FrameEncoder
// ---------------------------------------------------------------------------

/// Turns a sequence of [`ModelStreamItem`]s into durable [`AssistantFrame`]s.
///
/// Feed items with [`FrameEncoder::push`] as a provider stream produces them;
/// call [`FrameEncoder::into_frames`] (or read [`FrameEncoder::frames`]
/// incrementally) to get the encoded sequence. [`ModelStreamItem::Started`],
/// [`ModelStreamItem::MessageDelta`], and [`ModelStreamItem::ToolCallDelta`]
/// carry no information a block-aware reducer needs beyond what
/// `BlockStart`/`BlockDelta`/`BlockEnd` already carry, so they are not framed
/// — only the block-indexed and terminal items are.
#[derive(Debug, Default)]
pub struct FrameEncoder {
    frames: Vec<AssistantFrame>,
    /// Per-block running tool-argument JSON and fragment count since the
    /// last checkpoint, keyed by block index.
    tool_progress: BTreeMap<usize, (String, usize)>,
}

impl FrameEncoder {
    /// Creates an empty encoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Folds one stream item, appending zero or more frames.
    pub fn push(&mut self, item: &ModelStreamItem) {
        match item {
            ModelStreamItem::BlockStart { index, kind } => {
                if matches!(kind, BlockKind::ToolCall { .. }) {
                    self.tool_progress.insert(*index, (String::new(), 0));
                }
                self.frames.push(AssistantFrame::BlockStart {
                    index: *index,
                    kind: kind.clone(),
                });
            }
            ModelStreamItem::BlockDelta { index, delta } => {
                self.frames.push(AssistantFrame::BlockDelta {
                    index: *index,
                    delta: delta.clone(),
                });
                if let BlockDelta::ToolArgs(fragment) = delta
                    && let Some((json_so_far, count)) = self.tool_progress.get_mut(index)
                {
                    json_so_far.push_str(fragment);
                    *count += 1;
                    if *count >= TOOL_ARGS_CHECKPOINT_INTERVAL {
                        self.frames.push(AssistantFrame::ToolArgsCheckpoint {
                            index: *index,
                            json_so_far: json_so_far.clone(),
                        });
                        *count = 0;
                    }
                }
            }
            ModelStreamItem::BlockEnd { index, block } => {
                self.tool_progress.remove(index);
                self.frames.push(AssistantFrame::BlockEnd {
                    index: *index,
                    block: block.clone(),
                });
            }
            ModelStreamItem::UsageDelta(usage) => {
                self.frames.push(AssistantFrame::Usage(*usage));
            }
            ModelStreamItem::Completed(response) => {
                self.frames.push(AssistantFrame::Completed {
                    message: response.message.clone(),
                    stop_reason: response.finish_reason.clone(),
                });
            }
            ModelStreamItem::Failed(message) => {
                self.frames.push(AssistantFrame::Failed {
                    message: message.clone(),
                    partial: None,
                    stop_reason: None,
                });
            }
            ModelStreamItem::ProviderFailed(error) => {
                self.frames.push(AssistantFrame::Failed {
                    message: error.message.clone(),
                    partial: error.partial_message.clone(),
                    stop_reason: error.stop_reason.clone(),
                });
            }
            // No block-boundary information; the compatibility channel is
            // fully covered by the block-indexed items above for any
            // block-aware adapter. Adapters that only emit the flat
            // `MessageDelta`/`ToolCallDelta` shape (no block boundaries) have
            // nothing durable to frame here beyond what `Completed`/`Failed`
            // already capture. `Deferred` carries no message content either
            // (it is a handle to a response that will resolve later via
            // `ChatModel::fetch_deferred`, folded by
            // `StreamAccumulator::deferred` instead of this block reducer),
            // so it is likewise not framed.
            ModelStreamItem::Started
            | ModelStreamItem::MessageDelta(_)
            | ModelStreamItem::ToolCallDelta(_)
            | ModelStreamItem::Deferred(_) => {}
        }
    }

    /// Returns the frames encoded so far without consuming the encoder.
    pub fn frames(&self) -> &[AssistantFrame] {
        &self.frames
    }

    /// Consumes the encoder and returns the full encoded frame sequence.
    pub fn into_frames(self) -> Vec<AssistantFrame> {
        self.frames
    }
}

/// Encodes a complete slice of [`ModelStreamItem`]s into [`AssistantFrame`]s.
///
/// A convenience wrapper around [`FrameEncoder`] for callers that already
/// have the full item sequence (tests, post-processing).
pub fn encode_frames(items: &[ModelStreamItem]) -> Vec<AssistantFrame> {
    let mut encoder = FrameEncoder::new();
    for item in items {
        encoder.push(item);
    }
    encoder.into_frames()
}

// ---------------------------------------------------------------------------
// PartialAssistantMessage / reduce_frames
// ---------------------------------------------------------------------------

/// A block still open (no [`AssistantFrame::BlockEnd`] seen yet) while
/// reducing frames.
#[derive(Clone, Debug, PartialEq)]
enum OpenBlock {
    Text(String),
    Thinking(String),
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        json_so_far: String,
    },
}

/// The result of folding a (possibly truncated) [`AssistantFrame`] sequence.
///
/// Reflects exactly what the frames folded in describe: closed blocks are
/// merged into [`Self::content`] in index order, and a block with no
/// [`AssistantFrame::BlockEnd`] yet is exposed via [`Self::open_blocks`] so a
/// reconnecting consumer can still render in-progress content.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartialAssistantMessage {
    /// Closed content blocks, in block-index order.
    pub content: Vec<ContentBlock>,
    /// Blocks that opened but have not closed, as `(index, text_so_far)` for
    /// text/thinking blocks or `(index, json_so_far)` for tool-call blocks,
    /// in block-index order.
    pub open_blocks: Vec<(usize, String)>,
    /// Tool calls reconstructed from closed tool-use blocks. Malformed JSON
    /// becomes [`ToolCall::invalid`] rather than being dropped.
    pub tool_calls: Vec<ToolCall>,
    /// Most recent usage value seen.
    pub usage: Option<Usage>,
    /// Present once a terminal [`AssistantFrame::Completed`] or
    /// [`AssistantFrame::Failed`] frame has been folded in.
    pub terminal: Option<PartialTerminal>,
}

/// The terminal outcome folded into a [`PartialAssistantMessage`], when any.
#[derive(Clone, Debug, PartialEq)]
pub enum PartialTerminal {
    /// The stream completed successfully; carries the authoritative message.
    Completed {
        /// The complete assistant message.
        message: AssistantMessage,
        /// The provider's reported stop/finish reason, when known.
        stop_reason: Option<String>,
    },
    /// The stream failed; carries the human-readable message and, when the
    /// failure was mid-stream, the partial message and stop reason it
    /// interrupted.
    Failed {
        /// Human-readable failure message.
        message: String,
        /// The assistant message accumulated before the failure, when any.
        partial: Option<AssistantMessage>,
        /// The stop/finish reason reported before the failure, when known.
        stop_reason: Option<String>,
    },
}

/// Folds a (possibly truncated) [`AssistantFrame`] sequence into a
/// [`PartialAssistantMessage`].
///
/// A full sequence — one that ends in [`AssistantFrame::Completed`] or
/// [`AssistantFrame::Failed`] — reduces to a result whose `content` (in the
/// `Completed` case) matches the original [`AssistantMessage`]. A sequence
/// truncated mid-block reduces to a consistent partial: every fully closed
/// block lands in `content`, and the interrupted block's accumulated text or
/// argument JSON (using the most recent [`AssistantFrame::ToolArgsCheckpoint`]
/// as its base, when one was folded in) is exposed via `open_blocks`.
pub fn reduce_frames(frames: &[AssistantFrame]) -> PartialAssistantMessage {
    let mut open: BTreeMap<usize, OpenBlock> = BTreeMap::new();
    let mut closed: BTreeMap<usize, ContentBlock> = BTreeMap::new();
    let mut tool_calls = Vec::new();
    let mut usage = None;
    let mut terminal = None;

    for frame in frames {
        match frame {
            AssistantFrame::BlockStart { index, kind } => {
                let block = match kind {
                    BlockKind::Text => OpenBlock::Text(String::new()),
                    BlockKind::Thinking => OpenBlock::Thinking(String::new()),
                    BlockKind::ToolCall { id, name } => OpenBlock::ToolCall {
                        id: Some(id.clone()),
                        name: Some(name.clone()),
                        json_so_far: String::new(),
                    },
                };
                open.insert(*index, block);
            }
            AssistantFrame::BlockDelta { index, delta } => match (open.get_mut(index), delta) {
                (Some(OpenBlock::Text(text)), BlockDelta::Text(fragment)) => {
                    text.push_str(fragment);
                }
                (Some(OpenBlock::Thinking(text)), BlockDelta::Thinking(fragment)) => {
                    text.push_str(fragment);
                }
                (Some(OpenBlock::ToolCall { json_so_far, .. }), BlockDelta::ToolArgs(fragment)) => {
                    json_so_far.push_str(fragment);
                }
                _ => {}
            },
            AssistantFrame::ToolArgsCheckpoint { index, json_so_far } => {
                // A checkpoint is a full snapshot, not a delta: it replaces
                // whatever was accumulated so far. A reader that only has
                // frames from this checkpoint onward (its matching
                // `BlockStart` was pruned from the journal) still needs a
                // consistent partial, so a missing entry is created here
                // rather than the checkpoint being silently dropped.
                match open.get_mut(index) {
                    Some(OpenBlock::ToolCall {
                        json_so_far: current,
                        ..
                    }) => current.clone_from(json_so_far),
                    _ => {
                        open.insert(
                            *index,
                            OpenBlock::ToolCall {
                                id: None,
                                name: None,
                                json_so_far: json_so_far.clone(),
                            },
                        );
                    }
                }
            }
            AssistantFrame::BlockEnd { index, block } => {
                open.remove(index);
                if let ContentBlock::Json(value) = &block
                    && let (Some(id), Some(name)) = (
                        value.get("id").and_then(serde_json::Value::as_str),
                        value.get("name").and_then(serde_json::Value::as_str),
                    )
                {
                    let arguments = value
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    tool_calls.push(ToolCall::new(id, name, arguments));
                } else {
                    closed.insert(*index, block.clone());
                }
            }
            AssistantFrame::Usage(value) => {
                usage = Some(*value);
            }
            AssistantFrame::Completed {
                message,
                stop_reason,
            } => {
                terminal = Some(PartialTerminal::Completed {
                    message: message.clone(),
                    stop_reason: stop_reason.clone(),
                });
            }
            AssistantFrame::Failed {
                message,
                partial,
                stop_reason,
            } => {
                terminal = Some(PartialTerminal::Failed {
                    message: message.clone(),
                    partial: partial.clone(),
                    stop_reason: stop_reason.clone(),
                });
            }
        }
    }

    let content = closed.into_values().collect();
    let open_blocks = open
        .into_iter()
        .map(|(index, block)| {
            let text = match block {
                OpenBlock::Text(text) | OpenBlock::Thinking(text) => text,
                OpenBlock::ToolCall { json_so_far, .. } => json_so_far,
            };
            (index, text)
        })
        .collect();

    PartialAssistantMessage {
        content,
        open_blocks,
        tool_calls,
        usage,
        terminal,
    }
}

#[cfg(test)]
mod test;
