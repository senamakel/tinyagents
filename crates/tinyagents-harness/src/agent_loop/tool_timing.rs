//! How long a tool call took, shown to the model on the call's result row.
//!
//! Gated by [`RunPolicy::tool_result_durations`](crate::runtime::RunPolicy):
//! off by default, so a transcript stays byte-identical unless the host opts
//! in. When on, the fold appends one trailing `[took 12.3s]` line to every
//! executed call's tool row. Calls the loop answered without running a tool
//! (denials, unknown tools, invalid arguments) carry no line — there was no
//! execution to time — and neither do rows that are structured output (a JSON
//! block, or text that is one JSON document), which hosts parse.

use tinyinference_llm::message::{ContentBlock, ToolMessage};

/// Render a call's wall-clock duration as the line the model reads,
/// truncated to tenths of a second: `12_345` ms is `[took 12.3s]`.
pub(super) fn duration_suffix(duration_ms: u64) -> String {
    let tenths = duration_ms / 100;
    format!("[took {}.{}s]", tenths / 10, tenths % 10)
}

/// Append the duration line to a folded tool row as its own trailing text
/// block, so the tool's own blocks are left exactly as it returned them.
///
/// The block starts with a newline because providers join a tool row's text
/// blocks with no separator. A row marked `trusted_verbatim` must reach the
/// consumer byte-for-byte and is left alone.
pub(super) fn append_duration(message: &mut ToolMessage, duration_ms: u64) {
    if message.trusted_verbatim {
        tracing::trace!(
            target: "tinyagents::agent_loop",
            call_id = %message.tool_call_id,
            "[agent_loop::tool_timing] verbatim tool row; not appending its duration"
        );
        return;
    }
    if is_structured(message) {
        tracing::trace!(
            target: "tinyagents::agent_loop",
            call_id = %message.tool_call_id,
            "[agent_loop::tool_timing] structured tool row; not appending its duration"
        );
        return;
    }
    message.content.push(ContentBlock::Text(format!(
        "\n{}",
        duration_suffix(duration_ms)
    )));
}

/// Whether the row is structured output that must stay parseable: a JSON
/// block, or text that is one JSON document
/// (see [`crate::middleware::library::is_json_document`]).
fn is_structured(message: &ToolMessage) -> bool {
    if message
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Json(_)))
    {
        return true;
    }
    let text: String = message
        .content
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect();
    crate::middleware::library::is_json_document(&text)
}

#[cfg(test)]
#[path = "tool_timing_tests.rs"]
mod tests;
