//! How long a tool call took, shown to the model on the call's result row.
//!
//! Gated by [`RunPolicy::tool_result_durations`](crate::runtime::RunPolicy):
//! off by default, so a transcript stays byte-identical unless the host opts
//! in. When on, the fold appends one trailing `[took 12.3s]` line to every
//! executed call's tool row. Calls the loop answered without running a tool
//! (denials, unknown tools, invalid arguments) carry no line — there was no
//! execution to time.

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
    message.content.push(ContentBlock::Text(format!(
        "\n{}",
        duration_suffix(duration_ms)
    )));
}

#[cfg(test)]
#[path = "tool_timing_tests.rs"]
mod tests;
