//! Wire types for the `claude --output-format stream-json` NDJSON protocol.
//!
//! `mod.rs` in this directory spawns `claude -p --output-format stream-json`
//! and decodes each stdout line into an [`SdkMessage`] as it arrives. This
//! module owns only the message shape, not the process lifecycle or the
//! decision of what to do with a decoded message.

use serde::Deserialize;

/// One line of the `claude --output-format stream-json` NDJSON stream.
///
/// Tagged on the JSON `type` field (`rename_all = "snake_case"`); an unknown
/// or future variant deserializes to [`SdkMessage::Unknown`] rather than
/// failing the whole line, so the caller in `mod.rs` can skip it and keep
/// reading the rest of the stream.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SdkMessage {
    /// An incremental text chunk of the assistant's streamed response.
    Text {
        /// The text chunk itself.
        text: String,
    },
    /// The terminal message for a turn: either the final answer or an
    /// error, plus cost accounting when the CLI reports it.
    Result {
        /// The final response text, when the turn succeeded. `None` when
        /// `is_error` is `true` and the CLI supplied no summary text.
        result: Option<String>,
        /// Whether this result represents a failed turn rather than success.
        #[serde(rename = "is_error")]
        is_error: bool,
        /// Total USD cost of the turn, when the CLI reports it.
        #[serde(default)]
        total_cost_usd: Option<f64>,
    },
    /// An explicit protocol-level error emitted by the CLI.
    Error {
        /// The error payload.
        error: SdkError,
    },
    /// Any message type not recognized by this enum. Deserializing to this
    /// variant instead of failing keeps the reader forward-compatible with
    /// CLI versions that add new NDJSON message types.
    #[serde(other)]
    Unknown,
}

/// Error payload carried by an [`SdkMessage::Error`] line.
#[derive(Debug, Deserialize)]
pub struct SdkError {
    /// Human-readable error message from the CLI.
    pub message: String,
}
