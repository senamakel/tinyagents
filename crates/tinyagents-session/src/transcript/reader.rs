//! Pure decoder for the classic append-only transcript JSONL format.
//!
//! Hosts own path discovery and file reads. This module accepts the bytes they
//! read and returns both the model replay and display projection so each view
//! is derived from the same source snapshot.

use super::jsonl::{
    LineKind, MetaLine, classify_line, display_message_from_line, message_from_line,
    meta_from_payload,
};
use super::types::{
    CompactionMarker, DisplayRecord, DisplaySessionTranscript, SessionTranscript,
    TranscriptMessage, TranscriptMeta,
};
use anyhow::{Context, Result};

/// The model-context replay and display projection of one JSONL byte stream.
#[derive(Debug, Clone)]
pub struct DecodedTranscript {
    /// Messages and tool declarations visible to a resumed model turn.
    pub session: SessionTranscript,
    /// Every displayable record, including interrupted and pre-compaction rows.
    pub display: DisplaySessionTranscript,
}

/// Decodes a classic transcript JSONL byte stream without accessing the
/// filesystem.
///
/// The first non-empty line must be a valid `_meta` header. Later malformed,
/// unknown, or invalid-UTF-8 lines are skipped. Metadata and tool declarations
/// use the last valid record. Compactions replace model context and remain in
/// the display history; interrupted messages remain display-only.
///
/// # Errors
/// Returns an error when the first non-empty line is invalid UTF-8 or is not a
/// valid `_meta` header, or when the stream contains no header.
pub fn decode_transcript_jsonl(bytes: &[u8]) -> Result<DecodedTranscript> {
    let mut meta: Option<TranscriptMeta> = None;
    let mut messages = Vec::<TranscriptMessage>::new();
    let mut tools = None;
    let mut records = Vec::<DisplayRecord>::new();
    let mut seen_first = false;

    for (line_index, bytes) in bytes.split(|byte| *byte == b'\n').enumerate() {
        let line = match std::str::from_utf8(bytes) {
            Ok(line) => line.trim(),
            Err(error) if !seen_first => {
                anyhow::bail!(
                    "first non-empty transcript line is not valid UTF-8 (line {}): {error}",
                    line_index + 1
                );
            }
            Err(error) => {
                tracing::warn!(
                    line = line_index + 1,
                    %error,
                    "[transcript] skipping invalid UTF-8 record"
                );
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        if !seen_first {
            seen_first = true;
            let header: MetaLine = serde_json::from_str(line).map_err(|error| {
                anyhow::anyhow!(
                    "first non-empty transcript line (line {}) is not a valid _meta object: {error}",
                    line_index + 1
                )
            })?;
            meta = Some(meta_from_payload(header.meta));
            continue;
        }

        match classify_line(line) {
            Ok(LineKind::Meta(header)) => meta = Some(meta_from_payload(header.meta)),
            Ok(LineKind::Compaction(compaction)) => {
                messages = compaction
                    .replacement
                    .iter()
                    .cloned()
                    .map(message_from_line)
                    .collect();
                let replacement = compaction
                    .replacement
                    .into_iter()
                    .map(display_message_from_line)
                    .collect();
                records.push(DisplayRecord::Compaction(CompactionMarker {
                    replacement,
                    ts: compaction.ts,
                    request_id: compaction.request_id,
                }));
            }
            Ok(LineKind::Tools(declarations)) => tools = Some(declarations.tools),
            Ok(LineKind::Message(message)) => {
                if !message.interrupted {
                    messages.push(message_from_line(message.clone()));
                }
                records.push(DisplayRecord::Message(Box::new(display_message_from_line(
                    message,
                ))));
            }
            Err(error) => tracing::warn!(
                line = line_index + 1,
                %error,
                "[transcript] skipping malformed or unknown record"
            ),
        }
    }

    let meta = meta.context("missing _meta header line in JSONL transcript")?;
    Ok(DecodedTranscript {
        session: SessionTranscript {
            meta: meta.clone(),
            messages,
            tools,
        },
        display: DisplaySessionTranscript { meta, records },
    })
}

/// Rewrites the first non-interrupted system message in the transcript's
/// final model context, without rebuilding or dropping any other JSONL rows.
///
/// Hosts that perform content migrations can supply a small transform for the
/// message content while leaving file discovery and I/O on the host side. The
/// transform is called at most once. Its result is committed only when it
/// returns `true` and changes the content. All bytes outside the target record
/// remain unchanged; fields within the target record other than `content` are
/// retained when that record is serialized again. A compaction resets the
/// model context, so the target is selected from the latest valid compaction
/// replacement followed by subsequent non-interrupted messages.
///
/// The same first-line validation and later-line skip policy as
/// [`decode_transcript_jsonl`] applies. The output is identical to `bytes` when
/// no target exists or the transform declines to change it.
///
/// # Errors
/// Returns an error when the JSONL header is missing or invalid.
pub fn rewrite_first_system_message_jsonl(
    bytes: &[u8],
    mut rewrite: impl FnMut(&mut String) -> bool,
) -> Result<(Vec<u8>, bool)> {
    // Reuse the decoder as the authority for header validity and malformed
    // later-line policy, then locate the raw record that contributes the first
    // system message to the final model context.
    decode_transcript_jsonl(bytes)?;

    enum Target {
        Message(usize),
        CompactionReplacement(usize, usize),
    }

    let lines = bytes
        .split_inclusive(|byte| *byte == b'\n')
        .collect::<Vec<_>>();
    let mut target = None;
    let mut seen_header = false;

    for (line_index, raw_line) in lines.iter().enumerate() {
        let line = raw_line.strip_suffix(b"\n").unwrap_or(raw_line);
        let Ok(line) = std::str::from_utf8(line) else {
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if !seen_header {
            seen_header = true;
            continue;
        }

        match classify_line(line) {
            Ok(LineKind::Compaction(compaction)) => {
                // A later valid compaction supersedes every preceding model
                // message, including a previously selected system message.
                target = None;
                let replacement_index = compaction
                    .replacement
                    .iter()
                    .enumerate()
                    .find(|(_, message)| message.role == "system" && !message.interrupted)
                    .map(|(index, _)| index);
                if let Some(replacement_index) = replacement_index {
                    target = Some(Target::CompactionReplacement(line_index, replacement_index));
                }
            }
            Ok(LineKind::Message(message)) => {
                if target.is_none() && message.role == "system" && !message.interrupted {
                    target = Some(Target::Message(line_index));
                }
            }
            Ok(LineKind::Meta(_) | LineKind::Tools(_)) | Err(_) => {}
        }
    }

    let Some(target) = target else {
        return Ok((bytes.to_vec(), false));
    };
    let (line_index, replacement_index) = match target {
        Target::Message(line_index) => (line_index, None),
        Target::CompactionReplacement(line_index, replacement_index) => {
            (line_index, Some(replacement_index))
        }
    };

    let raw_line = lines[line_index];
    let line_without_newline = raw_line.strip_suffix(b"\n").unwrap_or(raw_line);
    let parsed = std::str::from_utf8(line_without_newline)
        .context("selected transcript message is not valid UTF-8")?;
    let mut value = serde_json::from_str::<serde_json::Value>(parsed.trim())?;
    let message = match replacement_index {
        Some(index) => value
            .get_mut("replacement")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|replacement| replacement.get_mut(index)),
        None => Some(&mut value),
    };
    let Some(message) = message else {
        return Ok((bytes.to_vec(), false));
    };
    let Some(content) = message.get("content").and_then(serde_json::Value::as_str) else {
        return Ok((bytes.to_vec(), false));
    };
    let original = content.to_owned();
    let mut updated = original.clone();
    if !rewrite(&mut updated) || updated == original {
        return Ok((bytes.to_vec(), false));
    }
    message["content"] = serde_json::Value::String(updated);

    let mut output = Vec::with_capacity(bytes.len());
    for (index, raw_line) in lines.iter().enumerate() {
        if index == line_index {
            output.extend_from_slice(&serde_json::to_vec(&value)?);
            if raw_line.ends_with(b"\r\n") {
                output.extend_from_slice(b"\r\n");
            } else if raw_line.ends_with(b"\n") {
                output.push(b'\n');
            }
        } else {
            output.extend_from_slice(raw_line);
        }
    }
    Ok((output, true))
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;

#[cfg(test)]
use super::legacy_md::read_transcript_legacy_md;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;

/// Read a session transcript from a path. The production host owns this I/O;
/// this wrapper remains test-only for compatibility tests of the old reader.
#[cfg(test)]
pub fn read_transcript(path: &Path) -> Result<SessionTranscript> {
    if path.extension().and_then(|extension| extension.to_str()) == Some("md") {
        return read_transcript_legacy_md(path);
    }
    let jsonl = if path.exists() {
        path.to_path_buf()
    } else {
        let markdown = path.with_extension("md");
        if markdown.exists() {
            return read_transcript_legacy_md(&markdown);
        }
        path.to_path_buf()
    };
    Ok(decode_transcript_jsonl(&fs::read(jsonl)?)?.session)
}

/// Test-only path wrapper; production callers should read bytes themselves
/// and call [`decode_transcript_jsonl`].
#[cfg(test)]
pub fn read_transcript_display(path: &Path) -> Result<DisplaySessionTranscript> {
    Ok(decode_transcript_jsonl(&fs::read(path)?)?.display)
}
