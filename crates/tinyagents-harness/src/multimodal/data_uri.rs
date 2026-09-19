//! `data:` URI parsing, including the compressed-attachment shape.
//!
//! A renderer that gzips an attachment before handing it over emits
//! `data:application/gzip;base64,…` with an `original_mime=` parameter naming
//! what the bytes decompress to. That parameter is **required** rather than
//! guessed: without it there is no way to validate the payload against a MIME
//! allowlist, and a gzip stream that decompresses to anything at all would pass
//! a check keyed on `application/gzip`.
//!
//! Decompression is bounded by [`Read::take`] against the caller's byte cap
//! plus one, so an over-cap payload is detected by the extra byte rather than
//! by allocating the whole thing first.

use std::io::Read;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use flate2::read::GzDecoder;

/// A decoded `data:` URI.
#[derive(Debug, Clone)]
pub struct ParsedDataUri {
    /// Lower-cased MIME type from the header.
    pub mime: String,
    /// Header parameters other than `base64`, lower-cased keys, percent-decoded
    /// values.
    pub params: Vec<(String, String)>,
    /// The decoded payload.
    pub bytes: Vec<u8>,
}

/// Parse a base64 or percent-encoded `data:` URI.
///
/// Size admission remains with the caller: this pure parser does not know
/// whether it is handling an image, a file, or a host-specific attachment
/// policy.  Percent-encoded payloads are decoded byte-for-byte, rather than
/// through UTF-8, so binary media is not lossy.
///
/// The `Err` is a plain reason string rather than a
/// [`MultimodalError`](super::error::MultimodalError) because the caller knows
/// whether it is resolving an image or a file, and that decides which variant
/// the reason belongs in.
pub fn parse_data_uri(source: &str) -> Result<ParsedDataUri, String> {
    let Some(comma_idx) = source.find(',') else {
        return Err("expected data URI payload".to_string());
    };

    let header = &source[..comma_idx];
    let payload = &source[comma_idx + 1..];

    if !header.starts_with("data:") {
        return Err("data URI must start with `data:`".to_string());
    }

    let mut parts = header.trim_start_matches("data:").split(';');
    let mime = parts.next().unwrap_or_default().trim().to_ascii_lowercase();
    let params = parts
        .filter_map(|part| {
            if part.eq_ignore_ascii_case("base64") {
                return None;
            }
            let (key, value) = part.split_once('=')?;
            Some((
                key.trim().to_ascii_lowercase(),
                percent_decode(value.trim()).unwrap_or_else(|| value.trim().to_string()),
            ))
        })
        .collect::<Vec<_>>();

    let is_base64 = header
        .split(';')
        .any(|part| part.trim().eq_ignore_ascii_case("base64"));
    let bytes = if is_base64 {
        STANDARD
            .decode(payload)
            .map_err(|error| format!("invalid base64 payload: {error}"))?
    } else {
        percent_decode_bytes(payload)?
    };

    Ok(ParsedDataUri {
        mime,
        params,
        bytes,
    })
}

/// Look up a header parameter, case-insensitively.
pub fn data_uri_param(params: &[(String, String)], key: &str) -> Option<String> {
    params
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(key))
        .map(|(_, value)| value.clone())
}

/// Percent-decode a parameter value, or `None` when it is malformed.
///
/// Malformed input yields `None` rather than a lossy best effort so the caller
/// can fall back to the raw value — a filename containing a bare `%` is far
/// more likely than a filename that meant to be percent-encoded and was not.
pub fn percent_decode(value: &str) -> Option<String> {
    String::from_utf8(percent_decode_bytes(value).ok()?).ok()
}

/// Percent-decode arbitrary data-URI payload bytes.
///
/// Unlike [`percent_decode`], this accepts non-UTF-8 results because image and
/// other binary attachments are valid data-URI payloads.
fn percent_decode_bytes(value: &str) -> Result<Vec<u8>, String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err("malformed percent escape in data URI payload".to_string());
            }
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3])
                .map_err(|_| "malformed percent escape in data URI payload".to_string())?;
            let byte = u8::from_str_radix(hex, 16)
                .map_err(|_| "malformed percent escape in data URI payload".to_string())?;
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(out)
}

/// Decompress a gzip payload, refusing anything over `max_decompressed_bytes`.
///
/// The reader is capped at `max + 1` so an over-cap stream is caught by the
/// spare byte instead of being fully materialised first — the difference
/// between rejecting a zip bomb and being one.
pub fn gunzip(bytes: &[u8], max_decompressed_bytes: usize) -> Result<Vec<u8>, String> {
    let limit = max_decompressed_bytes.saturating_add(1) as u64;
    let mut decoder = GzDecoder::new(bytes).take(limit);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|error| format!("invalid gzip payload: {error}"))?;
    if out.len() > max_decompressed_bytes {
        return Err(format!(
            "decompressed payload exceeds {max_decompressed_bytes} bytes"
        ));
    }
    Ok(out)
}

/// Encode `bytes` as a canonical `data:<mime>;base64,…` URI.
pub fn encode_data_uri(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}
