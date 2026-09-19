# harness::multimodal

Attachment resolution for `[IMAGE:…]` and `[FILE:…]` markers embedded in
message text.

## Why this exists

A user attaches a picture or a document; somewhere between the text box and
the provider that attachment has to become bytes the model can read —
validated, size-capped, MIME-checked, and rendered into the message. This
module is that pipeline, minus every decision that belongs to a host.
Attachments travel as markers inside message text rather than a parallel
structured field, so they survive every hop a host already has (persistence,
summarisation, delegation) without each of those learning about attachments.

Images and files are symmetrical up to the payload and diverge there: images
always inline as base64 `data:` URIs (the provider vision contract); files
never inline bytes — an extractable format contributes text, a binary-only
format contributes a header naming it plus a content hash.

## Public surface

- [`config`] — [`ImageLimits`], [`FileLimits`], [`ALLOWED_IMAGE_MIME_TYPES`]:
  crate-owned, host-mapped per-turn caps, including `FileLimits`'s
  `max_files == 0` hard-disable sentinel.
- [`markers`] — the marker vocabulary (`IMAGE_MARKER_PREFIX`,
  `FILE_MARKER_PREFIX`, placeholder tokens) and pure string transforms:
  `parse_image_markers`/`parse_file_markers`, placeholder
  render/detect/extract/rehydrate functions, `extract_ollama_image_payload`.
- [`mime`] — MIME detection (header → extension → magic bytes, in an order
  that differs deliberately between images and files) and the extension/magic
  lookup tables.
- [`data_uri`] — `data:` URI parsing (including gzip-compressed attachments
  with a required `original_mime` parameter), percent-decoding, and encoding.
- [`resolve`] — [`TextExtractor`] (host-pluggable document text extraction),
  [`NoTextExtractor`], and the two resolution entry points
  [`resolve_image`]/[`resolve_file`] that turn one marker reference into a
  payload, trying `data:` → `http(s)` (gated by `allow_remote_fetch`) → local
  path in that order.
- [`payload`] — [`FilePayload`] (`Extracted` / `Reference`),
  [`compose_multimodal_message`] (renders the final provider-bound message),
  and supporting helpers (`truncate_chars`, `sha256_prefix`, `format_size`,
  `escape_attr`).
- [`error`] — [`MultimodalError`] and its [`Result`] alias; every variant
  carries the offending `input` verbatim so a multi-attachment turn's failure
  is attributable.

## Files

| File          | Role                                                             |
| ------------- | ----------------------------------------------------------------- |
| `mod.rs`      | Module overview, sub-module wiring, and re-exports.                |
| `config.rs`   | `ImageLimits`, `FileLimits`, `ALLOWED_IMAGE_MIME_TYPES`.            |
| `markers.rs`  | Marker prefixes and pure string transforms over them.              |
| `mime.rs`     | MIME detection: header, extension, and magic-byte sniffing.        |
| `data_uri.rs` | `data:` URI parsing, gzip decompression, percent-decoding.         |
| `resolve.rs`  | `TextExtractor` trait, `resolve_image`/`resolve_file` pipelines.   |
| `payload.rs`  | `FilePayload`, message composition, truncation, hashing.           |
| `error.rs`    | `MultimodalError`, `Result`.                                       |
| `test.rs`     | Unit tests for the marker/MIME/size/path edge cases across both pipelines. |

## Operational constraints

- **Count before resolving.** `FileLimits::files_disabled` and the per-turn
  count caps must be checked against the raw markers, before any read
  happens — a cap enforced after the fetch is not a cap.
- **Check the sentinel before the clamp.** `max_files == 0` means *none*;
  `FileLimits::effective` clamps it up to `1`. Consulting only the clamped
  value would admit one attachment from a source that asked for zero.
- What stays with the host, deliberately: the `reqwest::Client` (proxy/timeout
  policy), the `TextExtractor` implementation (which parser, if any, and its
  timeout), the attachment stash (where bytes live between ingress and
  dispatch), and message-level marker counting (only the host knows its
  message type). This module never decides which local paths may be read —
  that is `FileLimits::files_disabled`'s lever, not a filesystem allowlist
  here.
- Every extraction/fetch/read failure degrades to a `FilePayload::Reference`
  or a skipped attachment rather than failing the whole turn — a damaged PDF
  should cost the model its text, not the conversation.
