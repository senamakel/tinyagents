//! Session transcripts: the store-port seam hosts persist conversations
//! through, and the types they exchange.
//!
//! A conversation's transcript lives in whatever store the host installed
//! (see [`crate::port`]): [`TranscriptLocator`] resolves transcripts by thread,
//! agent or [`SessionRef`] and binds a write handle, and
//! [`TranscriptHistory`]/[`TranscriptRead`] are those handles. This crate no
//! longer reads or writes a workspace directory on that path. Host-owned
//! migration code reads classic `session_raw/{stem}.jsonl` files and writes
//! their content through the target `TranscriptHistory` API.
//!
//! ## Append-only log
//!
//! A transcript is an **append-only event log** whichever store holds it. An
//! ordinary turn appends only its new rows. A reduction (context reduction
//! dropped or replaced earlier turns, so the previously-persisted set is not a
//! prefix of the new one) is recorded as a compaction that carries the full
//! reduced message set; the earlier rows stay in the log.
//!
//! - **Model context** ([`TranscriptRead::read_session`]) replays the log:
//!   rows accumulate, a compaction **replaces** the accumulator, and
//!   interrupted partials are **skipped** (they never entered the model's
//!   context).
//! - **Display** ([`TranscriptRead::read_display`]) returns **every** record in
//!   order: pre-compaction history, compaction markers and interrupted
//!   partials, so the UI projection ([`view`]) can render the full timeline.
//!
//! ## Typed rows
//!
//! The in-memory [`TranscriptMessage`] is typed: `content` is plain text, and a
//! native tool-call round or an image is carried in fields (`tool_calls`,
//! `tool_call_id`, `parts`) instead of a string envelope or an `[OH_IMAGE:..]`
//! marker inside `content`. Rows a classic file stored with the envelope or
//! marker in `content` are lifted into the typed fields on import
//! ([`TranscriptMessage::normalized`], lenient: any well-formed native
//! envelope, a tool-result envelope, `[OH_IMAGE:..]` markers); a line the
//! reader cannot type stays a plain text row.
//! [`TranscriptMessage::legacy_content`] rebuilds the old string for the
//! places that must keep it.
//!
//! ## Classic JSONL format (read by the legacy import only)
//!
//! **Line 1 (meta):**
//! ```json
//! {"_meta":{"agent":"code_executor","dispatcher":"native","created":"...","updated":"...","turn_count":3,"input_tokens":5000,"output_tokens":1200,"cached_input_tokens":3500,"charged_amount_usd":0.0045,"thread_id":"thr_abc123"}}
//! ```
//!
//! **Message lines:**
//! ```json
//! {"role":"system","content":"..."}
//! {"role":"user","content":"..."}
//! {"role":"assistant","content":"...","model":"claude-...","usage":{"input":1234,"output":567,"cached_input":1000,"cost_usd":0.0012},"ts":"2026-04-17T..."}
//! {"role":"tool","content":"..."}
//! ```
//!
//! Only `role` and `content` are required. Compaction records
//! (`{"kind":"compaction","replacement":[…]}`) and later `_meta` lines (the
//! last wins) are additive; unknown record kinds are skipped.
//!
//! ## Module layout
//!
//! | File            | Role                                                        |
//! |-----------------|-------------------------------------------------------------|
//! | `types`         | Public domain types (`TranscriptMeta`, projections, usage). |
//! | `history`       | The [`TranscriptLocator`]/[`TranscriptHistory`] store seam.  |
//! | `background`    | Out-of-band appends into a session's head generation.        |
//! | `turn_lock`     | The per-session turn lock.                                  |
//! | `spend`         | Token/cost totals read back from transcripts.               |
//! | `view`          | Display projection and pagination of a thread.              |
//! | `session`       | [`SessionRef`] identity and its deterministic stem.          |
//! | `jsonl`, `reader`, `writer`, `legacy_md`, `markdown`, `paths` | The classic JSONL codec the legacy import and rewrite use. |

mod adoption;
mod background;
mod display;
mod history;
mod jsonl;
pub mod legacy;
#[cfg(test)]
mod legacy_md;
#[cfg(test)]
mod markdown;
mod paths;
#[cfg(test)]
mod reader;
mod session;
pub mod spend;
#[cfg(test)]
pub(crate) mod test_locator;
mod turn_lock;
mod types;
pub mod view;
#[cfg(test)]
mod writer;

pub use adoption::SessionAdoption;
pub use background::append_background_message_in;
pub use display::BACKGROUND_METADATA_KEY;
pub(crate) use display::{display_from_session, display_message_of, partial_display_message};
pub(crate) use history::same_transcript_messages;
pub use history::{
    TranscriptHistory, TranscriptLocator, TranscriptPartial, TranscriptRead, TranscriptTurn,
    TruncateCut,
};
pub(crate) use jsonl::stamped_rows;
#[cfg(test)]
pub(crate) use reader::{read_transcript, read_transcript_display};
pub use session::{SessionRef, session_stem};
pub use turn_lock::{SessionTurnGuard, lock_session_turn};
pub use types::{
    BackgroundAppend, BackgroundAppendOutcome, BackgroundOrigin, CompactionMarker, DisplayMessage,
    DisplayRecord, DisplaySessionTranscript, LegacyText, MessageUsage, SessionTranscript,
    ThreadTranscript, ThreadTranscripts, ToolFailure, TranscriptMediaRef, TranscriptMessage,
    TranscriptMeta, TranscriptPart, TranscriptToolCall, TurnUsage, UsageCostSource,
};
#[cfg(test)]
pub(crate) use writer::{
    append_interrupted_partial, append_tools_record, append_transcript_turn, write_transcript,
    write_transcript_if_absent,
};

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) use paths::resolve_keyed_transcript_path;

#[cfg(test)]
#[path = "transcript/transcript_tests.rs"]
mod test;
#[cfg(test)]
#[path = "transcript/typed_rows_tests.rs"]
mod typed_rows_test;
