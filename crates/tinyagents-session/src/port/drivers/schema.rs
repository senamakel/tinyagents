//! The session crate's v2 storage declarations: [`storage_schema`].

use tinyagents_storage_kit::{Owner, Schema, ScopeClass};

use super::transcripts::{
    self, ENTRIES as ENTRIES_NAME, INDEX as INDEX_NAME, SCHEMA_VERSION as TRANSCRIPT_VERSION,
};
use super::turn_states::{self, COLLECTION as TURN_NAME, SCHEMA_VERSION as TURN_VERSION};

#[path = "schema_steps.rs"]
mod steps;

/// What the session crate stores, for the host's `Registry::compose` (owner
/// `session`, scope class Agent), and the steps that bring the v1 names and
/// shapes forward with `Registry::migrate`.
///
/// | v1 | v2 |
/// |---|---|
/// | `session_transcripts` (id `<len>:<stem>` or `h:<sha>`) | `session.transcripts`, id `Key::of([stem])`; `created_at` (RFC 3339) becomes application `created_ms` (indexed); unwritten reservation rows (`reserved_at`) are dropped |
/// | `session_transcript_entries` (id `<len>:<stem>#<seq>`) | `session.transcript_entries`, id `Key::of([stem, "<seq>"])` with `seq` zero-padded |
/// | `session_turn_states` (`{thread_id, request_id, lifecycle, turn: {camelCase}}`) | `session.turn_states`, id `Key::of([thread_id, request_id])`, bare snake_case body with `started_ms` / `updated_ms` (RFC 3339 to epoch ms) |
///
/// Things that move outside a document step:
///
/// - `session_kv` lives in `tsd.kv` (owner `session`); `open_agent_storage`
///   calls `tinyagents_harness::storage::import_legacy_kv` once per scope
///   before the session Registry migration.
/// - The generation reservations that were `reserved_at` rows are `tsd.leases`
///   records (owner `session`, key = the successor stem); nothing is migrated,
///   an in-flight reservation simply lapses.
/// - The transcript-import marker (`session_import_markers`) is a
///   `tsd.migrations` record (`session` / `session_raw_jsonl_v1`); the host
///   calls `legacy::import_legacy_marker` (which `import_transcripts_once` also
///   does itself) after `Registry::prepare`.
/// - The journal streams are the harness's: `harness.journal/<run>`, declared and
///   migrated by `tinyagents_harness::storage_schema()`.
///
/// Each migration runs against the already-bound scope supplied by the host.
/// Scope selection and any profile re-homing are host-owned operations; a step
/// here transforms data only within its supplied scope.
#[must_use]
pub fn storage_schema() -> Schema {
    let owner = Owner::new("session").expect("`session` is a valid owner");
    let [entries, index] = transcripts::collection_specs();
    Schema::new(owner)
        .collection_with(TRANSCRIPT_VERSION, entries, ScopeClass::Agent)
        .collection_with(TRANSCRIPT_VERSION, index, ScopeClass::Agent)
        .collection_with(
            TURN_VERSION,
            turn_states::collection_spec(),
            ScopeClass::Agent,
        )
        .step(steps::transcript_entries())
        .step(steps::transcripts())
        .step(steps::turn_states())
}
