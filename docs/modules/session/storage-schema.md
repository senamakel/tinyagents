# Session storage schema v2

`tinyagents_session::storage_schema()` declares what the session crate stores
for the host's `Registry::compose` (owner `session`, scope class Agent).
`DriverSessionStores` takes a tenant `Scope` and binds each agent to
`tenant.agent(id)`.

| v1 | v2 |
|---|---|
| `session_transcripts` (id `<len>:<stem>` or `h:<sha>`, `created_at` RFC 3339 nanos, `reserved_at`) | `session.transcripts`, id `Key::of([stem])`; creation time is converted to application `created_ms` (indexes and sorts use it, ties break on `_id`); reservation rows are dropped |
| `session_transcript_entries` (id `<len>:<stem>#<seq>`) | `session.transcript_entries`, id `Key::of([stem, "<seq>"])`, `seq` zero-padded to 10 digits |
| `session_turn_states` (`{thread_id, request_id, lifecycle, turn: {camelCase}}`) | `session.turn_states`, id `Key::of([thread_id, request_id])`, snake_case storage form of `TurnState` with `started_ms` / `updated_ms` |
| `reserved_at` + 30 s stale takeover on a transcript index row | `tsd.leases` (owner `session`, key = stem, 30 s); a takeover advances the lease epoch, which refuses the stalled owner's first write |
| `session_kv` | `tsd.kv` owner `session` (`import_legacy_kv`, see the [harness schema](../harness/storage-schema.md)) |
| `session_import_markers` | `tsd.migrations` (`session` / `session_raw_jsonl_v1`) via `legacy::import_legacy_marker` |

`TurnState` is a wire type (camelCase, RFC 3339) that the app mirrors; only its
stored form changes. Field names become snake_case at any depth, `startedAt` /
`updatedAt` become epoch-ms, a `thinking` transcript item's `startedAt` /
`endedAt` become `started_ms` / `ended_ms`, and a sub-agent tool call's `args` is
kept verbatim. Times keep millisecond precision and read back as UTC.

Steps (`transcripts_v2`, `transcript_entries_v2`, `turn_states_v2`) run with
`Registry::migrate` inside one scope. Moving a v1 agent scope (the raw agent id)
to `tenant.agent(id)` is the host's job. The journal stream is not part of this
change.
