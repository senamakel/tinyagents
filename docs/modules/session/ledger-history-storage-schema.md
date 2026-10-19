# Run ledger and history storage schema v2

`tinyagents_session::ledger_storage_schema()` (owner `ledger`) and
`history_storage_schema()` (owner `history`) declare what `DriverRunLedger`
stores, one `Schema` per owner as the registry requires. A host composes them
with `storage_schema()` (owner `session`) and runs `Registry::migrate` per
agent scope.

| v1 | v2 |
|---|---|
| `run_ledger_{workflow_runs,telemetry,agent_runs,teams,members,tool_effects,event_outbox,event_heads,tasks}` | `ledger.{...}`: id `Key::of`, snake_case body, `*_ms` times, no `{schemaVersion, record, *Sql}` envelope |
| stream `run_events:<sha>` | stream `ledger.events/<run>`; entries `{sequence, event_type, payload, ts_ms}` |
| `session_history_{sessions,messages,tool_calls}` | `history.{sessions,messages,tool_calls}`; `startedUs`/`createdUs` keys replaced by `started_ms` / `ended_ms` / `created_ms` |
| `session_history_fts` | `history.search`, rebuilt by the session, message and tool-call steps |
| `session_history_counters` | `tsd.counters` (owner `history`, `messages` / `tool_calls`) |
| `WorkflowRun.lease_owner` / `lease_expires_at` | `tsd.leases` (owner `ledger`, key `Key::of(["workflow", run_id])`, so the document id is `ledger/workflow%2F<run>`) |
| `run_ledger_agent_imports` | `tsd.migrations` (owner `ledger`, step `agent_import/<operation id>`) |

## Steps

Document steps (`*_v2`) decode the v1 envelope, convert the camelCase wire
record to its storage DTO and re-key it. `run_events_v2` is a stream step: the v1
stream name is a digest of the run id, so the run id is read from the entries.
`event_heads_v2` and `search_v2` drop their rows (the v1 head id is a digest of
the run id, and search rows are rebuilt from the records); the next sequence of a
migrated run is recomputed from its last event.

## Host importer

A step cannot write `tsd.*`, so `import_legacy_ledger(storage)` moves, once per
scope and **before** `Registry::migrate`, the live workflow leases and the
history counters. Run afterwards it finds nothing, and a held lease lapses.
Agent-import receipts are dropped (their id was a digest of the operation id):
re-running an old operation id after the upgrade conflicts on the imported rows.

## Behaviour changes

- Times are stored as whole milliseconds.
- Dense history ids come from `Counters::next`, no longer part of the batch
  that writes the row: a failed commit leaves a gap.
- A workflow lease and its run are separate documents, so claim, renew and the
  revision-fenced write are no longer one atomic change.
- `Repo::update` backs off with `tokio::time::sleep` on contention, so a
  runtime that drives the ledger needs timers enabled.
