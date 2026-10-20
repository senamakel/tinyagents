# Host import coverage handoff

The pre-migration TinyAgents host had a filesystem-backed SQLite source
snapshot importer. That behavior is not part of the current native run-ledger
port, so its source-opening and ownership policy belongs in OpenHuman's host
adapter (#440), not in this Mongo destination proof or a replacement
TinyAgents engine.

The source behavior and assertions to carry forward are recorded at canonical
TinyAgents `a5b4d0f6b8b962b28c4288a8518773743e252343`:

- `crates/tinyagents-session/src/run_ledger/drivers/import_tests.rs`:
  `preflight_rejects_wrong_owner_before_source_open`,
  `preflight_checkpointed_source_preserves_directory_bytes`,
  `preflight_rejects_wal_without_mutating_source_or_auxiliary_files`,
  `preflight_missing_owned_source_never_creates_database`, and
  `preflight_reserved_path_characters_preserve_original_source`. Preserve the
  owner check before opening a path, read-only source inspection, refusal of
  WAL/checkpoint ambiguity, no creation of missing source databases, path
  validation, and byte-for-byte preservation of source plus sidecar files.
- The same file's
  `admitted_agent_import_preserves_source_and_retry_never_overwrites_native_edits`,
  `later_event_collision_rolls_back_runs_and_import_receipt_together`,
  `imported_pruned_events_keep_original_ordinals_and_continue_allocation`, and
  `changed_source_cannot_reuse_a_committed_import_operation` pin import
  idempotency, collision atomicity, sequence preservation, and source-change
  refusal. The host adapter must preserve these semantics if it remains the
  owner of source admission/import orchestration.
- `integration-proofs/run-ledger-mongodb/tests/native_run_ledger_mongodb.rs`
  at that same SHA contained
  `terminal_event_append_uses_the_actual_mongo_bson_converter` and
  `terminal_event_import_uses_the_actual_mongo_bson_converter`. The append
  case's destination-side sequence boundary and the separate-connection
  persistence checks remain in the current Mongo proof. The import case's
  immutable SQLite source snapshot and destination import path are omitted
  here because the public native API no longer exposes that importer; the host
  import proof must retain the original source immutability and sequence
  boundary assertions when an approved import contract is available.

This handoff does not claim the host adapter implements the removed API. If
#440 requires the source-snapshot workflow, it needs an explicit host-owned
contract for admitted source paths and an atomic destination operation, or a
separately approved owner-side API change.
