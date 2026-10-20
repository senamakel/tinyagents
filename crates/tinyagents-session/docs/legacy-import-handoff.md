# Legacy session import handoff to OpenHuman

OpenHuman owns the former workspace layout and the source-reading workflow.
`tinyagents-session` accepts host-bound storage and exposes target writes and
owner-local in-storage migrations; it does not discover workspace files, open
SQLite sources, or rewrite JSONL files.

The removed implementation is recoverable at TinyAgents commit
[`bec73972c1a8c96ce0bc54dd77f56da2e145f875`](https://github.com/senamakel/tinyagents/tree/bec73972c1a8c96ce0bc54dd77f56da2e145f875/crates/tinyagents-session).
That is the parent of `c30844a4e`, which removed the source import and workspace
opening code. OpenHuman #440 should take over these paths and behaviors:

| Former source | Former entry point | Host responsibility |
| --- | --- | --- |
| `src/legacy/source.rs`, `rows.rs`, `workspace.rs` | `legacy::open_source`, `legacy::import_workspace` | Verify ownership/quiescence, read `session_db/sessions.db` as an immutable snapshot, enumerate old schema rows and coordinate the whole migration. |
| `src/run_ledger/drivers/import.rs` | `DriverRunLedger::import_legacy_database` | Decode classic sessions, messages, tool calls, runs, events, telemetry, teams and tasks, then write them to the host-selected target. The former entry-tree records have no target API after canonical TinyAgents removed that subsystem. |
| `src/run_ledger/drivers/legacy_import.rs` | `DriverRunLedger::import_agent_family` | Preserve the operation-id receipt and all-or-none target batch behavior while the host owns source authorization and bytes. |
| `src/legacy/transcripts.rs`, `transcripts_once.rs`, `fixtures.rs` | `legacy::import_transcripts`, `legacy::import_transcripts_once` | Read `session_raw/*.jsonl`, preserve stems and request identity, and track resumable completion. |
| `src/legacy/turn_states.rs` | `legacy::import_turn_states` | Read the former `memory/conversations/turn_states/` JSON snapshots and write turn-state ports. |
| `src/legacy/rewrite.rs`, `src/transcript/writer.rs` | `legacy::rewrite_legacy_transcripts` | Own any requested in-place JSONL rewrite, including backup/recovery and file lifecycle. |
| `src/port/workspace.rs`, `src/port/types.rs` | `WorkspaceStores::open(workspace_dir, agent_id)` | Resolve workspace and agent identity, select/open storage, import old files, and construct the host's `SessionStoreProvider`. |

For ordinary target writes, the host receives a `ScopedStorage` already bound
by its own identity policy and can build `DriverSessionStores` with an
`AgentStorageFactory` (`Arc<dyn Fn(&str) -> Result<ScopedStorage, StorageError>>`).
The normal `DriverSessionStores::try_for_agent(agent_id)` open path invokes
`open_agent_storage(&ScopedStorage)` before returning handles. That owner-local
entry point imports the legacy in-storage lease/counter rows, then runs the
session, ledger, history, and conversation schemas. The host supplies the
already-bound scope and does not compose consumer registries or run these model
migrations itself. `driver_run_ledger(agent_id)` and
`SessionStoreProvider::run_ledger_for_agent(agent_id)` use the same open path.
The open path also moves the session-owned `session_kv` rows into `tsd.kv`
through the harness's public `import_legacy_kv` API before acquiring the
session Registry lease; that importer owns its own lease. This is distinct from
the old workspace/SQLite import workflows listed above.
Session history uses `SessionHistoryStore` methods such
as `record_session_start`, `record_message`, `record_tool_call`, and
`record_session_end`; turn snapshots use `TurnStates`; transcripts use
`TranscriptLocator::open_stem` and `TranscriptHistory::append_turn`; chat rows
use `ConversationPersistence`. There is no entry-tree target API in this
branch. The imported filesystem/SQLite source is not passed to these methods.
None of these target methods accepts a path, SQLite connection, workspace,
tenant grammar, or source authority.

The source files are no longer present in this branch. Their importer tests
were `src/legacy/{files_tests.rs,mod_tests.rs,rewrite_tests.rs,
transcripts_once_tests.rs}` and `src/run_ledger/drivers/import_tests.rs`.
`src/run_ledger/drivers/legacy_state_tests.rs` is retained here because it
tests the in-storage self-migration described above. OpenHuman #440 should
port source lifecycle, read-only behavior, schema compatibility, idempotence
and resume tests to the host importer. Keep this crate's
`run_ledger/drivers/legacy_state.rs` self-migration: it reads old domain
documents already in the supplied storage scope and moves live workflow leases
and history counters to shared primitives before the owner's schema migration.

The existing `storage_schema()` steps likewise migrate records already in the
host-supplied backend. They do not inspect the former workspace files. Session
scope resolution and any profile re-homing remain host-owned operations.

Native-engine admission/fencing tests were removed with the engine-specific
contract. The former test names are preserved here for the host adapter's
review: `port/drivers/schema_admission_tests.rs` contained
`validation_only_conformance_never_attempts_schema_declaration`,
`validation_only_refusals_keep_native_sources_and_retry_without_caching`,
`explicit_actor_factory_selects_validation_without_changing_original_provider`,
`validation_only_missing_or_incompatible_schema_refuses_before_recovery_or_cache`,
`legacy_actor_factory_inherits_validation_policy_without_declaring_schema`,
and `validation_only_provider_ledger_uses_prepared_schema_without_declaration`.
`threads/driver_schema_tests.rs` contained
`memory_explicit_admission_refuses_missing_declaration_before_read`,
`sqlite_explicit_admission_refuses_missing_declaration_before_read`,
`memory_prepared_actor_retains_original_native_fence_and_scope`,
`sqlite_prepared_actor_retains_original_native_fence_and_scope`,
`validation_is_repeated_and_preserves_native_refusal_source_without_actor_ddl`,
`native_required_feature_validation_accepts_compatible_superset_declarations`,
and `schema_preparation_is_explicit_and_restricted_native_handles_refuse_it`.
The host-specific stale-fence assertion was
`threads/driver_tests.rs::native_scope_fence_rejects_metadata_takeover_without_local_fallback`;
append-phase native fence assertions were in
`threads/driver_append_tests.rs::memory_original_native_fence_refuses_each_append_commit_after_takeover`
and `sqlite_original_native_fence_refuses_each_append_commit_after_takeover`.
The reusable helper was `fenced_phases`. The workflow case was
`run_ledger/drivers/mod_tests.rs::scoped_workflow_writes_preserve_native_owner_fence_refusal`.
These tested backend admission and fencing, which the host selects and
composes; retain the separate session run-lease and transcript-reservation
behavior tests that exercise session-owned coordination.
