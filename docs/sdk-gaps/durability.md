# SDK Gaps: Durability And Storage

> Part of [SDK Gaps](README.md). Covers the durable orchestration task
> store, SQLite storage compatibility, and storage/graph conformance.

## Backlog

### 4. Durable Orchestration Task Store

Status: partially present.

TinyAgents defines a `TaskStore` trait and an `InMemoryTaskStore`. OpenHuman
still owns durable detached-sub-agent state, cancellation handles, wait/reuse
semantics, tombstones, and task lifecycle persistence around that store.

Implement:

- Add durable `TaskStore` implementations:
  - JSONL append store.
  - SQLite store behind a storage feature.
  - Optional caller-supplied store adapter.
- Persist task spec, status, timestamps, result, error, parent/root run ids,
  cancellation requests, timeouts, and control decisions.
- Add lifecycle history, not only latest state.
- Support replay/listing by parent run, root run, thread id, task kind, status,
  and created-at window.

Acceptance criteria:

- A process restart does not lose detached or awaiting orchestration tasks.
- Supervisors can list, wait, cancel, kill, and inspect tasks through the SDK
  store contract.
- OpenHuman can retire most bespoke task status/tombstone persistence in
  `running_subagents.rs`.

### 5. SQLite Storage Compatibility

Status: partially present.

TinyAgents has a `SqliteCheckpointer`, but enabling the `sqlite` feature pulls a
specific `rusqlite` / `libsqlite3-sys` version. OpenHuman already depends on a
different SQLite native-link version, so it cannot enable that feature and had
to implement `SqlRunLedgerCheckpointer`.

Implement one or more compatibility paths:

- Make SQLite support trait-first and allow external connection adapters.
- Provide a version-flexible storage layer, possibly via `sqlx` or a separate
  crate feature matrix.
- Split schema helpers from dependency ownership so apps can create the tables
  using their own SQLite connection.
- Expose a small `CheckpointStore` persistence trait below `Checkpointer`.

Acceptance criteria:

- Applications that already own SQLite can use TinyAgents durable checkpoints
  without native-link conflicts.
- OpenHuman can replace `SqlRunLedgerCheckpointer` with an SDK-supported adapter
  or a thin schema integration.
- Storage features remain opt-in and keep the default crate dependency-light.

### 17. Storage And Graph Conformance

Status: shipped.

Every persistence layer has a shared contract-test suite, so a backend swap or
a caller-supplied implementation can be certified rather than trusted:
`tinyagents_graph::testkit::conformance` (checkpointer + task-store, run
against memory/file/SQLite and JSONL in `tests/conformance.rs` and
`tests/persistence_conformance.rs`); `tinyagents_harness::store::conformance`
(`run_store_conformance`/`run_namespaced_store_conformance`, run against
`InMemoryStore`/`FileStore`/`InMemoryNamespacedStore` in
`tests/store_conformance.rs` — no in-tree SQLite `Store` exists yet); and
`tinyagents_session::testkit::conformance` (run ledger + transcript history,
run against SQLite and an in-memory double in `tests/session_conformance.rs`).
See `docs/modules/harness/store.md` and
`crates/tinyagents-session/src/README.md` for details. Parallel-agent
order/failure/timeout/cancellation regression tests remain open (see the
fuzz/e2e graph-agent orchestration tests instead).

