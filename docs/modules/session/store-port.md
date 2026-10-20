# The session store port

`tinyagents_session::port` is where a host plugs in its own storage for
everything an agent persists while it runs. A host implements one
`SessionStoreProvider`; the runtime asks it for each agent's `AgentStores`
and never names a file, a database or a directory itself.

```rust
pub trait SessionStoreProvider: Send + Sync {
    fn for_agent(&self, agent_id: &str) -> AgentStores;
    fn recover(&self) -> anyhow::Result<()> { Ok(()) }          // once, at boot
    fn destination_key(&self) -> Option<String> { None }        // for logs
    fn workspace_dir(&self) -> Option<PathBuf> { None }         // file-backed only
}

pub struct AgentStores {
    pub transcripts: Arc<dyn TranscriptLocator>,  // what the model sees
    pub turn_states: Arc<dyn TurnStates>,         // snapshots of turns in flight
    pub kv: Arc<dyn Store>,                       // run status, goals, todos
    pub journal: Arc<dyn AppendStore>,            // each run's events
}
```

## Why a port

The file and SQLite layout (`session_raw/`, `session_db/sessions.db`,
`tinyagents_store/`, turn-state files) suits one operator on one machine.
A cloud host serving many users from one process needs every agent's state
in a shared database, scoped by agent, so a conversation outlives the
process that served it. The port lets both use the same runtime.

## Isolation

`for_agent` and, with `storage-drivers`, `try_for_agent` obtain handles bound
to that agent. Hosts that must propagate admission or recovery failures use
`try_for_agent`; the legacy accessor can return refusing handles instead.
A provider serving several users from one backend must
make it impossible for one agent's handles to reach another's data; the
single-user desktop layout shares one workspace between its agents and does
not claim isolation.

## Sync and async

`TranscriptLocator`/`TranscriptHistory` and `TurnStates` are synchronous:
the turn path that commits a transcript is a chain of sync methods. An
implementation over an async client bridges at its own boundary (`DriverSessionStores` uses a dedicated runtime thread;
`block_in_place` also works, on a multi-threaded tokio runtime only). `Store` and `AppendStore` are
the harness's async traits; `Arc<dyn Store>` and `Arc<dyn AppendStore>`
implement them, so code generic over a store accepts injected handles, and
`FileStatusStore::over` keeps run status in any `Store`.

## Bundled pieces

- `InMemorySessionStores`: per-agent, process-lifetime stores
  (`InMemoryTranscriptLocator`, `InMemoryTurnStates`, the harness's
  in-memory stores). For tests and hosts that keep nothing.
- There is no file-backed transcript locator: the path-based transcript API
  (`FileTranscriptLocator`, `FileTranscriptHistory`, the workspace-path
  readers and writers) is removed. A host uses `DriverSessionStores` (or
  `WorkspaceStores`, which also performs the one-time read-only import of a
  classic `session_raw/` directory) and `InMemorySessionStores` for tests.
- `TranscriptLocator::append_interrupted_partial` carries the display-only
  partial of an interrupted turn without a file path.
- `DriverSessionStores` (feature `storage-drivers`): every agent's stores in
  one `tinystoragedrivers` backend, described below.

## `DriverSessionStores`

With the `storage-drivers` feature, a host that has opened a
`tinystoragedrivers` backend (SQLite on a desktop, MongoDB in the cloud,
memory in tests) gets a complete provider from it:

```rust
let backend: Arc<dyn StorageBackend> = Arc::new(SqliteStorage::open(path)?);
let provider = DriverSessionStores::new(backend, Scope::local())?.recover_on_open(true);
```

The provider is constructed with the **tenant** `Scope` it serves
(`Scope::local()` for the desktop operator, `Scope::tenant(id)` for a user) and
derives each agent's scope itself, `tenant.agent(id)` (`local/agent:<id>`); it
never uses a raw agent id as a scope. `Scope::agent` hashes an id that is too
long or outside the segment grammar (the segment only, so the agent stays under
its tenant), and an empty id has no scope. `new` refuses a scope that is not a
tenant. The v1 constructor took no tenant (`new(backend)`) and stored agents under
the raw id or `sha256:<hex>`; the driver enforces scopes, so the provider claims
isolation and passes `session_store_isolation_conformance`. The v2 names and
migration steps are in [storage-schema.md](storage-schema.md).

### Prepared actor schemas

Existing constructors keep declaring schema lazily. A trusted host can instead
prepare `DriverSessionStores::collection_specs()` through `prepare_schema`,
then create an actor provider with
`new_with_schema_admission(backend, tenant, SchemaAdmission::ValidateOnly)`.
Import the original admission policy from
`tinyagents_harness::store::SchemaAdmission`; the session driver constructors
consume that type directly.
The schema policy is fixed at construction. Checked opening validates all four
original declarations before creating handles or recovering turns. Transcript,
turn-state and KV handles use that same policy for their first operation;
validation never calls `ensure_collection` or repairs missing indexes.

Through an erased `SessionStoreProvider`, use
`bind_actor_with_schema_admission(backend, SchemaAdmission::ValidateOnly)`.
The driver implementation retains its bridge and recovery setting but creates
independent admission caches. Legacy providers return `Ok(None)` for this
policy unless they explicitly support it. `Declare` delegates to their existing
factory. Neither absence nor a native error authorizes a fallback.

The host still supplies an actor-authorized backend and handles lease,
generation and tenant admission. A schema mode does not grant that authority.
Checked opening preserves native missing, incompatible, unsupported and fenced
errors, including their sources. The older synchronous transcript/turn interfaces
and harness error conversion retain their existing string error representations.

### Checked opening

With `storage-drivers`, call `SessionStoreProvider::try_for_agent(agent_id)`
through the provider trait when admission must be authoritative. It returns
`Result<AgentStores, StorageError>`; driver scope admission and recovery failures
retain their original error kind and native source. The `Arc<P>` implementation
forwards the checked accessor. Legacy providers default to
`Ok(self.for_agent(agent_id))`, so existing providers and store literals remain
compatible.

Driver recovery uses the same turn-state handles and blocking bridge as normal
operations, without projecting errors through the legacy string interface.
Failed recovery caches neither stores nor a successful recovery marker, so a
later checked open retries. The existing infallible `for_agent` still returns
refusing handles on driver failure. Hosts selecting configured storage must
propagate checked errors rather than treating them as an absent provider.

Regression evidence: the generic provider and Arc accessors previously returned
successful refusing/legacy handles; recovery replaced native declaration and
query errors with `Unavailable`. Behavioral tests reproduced those failures
before the override, forwarding and typed recovery helper were added. Tests
cover source identity, unusual key mapping, legacy file compatibility,
successful cached handles, independent actor caches and recovery retry.

Local validation on 2026-10-11 passed 559 feature-enabled tests and two
doctests, 495 default-feature tests and two doctests, strict all-feature Clippy
and rustdoc. The full instrumented suite passed serially with 83.76% overall
line coverage; the changed provider modules had 100%, and the turn-state
module 96.48%. An initial parallel instrumented run encountered a pre-existing
transcript generation file-lock refusal; its failed log was retained and no
test was skipped in the successful full rerun.

### Actor-owned providers

With `storage-drivers`, `SessionStoreProvider::bind_actor(admitted_backend)`
optionally creates a provider for one host-admitted actor. `DriverSessionStores`
reuses its blocking bridge and recovery policy while creating a fresh agent cache,
recovery bookkeeping and provider identity. Repeated calls on the returned actor
share its handles. Agent scopes and persisted transcript stems remain unchanged;
old retained providers continue to use their original backend.

The host must supply a backend already bound to the actor's authority, such as an
immutable lease grant. This factory does not acquire a lease, resolve a profile,
or make a backend that selects the latest grant safe. Fresh actors after takeover
must receive a newly admitted backend; old actors must remain fenced by theirs.
Synchronous transcript operations retain native storage errors through the
blocking bridge. The legacy turn-state trait still exposes string errors.

File and arbitrary providers default to `Ok(None)`, meaning rebinding is
unsupported. Absence does not authorize a file or backend fallback. A configured
factory refusal remains an error with its native source. Existing `AgentStores`
literals and providers are unchanged.

| Store | Backend shape |
| --- | --- |
| transcripts | `session.transcripts`: one index document per stem (thread, agent, sub-agent flag, application `created_ms`); `session.transcript_entries`: an append-only log per stem |
| generation reservations | `tsd.leases` (owner `session`, key = the successor stem) |
| turn states | `session.turn_states`: one document per `(thread, request)` |
| key-value | `tsd.kv` (owner `session`) through the harness `DriverStore` |
| journal | streams `harness.journal/<name>` (the harness `DriverAppendStore::journal`; declared by `tinyagents_harness::storage_schema()`) |

- **Transcript log.** Each write is one entry, numbered from 0 and claimed
  with an insert-only write. An ordinary turn stores only its new rows
  (`extend`); a first write or a compaction stores the whole set (`set`). Two
  writers on one transcript (two processes on one database) serialize: the
  loser re-reads and decides again, which is also how a write racing a seal is
  refused. As in the JSONL writer, a turn whose `prev` is not what is stored
  is refused as stale rather than allowed to replace newer rows, and the
  written rows carry the turn's usage, request id and step stamps, built by
  the writer's own code (`transcript::stamped_rows`).
- **Index.** Each index document records the log entry it was built from
  (`indexed_seq`). A handle with an older replay leaves it alone, and the
  update is a compare-and-swap. A failed index refresh never fails the write
  that preceded it; the next write retries it.
- **Generations.** `begin_generation` reserves the successor in the index
  before sealing the predecessor, so two compactions cannot both open it. A
  reservation left unwritten for 30 seconds (its process stopped mid-way) may
  be taken over, so a sealed head never strands the conversation.
  `begin_generation_from_baseline` checks the baseline inside the seal
  itself, so a turn committed after the caller read it fails the compaction
  instead of vanishing from the successor. A failed seal releases the
  reservation only at the version this call wrote.
- **Turn states.** Conditional writes, settling, the interruption sweep and
  retention pruning are all conditional on the document version.
- **Sync seams.** Transcript and turn-state calls run on a
  `tinystoragedrivers` `Blocking` bridge (one dedicated runtime thread), so
  they work from any caller, inside a runtime or not.
- **Recovery.** A backend cannot list its scopes, so `recover` covers the
  agents this provider has opened. `recover_on_open(true)` also interrupts an
  agent's in-flight turns the first time it is opened. The agent counts as
  recovered only once the sweep succeeds, and a concurrent first open waits
  for that sweep. Use it only when a
  single process owns the database (the desktop app).
- **Failing closed.** If the backend cannot bind an agent's scope, `for_agent`
  returns stores that refuse every call with that error and does not cache
  them. `try_for_agent` returns the error itself.

## Trusted schema preparation

`DriverSessionStores::collection_specs()` returns the exact declarations used
by the transcript, turn-state and key-value adapters. Hosts can include this
manifest in their trusted bootstrap process without copying index definitions.
The journal uses streams and has no document collection declaration.

`DriverSessionStores::prepare_schema(documents).await` idempotently declares
all four collections in the addressed logical database. Supply a trusted
bootstrap document handle for each database, before issuing actors. It returns
the first native error with its kind and cause. Earlier declarations can remain
after a later failure; retrying is safe, but this is not an atomic migration.
Native internal stream/blob metadata preparation remains the driver's job.

This API does not grant a lease, bind actor identity, disable lazy declarations,
or validate restricted actor reads. A host must separately use the owning
native schema-validation contract when integrating actors without DDL authority.
`DriverStore::collection_spec()` also exposes the shared `tsd.kv`
declaration for hosts using that adapter independently.

## Conformance

`testkit::conformance::session_store_conformance(&provider)` exercises a
whole provider: transcripts (sessions, thread and agent lookups, compaction
generations, partials kept out of the replay), turn states (conditional
writes, settling, interrupted-marking) and the key-value and journal stores.
`session_store_conformance` runs against `InMemorySessionStores`, the file
building blocks and `DriverSessionStores` over the memory and SQLite drivers. It checks the common session-store behavior, not provider
isolation. `session_store_isolation_conformance` is a separate check that two
agents cannot see each other's data; it runs only against
`InMemorySessionStores`, `DriverSessionStores` and hosts whose providers claim
isolation.
