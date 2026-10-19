# `session::threads` — chat thread and message store

The product-facing chat log: the threads a UI lists (title, labels, parent,
personality, message count, last activity) and the messages it renders inside
each one, with substring search across every thread.

[`DriverConversationStore`] is the only implementation. It runs on
`tinystoragedrivers` ports, so the host picks the backend (SQLite, MongoDB,
memory, ...). The JSONL file store, `FileConversationStore` and the
`ConversationStore` free-function API are removed. A library host with no
storage backend uses `DriverConversationStore::in_memory().await`, a
process-lifetime store over the memory driver.

## Contract

`ConversationPersistence` is the object-safe async contract: thread and
message CRUD, `update_message`, `delete_messages_from`, `delete_thread`,
`purge_threads` and exact substring search. `ConversationError` keeps native
driver errors typed (`Driver`), refusals (`Unsupported`) and unresolved write
acknowledgments (`Uncertain`). `SessionStoreProvider::conversations_for_agent`
returns an optional shared handle; absence leaves selection to the host, and a
configured refusal must never cause a local fallback.

Turn claim, renew, complete and release return typed `Unsupported`: they need
durable completed-request deduplication and owner/epoch/expiry fencing that a
process mutex does not provide.

Wire types, serde names and deterministic message ids are unchanged from the
original TinyMemory `tinymemory-conversations` crate (`ThreadMessage` and
`ThreadMessagePatch` are the renamed `ConversationMessage` types).

## Layout in the driver

Metadata lives in the `conversations.threads` collection (one document per
thread, id `Key::of([thread_id])`, a flat snake_case body with `*_ms` times).
Committed messages live in an immutable stream per incarnation,
`conversations.messages/<thread>/<incarnation>`. Recreating a deleted thread,
editing a message and cutting a tail each open a fresh UUID incarnation. The
stored form is separate from the camelCase wire types; see
[the schema doc](../../../../docs/modules/session/conversations-storage-schema.md)
for the v1 -> v2 mapping and `conversations_storage_schema()` for the steps.

### Explicit schema preparation and actor admission

`DriverConversationStore::collection_specs()` publishes the thread and derived
search collection declarations. Trusted bootstrap calls
`DriverConversationStore::prepare_schema(documents)` on its raw native document
port. Native internal stream/database preparation remains a separate host step.
Restricted native handles refuse this declaration helper.

`new_validate_only(scoped)` captures immutable validation-only behavior and
validates required persisted features on every implemented operation, including
reads and empty searches. Validation success is not cached. Native compatible
supersets are accepted: this is required-feature validation, not exact schema
equality. Native failure kind and original source are retained. The caller must
supply existing-schema native handles with its original actor fence; selecting
this constructor alone does not restrict backend lazy initialization or grant
scope authority. Default turn-claim refusals perform no native operation.

The existing `new(scoped)` remains deliberately compatible and does not declare
or validate schema. It must not be advertised as actor schema safety. Neither
constructor selects a provider automatically, opens host files or enables imports.
Plain append requires the additional preparation below.

### Native plain append and recovery

`append_collection_specs()` adds `conversations.appends` to the existing
metadata declaration. Explicit trusted `prepare_append_schema(documents)`
prepares both; metadata-only readers retain their previous schema requirements.
Every append validates both declarations without actor DDL. It requires native
checked document batches and conditional stream append. An actual empty
conditional append probes the paired stream before reserving any intent: it
adds no message, but may affect native internal stream bookkeeping. A refusal
never selects file persistence.

Reservation atomically checks the observed thread revision and absent pending
intent. The intent holds an immutable attempt token, incarnation, expected
stream offset and the complete message as an object (the stream row it will
become). Writer-only
recovery appends or verifies the exact row, then atomically publishes the next
committed prefix and removes that exact intent version. Readers never recover
pending work or expose an uncommitted row. Metadata count/last activity follow
classic semantics, and concurrent unrelated title changes are retained.

Cancellation may leave accepted pending work for a later admitted writer.
An unknown acknowledgment is resolved only through durable ownership evidence:
matching row bytes alone cannot attribute an unobserved ordinary-ID reservation.
`ConversationError::Uncertain` preserves the original native error and source
when ownership remains unresolved; callers must not silently retry such ordinary
UUID appends. Deterministic `agent:` IDs or channel metadata retain classic
existing-ID deduplication and return the original stored message. Ordinary IDs
still append on each separate invocation; this is not a generic exactly-once API.

### Edits, tail cuts and purge

Committed streams are never edited in place. `update_message` and
`delete_messages_from` copy the replacement log onto a stream of a new
incarnation, then swap the metadata document onto it with a compare-and-swap on
the version they read; a concurrent metadata change retries the rewrite, and an
append racing the swap fails with a conflict because its incarnation is
retired. The old stream and any append intent that belonged to it are removed
after the swap. `delete_messages_from` returns `None` for an unknown cut point
or thread. `purge_threads` deletes every thread document (live and tombstoned),
stream and intent, and counts live threads and messages.

Plain message append does not authorize distributed agent execution. No
importer, provider selection or legacy source-ownership grant is added.

## Search

Search maintains a derived, scoped `conversations.search` collection for
FullText-capable stores. Its opaque ASCII tokens encode the existing normalized
trigram/CJK terms (plus code-point tokens for the legacy short-term cases), so
backend-specific text parsers do not change matching semantics. Appends publish
the message row atomically with thread metadata; edits and tail cuts write rows
for the replacement incarnation before publishing it. Searches use FullText
only to find candidates, then validate their live thread incarnation and
visible prefix and apply the existing exact substring ranking and pagination.
Old persisted conversations are backfilled once before the readiness marker is
published. No corpus scan is done on steady-state FullText searches. Stores
without FullText retain the exhaustive scan fallback, and FullText errors are
returned to the caller.

## Public surface

- **Store:** `DriverConversationStore` (`new`, `new_validate_only`,
  `in_memory`, `collection_specs`, `append_collection_specs`,
  `prepare_schema`, `prepare_append_schema`) implementing
  `ConversationPersistence`, plus `ConversationPurgeStats`.
- **Schema:** `conversations_storage_schema()` (owner `conversations`, scope
  class Tenant) for the host's `Registry::compose`.
- **Types:** `ConversationThread`, `ThreadMessage`,
  `CreateConversationThread`, `ThreadMessagePatch`, `CrossThreadHit`.
- **Deterministic ids:** `DETERMINISTIC_MESSAGE_ID_PREFIX`,
  `run_reply_message_id`, `is_deterministic_message_id`, `reply_run_id`.

A transient failure while helping another writer's pending intent returns the
original native driver error: the caller's own message has not yet been reserved
and can be retried. `Uncertain` describes only the caller's own accepted-attempt
ambiguity.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | module docs and re-exports |
| `persistence.rs` | async contract, typed failures, purge stats, unsupported turn claims |
| `driver.rs` | metadata CAS, schema policies, in-memory constructor |
| `driver/append.rs` | writer-only conditional append and checked publication |
| `driver/pending.rs` | attempt intent and strict pending-record validation |
| `driver/reader.rs` | committed-prefix readers |
| `driver/rewrite.rs` | message edit, tail cut and purge |
| `labels.rs` | label canonicalisation |
| `types.rs` | wire types and deterministic message ids |
| `dto.rs` | stored (v2) bodies of threads, messages and append intents; time conversion |
| `schema.rs` | `conversations_storage_schema()`, names and the stream-name rule |
| `schema_steps.rs` | the v1 -> v2 migration steps |
| `tokenize.rs` | normalization and n-gram tokenizer |
| `inverted_index.rs` | exact matching, ranking and pagination for search candidates |
| `driver/search.rs` | derived search rows, FullText query path and legacy backfill |
| `*_tests.rs` | unit tests |
