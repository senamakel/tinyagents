# harness::memory

Short-term (thread-scoped) conversation memory and its store boundary.

## Why this exists

Memory is the persistent-state side of the recursive runtime: a thread's
transcript survives across runs so an orchestrator can interrupt a sub-agent,
take human input, and re-enter the *same* thread later — the
reuse-with-accumulating-context pattern `crate::subagent` builds on.
[`MemoryScope`] separates that thread-local short-term layer from the
cross-thread long-term `crate::store::Store`; only short-term (chat history)
concerns live in this module, long-term memory is the `Store` trait itself.

## Public surface

- [`MemoryScope`] (`types.rs`) — labels which conceptual layer a record
  belongs to (`ShortTerm` / `LongTerm`). Primarily documentary.
- [`ChatHistory`] (`types.rs`) — the trait: ordered per-thread
  `messages`/`append`/`replace`/`clear`. `replace` has a default
  clear-then-re-append implementation that is non-atomic; both backends below
  override it with a single bulk write.
- [`InMemoryChatHistory`] (`types.rs` + `mod.rs`) — ephemeral, in-process
  implementation backed by a shared `Arc<Mutex<HashMap<...>>>`. No durability;
  for tests, examples, and prototyping.
- [`StoreChatHistory<S>`] (`types.rs` + `mod.rs`) — durable implementation
  over any `crate::store::Store`. Serializes each thread's full message list
  to JSON under namespace `StoreChatHistory::NAMESPACE`. `append` is
  serialized per-thread through an internal async mutex (`append_locks`) to
  close the read-modify-write race a naive `Store`-backed append would have.
- [`ShortTermMemory<H>`] (`types.rs` + `mod.rs`) — a thin wrapper scoping any
  `ChatHistory` to one fixed `thread_id`, with an optional trimming hook
  (`with_trim`) applied on both `load` and `save`.

## Files

| File       | Role                                                               |
| ---------- | ---------------------------------------------------------------------- |
| `types.rs` | `MemoryScope`, `ChatHistory` trait, `InMemoryChatHistory`, `StoreChatHistory`, `ShortTermMemory` struct definitions. |
| `mod.rs`   | Method implementations for the three concrete types.                  |
| `test.rs`  | Unit tests covering both `ChatHistory` backends and `ShortTermMemory` trimming. |

## Operational constraints

- `ChatHistory::replace` must be atomic (or as close as the backend allows):
  a mid-write failure must not leave a thread with its cleared prefix lost.
  Both concrete backends override the default clear-then-re-append with a
  single bulk write (or delete, for an empty list) — preserve that if a new
  backend is added.
- `StoreChatHistory::append`'s per-thread lock is an **in-process** guarantee
  only. Across processes sharing one `FileStore` directory the
  read-modify-write is still racy; closing that needs a compare-and-swap
  primitive the `Store` trait does not currently have. Prefer `replace` (a
  single write) when the caller already holds the full message list.
- `InMemoryChatHistory` and `StoreChatHistory` must give matching semantics
  for the same trait method — `InMemoryChatHistory::append` holds its lock for
  the whole operation, which is why `StoreChatHistory` needs its own
  serialization mechanism rather than relying on the store alone.
- `ShortTermMemory`'s trim hook runs on both `load` and `save`; a hook that is
  not idempotent (trimming further on every save) will progressively shrink
  the persisted history across repeated load/save cycles.
