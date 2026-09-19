# tinyagents-runtime

`tinyagents-runtime` provides the stateful session layer that sits between a
host's turn policy and TinyAgents' provider-neutral harness. A `Session` owns
the mutable model history, a stable prefix, and append-only transcript state.
Tool declarations are prepared afresh for every driver call. It has no host configuration, credentials,
prompt construction, tool authorization, model selection, or event system.

## Host responsibilities

The host supplies three narrow seams:

- `SessionDriver<C>` executes one prepared history snapshot. `HarnessDriver`
  adapts an `AgentHarness<State, C>` and passes the host's explicit `C` through
  unchanged. It fails closed unless the harness registry exactly matches the
  frozen request tool snapshot.
- `TranscriptCodec<C>` decodes the host's durable dialect and reconciles the
  previous durable rows with a model-history transition. The codec, rather
  than the runtime, retains fields inference messages cannot express. `C` is
  `Clone` so reconciliation receives the current host context plus request,
  thread, stream, and resume options after the live `RunContext` moves into the
  driver.
- `SessionHooks<C>` prepares a request and mutable `TurnOptions<C>` before
  handoff. Its `TurnPreparation` can install a first-turn prefix, select the
  one immutable `ToolSnapshot` for that request, and lazily choose a
  `TranscriptTarget`. `before_commit` validates the candidate; `after_commit`
  receives an exactly-once `CommitReceipt<C>` containing the explicit context
  snapshot and neutral transcript path/delta receipt. `on_terminal` receives
  one truthful terminal state. It does not make policy decisions.

```rust,no_run
use std::sync::Arc;
use tinyagents_runtime::{SessionBuilder, SessionDriver, TranscriptCodec};

# fn build(driver: Arc<dyn SessionDriver<()>>, codec: Arc<dyn TranscriptCodec<()>>) {
let session = SessionBuilder::new(driver)
    .codec(codec)
    .build();
# let _ = session;
# }
```

To supply a default lazy destination, add `SessionBuilder::transcript(locator, stem, meta)`.
It does not open a transcript while building: a selected target binds only on
resume/first append and cannot be redirected after that. The runtime
uses `tinyagents-session`'s `TranscriptHistory::append_turn_with_partial`, so a normal
extension appends only the new tail and a reduced context writes one compaction
record. A supplied partial driver outcome is represented through that single
history operation: logical history is replayable and interrupted display text
is not. Histories that cannot provide the combined operation reject a partial
rather than risk a two-step write. A persistence failure leaves the session's
in-memory history and persisted snapshot unchanged.

Every turn receives explicit `TurnOptions`, including its cancellation token
and `RunContext<C>`; no task-local data crosses the runtime boundary. The
stable prefix is reconciled after resume and driver compaction without
duplication. `Session::seed_history(history, raw)` is the explicit, lossless
resume/seed boundary; a host must not keep a second shadow history. Cancellation before the commit point leaves no durable mutation;
once it succeeds, the turn remains successful. `after_commit` and terminal
hooks get the committed outcome, but their error or a cooperative cancellation
cannot relabel it.
