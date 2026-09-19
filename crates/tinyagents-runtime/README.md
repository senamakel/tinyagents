# tinyagents-runtime

`tinyagents-runtime` provides the stateful session layer that sits between a
host's turn policy and TinyAgents' provider-neutral harness. A `Session` owns
the mutable model history, a stable prefix, a frozen tool declaration snapshot,
and append-only transcript state. It has no host configuration, credentials,
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
- `SessionHooks` prepares a request and validates the candidate before the
  commit point, then observes the durable result through `after_commit` and
  one terminal state. It does not make policy decisions.

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

To persist, add `SessionBuilder::transcript(locator, stem, meta)`. The runtime
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
duplication. Cancellation before the commit point leaves no durable mutation;
once it succeeds, the turn remains successful. `after_commit` and terminal
hooks get the committed outcome, but their error or a cooperative cancellation
cannot relabel it.
