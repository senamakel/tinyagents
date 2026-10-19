# Session persistence

`tinyagents-session` owns durable session history, transcripts, turn-state snapshots, chat threads and the run ledger. Hosts provide storage through the session store port; see [`store-port.md`](store-port.md).

## The tool-effect ledger (`run_ledger::tool_effects`, B5)

An independent piece of durable state lives in this crate alongside
the run ledger's `agent_runs`/`workflow_runs` tables:
`run_ledger::tool_effects` gives `tinyagents-harness` a crash-safe record of
tool-call side effects, backing
`tinyagents_harness::tool::ToolEffectLedger`. See
`docs/modules/harness/tool-effects.md` for the full design and the
`reconcile_tool_effects` resume logic this exists to support; this section
covers only the session-side storage.

Migration 7 adds one table, additive to everything above:

```
tool_effects(run_id, call_id, tool, status, idempotency_key,
             effect_summary, started_at, settled_at)
PRIMARY KEY (run_id, call_id)
```

`status` is one of `started` / `completed` / `failed` / `interrupted`. The
plain, synchronous operations mirror every other function in `run_ledger::ops`:

- `record_tool_started(workspace_dir, ToolEffectStart) -> Result<ToolEffectRow>`
  — upserts a `started` row, idempotent on `(run_id, call_id)`.
- `settle_tool_effect(workspace_dir, ToolEffectSettle) -> Result<Option<ToolEffectRow>>`
  — transitions to a terminal status; inserts a row even without a prior
  `started` write, so a ledger attached mid-run never wedges on an unseen
  call id.
- `list_unresolved_tool_effects(workspace_dir, run_id) -> Result<Vec<ToolEffectRow>>`
  — every row for `run_id` still `started`: the signature of a crash between
  admission and settlement.
- `mark_interrupted(workspace_dir, run_id, call_id) -> Result<Option<ToolEffectRow>>`
  — settles one row `interrupted`, the resume-time verdict for a call whose
  tool is not safe to blindly re-execute.

`RunLedgerToolEffects` (also in `run_ledger::tool_effects`) wraps these
behind the `async` `ToolEffectLedger` trait so a harness run can attach
durable tool-effect bookkeeping without either crate depending on the
other's concrete storage type:

```rust
use tinyagents_session::run_ledger::tool_effects::RunLedgerToolEffects;

let ledger = std::sync::Arc::new(RunLedgerToolEffects::new(workspace_dir));
let ctx = RunContext::new(config, ()).with_tool_effect_ledger(ledger);
```

Like every other bridge method in this crate, each trait method is a single
bounded synchronous SQLite statement — no `spawn_blocking`, matching the
tradeoff `store::with_connection`/`with_transaction` already make
throughout.

## Durable media parts

`TranscriptMessage::user_with_parts` preserves ordered text, image, audio, video
and document parts. Text remains in `content`; media lives in typed `parts` in
the JSONL row, including on replay. Existing image URL rows keep their wire shape.
Audio/video/document parts use `TranscriptMediaRef::Path { path }` or
`TranscriptMediaRef::Url { url }`, with a `mime_type`; there is deliberately no
inline-byte variant. Hosts store originals and resolve these references only on
ephemeral provider requests, applying their own workspace/fetch policy.

Legacy image markers continue to round-trip unchanged. `legacy_content` for new
media uses a `_tinyagents_media_parts` JSON envelope that `normalized` can lift
back into ordered parts. Older binaries cannot read these new media types.
Display projection shows compact `[AUDIO:<location>]`, `[VIDEO:<location>]` and
`[DOCUMENT:<location>]` markers without reading or encoding media bytes.
