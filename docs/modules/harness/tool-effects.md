# Tool-Effect Ledger And Replay (B5)

See [tool.md](tool.md) for the tool feature this extends. This covers the
durable bookkeeping the agent loop keeps around a tool call's side effects, and
the resume-time reconciliation that decides whether an interrupted call may be
safely re-run.

## Motivation

A tool call that mutates external state (sends an email, charges a card,
writes a file) is dangerous to blindly re-run after a process restart. Without
a durable record, the harness cannot tell "never started" from "started, effect
landed, process died before the result was folded into the transcript" from
"started, effect landed, crash, and now we are about to run it again". Pi and
pydantic-ai both solve this with a per-call effect record plus a per-tool
replay classification (`docs/runtime-comparison/pi.md` §4.7,
`pydantic-ai.md` §3.11); this module is TinyAgents' version of that pattern.

## Vocabulary (`src/tool/effects.rs`)

```rust
pub enum ToolEffectStatus { Started, Completed, Failed, Interrupted }

pub struct ToolEffectStart {
    pub run_id: RunId,
    pub call_id: CallId,
    pub tool: String,
    pub idempotency_key: String,
    pub effect_summary: Option<String>,
}

pub struct ToolEffectSettle {
    pub run_id: RunId,
    pub call_id: CallId,
    pub status: ToolEffectStatus,
    pub effect_summary: Option<String>,
}

pub struct ToolEffect { /* durable row, as read back */ }

#[async_trait]
pub trait ToolEffectLedger: Send + Sync {
    async fn started(&self, start: ToolEffectStart) -> Result<()>;
    async fn settled(&self, settle: ToolEffectSettle) -> Result<()>;
    async fn unresolved(&self, run_id: &str) -> Result<Vec<ToolEffect>>;
}

pub enum LedgerFailure { Abort, Continue }
```

`tinyagents-harness` owns only this vocabulary. The durable, SQLite-backed
implementation lives in `tinyagents-session::run_ledger::tool_effects`
(`RunLedgerToolEffects`) — see `docs/modules/session/README.md`. A host may
supply its own `ToolEffectLedger` for a different durability substrate.

## Attaching A Ledger

`RunContext` carries an optional ledger slot:

```rust
let ledger: Arc<dyn ToolEffectLedger> =
    Arc::new(RunLedgerToolEffects::new(workspace_dir));
let ctx = RunContext::new(config, ())
    .with_tool_effect_ledger(ledger)
    .with_tool_effect_ledger_failure(LedgerFailure::Abort); // default
```

`None` (the default) means no ledger writes happen at all — the loop behaves
exactly as it did before this existed. A child context inherits its parent's
ledger and failure mode, matching how `stores`/`events` propagate.

## What The Loop Writes

For every admitted tool call, `agent_loop::tools`:

1. Writes a `started` row (idempotency key: the tool's own key when it
   declares one via `ToolPolicy`, otherwise a SHA-256 hash of `(tool name,
   arguments)`) **before** the call executes.
2. Writes a `completed` row after the call returns a result (successful or a
   recoverable `ToolResult::error` — both count as "the effect ran and we know
   the outcome"), or a `failed` row if the call's execution future itself
   errored.

This holds in both the serial and the concurrent execution paths, and for the
concurrent path's sibling-abort handling (every already-started sibling gets a
`failed` settle alongside its `ToolFailed` event).

### Ledger-Write Failure

`started` failing is a decision point, governed by `RunContext::tool_effect_ledger_failure`:

- **`LedgerFailure::Abort`** (default): the tool call fails and the error
  propagates, exactly like any other admission failure. A ledger that cannot
  be trusted to record "started" cannot be trusted to detect "interrupted"
  either, so failing closed is the safer default for a tool with real
  effects.
- **`LedgerFailure::Continue`**: the failure is logged and the call proceeds
  unrecorded — appropriate for a host that would rather keep a run alive
  through a transient ledger outage than block on it.

A `settled` write failure is **always** best-effort and non-fatal: by the time
it runs the tool has already executed (or its execution future has already
failed), so aborting over a settle failure would discard a real result rather
than merely skip recording one. The row simply stays `started` and surfaces
again on the next `unresolved` read.

## Resuming After A Crash: `reconcile_tool_effects`

```rust
pub async fn reconcile_tool_effects(
    &self,
    ctx: &RunContext<Ctx>,
    run_id: &str,
    messages: &mut Vec<Message>,
) -> Result<Vec<Message>>
```

Called by a host that is resuming a run from durably-persisted `messages`
after a crash, before re-entering the agent loop. For every tool call on the
*last* assistant turn that has no `Message::Tool` answer yet **and** an
unresolved (`started`) ledger row, it consults the tool's declared
`tinytools::ToolPolicy::runtime.replay`:

- **`ToolReplay::Safe`**: the call is left unanswered. `messages` is not
  touched for that call, so the ordinary loop re-executes it exactly as it
  would a fresh call — the tool declared this safe.
- **`ToolReplay::Never`** (the default, and what a tool no longer registered
  on this harness is treated as — fail closed rather than blindly re-run an
  unknown effect): a synthesized tool-error result
  (`"interrupted before settlement"`) is appended in place of a real answer,
  the ledger row is settled `Interrupted`, and the loop never re-attempts the
  call.

Either branch emits `AgentEvent::ToolEffectReconciled { call_id, action }`
(`action` is `"re_execute"` or `"interrupted"`) so the decision is observable.

Returns the messages it synthesized (already appended to `messages` too), so a
caller that journals messages separately from the in-memory transcript knows
what changed. Returns an empty `Vec` immediately, without any ledger I/O, when
`ctx` has no ledger attached or the transcript has no pending tool calls.

### Not Auto-Wired On This Branch

There is no `resume_deferred`/deferred-results entry point on this branch for
`reconcile_tool_effects` to hook into — none exists yet. A host resuming a run
calls it explicitly:

```rust
let mut messages = /* durably-persisted transcript */;
harness.reconcile_tool_effects(&ctx, run_id, &mut messages).await?;
harness.invoke_in_context(&state, ctx, messages).await?;
```

When a deferred-results/resume entry point is added to this harness, it should
call `reconcile_tool_effects` immediately before re-entering the loop.
