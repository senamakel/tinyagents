# harness::steering

Policy-checked, observable orchestrator → sub-agent steering: how a *parent*
in the run tree exerts typed control over a *child* it is currently running,
without killing or restarting it.

This is the mid-run counterpart to `tinyagents_orchestration::subagent::SubAgentSession` reuse
(which resumes a *completed* child) — together they cover both ways an
orchestrator keeps a sub-agent "in play". An orchestrating agent, a human UI,
a graph supervisor, or a test harness attaches a `SteeringHandle` to a run's
`RunContext` and enqueues `SteeringCommand`s on it; the agent loop drains and
applies them at a safe checkpoint (before each model call), never mid-stream
or mid-tool-call.

## Public surface

- `SteeringCommand` — the typed instruction sent to a running loop: `Pause`,
  `PauseWith { reason }`, `Resume`, `Cancel`, `InjectMessage(Message)`,
  `Redirect { instruction }`, `SetMetadata { metadata }`.
  `Serialize`/`Deserialize` so commands can be logged, transported, and
  replayed. `.kind()` returns the payload-free `SteeringCommandKind`.
- `SteeringCommandKind` — the policy-relevant discriminant of a command;
  `ALL` lists every kind, `as_str()` gives a stable lower-snake-case label for
  logging/events.
- `SteeringPolicy` — an allowlist of permitted `SteeringCommandKind`s.
  `SteeringPolicy::new()` permits nothing (fail-closed default);
  `allow_all()` / `.allow(kind)` grant kinds explicitly.
- `SteeringHandle` — a cloneable, `Arc`-backed handle shared by the sender
  (orchestrator) and receiver (agent loop). `send` enqueues; `drain` empties
  the FIFO queue; `pending`/`is_empty` inspect it; `pause_state`/`is_paused`/
  `resume` read and clear the latched pause independent of the queue.
- `PauseState` — the latched state behind a `SteeringOutcome::Pause`: an
  optional human-readable `reason` and the zero-based `paused_at_checkpoint`
  index.
- `SteeringOutcome` — the control-flow decision from one checkpoint:
  `Continue`, `Pause` (a pause is latched — see the docs on this variant for
  the loop's required follow-up), `Cancel`. `.is_pause()` is a convenience
  check.
- `apply_pending_steering(ctx, messages)` — the single steering checkpoint:
  drains `ctx`'s handle (if any), validates the whole batch against the run's
  policy before applying anything, applies permitted commands to `messages`
  and `ctx.config`, and returns the resulting `SteeringOutcome`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: `SteeringCommand`, `SteeringCommandKind`, `SteeringPolicy`, `PauseState`, `SteeringOutcome`, `SteeringHandle` (and its private `SteeringInner`). |
| `mod.rs` | Behavioral code: `SteeringPolicy`/`SteeringHandle` methods and the `apply_pending_steering` checkpoint function. |
| `test.rs` | Unit tests against `apply_pending_steering` directly, plus integration-style tests driving a full `AgentHarness` run with a `SteeringHandle` attached and asserting both transcript outcome and `AgentEvent::Steered` events. |

## Key invariants

- **Fail-closed by default.** A fresh `SteeringPolicy` permits nothing; a run
  that wants steering must opt in per command kind.
- **Batch validation is atomic.** `apply_pending_steering` checks every
  drained command against the policy *before* applying any of them. A
  disallowed command anywhere in the batch aborts the whole checkpoint with
  `TinyAgentsError::Steering` and leaves the transcript/metadata untouched —
  it does not partially apply commands `0..n` and drop the rest.
- **`Cancel` takes precedence** over every other command in the same batch: it
  is applied and the function returns immediately, ignoring anything queued
  after it.
- **A pause is latched on the `SteeringHandle`, not scoped to a batch.** Once
  latched it survives across checkpoints until a `Resume` arrives — in the
  same batch or any later one. A caller distinguishes "paused" from "model
  produced an empty answer" via `SteeringHandle::pause_state`/`is_paused`
  (`AgentRun::paused` on the loop side, see `crate::middleware::AgentRun`).
- **Delivery is pull-based and checkpoint-scoped.** Commands enqueued via
  `SteeringHandle::send` become visible only at the next checkpoint (before a
  model call) — never mid-stream or mid-tool-call — so steering cannot
  interrupt a side-effecting operation partway through.

## Relation to neighbouring modules

- `crate::context::RunContext::with_steering` attaches a `SteeringHandle` to a
  run; the agent loop calls `apply_pending_steering` at its checkpoint.
- Applied commands emit `crate::events::AgentEvent::Steered` through the
  `RunContext`'s event sink, so steering activity is observable the same way
  as middleware and retry events.
- `SteeringCommand::InjectMessage`/`Redirect` operate on the same
  `tinyinference_llm::message::Message` transcript that `crate::prompt`
  assembles into a `ModelRequest`.
