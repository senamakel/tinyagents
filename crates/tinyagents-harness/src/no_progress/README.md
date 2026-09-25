# harness::no_progress

Foundational no-progress detector: recognises when an agent turn is stuck
re-issuing the same failing (or successful) tool call and hands back a
structured verdict a driver can turn into a corrective nudge or a halt.

## Why this exists

A model that hits an unproductive tool result tends to retry the *identical*
strategy — same tool, same arguments — instead of adapting. Left unchecked,
that only stops when a coarse limit like `RunLimits::max_tool_calls` trips,
burning dozens of wasted round trips first. This module is the reusable,
harness-type-free detector that breaks the pattern early: it tracks recent
`(tool, args) → outcome` across a turn and, on each failure, escalates through
an ordered ladder — keep going, **nudge** the model to change approach, or
**halt** once same-strategy retries are exhausted.

A sibling tracker, [`SuccessfulRepeatTracker`], covers the complementary shape
— a model that keeps *succeeding* at the same no-op call (or cycles through a
short repeating sequence) without making progress.

All trackers are free of harness types (no `RunContext`, no `Message`) so they
can be unit-tested in isolation. The agent loop drives
`StreamTextStallDetector` on visible streaming output and ends the model call
with non-retryable `GenerationStalled` when it fires. Tool-call trackers remain
host-wired through `after_tool`; see the "Driving this from an `after_tool`
hook" section in `mod.rs` for that contract.

## Public surface

- [`NoProgressTracker`] — holds the identical-failure and any-failure ladder
  state for one turn. `new(identical_halt_threshold)` builds it,
  `record(step, &ToolAttempt) -> NoProgress` feeds one outcome and returns the
  verdict, `reset()` clears all counters (called internally after a halt).
- [`ClassifiedFailureTracker`] — an additive ledger for equivalent failures
  keyed by class, operation, and resource or permission scope. `record` accepts
  a class-specific recovery budget; `clear` removes one group only after an
  observation shows its blocker changed. Intervening tool calls leave it intact.
  Hold one tracker per turn, or call `reset()` at a new turn boundary. A
  `NoProgress::Halt` verdict does not reset classified counts: retrying the same
  unchanged blocker in a resumed turn would halt again.
- [`ToolAttempt`] — one observed outcome, built with `success`/`failure` plus
  the `hard_reject()`/`recoverable_miss()` modifiers.
- [`NoProgress`] — the verdict enum: `Continue`, `Nudge(String)`,
  `Halt(String)`, with `message()`/`is_nudge()`/`is_halt()`/`as_str()` helpers.
- [`fingerprint_arguments`] — the canonical (key-order-independent) argument
  hash every driver must use so the identical-repeat rung compares correctly.
- [`SuccessfulRepeatTracker`] / [`SuccessfulRepeat`] — the successful-repeat
  counterpart: `record_output`, `record_call_batch`, `record_call_outcome`,
  and `reset`.
- [`StreamTextStallDetector`] — consumes visible text fragments during one
  model call and flags a long run of similarly opened sentences before the
  provider stream finishes.
- Threshold constants: [`DEFAULT_IDENTICAL_HALT_THRESHOLD`],
  [`DEFAULT_REPEAT_OUTPUT_THRESHOLD`], [`DEFAULT_REPEAT_CALL_THRESHOLD`] (all
  re-exported from `crate`).

## Files

| File | Role |
| --- | --- |
| `mod.rs` | The identical/any-failure escalation ladder (`NoProgressTracker::record`), argument fingerprinting, and the nudge/halt message builders. |
| `successful_repeat.rs` | The successful-repeat streak tracker (`SuccessfulRepeatTracker`) and its private `Streak` helper. |
| `stream_text/` | Chunk-independent streamed-text stall detector and focused tests. |
| `types.rs` | Public and crate-private type definitions shared by both trackers. |
| `test.rs` | Unit tests for the escalation ladder. |

## Operational constraints

- `identical_halt_threshold` passed to `NoProgressTracker::new` is clamped so
  it always sits strictly above the nudge threshold — a driver cannot
  accidentally configure a halt that fires before any nudge is given.
- A hard policy rejection (`ToolAttempt::hard_reject`) trips the ladder
  fastest (`HARD_REJECT_HALT_THRESHOLD = 2`), since a blocked call re-issued
  unchanged can never succeed.
- The unknown-tool recovery sentinel (`ToolAttempt::recoverable_miss`) feeds
  the identical-repeat counter but **not** the any-failure backstop, so a
  model that recovers from one bad tool name and then legitimately exhausts
  its budget does not trip the generic backstop early.
- On `NoProgress::Halt`, the tracker resets its own state, so a resumed run
  does not immediately re-trip on latched counters.
