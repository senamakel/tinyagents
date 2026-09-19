# harness::run_queue

A generic, thread-safe, multi-lane FIFO queue for messages that arrive while a
run is active.

## Why this exists

Hosts decide *which* incoming events should be queued rather than applied
immediately (a steering instruction, a deferred follow-up, extra context to
fold in later) and retain ownership of the queued payload. This module owns
only the reusable FIFO mechanics for the three lanes an agent runtime can
drain at safe iteration boundaries — it has no opinion on what `T` is or when
a lane should be drained.

## Public surface

- [`RunQueue<T>`] — the queue itself. `new`/`Default` create an empty queue;
  `push(lane, item)` appends; `drain(lane)` empties one lane in FIFO order and
  returns its contents; `status()` snapshots per-lane depth; `clear()` empties
  every lane and returns how many items were dropped.
- [`QueueLane`] — which lane an item belongs to: `Steer` (inject at the next
  safe boundary as an instruction), `Followup` (dispatch as a fresh turn once
  the active run completes), `Collect` (inject at the next safe boundary as
  additional context).
- [`QueueStatus`] — a `Serialize`-able snapshot of per-lane and total pending
  counts.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `RunQueue<T>` and its private `RunQueueInner<T>` storage. |
| `types.rs` | `QueueLane`, `QueueStatus`. |
| `test.rs` | Per-lane push/drain ordering, status snapshots, `clear`, and lane independence. |

## Operational constraints

- All mutating/reading methods are `async` and serialize through a single
  `tokio::sync::Mutex` guarding all three lanes together, so a caller draining
  one lane briefly blocks a concurrent push to another — acceptable given the
  queue is meant to hold a handful of pending items, not a high-throughput
  channel.
- `drain` removes and returns items in FIFO order; it does not re-queue on
  failure — a caller that fails to act on a drained item is responsible for
  re-pushing it if that is the desired behavior.
- The queue has no size limit; unbounded growth is a host responsibility to
  guard against if untrusted input can push into a lane.
