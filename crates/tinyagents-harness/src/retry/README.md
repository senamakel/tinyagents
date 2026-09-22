# harness::retry

Retry, fallback, and rate-limiting policy implementations for the harness's
model calls.

## Why this exists

Every level of the recursive harness — a top-level agent, a nested sub-agent,
a graph node calling out to a model — ultimately reduces to the same kind of
call: a request to a model provider. This module supplies the durability
policies applied uniformly at that one call site, so a flaky provider does not
collapse a deep recursion and a rate-limited provider does not burn every
retry attempt back-to-back.

Three independent policies live here:

- [`RetryPolicy`] — exponential backoff with optional jitter, a per-call
  attempt cap, and server-`Retry-After` awareness.
- [`FallbackPolicy`] — an ordered list of model names to try in sequence.
- [`RateLimiter`] — a token-bucket limiter for pacing provider calls.

A free function, [`is_retryable`], classifies a [`crate::TinyAgentsError`] so
callers can decide whether to retry or propagate immediately; provider error
taxonomy and `Retry-After` parsing are consumed directly from
`tinyinference_llm::failure` rather than reimplemented here.

## Public surface

- [`RetryPolicy`] — builder-style config (`with_max_attempts`,
  `with_initial_backoff_ms`, `with_max_backoff_ms`, `with_multiplier`,
  `with_jitter`, `with_backoff_sleep`, `with_max_retry_after_ms`,
  `with_retry_on`/`with_default_retry_on`) plus the decision/backoff API:
  `should_retry`, `should_retry_error`, `is_retryable_error`,
  `backoff_for_attempt`/`backoff_for_attempt_with`, `backoff_for_error`,
  `sleep_backoff`/`sleep_backoff_for_error`, and
  `max_attempts_capped_at` (reconciling with
  [`crate::limits::RunLimits::max_retries_per_call`]).
- [`RetryPredicate`] — the `Arc<dyn Fn(&TinyAgentsError) -> bool + Send +
  Sync>` type accepted by `RetryPolicy::with_retry_on`.
- [`is_retryable`] — the crate-wide default error classification (see its doc
  table for the per-variant heuristic).
- [`retry_after_hint`] — extracts a server-supplied `Retry-After` delay from an
  error, preferring the structured `ProviderError` field and falling back to
  message-text parsing.
- [`FallbackPolicy`] — `new`, `next_after` (advance to the next model on
  failure).
- [`RateLimiter`] — `new(capacity, refill_per_sec)`, `try_acquire(tokens,
  now)`, `available(now)`, `capacity`, `refill_per_sec`, `can_ever_acquire`.
- [`JITTER_FRACTION`] — the ± band jitter spreads backoff over.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Method implementations for all three policies, `is_retryable`, `retry_after_hint`. |
| `types.rs` | The declarative structs/type aliases (`RetryPolicy`, `FallbackPolicy`, `RateLimiter`, `RetryPredicate`) plus hand-written `Debug`/`PartialEq` where a boxed closure blocks the derive. |
| `jitter.rs` | A minimal, dependency-free `xorshift64*` RNG used only for backoff jitter spread — never security-relevant, and always bypassable in tests via an explicit `rand01`. |
| `test.rs` | Backoff growth/capping, jitter scaling/clamping, `should_retry` boundaries, `is_retryable` classification, `FallbackPolicy` traversal, and token-bucket behavior. |

## Operational constraints

- **Deterministically testable by construction**: `RateLimiter` methods take
  an explicit `now: Instant`, and `RetryPolicy::backoff_for_attempt_with`
  takes an explicit `rand01: f64`, so neither policy needs a clock/RNG
  injection seam — tests just pass fixed values.
- `RetryPolicy::backoff_sleep` defaults to `true`. Tests and other
  latency-sensitive callers must opt out explicitly with
  `with_backoff_sleep(false)`; leaving it on and calling a retry loop in a
  test will really sleep.
- Jitter is **additive**, never multiplicative — it can only widen the delay
  band around the base backoff, never collapse it toward zero. See
  `JITTER_FRACTION` and `RetryPolicy::backoff_for_attempt_with` for the
  historical bug this guards against.
- A server-supplied `Retry-After` only ever *lengthens* the wait
  (`RetryPolicy::backoff_for_error` takes the `max` of computed backoff and the
  hint), and is clamped by `max_retry_after_ms` so a hostile or buggy header
  cannot park a run indefinitely.
- Callers holding a `RetryPolicy` should always go through
  `RetryPolicy::is_retryable_error` / `should_retry_error` rather than calling
  the free `is_retryable` directly, so a caller-supplied `retry_on` predicate
  is honored.
