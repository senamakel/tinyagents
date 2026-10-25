//! Post-compaction loop guard.
//!
//! Context compaction can erase the evidence that a run was looping: the model
//! wakes up with a summary, repeats the call it was stuck on, gets the same
//! result and is back in the loop, with the repeat ledger cleared. This guard
//! remembers the last few `(call, result)` pairs that were *already repeating*
//! (recurred at least [`REPEATING_AT`] times) right before a compaction. For a
//! short window of calls after it, a call that repeats one of those pairs is
//! flagged so the host can warn the model.
//!
//! It is warning-only. Re-reading content a compaction evicted is correct
//! behaviour, so a single repeat is never blocked, and a pair that was not
//! repeating before the compaction is not remembered at all.

use std::sync::Mutex;

use super::types::{GuardState, PostCompactionGuard};
use super::util::{hash_pair, lock};

/// Recurrences of a `(call, result)` pair before a compaction that make it
/// part of the tail the guard remembers.
pub const REPEATING_AT: u32 = 2;
/// Tool calls watched after a compaction, and recent calls remembered before it.
pub const DEFAULT_POST_COMPACTION_WINDOW: u32 = 3;

impl Default for PostCompactionGuard {
    fn default() -> Self {
        Self::new(DEFAULT_POST_COMPACTION_WINDOW)
    }
}

impl PostCompactionGuard {
    /// `window` is both how many calls before a compaction are remembered and
    /// how many calls after it are watched. `0` disables the guard.
    pub fn new(window: u32) -> Self {
        Self {
            window,
            state: Mutex::default(),
        }
    }

    /// Records one successful call; `repeating` says whether the pair had
    /// already recurred `REPEATING_AT` times. Before a compaction only
    /// repeating pairs are remembered. Returns `true` when the guard is armed
    /// and this call repeats a remembered pair (and disarms, so one compaction
    /// flags at most one repeat).
    pub fn record(&self, call_signature: &str, outcome_identity: &str, repeating: bool) -> bool {
        if self.window == 0 {
            return false;
        }
        let pair = hash_pair(call_signature, outcome_identity);
        let mut state = lock(&self.state);
        if let Some((tail, remaining)) = state.armed.as_mut() {
            let repeated = tail.contains(&pair);
            *remaining -= 1;
            if repeated || *remaining == 0 {
                state.armed = None;
            }
            return repeated;
        }
        // Every successful call ages the tail, so a pair that stopped
        // repeating long ago cannot be flagged after a later compaction.
        state.calls += 1;
        let now = state.calls;
        if repeating {
            state.tail.push_back((pair, now));
        }
        let window = u64::from(self.window);
        while state
            .tail
            .front()
            .is_some_and(|(_, seen)| now - seen >= window)
        {
            state.tail.pop_front();
        }
        false
    }

    /// A compaction just removed results from context: watch the next
    /// `window` calls against the tail recorded so far. With nothing recorded
    /// since the previous compaction, an already armed guard is left as is.
    pub fn arm(&self) {
        if self.window == 0 {
            return;
        }
        let mut state = lock(&self.state);
        if state.tail.is_empty() {
            return;
        }
        let tail = state.tail.drain(..).map(|(pair, _)| pair).collect();
        state.armed = Some((tail, self.window));
    }

    /// Forgets the tail and disarms.
    pub fn reset(&self) {
        *lock(&self.state) = GuardState::default();
    }
}

#[cfg(test)]
#[path = "post_compaction_tests.rs"]
mod tests;
