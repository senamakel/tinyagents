//! Pure session semantics shared by the optional memory test provider and
//! production storage drivers.

use std::cmp::Ordering;

use crate::turn_state::{TurnLifecycle, TurnState};

pub(super) const MAX_GENERATIONS: u32 = 4096;
pub(super) const COMPLETED_RETENTION: usize = 20;

/// Newest first: by `started_at`, then `updated_at`.
pub(super) fn newest_first(a: &TurnState, b: &TurnState) -> Ordering {
    compare_rfc3339(&b.started_at, &a.started_at)
        .then_with(|| compare_rfc3339(&b.updated_at, &a.updated_at))
}

/// Completed-turn retention follows completion time; live-turn listing follows start time.
pub(super) fn completed_newest_first(a: &TurnState, b: &TurnState) -> Ordering {
    compare_rfc3339(&b.updated_at, &a.updated_at)
        .then_with(|| compare_rfc3339(&b.started_at, &a.started_at))
}

fn compare_rfc3339(left: &str, right: &str) -> Ordering {
    match (
        chrono::DateTime::parse_from_rfc3339(left),
        chrono::DateTime::parse_from_rfc3339(right),
    ) {
        (Ok(left), Ok(right)) => left.cmp(&right),
        _ => left.cmp(right),
    }
}

pub(super) fn is_terminal(state: &TurnState) -> bool {
    matches!(
        state.lifecycle,
        TurnLifecycle::Interrupted | TurnLifecycle::Completed
    )
}
