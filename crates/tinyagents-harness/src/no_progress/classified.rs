//! Failure equivalence and recovery budgets independent of literal call arguments.

use std::collections::HashMap;
use std::sync::Mutex;

use super::NoProgress;

/// A trusted classification of one failed operation. The caller supplies stable
/// operation and scope identifiers; neither error prose nor literal query text
/// belongs in this key.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ClassifiedFailure {
    pub class: String,
    pub operation: String,
    pub scope: String,
}

impl ClassifiedFailure {
    pub fn new(
        class: impl Into<String>,
        operation: impl Into<String>,
        scope: impl Into<String>,
    ) -> Self {
        Self {
            class: class.into(),
            operation: operation.into(),
            scope: scope.into(),
        }
    }
}

/// Counts equivalent failures across intervening calls. Only an observation
/// that the same blocker changed should call [`Self::clear`].
#[derive(Default)]
pub struct ClassifiedFailureTracker {
    counts: Mutex<HashMap<ClassifiedFailure, usize>>,
}

impl ClassifiedFailureTracker {
    /// `recovery_budget` is the number of further failed attempts permitted
    /// after the first failure. Zero stops on the first observation.
    pub fn record(&self, key: &ClassifiedFailure, recovery_budget: usize) -> NoProgress {
        let mut counts = self.counts.lock().unwrap();
        let attempts = counts.entry(key.clone()).or_default();
        *attempts += 1;
        if *attempts > recovery_budget {
            NoProgress::Halt(format!(
                "Stopping after {} attempt(s): failure class `{}` still blocks operation `{}` on `{}`. Resolve this blocker before retrying.",
                attempts, key.class, key.operation, key.scope
            ))
        } else {
            NoProgress::Continue
        }
    }

    /// Clear one blocker after an observation proves it changed or recovered.
    pub fn clear(&self, key: &ClassifiedFailure) {
        self.counts.lock().unwrap().remove(key);
    }

    /// Clear all groups when a new turn begins.
    pub fn reset(&self) {
        self.counts.lock().unwrap().clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn equivalent_failures_survive_intervening_calls_and_clear_by_scope() {
        let tracker = ClassifiedFailureTracker::default();
        let a = ClassifiedFailure::new("permission", "search", "account-a");
        let b = ClassifiedFailure::new("permission", "search", "account-b");
        assert_eq!(tracker.record(&a, 1), NoProgress::Continue);
        assert_eq!(tracker.record(&b, 1), NoProgress::Continue);
        assert!(
            matches!(tracker.record(&a, 1), NoProgress::Halt(message) if message.contains("2 attempt(s)"))
        );
        tracker.clear(&a);
        assert_eq!(tracker.record(&a, 1), NoProgress::Continue);
        assert!(matches!(tracker.record(&b, 1), NoProgress::Halt(_)));
    }

    #[test]
    fn zero_budget_stops_immediately() {
        let tracker = ClassifiedFailureTracker::default();
        let key = ClassifiedFailure::new("policy", "write", "project");
        assert!(matches!(tracker.record(&key, 0), NoProgress::Halt(_)));
    }
}
