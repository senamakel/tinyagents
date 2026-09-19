//! Per-node execution policy: retry, timeouts, caching, error recovery, and
//! deferred scheduling.
//!
//! A [`NodePolicy`] is attached to one node via
//! [`GraphBuilder::with_node_policy`](super::GraphBuilder::with_node_policy)
//! or to every node via
//! [`GraphBuilder::set_node_defaults`](super::GraphBuilder::set_node_defaults).
//! At run time the executor resolves an *effective* policy for each
//! activation field by field ([`NodePolicy::resolve`]): a per-node `Some`
//! wins, else the graph-wide default's field, else the older graph-wide
//! `with_node_retry` / `with_node_timeout` settings, else nothing.

use std::sync::Arc;
use std::time::Duration;

use crate::command::Command;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::retry::RetryPolicy;

/// Computes a task-cache key from the committed state snapshot and the
/// activation's [`crate::NodeContext::send_arg`] (if any).
pub type CacheKeyFn<State> =
    dyn Fn(&State, Option<&serde_json::Value>) -> String + Send + Sync;

/// Recovers from a node failure: given the state the node ran against and
/// the error that survived its retry policy, optionally produce a
/// [`Command`] that stands in for the node's result.
pub type OnErrorFn<State, Update> =
    dyn Fn(&State, &TinyAgentsError) -> Option<Command<Update>> + Send + Sync;

/// Opt-in result caching for one node.
///
/// `key` derives the cache key from the state snapshot and `send_arg`; an
/// identical key on a later activation replays the cached `Update` without
/// invoking the handler. `ttl` bounds how long an entry stays valid
/// (`None` = forever, until [`crate::cache::TaskCache::clear`]).
///
/// The policy itself is bound-free over `Update`. Actually (de)serializing
/// a cached update needs `Update: Serialize + DeserializeOwned`, which is
/// only required by
/// [`CompiledGraph::with_cached_node`](crate::CompiledGraph::with_cached_node)
/// (the entry point that installs the codec) — see its docs.
pub struct NodeCachePolicy<State> {
    /// Derives the cache key for one activation.
    pub key: Arc<CacheKeyFn<State>>,
    /// Optional time-to-live for a cached entry.
    pub ttl: Option<Duration>,
}

impl<State> NodeCachePolicy<State> {
    /// Builds a cache policy from a key function, with no TTL.
    pub fn new<F>(key: F) -> Self
    where
        F: Fn(&State, Option<&serde_json::Value>) -> String + Send + Sync + 'static,
    {
        Self {
            key: Arc::new(key),
            ttl: None,
        }
    }

    /// Sets the entry time-to-live.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

impl<State> Clone for NodeCachePolicy<State> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            ttl: self.ttl,
        }
    }
}

impl<State> std::fmt::Debug for NodeCachePolicy<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeCachePolicy")
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

/// Execution policy for one node (or, via `set_node_defaults`, every node).
///
/// Every field is optional and additive; the [`Default`] changes nothing.
/// See the module docs for how per-node and graph-wide policies compose.
pub struct NodePolicy<State, Update> {
    /// Retry policy applied around the handler; a
    /// [retryable](tinyagents_harness::retry::is_retryable) error is re-run
    /// from the node's start up to the policy's attempt cap.
    pub retry: Option<RetryPolicy>,
    /// Maximum total wall-clock time for one attempt of the handler,
    /// regardless of heartbeats.
    pub timeout: Option<Duration>,
    /// Maximum time between two [`crate::NodeContext::heartbeat`] calls (or
    /// between start and the first one). A handler that never heartbeats
    /// sees this as a flat timeout of the same duration.
    pub idle_timeout: Option<Duration>,
    /// Opt-in result caching; see [`NodeCachePolicy`].
    pub cache: Option<NodeCachePolicy<State>>,
    /// Error recovery hook consulted once the retry budget is exhausted (or
    /// the error was not retryable). Returning `Some(command)` makes the
    /// node complete with that command instead of failing the run.
    pub on_error: Option<Arc<OnErrorFn<State, Update>>>,
    /// Deferred scheduling: the node only runs once nothing *else* is left
    /// in the frontier (a "run when nothing else is ready" synthesis join).
    pub defer: bool,
}

impl<State, Update> Default for NodePolicy<State, Update> {
    fn default() -> Self {
        Self {
            retry: None,
            timeout: None,
            idle_timeout: None,
            cache: None,
            on_error: None,
            defer: false,
        }
    }
}

impl<State, Update> Clone for NodePolicy<State, Update> {
    fn clone(&self) -> Self {
        Self {
            retry: self.retry.clone(),
            timeout: self.timeout,
            idle_timeout: self.idle_timeout,
            cache: self.cache.clone(),
            on_error: self.on_error.clone(),
            defer: self.defer,
        }
    }
}

impl<State, Update> std::fmt::Debug for NodePolicy<State, Update> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodePolicy")
            .field("retry", &self.retry)
            .field("timeout", &self.timeout)
            .field("idle_timeout", &self.idle_timeout)
            .field("cache", &self.cache)
            .field("has_on_error", &self.on_error.is_some())
            .field("defer", &self.defer)
            .finish()
    }
}

impl<State, Update> NodePolicy<State, Update> {
    /// Sets the retry policy.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Sets the flat per-attempt timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the idle (heartbeat) timeout.
    pub fn with_idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.idle_timeout = Some(idle_timeout);
        self
    }

    /// Sets the cache policy.
    pub fn with_cache(mut self, cache: NodeCachePolicy<State>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Sets the error-recovery hook.
    pub fn with_on_error<F>(mut self, on_error: F) -> Self
    where
        F: Fn(&State, &TinyAgentsError) -> Option<Command<Update>> + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(on_error));
        self
    }

    /// Marks the node deferred.
    pub fn deferred(mut self) -> Self {
        self.defer = true;
        self
    }

    /// Resolves the effective policy for one node, field by field.
    ///
    /// Precedence per field: `per_node` (`Some` wins) → `defaults` →
    /// the legacy graph-wide `retry`/`timeout` (from
    /// `CompiledGraph::with_node_retry` / `GraphBuilder::with_node_timeout`).
    /// `defer` is the logical OR of the two policies' flags.
    pub(crate) fn resolve(
        per_node: Option<&Self>,
        defaults: Option<&Self>,
        legacy_retry: Option<&RetryPolicy>,
        legacy_timeout: Option<Duration>,
    ) -> Self {
        let pick = |f: fn(&Self) -> bool| -> Option<&Self> {
            per_node.filter(|p| f(p)).or(defaults.filter(|p| f(p)))
        };
        Self {
            retry: pick(|p| p.retry.is_some())
                .and_then(|p| p.retry.clone())
                .or_else(|| legacy_retry.cloned()),
            timeout: pick(|p| p.timeout.is_some())
                .and_then(|p| p.timeout)
                .or(legacy_timeout),
            idle_timeout: pick(|p| p.idle_timeout.is_some()).and_then(|p| p.idle_timeout),
            cache: pick(|p| p.cache.is_some()).and_then(|p| p.cache.clone()),
            on_error: pick(|p| p.on_error.is_some()).and_then(|p| p.on_error.clone()),
            defer: per_node.is_some_and(|p| p.defer) || defaults.is_some_and(|p| p.defer),
        }
    }
}
