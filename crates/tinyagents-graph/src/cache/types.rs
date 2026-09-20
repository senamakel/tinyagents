//! Task cache trait and key type.

use std::time::Duration;

use async_trait::async_trait;

use crate::Result;
use tinyagents_harness::ids::{GraphId, NodeId};

/// Identifies one cached task result: which graph and node produced it,
/// and the caller-computed `hash` of the inputs it was computed from
/// (see [`crate::NodeCachePolicy::key`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TaskCacheKey {
    /// The graph the node belongs to.
    pub graph_id: GraphId,
    /// The node whose result is cached.
    pub node_id: NodeId,
    /// The input hash the cache policy's key function produced.
    pub hash: String,
}

impl TaskCacheKey {
    /// Builds a key from its three parts.
    pub fn new(graph_id: GraphId, node_id: NodeId, hash: impl Into<String>) -> Self {
        Self {
            graph_id,
            node_id,
            hash: hash.into(),
        }
    }
}

/// A store for cached node results, keyed by [`TaskCacheKey`].
///
/// Implementations must honor `ttl` on [`Self::put`]: an entry older than
/// its TTL is reported as a miss by [`Self::get`]. The executor treats every
/// cache error as a miss (reads) or logs and continues (writes) — caching
/// is an optimization, never a correctness requirement.
#[async_trait]
pub trait TaskCache: Send + Sync {
    /// Looks up a live (non-expired) entry.
    async fn get(&self, key: &TaskCacheKey) -> Result<Option<serde_json::Value>>;
    /// Stores `value` under `key`, expiring after `ttl` when given.
    async fn put(
        &self,
        key: &TaskCacheKey,
        value: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<()>;
    /// Drops every entry belonging to `graph_id`.
    async fn clear(&self, graph_id: &GraphId) -> Result<()>;
}

/// Encodes an `Update` for cache storage.
type EncodeFn<Update> = dyn Fn(&Update) -> serde_json::Result<serde_json::Value> + Send + Sync;
/// Decodes a stored value back into an `Update`.
type DecodeFn<Update> = dyn Fn(serde_json::Value) -> serde_json::Result<Update> + Send + Sync;

/// One node's resolved cache policy plus the type-erased `Update` codec,
/// installed by [`crate::CompiledGraph::with_cached_node`].
///
/// [`crate::NodeCachePolicy`] itself is bound-free over `Update` (it only
/// ever touches `State`); actually storing a cached value needs
/// `Update: Serialize + DeserializeOwned`, which `with_cached_node` supplies
/// locally when it builds `encode`/`decode` — this struct carries the
/// resulting closures rather than the bound itself, so [`CompiledGraph`]
/// stays generic over any `Update`.
///
/// [`CompiledGraph`]: crate::CompiledGraph
pub(crate) struct CachedNode<State, Update> {
    /// Derives the cache key for one activation (state + optional send arg).
    pub(crate) key: std::sync::Arc<crate::builder::CacheKeyFn<State>>,
    /// Optional time-to-live for a cached entry.
    pub(crate) ttl: Option<Duration>,
    /// Encodes an `Update` for storage.
    pub(crate) encode: std::sync::Arc<EncodeFn<Update>>,
    /// Decodes a stored value back into an `Update`.
    pub(crate) decode: std::sync::Arc<DecodeFn<Update>>,
}

impl<State, Update> Clone for CachedNode<State, Update> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            ttl: self.ttl,
            encode: self.encode.clone(),
            decode: self.decode.clone(),
        }
    }
}
