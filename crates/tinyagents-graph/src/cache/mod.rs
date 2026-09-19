//! Task-level result caching for graph nodes.
//!
//! A [`TaskCache`] stores a node's serialized `Update` under a
//! [`TaskCacheKey`] (graph, node, and a caller-computed hash of the inputs)
//! so a later activation with the same key can skip the handler entirely.
//! Caching is opt-in per node through
//! [`NodeCachePolicy`](crate::NodeCachePolicy) and wired into a graph with
//! [`CompiledGraph::with_task_cache`](crate::CompiledGraph::with_task_cache)
//! plus [`CompiledGraph::with_cached_node`](crate::CompiledGraph::with_cached_node).
//!
//! Two backends ship here: [`InMemoryTaskCache`] (process-local, TTL-aware)
//! and, behind the `sqlite` feature, `SqliteTaskCache`.

mod memory;
mod types;

pub use memory::InMemoryTaskCache;
pub use types::{TaskCache, TaskCacheKey};

#[cfg(test)]
mod test;
