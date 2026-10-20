//! The `tsd.kv` body and id, for stores that need more than `Kv` offers
//! (expiry, creation times, range queries).
//!
//! `tinyagents_storage_kit::Kv` stores `{owner, ns, key, value}` under
//! `Key::of([owner, ns, key])` and cannot set an expiry. The namespaced store
//! and the response cache need one, so they write the same documented shape
//! through [`kv_body`] / [`kv_id`] and read with `Kv`-compatible bodies.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tinyagents_storage_kit::{Key, StorageError};

/// The `tsd.kv` collection name.
pub(crate) const KV: &str = tinyagents_storage_kit::primitives::KV;

/// `tsd.kv` namespace prefix of [`crate::store::namespaced::DriverNamespacedStore`]
/// entries (`namespaced:<segments joined with '.'>`).
pub(crate) const NAMESPACED_NS_PREFIX: &str = "namespaced:";

/// `tsd.kv` namespace prefix of [`crate::cache::DriverResponseCache`] entries
/// (`cache:<cache namespace>`).
pub(crate) const CACHE_NS_PREFIX: &str = "cache:";

/// The id of the entry `(owner, ns, key)`: `Key::of([owner, ns, key])`.
pub(crate) fn kv_id(owner: &str, ns: &str, key: &str) -> Result<String, StorageError> {
    Ok(Key::of([owner, ns, key])?.as_str().to_owned())
}

/// A `tsd.kv` document body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KvEntry {
    pub(crate) owner: String,
    pub(crate) ns: String,
    pub(crate) key: String,
    pub(crate) value: Value,
}

impl KvEntry {
    pub(crate) fn new(owner: &str, ns: &str, key: &str, value: Value) -> Self {
        Self {
            owner: owner.to_owned(),
            ns: ns.to_owned(),
            key: key.to_owned(),
            value,
        }
    }
}

/// The document body of an entry, as `Kv::put` would write it.
pub(crate) fn kv_body(owner: &str, ns: &str, key: &str, value: Value) -> Value {
    serde_json::json!({ "owner": owner, "ns": ns, "key": key, "value": value })
}
