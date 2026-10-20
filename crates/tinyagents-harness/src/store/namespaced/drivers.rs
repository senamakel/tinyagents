//! A [`NamespacedStore`] over `tinystoragedrivers` document ports.
//!
//! [`DriverNamespacedStore`] keeps the hierarchical, TTL-aware long-term store
//! on whatever backend the host opened. The handle is already bound to a tenant
//! scope, so two tenants never see each other's items.
//!
//! # Layout
//!
//! Items are documents of the shared `tsd.kv` collection, one per
//! `(namespace, key)`, owned by [`DriverNamespacedStore::DEFAULT_OWNER`]
//! (`harness`):
//!
//! ```text
//! id   Key::of([owner, "namespaced:<segments joined by '.'>", key])
//! body { owner, ns: "namespaced:<segments>", key, value }
//! meta created_ms, updated_ms, expires_ms
//! ```
//!
//! A namespace segment may not contain `.`, so the joined form is unambiguous;
//! the `namespaced:` prefix keeps these entries apart from the flat
//! [`DriverStore`](crate::store::DriverStore) of the same owner. The item's
//! creation, update and expiry times are the record metadata, kept by the
//! driver, and the expiry is enforced by it: an expired item reads as absent
//! everywhere. A subtree search or a namespace listing reads one root's range of
//! `ns` rather than the collection.
//!
//! # Semantics
//!
//! The matching rules are the in-memory store's, applied by the same functions:
//! field filters, case-insensitive substring `query`, `*` wildcards in
//! namespace listings, deterministic `(namespace, key)` order for paging. Prefix
//! narrowing by root is the only part the backend does; everything else runs on
//! the returned documents, so a subtree search costs the size of its root.
//!
//! Operations in one [`batch`](NamespacedStore::batch) run in order and each is
//! individually atomic; a batch is not a transaction.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tinyagents_storage_kit::{
    DocumentStore, DocumentStoreExt, EXPIRES_FIELD, Filter, Precondition, Query, Repo,
    StorageError, Versioned, WriteOptions,
};
use tokio::sync::OnceCell;

use super::{
    Item, ListNamespacesQuery, Namespace, NamespacedStore, SearchQuery, StoreOp, StoreResult,
    TtlConfig, expiry_at, matches_filter, matches_query, matches_wildcards,
};
use crate::error::{Result, TinyAgentsError};
use crate::ids::now_ms;
use crate::storage::{KV, KvEntry, NAMESPACED_NS_PREFIX, kv_id};
use crate::store::{DriverStore, SchemaAdmission, map_driver_error as map_error};

/// A [`NamespacedStore`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverNamespacedStore {
    docs: Arc<dyn DocumentStore>,
    owner: String,
    ttl: TtlConfig,
    declared: Arc<OnceCell<()>>,
}

impl DriverNamespacedStore {
    /// The `tsd.kv` owner used by [`DriverNamespacedStore::new`].
    pub const DEFAULT_OWNER: &'static str = "harness";

    /// Store under [`Self::DEFAULT_OWNER`] with no expiry.
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_owner(docs, Self::DEFAULT_OWNER)
    }

    /// Store under `owner`, so independent stores can share a backend.
    pub fn with_owner(docs: Arc<dyn DocumentStore>, owner: impl Into<String>) -> Self {
        Self {
            docs,
            owner: owner.into(),
            ttl: TtlConfig::default(),
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// Sets the TTL policy.
    #[must_use]
    pub fn with_ttl(mut self, ttl: TtlConfig) -> Self {
        self.ttl = ttl;
        self
    }

    /// The `tsd.kv` namespace of `namespace`: `namespaced:<segments joined by '.'>`.
    fn kv_ns(namespace: &Namespace) -> String {
        format!("{NAMESPACED_NS_PREFIX}{}", namespace.segments().join("."))
    }

    fn id(&self, namespace: &Namespace, key: &str) -> Result<String> {
        kv_id(&self.owner, &Self::kv_ns(namespace), key).map_err(map_error)
    }

    /// Writes made through this handle carry `expires_ms` and nothing else.
    fn repo(&self, expires_ms: Option<u64>) -> Result<Repo<KvEntry>> {
        let mut options = WriteOptions::new();
        if let Some(at) = expires_ms {
            options = options.expires_ms(i64::try_from(at).unwrap_or(i64::MAX));
        }
        Ok(Repo::new(&self.docs, KV)
            .map_err(map_error)?
            .with_options(options))
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                SchemaAdmission::Declare
                    .admit(self.docs.as_ref(), &DriverStore::collection_spec())
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }

    /// The item a `tsd.kv` document holds, with its times from the metadata.
    fn item(row: &Versioned<Value>) -> Result<Item> {
        let bad = |what: &str| TinyAgentsError::Storage(format!("namespaced item has no {what}"));
        let ns = row
            .doc
            .get("ns")
            .and_then(Value::as_str)
            .and_then(|ns| ns.strip_prefix(NAMESPACED_NS_PREFIX))
            .ok_or_else(|| bad("namespace"))?;
        let segments: Vec<String> = ns.split('.').map(str::to_owned).collect();
        Ok(Item {
            namespace: Namespace(segments),
            key: row
                .doc
                .get("key")
                .and_then(Value::as_str)
                .ok_or_else(|| bad("key"))?
                .to_owned(),
            value: row.doc.get("value").cloned().ok_or_else(|| bad("value"))?,
            created_at_ms: u64::try_from(row.meta.created_ms).unwrap_or(0),
            updated_at_ms: u64::try_from(row.meta.updated_ms).unwrap_or(0),
            expires_at_ms: row.meta.expires_ms.and_then(|at| u64::try_from(at).ok()),
        })
    }

    /// The `ns` range holding the subtree rooted at `root` (every root when
    /// `None`) of this owner. A range is a superset: `a-b` sorts between `a` and
    /// `a.x`, so [`Self::in_root`] decides each row.
    fn root_filter(&self, root: Option<&str>) -> Filter {
        let owner = Filter::eq("owner", self.owner.as_str());
        match root {
            Some(root) => {
                let low = format!("{NAMESPACED_NS_PREFIX}{root}");
                // `.` < `/`: everything below the root sorts before `<root>/`.
                owner
                    .and(Filter::gte("ns", low.as_str()))
                    .and(Filter::lt("ns", format!("{low}/")))
            }
            None => owner
                .and(Filter::gte("ns", NAMESPACED_NS_PREFIX))
                // `:` + 1 == `;`: the end of the prefix's range.
                .and(Filter::lt("ns", "namespaced;")),
        }
    }

    /// Whether a row of a [`Self::root_filter`] range really is in `root`'s subtree.
    fn in_root(row: &Versioned<Value>, root: Option<&str>) -> bool {
        let Some(ns) = row.doc.get("ns").and_then(Value::as_str) else {
            return false;
        };
        let Some(ns) = ns.strip_prefix(NAMESPACED_NS_PREFIX) else {
            return false;
        };
        root.is_none_or(|root| {
            ns == root
                || ns
                    .strip_prefix(root)
                    .is_some_and(|rest| rest.starts_with('.'))
        })
    }

    /// Live items of the subtrees rooted at `prefix`'s first segment (every
    /// root when the prefix is empty).
    async fn live_items(&self, prefix: &[String], now: u64) -> Result<Vec<Item>> {
        let root = prefix.first().map(String::as_str);
        let query = Query::filter(self.root_filter(root));
        let rows = self.docs.query_all(KV, &query).await.map_err(map_error)?;
        let mut items = Vec::with_capacity(rows.len());
        for row in rows.iter().filter(|row| Self::in_root(row, root)) {
            let item = Self::item(row)?;
            if !item.is_expired(now) {
                items.push(item);
            }
        }
        Ok(items)
    }

    async fn get_item(
        &self,
        namespace: &Namespace,
        key: &str,
        refresh_ttl: Option<bool>,
        now: u64,
    ) -> Result<Option<Item>> {
        let id = self.id(namespace, key)?;
        let Some(row) = self.docs.get(KV, &id).await.map_err(map_error)? else {
            return Ok(None);
        };
        let mut item = Self::item(&row)?;
        if item.is_expired(now) {
            return Ok(None);
        }
        let refresh = refresh_ttl.unwrap_or(self.ttl.refresh_on_read);
        let Some(minutes) = self.ttl.default_ttl_minutes.filter(|_| refresh) else {
            return Ok(Some(item));
        };
        // Slide the window with one compare-and-swap: the entry is rewritten
        // unchanged under its new expiry, and a concurrent delete wins.
        let expires = expiry_at(now, Some(minutes));
        let refreshed = self
            .repo(expires)?
            .update(&id, Ok::<_, StorageError>)
            .await
            .map_err(map_error)?;
        if refreshed.is_none() {
            return Ok(None);
        }
        item.expires_at_ms = expires;
        item.updated_at_ms = now;
        Ok(Some(item))
    }

    async fn put_item(
        &self,
        namespace: &Namespace,
        key: &str,
        value: Value,
        ttl_minutes: Option<f64>,
        now: u64,
    ) -> Result<()> {
        let id = self.id(namespace, key)?;
        let ttl = ttl_minutes.or(self.ttl.default_ttl_minutes);
        let entry = KvEntry::new(&self.owner, &Self::kv_ns(namespace), key, value);
        // The driver keeps the creation time of an overwritten item.
        self.repo(expiry_at(now, ttl))?
            .put(&id, &entry)
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    async fn search_items(&self, query: &SearchQuery, now: u64) -> Result<Vec<Item>> {
        let mut found: Vec<Item> = self
            .live_items(&query.namespace_prefix, now)
            .await?
            .into_iter()
            .filter(|item| item.namespace.starts_with(&query.namespace_prefix))
            .filter(|item| matches_filter(item, &query.filter))
            .filter(|item| {
                query
                    .query
                    .as_ref()
                    .is_none_or(|needle| matches_query(item, needle))
            })
            .collect();
        found.sort_by(|a, b| {
            a.namespace
                .cmp(&b.namespace)
                .then_with(|| a.key.cmp(&b.key))
        });
        Ok(found
            .into_iter()
            .skip(query.offset)
            .take(query.limit.unwrap_or(usize::MAX))
            .collect())
    }

    async fn list_namespace_items(
        &self,
        query: &ListNamespacesQuery,
        now: u64,
    ) -> Result<Vec<Namespace>> {
        // A wildcard first segment cannot narrow by root.
        let narrow: Vec<String> = query
            .prefix
            .iter()
            .flatten()
            .take(1)
            .filter(|segment| segment.as_str() != "*")
            .cloned()
            .collect();
        let mut seen: BTreeSet<Namespace> = BTreeSet::new();
        for item in self.live_items(&narrow, now).await? {
            let ns = &item.namespace;
            if let Some(prefix) = &query.prefix
                && !(ns.segments().len() >= prefix.len()
                    && matches_wildcards(&ns.segments()[..prefix.len()], prefix))
            {
                continue;
            }
            if let Some(suffix) = &query.suffix {
                let len = ns.segments().len();
                if !(len >= suffix.len()
                    && matches_wildcards(&ns.segments()[len - suffix.len()..], suffix))
                {
                    continue;
                }
            }
            let truncated = match query.max_depth {
                Some(depth) if ns.segments().len() > depth => {
                    Namespace(ns.segments()[..depth].to_vec())
                }
                _ => ns.clone(),
            };
            seen.insert(truncated);
        }
        Ok(seen
            .into_iter()
            .skip(query.offset)
            .take(query.limit.unwrap_or(usize::MAX))
            .collect())
    }
}

#[async_trait]
impl NamespacedStore for DriverNamespacedStore {
    fn ttl_config(&self) -> TtlConfig {
        self.ttl
    }

    async fn sweep_expired(&self) -> Result<usize> {
        self.declared().await?;
        // The driver already hides an expired item from every read; this
        // reclaims the rows of any it has not yet purged.
        let filter = self.root_filter(None).and(Filter::lte(
            EXPIRES_FIELD,
            i64::try_from(now_ms()).unwrap_or(i64::MAX),
        ));
        let reclaimed = self
            .docs
            .delete_where(KV, &filter)
            .await
            .map_err(map_error)? as usize;
        if reclaimed > 0 {
            tracing::debug!("[store:namespaced] sweep_expired reclaimed={reclaimed}");
        }
        Ok(reclaimed)
    }

    async fn batch(&self, ops: &[StoreOp]) -> Result<Vec<StoreResult>> {
        self.declared().await?;
        let now = now_ms();
        let mut out = Vec::with_capacity(ops.len());
        for op in ops {
            out.push(match op {
                StoreOp::Get {
                    namespace,
                    key,
                    refresh_ttl,
                } => {
                    namespace.validate()?;
                    StoreResult::Item(
                        self.get_item(namespace, key, *refresh_ttl, now)
                            .await?
                            .map(Box::new),
                    )
                }
                StoreOp::Put {
                    namespace,
                    key,
                    value,
                    ttl_minutes,
                } => {
                    namespace.validate()?;
                    match value {
                        None => {
                            self.docs
                                .delete(KV, &self.id(namespace, key)?, Precondition::None)
                                .await
                                .map_err(map_error)?;
                        }
                        Some(value) => {
                            self.put_item(namespace, key, value.clone(), *ttl_minutes, now)
                                .await?;
                        }
                    }
                    StoreResult::Ack
                }
                StoreOp::Search(query) => StoreResult::Items(self.search_items(query, now).await?),
                StoreOp::ListNamespaces(query) => {
                    StoreResult::Namespaces(self.list_namespace_items(query, now).await?)
                }
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
