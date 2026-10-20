//! [`Kv`]: a small key-value store per `(owner, namespace)`.

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::KV;
use super::repo::Repo;
use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_types::{Filter, Precondition, Query, Result, Sort, StorageError, Version};

/// Entries per page while [`Kv::list`] walks a namespace.
const PAGE: usize = 200;

/// A stored entry. Document bodies must be JSON objects, so a value that is a
/// number, string or array cannot be the body itself; the entry carries its
/// coordinates next to the value instead, which also lets the
/// `(owner, ns, key)` index list a namespace in key order.
#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    owner: String,
    ns: String,
    key: String,
    value: Value,
}

/// Key-value entries of one `(owner, namespace)` over [`KV`].
///
/// Values are any serde type; they are stored as JSON under the `value` field
/// of the entry document.
#[derive(Debug, Clone)]
pub struct Kv {
    repo: Repo<Entry>,
    docs: Arc<dyn DocumentStore>,
    owner: String,
    ns: String,
}

impl Kv {
    /// The namespace `ns` of `owner` on `docs`, which must be bound to the
    /// scope that owns the data.
    ///
    /// # Errors
    ///
    /// `InvalidInput` for an empty owner or namespace.
    pub fn ns(
        docs: &Arc<dyn DocumentStore>,
        owner: impl Into<String>,
        ns: impl Into<String>,
    ) -> Result<Self> {
        let (owner, ns) = (owner.into(), ns.into());
        if owner.is_empty() || ns.is_empty() {
            return Err(StorageError::invalid_input(
                "a key-value owner and namespace must not be empty",
            ));
        }
        Ok(Self {
            repo: Repo::new(docs, KV)?,
            docs: Arc::clone(docs),
            owner,
            ns,
        })
    }

    fn id(&self, key: &str) -> Result<String> {
        super::super::composite_key([self.owner.as_str(), self.ns.as_str(), key])
    }

    /// Read `key`.
    ///
    /// # Errors
    ///
    /// A storage error, or `Serialization` when the value is not a `T`.
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.repo.get(&self.id(key)?).await? {
            Some(found) => Ok(Some(serde_json::from_value(found.doc.value)?)),
            None => Ok(None),
        }
    }

    /// Store `value` under `key`, replacing any earlier one.
    ///
    /// # Errors
    ///
    /// A storage error, or `Serialization`.
    pub async fn put<T: Serialize + Sync>(&self, key: &str, value: &T) -> Result<Version> {
        let entry = Entry {
            owner: self.owner.clone(),
            ns: self.ns.clone(),
            key: key.to_owned(),
            value: serde_json::to_value(value)?,
        };
        self.repo.put(&self.id(key)?, &entry).await
    }

    /// Atomically update one value in this namespace.
    ///
    /// The closure may return `None` to leave the key unchanged. On a write
    /// conflict it is called again with the value written by the winner.
    ///
    /// # Errors
    ///
    /// A storage error, or `Serialization` when either value cannot be encoded.
    pub async fn update<T, F>(&self, key: &str, mut update: F) -> Result<Option<T>>
    where
        T: DeserializeOwned + Serialize + Sync,
        F: FnMut(Option<T>) -> Result<Option<T>> + Send,
    {
        let id = self.id(key)?;
        let owner = self.owner.clone();
        let ns = self.ns.clone();
        let changed = self
            .repo
            .update(&id, move |current: Option<Entry>| {
                let value = current
                    .map(|entry| serde_json::from_value(entry.value))
                    .transpose()?;
                let Some(value) = update(value)? else {
                    return Ok(None);
                };
                Ok(Some(Entry {
                    owner: owner.clone(),
                    ns: ns.clone(),
                    key: key.to_owned(),
                    value: serde_json::to_value(value)?,
                }))
            })
            .await?;
        Ok(changed
            .map(|(entry, _)| serde_json::from_value(entry.value))
            .transpose()?)
    }

    /// Remove `key` and report whether it existed.
    ///
    /// # Errors
    ///
    /// A storage error.
    pub async fn delete(&self, key: &str) -> Result<bool> {
        self.docs
            .delete(KV, &self.id(key)?, Precondition::None)
            .await
    }

    /// Every entry whose key starts with `prefix` (all entries for `""`), in
    /// key order, paging through the namespace internally.
    ///
    /// # Errors
    ///
    /// A storage error, or `Serialization` when a value is not a `T`.
    pub async fn list<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<(String, T)>> {
        let mut filter =
            Filter::eq("owner", self.owner.as_str()).and(Filter::eq("ns", self.ns.as_str()));
        if !prefix.is_empty() {
            filter = filter.and(Filter::gte("key", prefix));
        }
        let mut query = Query::filter(filter).sort(Sort::asc("key")).limit(PAGE);
        let mut out = Vec::new();
        loop {
            let page = self.repo.query(&query).await?;
            for item in page.items {
                if !item.doc.key.starts_with(prefix) {
                    // Sorted by key: the first miss past the lower bound ends the range.
                    return Ok(out);
                }
                out.push((item.doc.key, serde_json::from_value(item.doc.value)?));
            }
            match page.next {
                Some(cursor) => query = query.after(cursor),
                None => return Ok(out),
            }
        }
    }
}

#[cfg(test)]
#[path = "kv_tests.rs"]
mod tests;
