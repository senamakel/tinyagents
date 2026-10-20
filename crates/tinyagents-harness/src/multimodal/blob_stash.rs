//! Attachment stash on `tinystoragedrivers` blob ports: content-addressed image
//! bytes kept out of band so conversations persist a compact
//! `[Image: … #att:<id>]` placeholder instead of a multi-megabyte `data:` URI.
//!
//! [`BlobAttachmentStash`] is the only attachment stash: dedup'd writes,
//! size-cap eviction, TTL sweep and an id index over the host's blob and
//! document ports, with no directory of files (the earlier directory-backed
//! stash is removed). The host owns the policy (how large, how long); the tenant scope is
//! already bound into the handles it passes.
//!
//! # Layout
//!
//! - bytes: blob `harness/attachments/<id>` with the image MIME as content type;
//! - index: one document per id in [`BlobAttachmentStash::INDEX_COLLECTION`],
//!   `{ id, key, mime, len, touched_ms }`. Blob metadata carries no timestamps,
//!   so the index is what orders eviction and expiry. Retrieval is by id; there
//!   is no path to hand a file reader, so a host rehydrates placeholders from
//!   [`read_data_uri`](BlobAttachmentStash::read_data_uri).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tinyagents_storage_kit::{
    BlobStore, CollectionSpec, DocumentStore, DocumentStoreExt, Filter, Precondition, Query, Sort,
    WriteOptions,
};
use tokio::sync::{Mutex, OnceCell};

use super::data_uri;
use crate::ids::now_ms;

/// Where a stashed attachment lives and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StashedAttachment {
    /// The content-addressed id.
    pub id: String,
    /// The blob key holding the bytes.
    pub key: String,
    /// The image MIME type.
    pub mime: String,
    /// Size in bytes.
    pub len: u64,
}

/// A size-capped, TTL-swept attachment stash on blob and document ports.
#[derive(Debug, Clone)]
pub struct BlobAttachmentStash {
    blobs: Arc<dyn BlobStore>,
    index: Arc<dyn DocumentStore>,
    max_bytes: u64,
    ttl: Duration,
    declared: Arc<OnceCell<()>>,
    /// Serialises writes and cap enforcement across clones, so two writers
    /// never evict from the same stale snapshot of the index.
    guard: Arc<Mutex<()>>,
}

impl BlobAttachmentStash {
    /// The index collection.
    pub const INDEX_COLLECTION: &'static str = crate::storage::ATTACHMENT_STASH;
    /// Prefix of every blob key this stash writes.
    pub const KEY_PREFIX: &'static str = "harness/attachments/";

    /// A stash that, after each new write, evicts the least recently used
    /// attachments until the total is back under `max_bytes`;
    /// [`Self::sweep_stale`] removes attachments untouched for `ttl`.
    pub fn new(
        blobs: Arc<dyn BlobStore>,
        index: Arc<dyn DocumentStore>,
        max_bytes: u64,
        ttl: Duration,
    ) -> Self {
        Self {
            blobs,
            index,
            max_bytes,
            ttl,
            declared: Arc::new(OnceCell::new()),
            guard: Arc::new(Mutex::new(())),
        }
    }

    /// The declaration the index needs.
    pub fn collection_spec() -> CollectionSpec {
        crate::storage::attachment_stash_spec()
    }

    fn key(id: &str) -> String {
        format!("{}{id}", Self::KEY_PREFIX)
    }

    async fn declared(&self) -> anyhow::Result<()> {
        self.declared
            .get_or_try_init(|| async {
                self.index.ensure_collection(&Self::collection_spec()).await
            })
            .await?;
        Ok(())
    }

    /// Persist a canonical image data URI under `id`.
    ///
    /// Deduplicated: an existing attachment only has its last-use time
    /// refreshed. `id` must be non-empty `[A-Za-z0-9_-]`, so it can never name
    /// a key outside the stash. An attachment larger than the whole cap is
    /// rejected before anything is written or evicted. After a fresh write the
    /// size cap is enforced.
    pub async fn write(&self, id: &str, data_uri_text: &str) -> anyhow::Result<StashedAttachment> {
        if !is_safe_id(id) {
            anyhow::bail!("invalid attachment id: must be non-empty [A-Za-z0-9_-]");
        }
        let parsed = data_uri::parse_data_uri(data_uri_text)
            .map_err(|reason| anyhow::anyhow!("cannot decode stashed data URI: {reason}"))?;
        let len = parsed.bytes.len() as u64;
        if len > self.max_bytes {
            anyhow::bail!(
                "attachment of {len} bytes exceeds the stash cap of {} bytes",
                self.max_bytes
            );
        }
        self.declared().await?;
        let _writing = self.guard.lock().await;
        let key = Self::key(id);
        let attachment = StashedAttachment {
            id: id.to_owned(),
            key: key.clone(),
            mime: parsed.mime.clone(),
            len,
        };
        let present = self.index.get(Self::INDEX_COLLECTION, id).await?.is_some()
            && self.blobs.head(&key).await?.is_some();
        if !present {
            self.blobs
                .put(&key, parsed.bytes, Some(&parsed.mime))
                .await?;
        }
        // Content-addressing deduplicates identical images, but a new message
        // can reference one whose previous use is older than the TTL: refresh
        // so the immediately following sweep cannot reclaim it.
        self.index
            .put_with(
                Self::INDEX_COLLECTION,
                id,
                json!({
                    "id": id,
                    "key": key,
                    "mime": attachment.mime,
                    "len": len,
                    "touched_ms": now_ms(),
                }),
                Precondition::None,
                WriteOptions::new().schema(crate::storage::SCHEMA_VERSION),
            )
            .await?;
        if !present {
            self.enforce_cap_locked().await?;
        }
        // Never report an attachment that is gone: a cap eviction by another
        // process sharing the backend may have reclaimed it.
        if self.blobs.head(&attachment.key).await?.is_none() {
            anyhow::bail!("attachment was evicted by the stash size cap before it could be used");
        }
        Ok(attachment)
    }

    /// The bytes and MIME type of `id`, if it is still stashed.
    pub async fn read(&self, id: &str) -> anyhow::Result<Option<(Vec<u8>, String)>> {
        if !is_safe_id(id) {
            return Ok(None);
        }
        self.declared().await?;
        let Some(row) = self.index.get(Self::INDEX_COLLECTION, id).await? else {
            return Ok(None);
        };
        let mime = row
            .doc
            .get("mime")
            .and_then(Value::as_str)
            .unwrap_or("application/octet-stream")
            .to_owned();
        Ok(self
            .blobs
            .get(&Self::key(id))
            .await?
            .map(|blob| (blob.bytes, mime)))
    }

    /// `id` as a canonical `data:` URI, for rehydrating a placeholder.
    pub async fn read_data_uri(&self, id: &str) -> anyhow::Result<Option<String>> {
        Ok(self
            .read(id)
            .await?
            .map(|(bytes, mime)| data_uri::encode_data_uri(&mime, &bytes)))
    }

    /// Every stashed attachment, keyed by id.
    pub async fn build_index(&self) -> anyhow::Result<HashMap<String, StashedAttachment>> {
        self.declared().await?;
        let rows = self
            .index
            .query_all(Self::INDEX_COLLECTION, &Query::default())
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| attachment_of(&row.doc).map(|a| (a.id.clone(), a)))
            .collect())
    }

    /// Evict the least recently used attachments until the stash is under the
    /// size cap. Best-effort per entry.
    pub async fn enforce_cap(&self) -> anyhow::Result<()> {
        self.declared().await?;
        let _evicting = self.guard.lock().await;
        self.enforce_cap_locked().await
    }

    async fn enforce_cap_locked(&self) -> anyhow::Result<()> {
        let rows = self
            .index
            .query_all(
                Self::INDEX_COLLECTION,
                &Query::default().sort(Sort::asc("touched_ms")),
            )
            .await?;
        let mut total: u64 = rows
            .iter()
            .filter_map(|row| row.doc.get("len").and_then(Value::as_u64))
            .fold(0, u64::saturating_add);
        for row in rows {
            if total <= self.max_bytes {
                break;
            }
            let len = row.doc.get("len").and_then(Value::as_u64).unwrap_or(0);
            let Some(attachment) = attachment_of(&row.doc) else {
                continue;
            };
            self.blobs.delete(&attachment.key).await?;
            self.index
                .delete(Self::INDEX_COLLECTION, &row.id, Precondition::None)
                .await?;
            total = total.saturating_sub(len);
            tracing::debug!(
                target: "multimodal",
                id = %attachment.id,
                "[multimodal::images][gc] evicted attachment over size cap"
            );
        }
        Ok(())
    }

    /// Delete attachments untouched for longer than the TTL. Returns reclaimed bytes.
    pub async fn sweep_stale(&self) -> anyhow::Result<u64> {
        self.declared().await?;
        let _sweeping = self.guard.lock().await;
        let ttl_ms = u64::try_from(self.ttl.as_millis()).unwrap_or(u64::MAX);
        let cutoff = now_ms().saturating_sub(ttl_ms);
        let rows = self
            .index
            .query_all(
                Self::INDEX_COLLECTION,
                &Query::filter(Filter::lt("touched_ms", cutoff)),
            )
            .await?;
        let mut reclaimed = 0u64;
        for row in rows {
            let Some(attachment) = attachment_of(&row.doc) else {
                continue;
            };
            self.blobs.delete(&attachment.key).await?;
            self.index
                .delete(Self::INDEX_COLLECTION, &row.id, Precondition::None)
                .await?;
            reclaimed = reclaimed.saturating_add(attachment.len);
        }
        if reclaimed > 0 {
            tracing::info!(
                target: "multimodal",
                reclaimed_bytes = reclaimed,
                "[multimodal::images][gc] sweep removed stale attachments"
            );
        }
        Ok(reclaimed)
    }
}

fn attachment_of(doc: &Value) -> Option<StashedAttachment> {
    Some(StashedAttachment {
        id: doc.get("id")?.as_str()?.to_owned(),
        key: doc.get("key")?.as_str()?.to_owned(),
        mime: doc.get("mime")?.as_str()?.to_owned(),
        len: doc.get("len")?.as_u64()?,
    })
}

/// Whether `id` is non-empty and made only of ASCII letters, digits, `-` and `_`.
fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

#[cfg(test)]
#[path = "blob_stash_tests.rs"]
mod tests;
