//! Owners, stored items, and transform records.

use super::seal::SealPolicy;
use crate::{CollectionSpec, Meta, Result, StorageError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

/// The owner reserved for the crate's own shared collections.
const RESERVED_OWNER: &str = "tsd";

/// The crate or domain that defines a record type: lowercase
/// `[a-z][a-z0-9_]{1,23}`. Every name an owner declares starts with its owner
/// (`<owner>.` for collections and streams, `<owner>/` for blob and secret
/// prefixes).
///
/// `tsd` is reserved for the shared primitives and is refused by
/// [`Owner::new`].
///
/// ```
/// use tinyagents_storage_kit::Owner;
///
/// assert_eq!(Owner::new("cron").unwrap().as_str(), "cron");
/// assert!(Owner::new("tsd").is_err());
/// assert!(Owner::new("Cron").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Owner(String);

impl Owner {
    /// Validate an owner name.
    ///
    /// # Errors
    ///
    /// `ErrorKind::InvalidInput` when the
    /// name is outside `[a-z][a-z0-9_]{1,23}` or is the reserved `tsd`.
    pub fn new(name: &str) -> Result<Self> {
        let bytes = name.as_bytes();
        let ok = (2..=24).contains(&bytes.len())
            && bytes.first().is_some_and(u8::is_ascii_lowercase)
            && bytes
                .iter()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_');
        if !ok {
            return Err(StorageError::invalid_input(format!(
                "owner `{name}` must match [a-z][a-z0-9_]{{1,23}}"
            )));
        }
        if name == RESERVED_OWNER {
            return Err(StorageError::invalid_input(format!(
                "owner `{RESERVED_OWNER}` is reserved for the storage crate's own collections"
            )));
        }
        Ok(Self(name.to_owned()))
    }

    /// The owner as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Owner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which scope an item lives in. The registry records it so a host can
/// derive grants and pick the right `Scope` for each item; the
/// storage drivers do not look at it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScopeClass {
    /// Shared across the tenant (`u:<id>` or `local`).
    Tenant,
    /// One agent inside the tenant (`<tenant>/agent:<id>`).
    Agent,
    /// One module inside the tenant (`<tenant>/module:<id>`).
    Module,
    /// The tenant-less `cluster` scope.
    Cluster,
}

/// One thing an owner stores.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Item {
    /// A document collection, named `<owner>.<entity>`, at body `version`
    /// (starting at 1).
    Collection {
        /// The collection name.
        name: String,
        /// The declared body version.
        version: u32,
        /// Indexes, search and so on, declared when the registry is prepared.
        spec: CollectionSpec,
        /// Where the collection lives.
        scope: ScopeClass,
    },
    /// A stream family, declared by its base name `<owner>.<stream>`. Entries
    /// are appended to `<base>` or `<base>/<key>` at use time.
    Stream {
        /// The base stream name.
        name: String,
        /// Where the streams live.
        scope: ScopeClass,
        /// When the owner seals these streams and how long sealed ones are
        /// kept.
        seal: SealPolicy,
    },
    /// A blob key prefix `<owner>/…`.
    Blobs {
        /// The key prefix.
        prefix: String,
        /// Where the blobs live.
        scope: ScopeClass,
    },
    /// A secret key prefix `<owner>/…`.
    Secrets {
        /// The key prefix.
        prefix: String,
        /// Where the secrets live.
        scope: ScopeClass,
    },
}

impl Item {
    /// The name (collection or stream) or prefix (blobs or secrets) the item
    /// claims.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Collection { name, .. } | Self::Stream { name, .. } => name,
            Self::Blobs { prefix, .. } | Self::Secrets { prefix, .. } => prefix,
        }
    }

    /// The scope class of the item.
    #[must_use]
    pub fn scope(&self) -> ScopeClass {
        match self {
            Self::Collection { scope, .. }
            | Self::Stream { scope, .. }
            | Self::Blobs { scope, .. }
            | Self::Secrets { scope, .. } => *scope,
        }
    }
}

/// A document handed to a `Step`, as stored in its source collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRecord {
    /// The document id.
    pub id: String,
    /// The document body.
    pub body: Value,
    /// The driver's metadata for it; `meta.created_ms` is what a step passes to
    /// [`TargetRecord::created_ms`] to keep history.
    pub meta: Meta,
}

/// A document a `Step` wants written.
///
/// It is written to `collection` (an item of the same owner) with the schema
/// version implied by the step's `to` (or by the declared version of a
/// different item), and with `Precondition::Absent`, so a record that already
/// exists is kept. Its expiry defaults to the source record's when the output
/// rewrites that record under its own id ([`Expiry::Carry`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetRecord {
    /// Destination collection.
    pub collection: String,
    /// Destination id.
    pub id: String,
    /// The new body.
    pub body: Value,
    /// Source creation time for checking an idempotent same-row rewrite.
    /// The owner runner does not write host-owned creation metadata through
    /// the module storage port; a newly created target uses the host clock.
    pub created_ms: Option<i64>,
    /// What to record as the expiry. See [`Expiry`].
    pub expires: Expiry,
}

/// The expiry a [`TargetRecord`] is written with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Expiry {
    /// The expiry of the source record with the same id, when the output is
    /// that record rewritten (an in-place step, or a step that keeps ids);
    /// none for a record with a new id, which the step must date itself.
    #[default]
    Carry,
    /// The record never expires, whatever the source did.
    Never,
    /// The record reads as absent from this Unix epoch millisecond.
    At(i64),
}

impl TargetRecord {
    /// A record for `collection` with no creation time of its own.
    #[must_use]
    pub fn new(collection: impl Into<String>, id: impl Into<String>, body: Value) -> Self {
        Self {
            collection: collection.into(),
            id: id.into(),
            body,
            created_ms: None,
            expires: Expiry::Carry,
        }
    }

    /// Carry the source time for same-row rewrite conflict checks. New target
    /// rows remain dated by the storage host.
    #[must_use]
    pub fn created_ms(mut self, created_ms: i64) -> Self {
        self.created_ms = Some(created_ms);
        self
    }

    /// Record `expires_ms` as the expiry (`Some(ms)`), or none (`None`),
    /// instead of carrying the source record's. A step moving a legacy TTL
    /// body field into metadata passes it here; one that renames a record
    /// passes `record.meta.expires_ms` to keep the old deadline.
    #[must_use]
    pub fn expires_ms(mut self, expires_ms: Option<i64>) -> Self {
        self.expires = expires_ms.map_or(Expiry::Never, Expiry::At);
        self
    }
}

impl<'de> Deserialize<'de> for Owner {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::new(&name).map_err(serde::de::Error::custom)
    }
}
