//! Storage-port models shared by TinyAgents persistence adapters.
//!
//! The crate owns the small typed helpers used by TinyAgents. Runtime engine
//! selection remains with the host, which supplies handles from
//! `tinystoragedrivers-api`.

pub use tinystoragedrivers_api::{
    BlobStore, DocumentStore, DocumentWatch, KvStore, QueueStore, SecretStore, Storage,
    StorageBackend as PortStorageBackend, StreamStore,
};
pub use tinystoragedrivers_types::*;

/// Scope-bound bundle used by owner-local migrations and typed helpers.
#[derive(Clone, Debug)]
pub struct ScopedStorage {
    storage: Storage,
    scope: tinystoragedrivers_types::Scope,
}

impl ScopedStorage {
    pub fn new(storage: Storage, scope: tinystoragedrivers_types::Scope) -> Self {
        Self { storage, scope }
    }

    pub fn scope(&self) -> &tinystoragedrivers_types::Scope {
        &self.scope
    }
    pub fn documents(&self) -> &std::sync::Arc<dyn DocumentStore> {
        self.storage.documents()
    }
    pub fn streams(&self) -> &std::sync::Arc<dyn StreamStore> {
        self.storage.streams()
    }
    pub fn kv(&self) -> &std::sync::Arc<dyn KvStore> {
        self.storage.kv()
    }
    pub fn blobs(&self) -> &std::sync::Arc<dyn BlobStore> {
        self.storage.blobs()
    }
    pub fn queues(&self) -> &std::sync::Arc<dyn QueueStore> {
        self.storage.queues()
    }
    pub fn secrets(&self) -> &std::sync::Arc<dyn SecretStore> {
        self.storage.secrets()
    }
}

/// Adapter for opening a scope-bound storage bundle from a port backend.
pub trait StorageBackend: Send + Sync {
    fn for_scope(&self, scope: &tinystoragedrivers_types::Scope) -> error::Result<ScopedStorage>;
}

impl<T: PortStorageBackend> StorageBackend for T {
    fn for_scope(&self, scope: &tinystoragedrivers_types::Scope) -> error::Result<ScopedStorage> {
        Ok(ScopedStorage::new(
            PortStorageBackend::for_scope(self, scope)?,
            scope.clone(),
        ))
    }
}

pub mod blocking;
pub use blocking::Blocking;
mod migration_lease;
mod migrations;
pub mod primitives;
pub mod schema;

pub mod backend {
    pub use crate::{ScopedStorage, StorageBackend};
}

/// Legacy-compatible composite key helper, backed by the current serialized type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key(tinystoragedrivers_types::Key);

impl Key {
    pub fn of<I, S>(parts: I) -> error::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let parts = parts
            .into_iter()
            .map(|part| part.as_ref().to_owned())
            .collect::<Vec<_>>();
        let refs = parts.iter().map(String::as_str).collect::<Vec<_>>();
        Ok(Self(tinystoragedrivers_types::Key::new(composite_key(
            refs,
        )?)?))
    }
    pub fn new(value: impl Into<String>) -> error::Result<Self> {
        Ok(Self(tinystoragedrivers_types::Key::new(value)?))
    }
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

pub use migration_lease::{MigrationLease, with_migration_lease};
pub use migrations::{MIGRATIONS_COLLECTION, Migrations};
pub use primitives::{Counters, Kv, Leases, Repo, RetryPolicy};
pub use schema::{
    BlobOutput, BlobStepFn, ComposedRegistry, Expiry, Item, MigrateOptions, MigrationReport, Owner,
    Registry, Schema, SchemaMetadata, ScopeClass, SealPolicy, SealWhen, SourceRecord, Step,
    StepExecution, StepFn, StepMetadata, StepReport, StreamStepFn, TargetRecord,
};
pub use tinystoragedrivers_api::DocumentStoreExt;

/// Compatibility namespaces for owner-local storage adapters.
pub mod document {
    pub use tinystoragedrivers_api::{DocumentStore, DocumentStoreExt};
    pub use tinystoragedrivers_types::{
        CollectionSpec, Cursor, IndexSpec, MAX_COLLECTION_LEN, Page, Precondition, Query, Version,
        Versioned, WriteOptions, validate_collection, validate_doc, validate_id,
    };
}
pub mod error {
    pub use tinystoragedrivers_types::{ErrorKind, Result, StorageError};
}
pub mod filter {
    pub use tinystoragedrivers_types::{Filter, ID_FIELD, SCHEMA_FIELD, Sort};
}
pub mod key {
    pub use crate::Key;
}
pub mod scope {
    pub use tinystoragedrivers_types::Scope;
}
pub mod stream {
    pub use tinystoragedrivers_api::StreamStore;
    pub use tinystoragedrivers_types::{SEALED_BLOB_PREFIX, SealOutcome, StreamEntry, StreamInfo};

    /// Validate a persisted stream name.
    pub fn validate_stream(name: &str) -> crate::error::Result<()> {
        tinystoragedrivers_types::validate_id(name)
    }
}
pub mod blob {
    pub use tinystoragedrivers_api::{BlobReader, BlobStore, BlobStream};
    pub use tinystoragedrivers_types::{Blob, BlobMeta, TreeEntry, TreeFile};
    pub const MAX_BLOB_KEY_LEN: usize = 1024;

    /// Validate a user-owned blob key.
    pub fn validate_user_blob_key(key: &str) -> crate::error::Result<()> {
        if key.is_empty() || key.len() > MAX_BLOB_KEY_LEN || key.contains('\0') {
            return Err(crate::error::StorageError::invalid_input(
                "invalid blob key",
            ));
        }
        if key.starts_with(tinystoragedrivers_types::SEALED_BLOB_PREFIX) {
            return Err(crate::error::StorageError::invalid_input(
                "reserved blob key",
            ));
        }
        Ok(())
    }
}

/// Build the same reversible composite document id used by older TSD consumers.
pub fn composite_key<'a>(parts: impl IntoIterator<Item = &'a str>) -> error::Result<String> {
    let mut out = String::new();
    let mut count = 0;
    for part in parts {
        if count > 0 {
            out.push('/');
        }
        count += 1;
        for (offset, ch) in part.chars().enumerate() {
            match ch {
                '%' => out.push_str("%25"),
                '/' => out.push_str("%2F"),
                '\0' => out.push_str("%00"),
                '~' if count == 1 && offset == 0 => out.push_str("%7E"),
                _ => out.push(ch),
            }
        }
    }
    if count == 0 {
        return Err(tinystoragedrivers_types::StorageError::invalid_input(
            "a key needs at least one part",
        ));
    }
    if out.len() > 512 {
        use sha2::{Digest, Sha256};
        out = format!(
            "~{}",
            data_encoding::BASE32_NOPAD
                .encode(&Sha256::digest(out.as_bytes()))
                .to_ascii_lowercase()
        );
    }
    Ok(tinystoragedrivers_types::Key::new(out)?.as_str().to_owned())
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
