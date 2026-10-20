//! Schema declarations and migration helpers for storage-owning crates.
//!
//! Every crate that owns a record type may declare `storage_schema() -> Schema`:
//! its [`Item`]s (collections, streams, blob and secret prefixes, each with a
//! [`ScopeClass`]) and the [`Step`]s that rewrite older versions of a
//! collection. The owner applies its declaration to the storage handle already
//! bound by the host; owner migrations stay local to that crate:
//!
//! 1. [`Registry::compose`] refuses bad names, names outside an owner's prefix,
//!    duplicate claims and owners, step gaps and descending versions. It is
//!    pure: nothing touches a backend.
//! 2. [`Registry::prepare`] declares the owner's collections (trusted bootstrap).
//! 3. [`Registry::migrate`] runs that owner's pending steps over a [`ScopedStorage`](crate::ScopedStorage),
//!    recording progress in the `tsd.migrations` collection so a crash resumes
//!    at the last finished batch.
//! 4. Typed repositories call [`Registry::check_ready`] before they write;
//!    a collection whose steps have not finished answers
//!    [`ErrorKind::Unavailable`](crate::ErrorKind::Unavailable) with
//!    `migrating <owner>.<item>`. Drivers do not enforce it.
//!
//! # Example
//!
//! ```
//! use serde_json::json;
//! use tinyagents_storage_kit::schema::{MigrateOptions, Owner, Registry, Schema, ScopeClass, Step, TargetRecord};
//! use tinyagents_storage_kit::{Scope, StorageBackend};
//! use tinystoragedrivers_testkit::memory_backend;
//!
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! let schema = Schema::new(Owner::new("shop")?)
//!     .collection("shop.orders", 2, ScopeClass::Tenant)
//!     .step(Step::new("rename", "shop.orders", 1, 2, |batch| {
//!         Ok(batch
//!             .into_iter()
//!             .map(|r| TargetRecord::new("shop.orders", r.id, json!({"total_ms": r.body["t"]})))
//!             .collect())
//!     }));
//! let registry = Registry::compose([schema])?;
//! let backend = memory_backend();
//! registry.prepare(&backend).await?;
//! let storage = backend.for_scope(&Scope::new("local")?)?;
//! assert!(registry.check_ready("shop.orders").is_err());
//! registry.migrate(&storage, &MigrateOptions::new()).await?;
//! registry.check_ready("shop.orders")?;
//! # Ok::<_, tinyagents_storage_kit::error::StorageError>(())
//! # }).unwrap();
//! ```
//!
//! A step targets one kind of item: [`Step::new`] rewrites collection
//! documents, [`Step::stream`] moves legacy streams (read whole, mapped,
//! rewritten, sealed again when the source was sealed, then deleted) and
//! [`Step::blobs`] moves legacy blobs. Stream and blob steps name their legacy
//! source with [`Step::source_prefix`]; see [`Registry::migrate`] for how they
//! resume.
//!
//! Out of scope here: legacy-file importers (steps with no source) are run by
//! the owner, which records completion through a [`MigrationLease`](crate::MigrationLease).
//! A stream's seal policy is declarative; its owner or host is responsible for
//! enforcing it.

mod compose;
mod declaration;
mod executor;
mod metadata;
mod migrations;
mod prepared;
mod runner;
mod runner_blobs;
mod runner_streams;
mod seal;
mod transform;
mod types;

pub use compose::{ComposedRegistry, Registry};
pub use executor::TransformExecutor;
pub use metadata::{SchemaMetadata, StepExecution, StepMetadata};
pub use runner::{MigrateOptions, MigrationReport, StepReport};
pub use seal::{SealPolicy, SealWhen};
pub use transform::{
    BlobTransformOutput, BlobTransformRequest, DocumentsTransformRequest, SchemaDeclaration,
    StepDeclaration, StreamTransformOutput, StreamTransformRequest, TransformDescriptor,
    TransformError, TransformErrorKind, TransformKind, TransformResult,
};
pub use types::{
    BlobOutput, BlobStepFn, Expiry, Item, Owner, Schema, ScopeClass, SourceRecord, Step, StepBody,
    StepFn, StreamStepFn, TargetRecord,
};
