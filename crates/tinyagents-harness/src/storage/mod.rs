//! The harness's v2 storage declarations: its `storage_schema()`, the shared
//! `tsd.kv` body helpers and the importer for the key-value stores that moved
//! onto `tsd.kv`.

mod kv;
mod kv_import;
mod steps;

use tinyagents_storage_kit::{
    CollectionSpec, IndexSpec, MigrateOptions, Owner, Registry, Schema, ScopeClass, ScopedStorage,
    SealPolicy, SealWhen,
};

pub(crate) use kv::{CACHE_NS_PREFIX, KV, KvEntry, NAMESPACED_NS_PREFIX, kv_id};
pub(crate) use kv_import::import_legacy_kv_with_lease;
pub use kv_import::{LegacyKv, import_legacy_kv};

/// The body version of every harness collection a v1 -> v2 step produces.
pub(crate) const SCHEMA_VERSION: u32 = 2;

/// Per-run status snapshots.
pub(crate) const RUN_STATUS: &str = "harness.run_status";

/// Cached tool-catalogue embedding vectors.
pub(crate) const EMBEDDING_CACHE: &str = "harness.embedding_cache";

/// The run journal's stream family: `harness.journal/<run id>`.
pub(crate) const JOURNAL: &str = "harness.journal";

/// How long a sealed run journal is kept before the host's maintenance deletes
/// it: 30 days.
const JOURNAL_RETENTION_MS: u64 = 30 * 24 * 60 * 60 * 1000;

/// The attachment stash's index: one row per stashed attachment.
pub(crate) const ATTACHMENT_STASH: &str = "harness.attachment_stash";

/// The attachment stash's index declaration; the stash itself is behind the
/// `multimodal` feature, the schema is not.
pub(crate) fn attachment_stash_spec() -> CollectionSpec {
    CollectionSpec::new(ATTACHMENT_STASH).index(IndexSpec::new("touched", ["touched_ms"]))
}

/// The harness's storage declaration and the steps that bring its v1 names
/// and shapes forward. Harness open runs this owner-local migration against
/// the already-bound scope before returning the handle.
///
/// | v1 | v2 |
/// |---|---|
/// | `harness_run_status` | `harness.run_status` |
/// | `harness_embedding_cache` | `harness.embedding_cache` |
/// | `attachment_stash_index` | `harness.attachment_stash` |
/// | stream `16:session_journal/<run>` | stream `harness.journal/<run>`, sealed when the run finishes, kept 30 days after |
/// | blobs `attachments/`, `tool-results/`, `outputs/`, `workspace/`, `generated-media/` | `harness/attachments/`, `harness/tool_results/`, `harness/outputs/`, `harness/workspace/`, `harness/generated_media/` |
///
/// The key-value stores (`harness_store`, `harness_namespaced_store`,
/// `harness_response_cache`) live in `tsd.kv`, which `tsd` owns, so a document
/// step cannot write them: the host calls [`import_legacy_kv`] for each.
///
/// The journal entries drop the run id, root run id and event id (see
/// `HarnessEventJournal::seal_run` for sealing). The host seals a run's journal
/// when it ends; `reap_orphaned_runs_and_seal` does it for orphans.
///
/// Blob prefixes are declared one per directory (`harness/<dir>/`): a bare
/// `harness/` is not a valid prefix, which needs at least one path segment.
#[must_use]
pub fn storage_schema() -> Schema {
    let owner = Owner::new("harness").expect("`harness` is a valid owner");
    let mut schema = Schema::new(owner)
        .collection_with(
            SCHEMA_VERSION,
            crate::observability::DriverStatusStore::collection_spec(),
            ScopeClass::Agent,
        )
        .collection_with(
            SCHEMA_VERSION,
            crate::tool::discover::DriverEmbeddingCache::collection_spec(),
            ScopeClass::Agent,
        )
        .collection_with(SCHEMA_VERSION, attachment_stash_spec(), ScopeClass::Agent)
        .stream_with(
            JOURNAL,
            ScopeClass::Agent,
            SealPolicy::new(SealWhen::OnFinish).retaining_ms(JOURNAL_RETENTION_MS),
        );
    for (_, to) in steps::BLOB_PREFIXES {
        schema = schema.blobs(to, ScopeClass::Agent);
    }
    schema = schema
        .step(steps::run_status())
        .step(steps::embedding_cache())
        .step(steps::attachment_stash())
        .step(steps::run_journal());
    for (index, (from, to)) in steps::BLOB_PREFIXES.into_iter().enumerate() {
        schema = schema.step(steps::blobs(BLOB_STEP_IDS[index], from, to));
    }
    schema
}

/// Collections owned by the harness; callers declare these on their chosen backend.
#[must_use]
pub fn collections() -> Vec<CollectionSpec> {
    storage_schema().collections()
}

/// Open harness storage for one scope, declaring its collections and applying
/// the harness-owned migrations before returning the scope-bound ports.
pub async fn open_storage(
    storage: ScopedStorage,
) -> tinyagents_storage_kit::error::Result<ScopedStorage> {
    let owned_storage = storage.clone();
    tinyagents_storage_kit::with_migration_lease(&storage, "harness", move |lease| async move {
        let storage = owned_storage;
        let registry = Registry::compose([storage_schema()])?;
        for spec in collections() {
            storage.documents().ensure_collection(&spec).await?;
        }
        storage
            .documents()
            .ensure_collection(&tinyagents_storage_kit::primitives::kv_collection_spec())
            .await?;
        registry.migrate(&storage, &MigrateOptions::new()).await?;
        lease.renew().await?;
        for (collection, kind) in [
            ("harness_store", LegacyKv::Flat),
            ("harness_namespaced_store", LegacyKv::Namespaced),
            ("harness_response_cache", LegacyKv::ResponseCache),
        ] {
            import_legacy_kv_with_lease(&storage, "harness", collection, kind, &lease).await?;
        }
        Ok(storage)
    })
    .await
}

const BLOB_STEP_IDS: [&str; 5] = [
    "attachment_blobs_v2",
    "tool_result_blobs_v2",
    "output_blobs_v2",
    "workspace_blobs_v2",
    "generated_media_blobs_v2",
];

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
