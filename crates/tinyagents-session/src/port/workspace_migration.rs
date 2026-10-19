//! Resumable copy from the former private workspace database.

use tinystoragedrivers_core::{
    CollectionSpec, DocumentStoreExt, Filter, Migrations, Precondition, Query,
};

use super::DriverSessionStores;

const OWNED_COLLECTIONS: [&str; 5] = [
    "session.transcripts",
    "session.transcript_entries",
    "session.turn_states",
    "tsd.kv",
    "tsd.migrations",
];
const JOURNAL_PREFIX: &str = "harness.journal/";
const STREAM_BATCH: usize = 256;
const IMPORT_OWNER: &str = "tinyagents_session_workspace";
const IMPORT_STEP: &str = "private_sqlite_v1";

/// Whether the former private store was already copied into this agent scope.
pub(super) fn import_is_complete(
    destination: &DriverSessionStores,
    agent_id: &str,
) -> anyhow::Result<bool> {
    let (destination_scope, _) = destination.scoped_for(agent_id)?;
    let complete = destination.bridge().run(async move {
        destination_scope
            .documents()
            .ensure_collection(&CollectionSpec::new("tsd.migrations"))
            .await?;
        Migrations::new(&destination_scope)
            .done(IMPORT_OWNER, IMPORT_STEP)
            .await
    })??;
    Ok(complete)
}

/// Copies only session-owned records through the storage ports. Documents
/// enter only when absent; an existing document must match its source or the
/// import fails without changing it. Existing journals must match the copied
/// prefix before a partial copy can resume; the source database stays intact.
pub(super) fn copy_agent_storage(
    source: &DriverSessionStores,
    destination: &DriverSessionStores,
    agent_id: &str,
) -> anyhow::Result<()> {
    let (source_scope, _) = source.scoped_for(agent_id)?;
    let (destination_scope, _) = destination.scoped_for(agent_id)?;
    source.bridge().run(async move {
        DriverSessionStores::prepare_schema(destination_scope.documents().as_ref()).await?;
        destination_scope
            .documents()
            .ensure_collection(&CollectionSpec::new("tsd.migrations"))
            .await?;
        let migrations = Migrations::new(&destination_scope);
        if migrations.done(IMPORT_OWNER, IMPORT_STEP).await? {
            return Ok(());
        }
        copy_documents(&source_scope, &destination_scope).await?;
        copy_journals(&source_scope, &destination_scope).await?;
        migrations.mark_done(IMPORT_OWNER, IMPORT_STEP).await
    })??;
    Ok(())
}

async fn copy_documents(
    source: &tinystoragedrivers_core::ScopedStorage,
    destination: &tinystoragedrivers_core::ScopedStorage,
) -> Result<(), tinystoragedrivers_core::StorageError> {
    let source_documents = source.documents();
    let destination_documents = destination.documents();
    for collection in OWNED_COLLECTIONS {
        let query = match collection {
            "tsd.kv" | "tsd.migrations" => Query::filter(Filter::eq("owner", "session")),
            _ => Query::all(),
        };
        for row in source_documents.query_all(collection, &query).await? {
            match destination_documents.get(collection, &row.id).await? {
                Some(existing) if existing.doc != row.doc => {
                    return Err(tinystoragedrivers_core::StorageError::backend(format!(
                        "cannot safely import {collection}/{}: destination document differs",
                        row.id
                    )));
                }
                Some(_) => {}
                None => {
                    destination_documents
                        .put(collection, &row.id, row.doc, Precondition::Absent)
                        .await?;
                }
            }
        }
    }
    Ok(())
}

async fn copy_journals(
    source: &tinystoragedrivers_core::ScopedStorage,
    destination: &tinystoragedrivers_core::ScopedStorage,
) -> Result<(), tinystoragedrivers_core::StorageError> {
    let source_streams = source.streams();
    let destination_streams = destination.streams();
    let source_names = source_streams.streams(JOURNAL_PREFIX).await?;
    for stream in &source_names {
        let Some(source_info) = source_streams.info(stream).await? else {
            continue;
        };
        let destination_info = destination_streams.info(stream).await?;
        let mut next = destination_info.as_ref().map_or(0, |found| found.len);
        if let Some(found) = destination_info {
            if found.base != 0 && found.base != source_info.base {
                return Err(storage_conflict(stream, "retained offsets differ"));
            }
            verify_copied_prefix(
                source_streams.as_ref(),
                destination_streams.as_ref(),
                stream,
                source_info.base,
                found.base,
                next,
                source_info.len,
            )
            .await?;
        }

        // A truncated stream retains its original next offset. Null entries
        // recreate only the discarded prefix, removed after retained events
        // have been copied. This preserves offsets and is safe to resume.
        while next < source_info.base {
            let count = (source_info.base - next).min(STREAM_BATCH as u64) as usize;
            destination_streams
                .append_batch_if_offset(stream, next, vec![serde_json::Value::Null; count])
                .await?;
            next += count as u64;
        }
        while next < source_info.len {
            let count = (source_info.len - next).min(STREAM_BATCH as u64) as usize;
            let entries = source_streams.read_window(stream, next, count).await?;
            if entries.len() != count
                || entries
                    .iter()
                    .enumerate()
                    .any(|(index, entry)| entry.offset != next + index as u64)
            {
                return Err(storage_conflict(
                    stream,
                    "source stream changed during copy",
                ));
            }
            let values = entries
                .into_iter()
                .map(|entry| entry.value)
                .collect::<Vec<_>>();
            let count = values.len() as u64;
            destination_streams
                .append_batch_if_offset(stream, next, values)
                .await?;
            next += count;
        }
        if source_info.base > 0 {
            destination_streams
                .truncate_before(stream, source_info.base)
                .await?;
        }
        if source_info.is_sealed() && next == source_info.len {
            destination_streams.seal(stream).await?;
        }
        if source_streams.info(stream).await? != Some(source_info) {
            return Err(storage_conflict(
                stream,
                "source stream changed during copy",
            ));
        }
    }
    if source_streams.streams(JOURNAL_PREFIX).await? != source_names {
        return Err(tinystoragedrivers_core::StorageError::backend(
            "cannot safely import journals: source stream set changed during copy",
        ));
    }
    Ok(())
}

async fn verify_copied_prefix(
    source: &dyn tinystoragedrivers_core::StreamStore,
    destination: &dyn tinystoragedrivers_core::StreamStore,
    stream: &str,
    source_base: u64,
    destination_base: u64,
    destination_len: u64,
    source_len: u64,
) -> Result<(), tinystoragedrivers_core::StorageError> {
    if destination_len > 0 && source_base > 0 && destination_base == 0 {
        let placeholder_end = destination_len.min(source_base);
        let mut offset = 0;
        while offset < placeholder_end {
            let count = (placeholder_end - offset).min(STREAM_BATCH as u64) as usize;
            let placeholders = destination.read_window(stream, offset, count).await?;
            if placeholders.len() != count
                || placeholders.iter().enumerate().any(|(index, entry)| {
                    entry.offset != offset + index as u64 || !entry.value.is_null()
                })
            {
                return Err(storage_conflict(stream, "destination prefix differs"));
            }
            offset += count as u64;
        }
    }
    if destination_len > source_base {
        let overlap_end = destination_len.min(source_len);
        let mut offset = source_base;
        while offset < overlap_end {
            let count = (overlap_end - offset).min(STREAM_BATCH as u64) as usize;
            let old = source.read_window(stream, offset, count).await?;
            let copied = destination.read_window(stream, offset, count).await?;
            if old.len() != count
                || copied.len() != count
                || old
                    .iter()
                    .enumerate()
                    .any(|(index, entry)| entry.offset != offset + index as u64)
                || copied
                    .iter()
                    .enumerate()
                    .any(|(index, entry)| entry.offset != offset + index as u64)
                || old != copied
            {
                return Err(storage_conflict(stream, "destination prefix differs"));
            }
            offset += count as u64;
        }
    } else if destination_len > 0 && source_base == 0 {
        let mut offset = 0;
        while offset < destination_len {
            let count = (destination_len - offset).min(STREAM_BATCH as u64) as usize;
            let placeholders = destination.read_window(stream, offset, count).await?;
            if placeholders.len() != count
                || placeholders.iter().enumerate().any(|(index, entry)| {
                    entry.offset != offset + index as u64 || !entry.value.is_null()
                })
            {
                return Err(storage_conflict(stream, "destination prefix differs"));
            }
            offset += count as u64;
        }
    }
    Ok(())
}

fn storage_conflict(stream: &str, reason: &str) -> tinystoragedrivers_core::StorageError {
    tinystoragedrivers_core::StorageError::backend(format!(
        "cannot safely import journal {stream}: {reason}"
    ))
}
