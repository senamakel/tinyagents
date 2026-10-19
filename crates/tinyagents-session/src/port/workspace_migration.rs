//! Resumable copy from the former private workspace database.

use tinystoragedrivers_core::{CollectionSpec, DocumentStoreExt, Precondition, Query};

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

/// Copies only session-owned records through the storage ports. Documents
/// enter only when absent. Existing journals must match the copied prefix
/// before a partial copy can resume; the source database stays intact.
pub(super) fn copy_agent_storage(
    source: &DriverSessionStores,
    destination: &DriverSessionStores,
    agent_id: &str,
) -> anyhow::Result<()> {
    let (source_scope, _) = source.scoped_for(agent_id)?;
    let (destination_scope, _) = destination.scoped_for(agent_id)?;
    source.bridge().run(async move {
        copy_documents(&source_scope, &destination_scope).await?;
        copy_journals(&source_scope, &destination_scope).await
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
        destination_documents
            .ensure_collection(&CollectionSpec::new(collection))
            .await?;
        for row in source_documents
            .query_all(collection, &Query::all())
            .await?
        {
            if destination_documents
                .get(collection, &row.id)
                .await?
                .is_none()
            {
                destination_documents
                    .put(collection, &row.id, row.doc, Precondition::Absent)
                    .await?;
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
    for stream in source_streams.streams(JOURNAL_PREFIX).await? {
        let Some(source_info) = source_streams.info(&stream).await? else {
            continue;
        };
        let destination_info = destination_streams.info(&stream).await?;
        let mut next = destination_info.map_or(0, |found| found.len);
        if let Some(found) = destination_info {
            if found.base != 0 && found.base != source_info.base {
                return Err(storage_conflict(&stream, "retained offsets differ"));
            }
            verify_copied_prefix(
                source_streams.as_ref(),
                destination_streams.as_ref(),
                &stream,
                source_info.base,
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
                .append_batch_if_offset(&stream, next, vec![serde_json::Value::Null; count])
                .await?;
            next += count as u64;
        }
        while next < source_info.len {
            let entries = source_streams
                .read_window(&stream, next, STREAM_BATCH)
                .await?;
            if entries.is_empty() {
                return Err(storage_conflict(
                    &stream,
                    "source stream changed during copy",
                ));
            }
            let values = entries
                .into_iter()
                .map(|entry| entry.value)
                .collect::<Vec<_>>();
            let count = values.len() as u64;
            destination_streams
                .append_batch_if_offset(&stream, next, values)
                .await?;
            next += count;
        }
        if source_info.base > 0 {
            destination_streams
                .truncate_before(&stream, source_info.base)
                .await?;
        }
        if source_info.is_sealed() {
            destination_streams.seal(&stream).await?;
        }
    }
    Ok(())
}

async fn verify_copied_prefix(
    source: &dyn tinystoragedrivers_core::StreamStore,
    destination: &dyn tinystoragedrivers_core::StreamStore,
    stream: &str,
    source_base: u64,
    destination_len: u64,
    source_len: u64,
) -> Result<(), tinystoragedrivers_core::StorageError> {
    if destination_len > source_base {
        let overlap_end = destination_len.min(source_len);
        let count = (overlap_end - source_base) as usize;
        let old = source.read_window(stream, source_base, count).await?;
        let copied = destination.read_window(stream, source_base, count).await?;
        if old != copied {
            return Err(storage_conflict(stream, "destination prefix differs"));
        }
    } else if destination_len > 0 {
        let placeholders = destination
            .read_window(stream, 0, destination_len as usize)
            .await?;
        if placeholders.len() != destination_len as usize
            || placeholders.iter().any(|entry| !entry.value.is_null())
        {
            return Err(storage_conflict(stream, "destination prefix differs"));
        }
    }
    Ok(())
}

fn storage_conflict(stream: &str, reason: &str) -> tinystoragedrivers_core::StorageError {
    tinystoragedrivers_core::StorageError::backend(format!(
        "cannot safely import journal {stream}: {reason}"
    ))
}
