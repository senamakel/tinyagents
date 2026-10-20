//! The stream step runner: each legacy stream is read whole, mapped by the
//! step, written under its new name in batches, sealed when the source was,
//! and only then deleted.
//!
//! Progress lives in `tsd.migrations`, one document per source stream under
//! `Key::of([owner, item, step, source])`, holding the target name (`cursor`)
//! and the entries appended so far (`written`). It is saved before the first
//! append and after every batch, and deleted once the source is gone.
//!
//! A step function is pure, so a resumed run recomputes the target's entries
//! and appends the ones the target does not have yet. The target's own length
//! is the authority for where to continue: a crash between an append and its
//! progress save leaves the target one batch ahead of `written`, which is
//! fine. A target that is shorter than the recorded progress, longer than the
//! mapped entries, or non-empty with no recorded progress was not written by
//! this step and is refused with `Conflict`.
//!
//! Every append is conditional on the length the run observed
//! (`append_batch_if_offset`), so a second migrator working on the same stream
//! gets a `Conflict` rather than duplicating entries. Whether the source was
//! sealed is read from the stream backend (`StreamStore::info`), never from
//! the blob handle: a composite keeps sealed blobs with the records.

use crate::document::Precondition;
use crate::error::{ErrorKind, Result, StorageError};
use crate::key::Key;
use crate::stream::StreamStore;

use super::migrations::Progress;
use super::prepared::Transform;
use super::runner::Context;
use super::types::StreamStepFn;

pub(super) async fn run(
    context: &Context<'_>,
    run: &Transform<StreamStepFn>,
    progress: Option<Progress>,
) -> Result<u64> {
    let step = context.step;
    let prefix = step.source.as_deref().unwrap_or_default();
    let streams = context.storage.streams();
    let mut progress = progress.unwrap_or_else(|| Progress {
        owner: context.owner.to_string(),
        step: step.id.clone(),
        from: step.from,
        to: step.to,
        cursor: None,
        written: 0,
        done: false,
    });
    for source in streams.streams(prefix).await? {
        progress.written += migrate_stream(context, run, &source).await?;
        context.migrations.save(context.key, &progress).await?;
    }
    progress.done = true;
    context.migrations.save(context.key, &progress).await?;
    Ok(progress.written)
}

/// Move one source stream; returns the entries its target ends up holding.
async fn migrate_stream(
    context: &Context<'_>,
    run: &Transform<StreamStepFn>,
    source: &str,
) -> Result<u64> {
    let step = context.step;
    let streams = context.storage.streams();
    let entries = read_all(streams.as_ref(), source, context.batch_size).await?;
    let sealed = is_sealed(streams.as_ref(), source).await?;
    let source_key = Key::of([
        context.owner.as_str(),
        step.item.as_str(),
        step.id.as_str(),
        source,
    ])?;
    let output = match run {
        Transform::Local(run) => run(source.to_owned(), entries)?,
        Transform::External(descriptor) => super::executor::required(context.executor)?
            .stream(
                context.owner,
                super::StreamTransformRequest {
                    operation: descriptor.operation.clone(),
                    version: descriptor.version,
                    source: source.to_owned(),
                    entries,
                },
            )
            .await?
            .map(|out| (out.target, out.entries)),
    };
    let Some((target, values)) = output else {
        streams.delete_stream(source).await?;
        delete_progress(context, &source_key).await?;
        return Ok(0);
    };
    check_target(context, &target)?;
    let total = values.len() as u64;
    let mut have = streams.len(&target).await?;
    let recorded = context.migrations.load(&source_key).await?;
    match &recorded {
        None if have != 0 => {
            return Err(StorageError::conflict(format!(
                "step `{}` of `{}`: stream `{target}` already holds {have} entries that this step did not write; refusing to append to it",
                step.id, step.item
            )));
        }
        Some(found) if found.cursor.as_deref() != Some(target.as_str()) => {
            return Err(StorageError::conflict(format!(
                "step `{}` of `{}`: the recorded target of `{source}` differs from `{target}`; the step function is not deterministic",
                step.id, step.item
            )));
        }
        Some(found) if have < found.written || have > total => {
            return Err(StorageError::conflict(format!(
                "step `{}` of `{}`: stream `{target}` holds {have} entries but {} were recorded and {total} are expected",
                step.id, step.item, found.written
            )));
        }
        _ => {}
    }
    let mut progress = recorded.unwrap_or_else(|| Progress {
        owner: context.owner.to_string(),
        step: step.id.clone(),
        from: step.from,
        to: step.to,
        cursor: Some(target.clone()),
        written: 0,
        done: false,
    });
    if have == 0 {
        context.migrations.save(&source_key, &progress).await?;
    }
    let mut rest = values
        .into_iter()
        .skip(usize::try_from(have).unwrap_or(usize::MAX));
    loop {
        let batch: Vec<_> = rest.by_ref().take(context.batch_size).collect();
        if batch.is_empty() {
            break;
        }
        // Conditional on the length this run observed: a competing migrator
        // that got there first makes this a `Conflict` instead of a duplicate.
        let added = batch.len() as u64;
        streams
            .append_batch_if_offset(&target, have, batch)
            .await
            .map_err(|error| competing_runner(error, &step.id, &step.item, &target))?;
        have += added;
        progress.written = have;
        context.migrations.save(&source_key, &progress).await?;
    }
    if sealed && total > 0 {
        streams.seal(&target).await?;
    }
    streams.delete_stream(source).await?;
    delete_progress(context, &source_key).await?;
    Ok(total)
}

/// Whether the stream backend reports `stream` sealed. The backend is asked
/// rather than the blob handle, because a composite keeps sealed blobs with
/// the records. A driver that cannot seal has no sealed streams.
async fn is_sealed(streams: &dyn StreamStore, stream: &str) -> Result<bool> {
    match streams.info(stream).await {
        Ok(info) => Ok(info.is_some_and(|info| info.sealed_ms.is_some())),
        Err(error) if matches!(error.kind(), ErrorKind::Unsupported(_)) => Ok(false),
        Err(error) => Err(error),
    }
}

fn competing_runner(error: StorageError, step: &str, item: &str, target: &str) -> StorageError {
    if error.kind() != ErrorKind::Conflict {
        return error;
    }
    StorageError::conflict(format!(
        "step `{step}` of `{item}`: stream `{target}` changed under this run; another migrator is working on it"
    ))
}

/// The target must be the item itself or live under `item/`.
fn check_target(context: &Context<'_>, target: &str) -> Result<()> {
    let item = context.step.item.as_str();
    let inside = target == item
        || target
            .strip_prefix(item)
            .is_some_and(|rest| rest.starts_with('/'));
    if !inside {
        return Err(StorageError::invalid_input(format!(
            "step `{}` of `{item}` mapped a stream to `{target}`, which is not `{item}` or under `{item}/`",
            context.step.id
        )));
    }
    crate::stream::validate_stream(target)
}

async fn read_all(
    streams: &dyn StreamStore,
    stream: &str,
    page: usize,
) -> Result<Vec<serde_json::Value>> {
    let mut values = Vec::new();
    let mut from = 0;
    loop {
        let window = streams.read_window(stream, from, page).await?;
        let Some(last) = window.last() else {
            return Ok(values);
        };
        from = last.offset + 1;
        values.extend(window.into_iter().map(|entry| entry.value));
    }
}

async fn delete_progress(context: &Context<'_>, key: &Key) -> Result<()> {
    context
        .storage
        .documents()
        .delete(
            super::migrations::MIGRATIONS_COLLECTION,
            key.as_str(),
            Precondition::None,
        )
        .await
        .map(|_| ())
}
