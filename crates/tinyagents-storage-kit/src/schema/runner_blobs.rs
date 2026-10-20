//! The blob step runner: each legacy blob is read, mapped by the step,
//! written under its new key and only then deleted.
//!
//! The stores are their own progress. A blob moves in two writes (put the
//! target, delete the source), so a crash between them leaves both; the next
//! run lists the source again and finds the target present. A present target
//! is the interrupted write only when its bytes and content type equal the
//! step's output; anything else (an unrelated blob, or the target of another
//! source that maps to the same key) is a `Conflict`: the target is never
//! overwritten and the source is kept, so no transformed bytes are discarded.
//! `tsd.migrations` gets the step's running count after every blob.

use crate::error::{Result, StorageError};

use super::migrations::Progress;
use super::prepared::Transform;
use super::runner::Context;
use super::types::BlobStepFn;

pub(super) async fn run(
    context: &Context<'_>,
    run: &Transform<BlobStepFn>,
    progress: Option<Progress>,
) -> Result<u64> {
    let step = context.step;
    let prefix = step.source.as_deref().unwrap_or_default();
    let blobs = context.storage.blobs();
    let mut progress = progress.unwrap_or_else(|| Progress {
        owner: context.owner.to_string(),
        step: step.id.clone(),
        from: step.from,
        to: step.to,
        cursor: None,
        written: 0,
        done: false,
    });
    for listed in blobs.list(prefix).await? {
        // Listed a moment ago and gone now: another runner moved it.
        let Some(blob) = blobs.get(&listed.key).await? else {
            continue;
        };
        let output = match run {
            Transform::Local(run) => run(listed.key.clone(), blob.meta.content_type, blob.bytes)?,
            Transform::External(descriptor) => super::executor::required(context.executor)?
                .blob(
                    context.owner,
                    super::BlobTransformRequest {
                        operation: descriptor.operation.clone(),
                        version: descriptor.version,
                        source: listed.key.clone(),
                        content_type: blob.meta.content_type,
                        bytes: blob.bytes,
                    },
                )
                .await?
                .map(|out| (out.target, out.content_type, out.bytes)),
        };
        if let Some((target, content_type, bytes)) = output {
            check_target(context, &target)?;
            write_target(context, &target, bytes, content_type).await?;
            progress.written += 1;
        }
        blobs.delete(&listed.key).await?;
        context.migrations.save(context.key, &progress).await?;
    }
    progress.done = true;
    context.migrations.save(context.key, &progress).await?;
    Ok(progress.written)
}

/// Put the step's output under `target`, or accept an equal blob that an
/// interrupted run left there. A different blob is a `Conflict`.
async fn write_target(
    context: &Context<'_>,
    target: &str,
    bytes: Vec<u8>,
    content_type: Option<String>,
) -> Result<()> {
    let blobs = context.storage.blobs();
    match blobs.get(target).await? {
        None => {
            blobs.put(target, bytes, content_type.as_deref()).await?;
            Ok(())
        }
        Some(existing) if existing.bytes == bytes && existing.meta.content_type == content_type => {
            Ok(())
        }
        Some(_) => Err(StorageError::conflict(format!(
            "step `{}` of `{}` maps a blob to `{target}`, which already holds different content; the source is kept",
            context.step.id, context.step.item
        ))),
    }
}

/// The target key must lie under the item's prefix.
fn check_target(context: &Context<'_>, target: &str) -> Result<()> {
    let item = context.step.item.as_str();
    if !target.starts_with(item) {
        return Err(StorageError::invalid_input(format!(
            "step `{}` of `{item}` mapped a blob to `{target}`, which is not under `{item}`",
            context.step.id
        )));
    }
    crate::blob::validate_user_blob_key(target)
}
