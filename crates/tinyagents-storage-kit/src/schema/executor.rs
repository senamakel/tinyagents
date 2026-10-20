//! Async host port for deterministic owner transforms; no transport is assumed.
use super::{
    BlobTransformOutput, BlobTransformRequest, DocumentsTransformRequest, Owner,
    StreamTransformOutput, StreamTransformRequest, TargetRecord,
};
use crate::{Result, StorageError};
use std::{future::Future, pin::Pin};

/// A sendable future borrowing its executor and owner.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Runs a named external transform. Implementations must return deterministic
/// complete outputs and must not mutate migration storage themselves.
pub trait TransformExecutor: Send + Sync {
    /// Transform one document page, including its original metadata.
    fn documents<'a>(
        &'a self,
        owner: &'a Owner,
        request: DocumentsTransformRequest,
    ) -> BoxFuture<'a, Result<Vec<TargetRecord>>>;
    /// Transform one complete retained stream.
    fn stream<'a>(
        &'a self,
        owner: &'a Owner,
        request: StreamTransformRequest,
    ) -> BoxFuture<'a, Result<Option<StreamTransformOutput>>>;
    /// Transform one complete blob.
    fn blob<'a>(
        &'a self,
        owner: &'a Owner,
        request: BlobTransformRequest,
    ) -> BoxFuture<'a, Result<Option<BlobTransformOutput>>>;
}

pub(super) fn required(executor: Option<&dyn TransformExecutor>) -> Result<&dyn TransformExecutor> {
    executor.ok_or_else(|| {
        StorageError::unavailable("external migration requires a transform executor")
    })
}
