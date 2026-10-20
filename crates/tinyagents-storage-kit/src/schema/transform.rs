//! Version-one declarations and transform request/response data.
use crate::{Item, Owner, SourceRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An owner's stored items and explicitly named versioned transforms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaDeclaration {
    /// Owner of every declared item.
    pub owner: Owner,
    /// Stored items, in declaration order.
    pub items: Vec<Item>,
    /// Transform steps, in execution order.
    pub steps: Vec<StepDeclaration>,
}

/// A named transform bringing one item between schema versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDeclaration {
    /// Stable progress identifier within the item.
    pub id: String,
    /// Destination item name or prefix.
    pub item: String,
    /// Input schema version.
    pub from: u32,
    /// Output schema version.
    pub to: u32,
    /// Legacy source collection, stream prefix, or blob prefix.
    pub source: Option<String>,
    /// The owner-provided transform operation.
    pub transform: TransformDescriptor,
}

/// Identifies a transform implementation without embedding executable code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformDescriptor {
    /// Owner-defined operation name.
    pub operation: String,
    /// Input and output record family.
    pub kind: TransformKind,
    /// Operation contract version; the initial wire contract is version 1.
    pub version: u32,
}

/// Record family accepted by a transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TransformKind {
    /// A batch of document records.
    Documents,
    /// One complete ordered stream.
    Stream,
    /// One opaque byte sequence.
    Blob,
}

/// A document batch passed to an owner-defined transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentsTransformRequest {
    /// Operation from the step descriptor.
    pub operation: String,
    /// Operation contract version.
    pub version: u32,
    /// Stored source records, including original metadata.
    pub records: Vec<SourceRecord>,
}

/// One stream passed to an owner-defined transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamTransformRequest {
    /// Operation from the step descriptor.
    pub operation: String,
    /// Operation contract version.
    pub version: u32,
    /// Source stream name.
    pub source: String,
    /// Retained entries in their original order.
    pub entries: Vec<Value>,
}

/// Replacement stream; an optional output of `None` means skip/drop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamTransformOutput {
    /// Destination stream name.
    pub target: String,
    /// Replacement entries in append order.
    pub entries: Vec<Value>,
}

/// One blob passed to an owner-defined transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobTransformRequest {
    /// Operation from the step descriptor.
    pub operation: String,
    /// Operation contract version.
    pub version: u32,
    /// Source blob key.
    pub source: String,
    /// Original media type, if known.
    pub content_type: Option<String>,
    /// Original bytes; no UTF-8 assumption is made.
    pub bytes: Vec<u8>,
}

/// Replacement blob; an optional output of `None` means skip/drop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobTransformOutput {
    /// Destination blob key.
    pub target: String,
    /// Replacement media type, if known.
    pub content_type: Option<String>,
    /// Replacement bytes.
    pub bytes: Vec<u8>,
}

/// Portable transform failure category, independent of backend errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TransformErrorKind {
    /// The input cannot be transformed.
    InvalidInput,
    /// The owner does not recognize the operation.
    UnsupportedOperation,
    /// The owner does not support this operation version.
    UnsupportedVersion,
}

/// Wire-safe transform failure with no backend source or executable state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformError {
    /// Failure category a caller may branch on.
    pub kind: TransformErrorKind,
    /// Safe diagnostic text supplied by the transform owner.
    pub message: String,
}

/// Transform success or portable failure; optional successes distinguish skips.
pub type TransformResult<T> = std::result::Result<T, TransformError>;
