//! Provider-neutral retrieval contracts.
//!
//! Embedding provider construction belongs to the host or directly to
//! `tinyinference-embeddings`.  This module only names the data an agent needs
//! after a host has decided what it is permitted to retrieve.

use async_trait::async_trait;
use serde_json::Value;

use crate::Result;
use crate::CancellationToken;

/// A retrieval request supplied by context composition.
#[derive(Clone, Debug, PartialEq)]
pub struct RetrievalRequest {
    /// Query text. Hosts are responsible for authorization and source scope
    /// before creating this request.
    pub query: String,
    /// Maximum number of ranked documents to return.
    pub limit: usize,
    /// Optional caller metadata, preserved for a host adapter but never
    /// interpreted by the generic contract.
    pub metadata: Value,
    /// Cooperative cancellation signal for expensive retrieval work.
    pub cancellation: CancellationToken,
}

impl RetrievalRequest {
    /// Creates a request with no host-specific metadata.
    pub fn new(query: impl Into<String>, limit: usize) -> Self {
        Self {
            query: query.into(),
            limit,
            metadata: Value::Null,
            cancellation: CancellationToken::new(),
        }
    }

    /// Attaches opaque host metadata.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Uses the caller's cancellation tree for retrieval work.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }
}

/// One ranked document returned by a [`Retriever`].
#[derive(Clone, Debug, PartialEq)]
pub struct RetrievedDocument {
    /// Stable document identifier, used for host-owned deduplication.
    pub id: String,
    /// Model-facing document text. The host decides redaction and byte caps.
    pub content: String,
    /// Relevance score in the retriever's documented ranking scale.
    pub score: f32,
    /// Source metadata retained for citations and host projection.
    pub metadata: Value,
}

impl RetrievedDocument {
    /// Creates a retrieval result with no source metadata.
    pub fn new(id: impl Into<String>, content: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            score,
            metadata: Value::Null,
        }
    }

    /// Attaches source metadata.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Provider- and storage-neutral ranked retrieval interface.
///
/// An implementation may use `tinyinference_embeddings` directly, a remote
/// vector store, or a host memory adapter.  Credentials, provider settings,
/// file access, and collection scope deliberately remain outside this trait.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Returns at most `request.limit` documents in descending relevance order.
    async fn retrieve(&self, request: RetrievalRequest) -> Result<Vec<RetrievedDocument>>;
}
