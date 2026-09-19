//! Generic retrieval contracts used by agent context composition.
//!
//! The harness does not construct embedding providers or assign memory scope.
//! Those are host concerns.  It provides only the narrow request/result/trait
//! seam needed to inject ranked context into a prompt.  Embedding callers use
//! the direct `tinyinference_embeddings` 0.3 types; no local embedding facade
//! is introduced here.

mod types;

pub use types::*;

/// Retrieves ranked context and returns one caller-insertable prompt section.
///
/// The caller owns source policy and where the returned section is placed; this
/// function only preserves retriever order and uses the request's cancellation
/// and limit unchanged.
pub async fn compose_retrieval_context(
    retriever: &dyn Retriever,
    request: RetrievalRequest,
    section_name: impl Into<String>,
) -> crate::Result<crate::prompt::PromptSection> {
    let documents = retriever.retrieve(request).await?;
    Ok(crate::prompt::PromptSection::new(
        section_name,
        crate::prompt::render_retrieved_documents(&documents),
    ))
}

#[cfg(test)]
mod test;
