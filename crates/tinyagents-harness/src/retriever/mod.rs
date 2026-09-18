//! Generic retrieval contracts used by agent context composition.
//!
//! The harness does not construct embedding providers or assign memory scope.
//! Those are host concerns.  It provides only the narrow request/result/trait
//! seam needed to inject ranked context into a prompt.  Embedding callers use
//! the direct `tinyinference_embeddings` 0.3 types; no local embedding facade
//! is introduced here.

mod types;

pub use types::*;

#[cfg(test)]
mod test;
