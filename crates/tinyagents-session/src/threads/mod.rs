//! Chat thread/message store over `tinystoragedrivers` ports.
//!
//! This is the *product-facing* chat log a UI lists and renders (threads with
//! titles, labels, message counts), plus a trigram / CJK-bigram inverted index
//! for cross-thread substring search. It is independent of the driver-backed
//! session history ([`crate::SessionHistoryStore`]) and of the model-facing
//! [`crate::transcript`]: nothing here touches `session_db/` or `session_raw/`.
//!
//! [`DriverConversationStore`] is the only implementation. The host supplies
//! its admitted storage backend; this crate never opens an engine or selects a
//! memory fallback.
//! The legacy API retained textual errors; the persistence contract preserves
//! typed failures. See `README.md` beside this file for the layout and design.
//!
//! ## Layout
//!
//! - `types` - the on-disk wire types (threads, messages, patches, hits).
//! - `tokenize` - multilingual normalization + character n-gram tokenizer.
//! - `inverted_index` - in-memory index over message content.
//! - `persistence` - the async [`ConversationPersistence`] contract.
//! - `driver` - the [`DriverConversationStore`] implementation.
//! - `dto` - the stored (v2) form of threads, messages and append intents.
//! - `schema` - [`conversations_storage_schema`], with the v1 -> v2 steps.

pub mod persistence;
pub use persistence::{
    ConversationError, ConversationPersistence, ConversationPurgeStats, ConversationResult,
    ConversationTurnClaim, ConversationTurnClaimOutcome, ConversationTurnClaimRequest,
};

mod dto;
mod inverted_index;
mod labels;
mod schema;
mod schema_steps;
mod tokenize;
mod types;

pub use schema::conversations_storage_schema;

pub use types::{
    ConversationThread, CreateConversationThread, CrossThreadHit, DETERMINISTIC_MESSAGE_ID_PREFIX,
    ThreadMessage, ThreadMessagePatch, is_deterministic_message_id, reply_run_id,
    run_reply_message_id,
};

#[cfg(test)]
#[path = "persistence_tests.rs"]
mod persistence_tests;

pub mod driver;
pub use driver::DriverConversationStore;
