//! Durable session history and run ledger.
//!
//! Everything this crate persists goes through the `tinystoragedrivers` ports:
//! session history (sessions, messages, tool calls, with full-text search), a
//! [`run_ledger`] for background agent and workflow execution state, the
//! transcripts, turn-state snapshots and chat
//! threads. A host selects its backend and supplies storage already bound to
//! each authorized agent identity to [`DriverSessionStores`]. This is the
//! runtime's *history* layer: what ran, what it cost, what it called, and how
//! runs nest.
//!
//! # Why this is a top-level module
//!
//! Session history is a persistence domain in its own right, not a part of the
//! agent loop. Nothing in `tinyagents_harness` reads from it, and a host can use
//! it without running a harness at all — indexing sessions produced elsewhere,
//! or recovering orchestration state at boot before any agent exists. Filing it
//! under `harness::` would imply a dependency that does not exist in either
//! direction.
//!
//! # Relationship to the other persistence layers
//!
//! - `tinyagents_harness::store` is namespaced key-value storage for live
//!   runtime data. It is a substrate runs read and write during execution.
//! - `tinyagents_graph::checkpoint` is durability for *resuming* an interrupted
//!   graph run.
//! - This crate is queryable history. Nothing resumes from it; it answers
//!   "what happened", supports cross-session search, and lets a host recover
//!   orchestration state after a restart.
//!
//! # Session history and the run ledger
//!
//! [`run_ledger::DriverRunLedger`] is the one implementation of both
//! [`run_ledger::RunLedgerStore`] (runs, workflows, teams and tasks with atomic
//! claim/complete/shutdown, events, telemetry, tool effects, entry trees,
//! retention) and [`run_ledger::SessionHistoryStore`] (the session, message and
//! tool-call history and its full-text index). Retention, trimming and
//! `reindex_fts` bound and repair exactly the records those writers produce.
//!
//! # Chat threads
//!
//! [`threads`] is the product-facing chat log: threads and messages with a
//! cross-thread inverted index. See `src/threads/README.md`.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use tinyagents_session::run_ledger::{
//!     DriverRunLedger, NewSessionMessage, SessionEnd, SessionHistoryStore, SessionStart,
//! };
//! use tinyagents_session::SessionStatus;
//! use tinystoragedrivers_api::StorageBackend;
//! use tinystoragedrivers_testkit::memory_backend;
//! use tinyagents_storage_kit::ScopedStorage;
//! use tinystoragedrivers_types::Scope;
//!
//! # tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(async {
//! let scope = Scope::new("local")?;
//! let storage = memory_backend().for_scope(&scope)?;
//! let ledger = DriverRunLedger::new(ScopedStorage::new(storage, scope));
//!
//! ledger
//!     .record_session_start(SessionStart {
//!         id: "sess-1".into(),
//!         agent_definition_id: "researcher".into(),
//!         agent_definition_name: "Researcher".into(),
//!         session_key: "sess-1".into(),
//!         ..Default::default()
//!     })
//!     .await?;
//! ledger
//!     .record_message(NewSessionMessage {
//!         session_id: "sess-1".into(),
//!         role: "user".into(),
//!         content: "summarize the repo".into(),
//!         ..Default::default()
//!     })
//!     .await?;
//! ledger
//!     .record_session_end(
//!         "sess-1",
//!         SessionEnd {
//!             status: SessionStatus::Completed,
//!             turn_count: 1,
//!             input_tokens: 120,
//!             output_tokens: 340,
//!             cached_input_tokens: 0,
//!             cost_usd: 0.004,
//!         },
//!     )
//!     .await?;
//! # let _ = Arc::new(());
//! # Ok::<(), tinyagents_session::LedgerError>(())
//! # }).unwrap();
//! ```
//!
//! See [`README.md`](./README.md) for the layout, the FTS behaviour, and the
//! coordination guarantees.

pub mod fts;
pub mod port;
pub mod retention;
pub mod run_ledger;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod threads;
pub mod transcript;
pub mod turn_state;
pub mod types;

pub use tinyagents_harness::error::{Result, TinyAgentsError};

pub use fts::{DEFAULT_FTS_SNIPPET_BYTES, fts_snippet_bytes, set_fts_snippet_bytes};
pub use port::{AgentStores, SessionStoreProvider, TurnStates};
pub use port::{
    DriverSessionStores, DriverTranscriptHistory, DriverTranscriptLocator, DriverTurnStates,
    RebindableSessionStores, SessionStoreRegistry, collections, open_agent_storage, storage_schema,
};
#[cfg(any(test, feature = "testkit"))]
pub use port::{InMemorySessionStores, InMemoryTranscriptLocator, InMemoryTurnStates};
pub use retention::RetentionReport;
pub use run_ledger::DriverRunLedger;
pub use run_ledger::command_center::{CommandCenterView, ControlError, ControlVerb, build_view};
pub use run_ledger::drivers::{
    history_storage_schema, import_legacy_ledger, ledger_storage_schema,
};
pub use run_ledger::{LedgerError, LedgerResult, RunLedgerStore, SessionHistoryStore};
pub use threads::{
    ConversationPurgeStats, ConversationThread, CreateConversationThread, CrossThreadHit,
    ThreadMessage, ThreadMessagePatch, conversations_storage_schema,
};
pub use transcript::spend::{ThreadSpend, TranscriptSpend, thread_spend_in, transcript_spend};
pub use turn_state::TurnStateMirror;
pub use types::{
    SessionMessage, SessionRecord, SessionSearchParams, SessionSearchResult, SessionStatus,
    SessionToolCall,
};
