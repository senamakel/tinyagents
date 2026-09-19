//! Session-persistence test building blocks.
//!
//! This is the session-layer counterpart to the harness
//! [`testkit`](tinyagents_harness::testkit) and the graph
//! [`testkit`](../../tinyagents_graph/testkit/index.html): reusable doubles and
//! **conformance** (contract) suites so a downstream backend author can prove
//! their implementation behaves like the bundled ones, and the bundled ones
//! can be pinned against each other.
//!
//! # Conformance suites
//!
//! See [`conformance`] for the run-ledger and transcript-history contract
//! suites. Call them from a `#[test]` against any implementation:
//!
//! ```rust
//! use tinyagents_session::testkit::conformance::run_ledger_conformance;
//!
//! let dir = tempfile::tempdir().unwrap();
//! run_ledger_conformance(dir.path());
//! ```
//!
//! # In-memory transcript double
//!
//! [`InMemoryTranscriptHistory`] implements
//! [`TranscriptHistory`](crate::transcript::history::TranscriptHistory) purely
//! in memory, so the transcript-history conformance suite (and any test that
//! needs a cheap stand-in for a real `session_raw/*.jsonl` file) does not have
//! to touch a filesystem.

pub mod conformance;

mod in_memory_transcript;
pub use in_memory_transcript::InMemoryTranscriptHistory;
