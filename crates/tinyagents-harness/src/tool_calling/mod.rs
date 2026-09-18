//! Compatibility re-export for agent-facing tool-call protocols.
//!
//! New consumers can depend on `tinytools-agent` directly. The harness
//! keeps this module so existing `tinyagents_harness::tool_calling` imports do
//! not break.

pub use tinytools_agent::*;
