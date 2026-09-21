//! On-demand tool discovery: keeping deferred tool schemas out of the context
//! window until a model asks for them.
//!
//! See `README.md` in this directory for the design and the cache rationale.
//! The pieces:
//!
//! - [`DeferredCatalog`] — the run's deferred schemas, BM25-indexed, ranked
//!   by the host's [`tinytools::ToolRanker`] when one is installed.
//! - [`ToolDiscoveryPolicy`] — the knobs, carried on
//!   [`crate::runtime::RunPolicy::discovery`], including the ranker and the
//!   [`DiscoveryRankMode`] that says how it is used.
//! - [`bridge_schemas`] / [`answer_tool_search`] / [`unwrap_tool_call`] — the
//!   two intrinsic bridge tools the agent loop advertises and answers.
//! - [`render_manifest`] — the budgeted listing inside `tool_search`'s
//!   description.

mod bridge;
mod manifest;
mod types;

pub use bridge::{
    SearchAnswer, TOOL_CALL_NAME, TOOL_SEARCH_NAME, answer_tool_search, bridge_schemas,
    unwrap_tool_call,
};
// The BM25 arithmetic lives in `tinytools::rank` now, so a host ranks with the
// same index the bridge does; the old paths keep resolving.
pub use tinytools::rank::{Bm25Index, tokenize};
pub use manifest::{MANIFEST_DESCRIPTION_CHARS, first_sentence, render_manifest};
pub use types::{
    DeferredCatalog, DeferredTool, DiscoveryRankMode, RankedSearch, ToolDiscoveryPolicy,
};

#[cfg(test)]
mod test;
