//! Types for on-demand tool discovery.

use tinyinference_llm::tool::ToolSchema;

use super::index::Bm25Index;

/// How the agent loop exposes [`tinytools::ToolExposure::Deferred`] tools.
///
/// Deferred tools are never part of a request's `tools` array. When a run has
/// at least one, the loop appends two small bridge tools instead —
/// `tool_search` (find a deferred tool by describing what you need) and
/// `tool_call` (invoke one by name) — and answers both itself. The `tools`
/// array therefore stays byte-identical for the whole run, which is what a
/// provider prompt cache keys on; revealing a schema costs one tool result,
/// not a cache miss on every later turn.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolDiscoveryPolicy {
    /// Whether the bridge tools are offered at all.
    ///
    /// When `false`, deferred tools are simply absent from the run: neither
    /// advertised nor searchable, though a direct call by name still resolves
    /// (deferral only ever subtracts from what a host registered). Defaults
    /// to `true`.
    pub enabled: bool,
    /// Token budget (at ~4 bytes per token) for the manifest of deferred tools
    /// embedded in `tool_search`'s description. The manifest degrades from
    /// `name: first sentence` lines to names only to a bare count until it
    /// fits. Defaults to 4,000 tokens — the description is paid on every
    /// request, so this is the ceiling on what discovery itself costs.
    pub manifest_token_budget: usize,
    /// Matches returned by `tool_search` when the model does not say.
    pub default_limit: usize,
    /// Ceiling on the model-supplied `limit`, so one call cannot undo the
    /// saving by asking for everything.
    pub max_limit: usize,
}

impl Default for ToolDiscoveryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            manifest_token_budget: 4_000,
            default_limit: 5,
            max_limit: 20,
        }
    }
}

/// One deferred tool as the catalogue sees it.
#[derive(Clone, Debug)]
pub struct DeferredTool {
    /// The model-facing schema (host-injected arguments already projected out).
    pub schema: ToolSchema,
    /// `name` (also split into words) + description + top-level property
    /// names: the text the ranker sees.
    searchable: String,
}

impl DeferredTool {
    fn from_schema(schema: ToolSchema) -> Self {
        let mut searchable = String::with_capacity(schema.description.len() + 64);
        searchable.push_str(&schema.name);
        searchable.push(' ');
        searchable.push_str(&schema.name.replace('_', " "));
        searchable.push(' ');
        searchable.push_str(&schema.description);
        if let Some(properties) = schema
            .parameters
            .get("properties")
            .and_then(|value| value.as_object())
        {
            for key in properties.keys() {
                searchable.push(' ');
                searchable.push_str(key);
            }
        }
        Self { schema, searchable }
    }
}

/// The deferred tools of one run, indexed for `tool_search`.
///
/// Built once per run from the registry's deferred schemas (after the host's
/// allow-list is applied), sorted by name so both the manifest and search
/// output are deterministic.
#[derive(Clone, Debug, Default)]
pub struct DeferredCatalog {
    tools: Vec<DeferredTool>,
    index: Bm25Index,
}

impl DeferredCatalog {
    /// Indexes `schemas`, sorting them by name.
    #[must_use]
    pub fn build(mut schemas: Vec<ToolSchema>) -> Self {
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        let tools: Vec<DeferredTool> = schemas.into_iter().map(DeferredTool::from_schema).collect();
        let index = Bm25Index::build(
            tools
                .iter()
                .map(|tool| (tool.schema.name.as_str(), tool.searchable.as_str())),
        );
        Self { tools, index }
    }

    /// `true` when no tool is deferred.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Number of deferred tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Every deferred schema, sorted by name.
    pub fn schemas(&self) -> impl Iterator<Item = &ToolSchema> {
        self.tools.iter().map(|tool| &tool.schema)
    }

    /// Looks a deferred tool up by exact name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolSchema> {
        self.tools
            .binary_search_by(|tool| tool.schema.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tools[index].schema)
    }

    /// Ranks the catalogue against `query`, best first, at most `limit` hits.
    /// Only positively scored tools are returned.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<&ToolSchema> {
        self.index
            .search(query, limit)
            .into_iter()
            .map(|index| &self.tools[index].schema)
            .collect()
    }
}
