//! The two bridge tools the agent loop answers itself: `tool_search` and
//! `tool_call`.
//!
//! Neither is a registered [`tinytools::Tool`]. They are intrinsic to the
//! loop so they can read the run's deferred catalogue and, for `tool_call`,
//! be unwrapped *before* admission — every `before_tool` hook, allow-list,
//! policy middleware, and host authorization gate then sees the real tool
//! name and arguments, exactly as if the model had called it directly. A
//! host that registers its own tool under either name keeps it: registry
//! lookups win over the intrinsic answer.

use serde_json::{Value, json};
use tinyinference_llm::tool::{ToolFormat, ToolSchema};
use tinytools::{RankContext, ToolResult};

use super::manifest::render_manifest;
use super::types::{DeferredCatalog, RankedSearch, ToolDiscoveryPolicy};

/// Name of the intrinsic search bridge.
pub const TOOL_SEARCH_NAME: &str = "tool_search";
/// Name of the intrinsic call bridge.
pub const TOOL_CALL_NAME: &str = "tool_call";

/// Longest `description` returned per search hit. The full schema is what the
/// model needs to call the tool; the prose only needs to confirm the match.
const HIT_DESCRIPTION_CHARS: usize = 500;

/// The bridge schemas for a run, in the order they are appended to the
/// request: `tool_search` then `tool_call`.
#[must_use]
pub fn bridge_schemas(catalog: &DeferredCatalog, policy: &ToolDiscoveryPolicy) -> [ToolSchema; 2] {
    [tool_search_schema(catalog, policy), tool_call_schema()]
}

fn tool_search_schema(catalog: &DeferredCatalog, policy: &ToolDiscoveryPolicy) -> ToolSchema {
    let manifest = render_manifest(catalog, policy.manifest_token_budget);
    // Normalized, not the raw policy fields: a misconfigured `max_limit: 0`
    // must not advertise `"minimum": 1, "maximum": 0`, and the advertised
    // default must never exceed the advertised maximum. `answer_tool_search`
    // clamps against this same pair.
    let (default_limit, max_limit) = policy.effective_limits();
    ToolSchema {
        name: TOOL_SEARCH_NAME.to_string(),
        description: format!(
            "Find a tool that is not in your tool list. Not every capability is \
             advertised up front; describe what you need in plain words and this \
             returns the matching tools with their full argument schemas. Invoke a \
             match with `{TOOL_CALL_NAME}` (or by its own name). Use it before \
             telling the user something is impossible.\n\n{manifest}"
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you need to do, in plain words."
                },
                "limit": {
                    "type": "integer",
                    "description": format!(
                        "How many matches to return (default {default_limit}, max {max_limit})."
                    ),
                    "minimum": 1,
                    "maximum": max_limit
                }
            },
            "required": ["query"]
        }),
        format: ToolFormat::Json,
    }
}

/// `arguments` is declared as a **JSON-encoded string**, not an object.
///
/// Its shape depends on whichever tool the search returned, so as an object it
/// could only be declared open-ended (`{"type": "object"}` with no
/// `properties`). Providers that constrain decoding to the schema read that as
/// "no keys allowed": OpenRouter's Sail Research route for DeepSeek V4 Flash
/// answered every `tool_call` with `{}`, dropping `name` as well. Marking the
/// object open (`additionalProperties: true`) fixes that route, but the strict
/// sanitizer rewrites it to `false` and the conservative and Gemini
/// projections strip it, so the failure returns on those routes. A string
/// survives every projection. [`unwrap_tool_call`] still accepts an object
/// from models that send one anyway.
fn tool_call_schema() -> ToolSchema {
    ToolSchema {
        name: TOOL_CALL_NAME.to_string(),
        description: format!(
            "Invoke a tool found with `{TOOL_SEARCH_NAME}`. `name` is the tool's name \
             and `arguments` is its argument object encoded as a JSON string, matching \
             the schema the search returned."
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact name of the tool to invoke."
                },
                "arguments": {
                    "type": "string",
                    "description": "That tool's arguments as a JSON object string, per its \
                                    schema, e.g. \"{\\\"path\\\":\\\"a.pdf\\\"}\". Use \"{}\" for none."
                }
            },
            "required": ["name", "arguments"]
        }),
        format: ToolFormat::Json,
    }
}

/// What a `tool_search` produced: the result to hand the model plus the
/// facts the loop reports in its `ToolSearched` event.
#[derive(Debug)]
pub struct SearchAnswer {
    /// The tool result the model sees.
    pub result: ToolResult,
    /// How many tools it named.
    pub matched: usize,
    /// Names whose typed declarations should be offered on the next model call.
    pub matched_names: Vec<String>,
    /// Which ranker's answer was served, and how it went. `None` when the
    /// query was rejected before ranking.
    pub ranking: Option<RankedSearch>,
}

/// Answers a `tool_search` call against the run's catalogue.
///
/// Ranks as `policy` says — the host ranker when one is active, BM25
/// otherwise or on failure — and returns the full schema of every hit so
/// the model can call it.
pub async fn answer_tool_search(
    catalog: &DeferredCatalog,
    policy: &ToolDiscoveryPolicy,
    arguments: &Value,
) -> SearchAnswer {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if query.is_empty() {
        return SearchAnswer {
            result: ToolResult::error(format!(
                "`{TOOL_SEARCH_NAME}` needs a `query` describing what you want to do."
            )),
            matched: 0,
            matched_names: Vec::new(),
            ranking: None,
        };
    }
    let (default_limit, max_limit) = policy.effective_limits();
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(default_limit, |n| {
            usize::try_from(n).unwrap_or(usize::MAX).clamp(1, max_limit)
        });

    let ranking = catalog
        .rank(policy, query, &RankContext::empty(), limit)
        .await;
    let matches: Vec<&ToolSchema> = ranking
        .names
        .iter()
        .filter_map(|name| catalog.get(name))
        .collect();
    if matches.is_empty() {
        return SearchAnswer {
            result: ToolResult::success(format!(
                "No deferred tool matches \"{query}\". {} tool(s) are searchable; everything \
                 else you can use is already in your tool list.",
                catalog.len()
            )),
            matched: 0,
            matched_names: Vec::new(),
            ranking: Some(ranking),
        };
    }
    let payload: Vec<Value> = matches
        .iter()
        .map(|schema| {
            json!({
                "name": schema.name,
                "description": clip(&schema.description, HIT_DESCRIPTION_CHARS),
                "parameters": schema.parameters,
            })
        })
        .collect();
    let rendered = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "[]".to_string());
    let matched = payload.len();
    SearchAnswer {
        result: ToolResult::success(format!(
            "{matched} match(es). Invoke one with `{TOOL_CALL_NAME}` {{\"name\", \"arguments\"}} \
             (`arguments` as a JSON object string) or by its own name, using the parameters \
             shown.\n{rendered}"
        )),
        matched,
        matched_names: matches.iter().map(|schema| schema.name.clone()).collect(),
        ranking: Some(ranking),
    }
}

/// Unwraps a `tool_call` payload into the real `(name, arguments)` pair.
///
/// Returns the message to answer the model with when the payload is
/// malformed. `arguments` is advertised as a JSON string (see
/// [`tool_call_schema`]) but an object is accepted too, and it defaults to an
/// empty object when omitted so a zero-argument tool is callable without
/// ceremony.
pub fn unwrap_tool_call(arguments: &Value) -> Result<(String, Value), String> {
    let name = arguments
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            format!("`{TOOL_CALL_NAME}` needs a non-empty string `name` naming the tool to invoke.")
        })?;
    let inner = match arguments.get("arguments") {
        None | Some(Value::Null) => json!({}),
        Some(Value::Object(object)) => Value::Object(object.clone()),
        Some(Value::String(raw)) => serde_json::from_str::<Value>(raw)
            .ok()
            .filter(Value::is_object)
            .ok_or_else(|| {
                format!(
                    "`{TOOL_CALL_NAME}.arguments` must be a JSON object (got a string that is \
                     not one)."
                )
            })?,
        Some(_) => {
            return Err(format!(
                "`{TOOL_CALL_NAME}.arguments` must be a JSON object."
            ));
        }
    };
    Ok((name.to_string(), inner))
}

fn clip(text: &str, max_chars: usize) -> String {
    let mut clipped: String = text.chars().take(max_chars).collect();
    if clipped.len() < text.len() {
        clipped.push('…');
    }
    clipped
}
