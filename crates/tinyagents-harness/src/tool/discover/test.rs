use serde_json::json;
use tinyinference_llm::tool::ToolSchema;

use super::*;

fn schema(name: &str, description: &str, properties: &[&str]) -> ToolSchema {
    let props: serde_json::Map<String, serde_json::Value> = properties
        .iter()
        .map(|key| ((*key).to_string(), json!({"type": "string"})))
        .collect();
    ToolSchema::new(
        name,
        description,
        json!({"type": "object", "properties": props}),
    )
}

fn catalog() -> DeferredCatalog {
    DeferredCatalog::build(vec![
        schema(
            "stock_quote",
            "Fetch the latest price for a ticker symbol. Returns bid, ask and volume.",
            &["symbol"],
        ),
        schema(
            "calendar_invite",
            "Send a calendar invite to one or more attendees.",
            &["attendees", "start", "end"],
        ),
        schema(
            "pdf_read",
            "Read the text of a PDF file on disk.",
            &["path"],
        ),
    ])
}

#[test]
fn tokenize_splits_identifiers_and_camel_case() {
    assert_eq!(
        tokenize("memory_hybrid_search readWorkflowResource v2"),
        vec![
            "memory", "hybrid", "search", "read", "workflow", "resource", "v2"
        ]
    );
}

#[test]
fn catalog_is_name_sorted_and_searchable_by_description() {
    let catalog = catalog();
    let names: Vec<_> = catalog.schemas().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["calendar_invite", "pdf_read", "stock_quote"]);
    assert!(catalog.get("pdf_read").is_some());
    assert!(catalog.get("nope").is_none());

    let hits = catalog.search("send an invite to attendees", 5);
    assert_eq!(hits[0].name, "calendar_invite");
    // Unrelated tools are not padded in.
    assert!(hits.iter().all(|s| s.name != "pdf_read"));
}

#[test]
fn search_matches_on_property_names_and_split_identifiers() {
    let catalog = catalog();
    let hits = catalog.search("ticker symbol", 5);
    assert_eq!(hits[0].name, "stock_quote");
    let hits = catalog.search("pdf", 5);
    assert_eq!(hits[0].name, "pdf_read");
}

#[test]
fn search_on_one_document_corpus_still_finds_it() {
    let catalog = DeferredCatalog::build(vec![schema("only_tool", "Provision a cluster.", &[])]);
    assert_eq!(catalog.search("provision a cluster", 3).len(), 1);
}

#[test]
fn manifest_degrades_full_to_names_to_count() {
    let catalog = catalog();
    let full = render_manifest(&catalog, 4_000);
    assert!(full.starts_with("3 deferred tool(s) are searchable:\n"));
    assert!(full.contains("- calendar_invite: Send a calendar invite to one or more attendees\n"));
    assert!(full.contains("- stock_quote: Fetch the latest price for a ticker symbol\n"));

    // Too small for descriptions, big enough for names.
    let names = render_manifest(&catalog, 20);
    assert!(names.contains("- calendar_invite\n"));
    assert!(!names.contains("Send a calendar"));

    // Too small for names, big enough for the bare count.
    let count = render_manifest(&catalog, 10);
    assert_eq!(count, "3 deferred tool(s) are searchable.\n");

    // Too small even for the count form: regression for the fallback that
    // used to return the count unconditionally, silently exceeding the
    // budget it was supposed to respect. The empty manifest itself always
    // respects any budget, including zero.
    assert_eq!(render_manifest(&catalog, 2), "");
    assert_eq!(render_manifest(&catalog, 0), "");
}

#[test]
fn first_sentence_clips_and_ignores_inline_dots() {
    assert_eq!(
        first_sentence("Read v1.2 files. Then more.", 60),
        "Read v1.2 files"
    );
    assert_eq!(
        first_sentence("No terminator here", 60),
        "No terminator here"
    );
    assert_eq!(first_sentence("abcdefghij", 4), "abcd…");
    assert_eq!(first_sentence("  spaced\n\nout  ", 60), "spaced out");
}

#[test]
fn bridge_schemas_are_search_then_call_and_embed_manifest() {
    let policy = ToolDiscoveryPolicy::default();
    let [search, call] = bridge_schemas(&catalog(), &policy);
    assert_eq!(search.name, TOOL_SEARCH_NAME);
    assert_eq!(call.name, TOOL_CALL_NAME);
    assert!(
        search
            .description
            .contains("- pdf_read: Read the text of a PDF file on disk")
    );
    assert_eq!(search.parameters["required"], json!(["query"]));
    assert_eq!(call.parameters["required"], json!(["name", "arguments"]));
    assert_eq!(
        search.parameters["properties"]["limit"]["maximum"],
        json!(20)
    );
}

#[test]
fn bridge_schemas_are_byte_stable_across_builds() {
    let policy = ToolDiscoveryPolicy::default();
    let a = serde_json::to_string(&bridge_schemas(&catalog(), &policy)).unwrap();
    let b = serde_json::to_string(&bridge_schemas(&catalog(), &policy)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn answer_tool_search_returns_full_schemas_for_hits() {
    let policy = ToolDiscoveryPolicy::default();
    let (result, matched) = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "read a pdf", "limit": 1}),
    );
    assert!(!result.is_error);
    assert_eq!(matched, 1);
    let text = result.text();
    assert!(text.starts_with("1 match(es)."));
    assert!(text.contains("\"name\": \"pdf_read\""));
    assert!(text.contains("\"path\""));
}

#[test]
fn answer_tool_search_clamps_limit_and_handles_misses() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 2,
        ..ToolDiscoveryPolicy::default()
    };
    let (_, matched) = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "pdf invite quote symbol attendees", "limit": 50}),
    );
    assert!(matched <= 2);

    let (result, matched) = answer_tool_search(&catalog(), &policy, &json!({"query": "zzzz qqqq"}));
    assert!(!result.is_error);
    assert_eq!(matched, 0);
    assert!(result.text().starts_with("No deferred tool matches"));

    let (result, _) = answer_tool_search(&catalog(), &policy, &json!({"query": "  "}));
    assert!(result.is_error);
}

/// Regression: `max_limit: 0` used to reach `usize::clamp(1, 0)`, which
/// panics because its minimum exceeds its maximum — a model-supplied numeric
/// `limit` could crash the process. It must instead clamp against a
/// normalized effective maximum of at least 1.
#[test]
fn answer_tool_search_does_not_panic_on_a_zero_max_limit() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 0,
        default_limit: 5,
        ..ToolDiscoveryPolicy::default()
    };
    let (result, matched) = answer_tool_search(
        &catalog(),
        &policy,
        &json!({"query": "pdf invite quote symbol attendees", "limit": 50}),
    );
    assert!(!result.is_error);
    assert!(matched <= 1, "effective max_limit must clamp to at least 1");
}

/// Regression: the `tool_search` schema advertised `"minimum": 1, "maximum":
/// policy.max_limit` verbatim, so `max_limit: 0` produced an inconsistent
/// (and provider-invalid) `minimum > maximum` pair, and a `default_limit`
/// above `max_limit` advertised a default outside the advertised bounds. Both
/// must be normalized before they reach the wire.
#[test]
fn tool_search_schema_normalizes_inconsistent_limits() {
    let policy = ToolDiscoveryPolicy {
        max_limit: 0,
        default_limit: 5,
        ..ToolDiscoveryPolicy::default()
    };
    let schemas = bridge_schemas(&catalog(), &policy);
    let limit = &schemas[0].parameters["properties"]["limit"];
    let minimum = limit["minimum"].as_u64().unwrap();
    let maximum = limit["maximum"].as_u64().unwrap();
    assert!(minimum <= maximum, "minimum must not exceed maximum");
    assert!(maximum >= 1);
}

#[test]
fn unwrap_tool_call_accepts_object_string_and_missing_arguments() {
    let (name, args) =
        unwrap_tool_call(&json!({"name": "pdf_read", "arguments": {"path": "a.pdf"}})).unwrap();
    assert_eq!(name, "pdf_read");
    assert_eq!(args, json!({"path": "a.pdf"}));

    let (_, args) =
        unwrap_tool_call(&json!({"name": "pdf_read", "arguments": "{\"path\":\"b\"}"})).unwrap();
    assert_eq!(args, json!({"path": "b"}));

    let (_, args) = unwrap_tool_call(&json!({"name": " pdf_read "})).unwrap();
    assert_eq!(args, json!({}));
}

#[test]
fn unwrap_tool_call_rejects_malformed_payloads() {
    assert!(unwrap_tool_call(&json!({"arguments": {}})).is_err());
    assert!(unwrap_tool_call(&json!({"name": ""})).is_err());
    assert!(unwrap_tool_call(&json!({"name": "x", "arguments": 3})).is_err());
    assert!(unwrap_tool_call(&json!({"name": "x", "arguments": "not json"})).is_err());
}
