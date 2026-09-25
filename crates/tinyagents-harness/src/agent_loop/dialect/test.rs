//! Focused tests for model-aware tool dialect selection.

use super::RunDialect;
use crate::config::ToolDispatcher;
use tinyinference_llm::model::ModelProfile;

#[test]
fn auto_uses_xml_when_the_model_disables_native_tool_calling() {
    let profile = ModelProfile {
        tool_calling: false,
        ..ModelProfile::default()
    };

    let dialect = RunDialect::resolve(ToolDispatcher::Auto, &[], Some(profile.tool_calling));

    assert!(matches!(dialect, RunDialect::Xml));
}

#[test]
fn text_dialect_keeps_latest_request_after_a_tool_search_result() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::ModelRequest;
    use tinyinference_llm::tool::ToolSchema;

    let tool = ToolSchema::new("tool_search", "find tools", serde_json::json!({}));
    let dialect = RunDialect::resolve(
        ToolDispatcher::Xml,
        std::slice::from_ref(&tool),
        Some(false),
    );
    let mut request = ModelRequest::new(vec![
        Message::system("system"),
        Message::user("hey"),
        Message::assistant("Hey! What's up?"),
        Message::user("fetch my latest email"),
        Message::assistant("searching"),
        Message::tool("search-1", "GMAIL_FETCH_EMAILS schema"),
    ])
    .with_tools(vec![tool]);

    dialect.apply_to_request(&mut request, true, &[]);

    assert!(
        request
            .messages
            .last()
            .unwrap()
            .text()
            .contains("GMAIL_FETCH_EMAILS")
    );
    assert!(
        request
            .messages
            .last()
            .unwrap()
            .text()
            .contains("fetch my latest email")
    );
}

#[test]
fn code_dialects_are_opt_in_and_share_the_positional_registry() {
    use tinyinference_llm::tool::ToolSchema;
    use tinytools_agent::dialect::CodeStyle;

    let schema = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    let tools = [schema];

    // Auto never picks a code dialect, even without native tool calling.
    assert!(matches!(
        RunDialect::resolve(ToolDispatcher::Auto, &tools, Some(false)),
        RunDialect::Xml
    ));

    let python = RunDialect::resolve(ToolDispatcher::Python, &tools, Some(false));
    let RunDialect::Code(style, registry) = &python else {
        panic!("expected a code dialect, got {python:?}");
    };
    assert_eq!(*style, CodeStyle::Python);
    assert!(registry.contains_key("lookup"));
    assert!(python.is_text());
    assert!(python.registry_for(&tools).is_some());

    let typescript = RunDialect::resolve(ToolDispatcher::Typescript, &tools, Some(true));
    assert!(matches!(
        typescript,
        RunDialect::Code(CodeStyle::TypeScript, _)
    ));
}

#[test]
fn a_host_that_renders_the_catalogue_gets_the_schemas_stripped_but_nothing_appended() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::{ModelRequest, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tools = vec![ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    )];
    let dialect = RunDialect::resolve(ToolDispatcher::Python, &tools, Some(true));
    let messages = vec![
        Message::system("host prompt with its own ## Tools block"),
        Message::user("hi"),
    ];

    // Default: the loop appends the protocol block and the catalogue.
    let mut appended = ModelRequest::new(messages.clone()).with_tools(tools.clone());
    dialect.apply_to_request(&mut appended, false, &[]);
    assert!(appended.tools.is_empty());
    let system = appended.messages[0].text();
    assert!(system.contains("def lookup("), "{system}");

    // Host-rendered: schemas still leave the wire, the prompt is untouched.
    let mut host = ModelRequest::new(messages.clone()).with_tools(tools.clone());
    dialect.apply_to_request(&mut host, true, &[]);
    assert!(host.tools.is_empty());
    assert_eq!(host.messages, messages);
    assert_eq!(host.tool_choice, ToolChoice::Auto);

    // A forced choice is the one thing the host could not have said.
    let mut forced = ModelRequest::new(messages).with_tools(tools);
    forced.tool_choice = ToolChoice::Tool("lookup".into());
    dialect.apply_to_request(&mut forced, true, &[]);
    let system = forced.messages[0].text();
    assert!(
        system.contains("You must call the `lookup` tool."),
        "{system}"
    );
    assert!(!system.contains("def lookup("));
    assert_eq!(forced.tool_choice, ToolChoice::Auto);
}

/// Every case above starts from a transcript that already has a leading
/// system message, so `prompt_tools::append_system_block`'s *other* branch —
/// inserting a brand-new leading message when none exists yet — is never
/// exercised. A regression there (failing to insert, inserting more than
/// one, or inserting it somewhere other than the front) would pass every
/// other test in this file undetected.
#[test]
fn a_run_with_no_leading_system_message_gets_exactly_one_synthesized_by_the_rewrite() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::{ModelRequest, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    let tools = vec![ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    )];
    let dialect = RunDialect::resolve(ToolDispatcher::Python, &tools, Some(true));
    let no_leading_system = vec![Message::user("hi")];

    // Ordinary rewrite: the loop's own protocol block and catalogue become
    // the sole, newly-inserted leading system message.
    let mut appended = ModelRequest::new(no_leading_system.clone()).with_tools(tools.clone());
    dialect.apply_to_request(&mut appended, false, &[]);
    assert_eq!(
        appended.messages.len(),
        2,
        "exactly one system message must be inserted, not folded into an \
         existing one or duplicated: {:?}",
        appended.messages
    );
    assert!(
        matches!(appended.messages[0], Message::System(_)),
        "the synthesized message must be the new leading one: {:?}",
        appended.messages
    );
    let system = appended.messages[0].text();
    assert!(system.contains("def lookup("), "{system}");
    assert_eq!(appended.messages[1], Message::user("hi"));

    // Host-rendered with a forced choice: the host had no prompt at all to
    // predate the synthesis, so the forced-choice sentence is what lands in
    // the newly-inserted message; the catalogue stays the host's job.
    let mut forced = ModelRequest::new(no_leading_system).with_tools(tools);
    forced.tool_choice = ToolChoice::Tool("lookup".into());
    dialect.apply_to_request(&mut forced, true, &[]);
    assert_eq!(forced.messages.len(), 2, "{:?}", forced.messages);
    assert!(matches!(forced.messages[0], Message::System(_)));
    let system = forced.messages[0].text();
    assert!(
        system.contains("You must call the `lookup` tool."),
        "{system}"
    );
    assert!(!system.contains("def lookup("));
    assert_eq!(forced.tool_choice, ToolChoice::Auto);
}

#[test]
fn a_host_that_renders_the_catalogue_still_learns_a_turn_synthesized_tool() {
    use tinyinference_llm::message::Message;
    use tinyinference_llm::model::{ModelRequest, ToolChoice};
    use tinyinference_llm::tool::ToolSchema;

    // The base tool the host's own static prompt already advertises.
    let base = ToolSchema::new(
        "lookup",
        "Looks something up.",
        serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        }),
    );
    // The structured-output fallback tool minted for this turn only, after
    // the host's prompt was already composed — see
    // `RunPolicy::host_renders_tool_catalogue` and
    // `StructuredStrategy::ToolCall`.
    let synthesized = ToolSchema::new(
        "emit_result",
        "Return the result as `emit_result`.",
        serde_json::json!({
            "type": "object",
            "properties": {"total": {"type": "number"}},
            "required": ["total"]
        }),
    );
    let dialect = RunDialect::resolve(
        ToolDispatcher::Python,
        std::slice::from_ref(&base),
        Some(true),
    );
    let messages = vec![
        Message::system("host prompt with its own ## Tools block"),
        Message::user("hi"),
    ];

    let mut request =
        ModelRequest::new(messages).with_tools(vec![base.clone(), synthesized.clone()]);
    request.tool_choice = ToolChoice::Tool("emit_result".into());
    dialect.apply_to_request(&mut request, true, std::slice::from_ref(&synthesized));

    assert!(request.tools.is_empty());
    let system = request.messages[0].text();
    // The base tool is left to the host's own (untouched) catalogue...
    assert!(!system.contains("def lookup("), "{system}");
    // ...but the synthesized one, the host could never have known about, is
    // appended so the forced call has a schema to answer against.
    assert!(system.contains("def emit_result("), "{system}");
    assert!(system.contains("You must call the `emit_result` tool."));
    assert_eq!(request.tool_choice, ToolChoice::Auto);
}
