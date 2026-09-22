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
    dialect.apply_to_request(&mut appended, false);
    assert!(appended.tools.is_empty());
    let system = appended.messages[0].text();
    assert!(system.contains("def lookup("), "{system}");

    // Host-rendered: schemas still leave the wire, the prompt is untouched.
    let mut host = ModelRequest::new(messages.clone()).with_tools(tools.clone());
    dialect.apply_to_request(&mut host, true);
    assert!(host.tools.is_empty());
    assert_eq!(host.messages, messages);
    assert_eq!(host.tool_choice, ToolChoice::Auto);

    // A forced choice is the one thing the host could not have said.
    let mut forced = ModelRequest::new(messages).with_tools(tools);
    forced.tool_choice = ToolChoice::Tool("lookup".into());
    dialect.apply_to_request(&mut forced, true);
    let system = forced.messages[0].text();
    assert!(system.contains("You must call the `lookup` tool."), "{system}");
    assert!(!system.contains("def lookup("));
    assert_eq!(forced.tool_choice, ToolChoice::Auto);
}
