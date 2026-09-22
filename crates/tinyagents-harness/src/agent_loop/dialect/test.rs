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
