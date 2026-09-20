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
