//! Unit tests for the [`CapabilityRegistry`](super::CapabilityRegistry):
//! registration and lookup of models and tools, kind-scoped namespacing,
//! duplicate rejection, `replace_*` overwrite semantics, alias resolution and
//! validation, and harness registry hand-off builders.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::component::ComponentKind;
use tinyagents_definition::AgentDefinition;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinytools::{Tool, ToolResult};

struct FakeModel(&'static str);

#[async_trait]
impl ChatModel<()> for FakeModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant(self.0))
    }
}

struct FakeTool(&'static str);

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        self.0
    }
    fn description(&self) -> &str {
        "fake tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

#[test]
fn registers_and_looks_up_models_and_tools() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("hi")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_router("classify").unwrap();
    reg.register_reducer("append").unwrap();

    assert!(reg.model("default").is_some());
    assert!(reg.tool("lookup_user").is_some());

    assert!(reg.has(ComponentKind::Model, "default"));
    assert!(reg.has(ComponentKind::Tool, "lookup_user"));
    assert!(reg.has(ComponentKind::Router, "classify"));
    assert!(reg.has(ComponentKind::Reducer, "append"));

    assert!(!reg.has(ComponentKind::Model, "missing"));
    assert!(reg.model("missing").is_none());
}

#[test]
fn registers_declarative_agents_without_an_executable_graph_adapter() {
    let mut registry = CapabilityRegistry::<()>::new();
    registry
        .register_agent(
            AgentDefinition::new("planner", "Planner", "Plans work")
                .with_tools(["todo"])
                .with_subagents(["researcher"]),
        )
        .unwrap();
    registry
        .alias(ComponentKind::Agent, "default", "planner")
        .unwrap();

    let definition = registry.agent("default").unwrap();
    assert_eq!(definition.id, "planner");
    assert_eq!(definition.subagents, vec!["researcher"]);
    assert_eq!(registry.names(ComponentKind::Agent), vec!["planner"]);
}

#[test]
fn names_are_sorted_and_kind_scoped() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("b", Arc::new(FakeModel("x"))).unwrap();
    reg.register_model("a", Arc::new(FakeModel("y"))).unwrap();
    reg.register_tool(Arc::new(FakeTool("t"))).unwrap();

    assert_eq!(reg.names(ComponentKind::Model), vec!["a", "b"]);
    assert_eq!(reg.names(ComponentKind::Tool), vec!["t"]);
    assert!(reg.names(ComponentKind::Graph).is_empty());
}

#[test]
fn same_kind_name_namespaces_are_independent() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("shared", Arc::new(FakeModel("m")))
        .unwrap();
    // Same name under a different kind is allowed.
    reg.register_router("shared").unwrap();
    assert!(reg.has(ComponentKind::Model, "shared"));
    assert!(reg.has(ComponentKind::Router, "shared"));
}

#[test]
fn duplicate_registration_is_rejected() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("a")))
        .unwrap();
    let err = reg
        .register_model("default", Arc::new(FakeModel("b")))
        .unwrap_err();
    assert!(matches!(err, TinyAgentsError::DuplicateComponent(_)));

    reg.register_tool(Arc::new(FakeTool("t"))).unwrap();
    assert!(matches!(
        reg.register_tool(Arc::new(FakeTool("t"))).unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));

    reg.register_router("r").unwrap();
    assert!(matches!(
        reg.register_router("r").unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
}

#[test]
fn replace_overwrites_without_error_and_keeps_metadata() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("default", Arc::new(FakeModel("first")))
        .unwrap();
    reg.alias(ComponentKind::Model, "fast", "default").unwrap();

    // Replacing the value does not error and preserves the alias metadata.
    reg.replace_model("default", Arc::new(FakeModel("second")));
    assert!(reg.model("default").is_some());
    assert!(reg.model("fast").is_some());
    let meta = reg.metadata(ComponentKind::Model, "default").unwrap();
    assert_eq!(meta.aliases, vec!["fast".to_string()]);
}

#[test]
fn aliases_resolve_in_lookups() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_descriptor(ComponentKind::Graph, "flow")
        .unwrap();

    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    reg.alias(ComponentKind::Tool, "user", "lookup_user")
        .unwrap();
    reg.alias(ComponentKind::Graph, "main", "flow").unwrap();

    assert!(reg.model("default").is_some());
    assert!(reg.tool("user").is_some());
    assert!(reg.has(ComponentKind::Graph, "main"));
    assert!(reg.has(ComponentKind::Model, "default"));
    assert_eq!(
        reg.resolve_name(ComponentKind::Model, "default").as_deref(),
        Some("gpt-4o")
    );

    // metadata via alias resolves to the canonical entry.
    let meta = reg.metadata(ComponentKind::Model, "default").unwrap();
    assert_eq!(meta.name(), "gpt-4o");
    assert!(meta.aliases.contains(&"default".to_string()));

    // Aliases are not listed in names().
    assert_eq!(reg.names(ComponentKind::Model), vec!["gpt-4o"]);
}

#[test]
fn alias_validation_rejects_unknown_target_and_duplicates() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();

    // Unknown target.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "x", "missing").unwrap_err(),
        TinyAgentsError::Capability(_)
    ));

    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    // Duplicate alias.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "default", "gpt-4o")
            .unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
    // Alias colliding with a registered component name.
    assert!(matches!(
        reg.alias(ComponentKind::Model, "gpt-4o", "gpt-4o")
            .unwrap_err(),
        TinyAgentsError::DuplicateComponent(_)
    ));
}

#[tokio::test]
async fn builds_harness_registries_with_model_aliases() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("hello")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    let models = reg.to_model_registry();
    assert!(models.get("gpt-4o").is_some());
    assert!(models.get("default").is_some());

    let tools = reg.to_tool_registry::<()>();
    assert_eq!(tools.names(), vec!["lookup_user"]);
}

#[test]
fn snapshot_lists_components_sorted_by_kind_and_name() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_tool(Arc::new(FakeTool("lookup_user")))
        .unwrap();
    reg.register_router("classify").unwrap();

    let snapshot = reg.snapshot();
    assert_eq!(snapshot.len(), 3);
    assert_eq!(snapshot.count(ComponentKind::Model), 1);
    assert_eq!(snapshot.by_kind(ComponentKind::Tool)[0].id.0, "lookup_user");
    // Round-trips for audit logs / UIs.
    let json = serde_json::to_string(&snapshot).unwrap();
    let back: crate::RegistrySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, snapshot);
    // DOT export clusters every registered kind.
    let dot = snapshot.to_dot();
    assert!(dot.contains("digraph registry"));
    assert!(dot.contains("cluster_model"));
}

#[test]
fn snapshot_enumerates_aliases() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();

    let snapshot = reg.snapshot();
    assert_eq!(snapshot.aliases.len(), 1);
    assert_eq!(snapshot.aliases[0].alias, "default");
    assert_eq!(snapshot.aliases[0].canonical, "gpt-4o");
    assert_eq!(snapshot.aliases[0].kind, ComponentKind::Model);
    // Aliases survive serialization round-trips for audit logs.
    let json = serde_json::to_string(&snapshot).unwrap();
    let back: crate::RegistrySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back, snapshot);
}

#[test]
fn diagnostics_flag_name_reused_across_kinds() {
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("shared", Arc::new(FakeModel("m")))
        .unwrap();
    reg.register_router("shared").unwrap();

    let diags = reg.diagnostics();
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].name, "shared");
    assert!(diags[0].message.contains("multiple kinds"));
}

#[test]
fn diagnostics_are_clean_for_a_healthy_registry() {
    // `alias()` is fail-closed against shadowing and dangling targets, so a
    // registry built through the public API always passes the integrity check.
    let mut reg = CapabilityRegistry::<()>::new();
    reg.register_model("gpt-4o", Arc::new(FakeModel("m")))
        .unwrap();
    reg.alias(ComponentKind::Model, "default", "gpt-4o")
        .unwrap();
    assert!(reg.diagnostics().is_empty());
}
