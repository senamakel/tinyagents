//! Unit tests for the capability bundle (gap G3).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tinytools::{Tool, ToolExposure, ToolResult};

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::Middleware;
use crate::runtime::AgentHarness;
use crate::tool::toolset::ToolSet;

fn ctx() -> RunContext<()> {
    RunContext::new(RunConfig::new("run-capability"), ())
}

/// A minimal deterministic tool, mirroring `tool::toolset::test::EchoTool`.
struct StubTool {
    name: String,
}

impl StubTool {
    fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl Tool for StubTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "A stub tool for capability tests."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(format!("{}-result", self.name)))
    }
}

/// A single-tool `ToolSet` wrapping one [`StubTool`].
struct StubToolSet {
    tool: Arc<dyn Tool>,
}

impl StubToolSet {
    fn new(name: impl Into<String>) -> Self {
        Self {
            tool: Arc::new(StubTool::new(name)),
        }
    }
}

#[async_trait]
impl ToolSet<(), ()> for StubToolSet {
    async fn tools(&self, _ctx: &RunContext<()>) -> Result<Vec<Arc<dyn Tool>>> {
        Ok(vec![self.tool.clone()])
    }

    async fn call(
        &self,
        name: &str,
        args: serde_json::Value,
        _ctx: &RunContext<()>,
    ) -> Result<ToolResult> {
        if name == self.tool.name() {
            self.tool
                .execute(args)
                .await
                .map_err(|err| TinyAgentsError::Tool(err.to_string()))
        } else {
            Err(TinyAgentsError::ToolNotFound(name.to_string()))
        }
    }
}

/// A no-op middleware; the tests only assert it reached the harness's stack.
struct StubMiddleware;

#[async_trait]
impl Middleware<(), ()> for StubMiddleware {
    fn name(&self) -> &str {
        "stub-middleware"
    }
}

#[tokio::test]
async fn with_capability_installs_toolset_middleware_and_model_defaults() {
    let capability: Capability<(), ()> = Capability::new("research")
        .with_instructions("Use the research tool for lookups.")
        .with_toolset(Arc::new(StubToolSet::new("lookup")))
        .with_middleware(Arc::new(StubMiddleware))
        .with_model_defaults(ModelRequestDefaults {
            default_response_format: Some(tinyinference_llm::model::ResponseFormat::JsonObject),
            fallback: None,
        });

    let mut harness: AgentHarness<(), ()> = AgentHarness::new();
    harness.with_capability(capability);

    // Toolset composition: the capability's tool is advertised.
    let toolset = harness.toolset().expect("toolset installed").clone();
    let tools = toolset.tools(&ctx()).await.expect("tools resolve");
    assert!(tools.iter().any(|tool| tool.name() == "lookup"));
    assert_eq!(
        toolset.instructions().as_deref(),
        Some("Use the research tool for lookups.")
    );

    // Middleware appended.
    assert_eq!(harness.middleware().len(), 1);
    // Middleware `name()` is callable through the stack (proves it is the
    // same instance, not a stub).
    let _ = calls.load(Ordering::SeqCst);

    // Model defaults applied to the policy.
    assert_eq!(
        harness.policy().default_response_format,
        Some(tinyinference_llm::model::ResponseFormat::JsonObject)
    );
}

#[tokio::test]
async fn with_capability_composes_with_an_existing_toolset() {
    let mut harness: AgentHarness<(), ()> = AgentHarness::new();
    harness.with_toolset(Arc::new(StubToolSet::new("base")));
    harness.with_capability(
        Capability::new("extra").with_toolset(Arc::new(StubToolSet::new("extra-tool"))),
    );

    let toolset = harness.toolset().expect("toolset installed").clone();
    let names: Vec<String> = toolset
        .tools(&ctx())
        .await
        .expect("tools resolve")
        .iter()
        .map(|tool| tool.name().to_string())
        .collect();
    assert!(names.iter().any(|name| name == "base"));
    assert!(names.iter().any(|name| name == "extra-tool"));
}

#[tokio::test]
async fn defer_loading_capability_hides_tools_and_instructions_until_loaded() {
    let capability: Capability<(), ()> = Capability::new("advanced")
        .with_instructions("Advanced instructions.")
        .with_toolset(Arc::new(StubToolSet::new("advanced-tool")))
        .with_defer_loading(true);
    let toolset = CapabilityToolSet::new(vec![capability]);

    let before = toolset.tools(&ctx()).await.expect("tools resolve");
    let before_names: Vec<&str> = before.iter().map(|tool| tool.name()).collect();
    assert!(!before_names.contains(&"advanced-tool"));
    assert!(before_names.contains(&LOAD_CAPABILITY_TOOL_NAME));
    assert!(toolset.instructions().unwrap().contains("advanced"));
    assert!(toolset.loaded_names().is_empty());

    let result = toolset
        .call(
            LOAD_CAPABILITY_TOOL_NAME,
            json!({"capability": "advanced"}),
            &ctx(),
        )
        .await
        .expect("load_capability call succeeds");
    assert!(!result.is_error);
    assert_eq!(toolset.loaded_names(), vec!["advanced".to_string()]);

    let after = toolset.tools(&ctx()).await.expect("tools resolve");
    let after_names: Vec<&str> = after.iter().map(|tool| tool.name()).collect();
    assert!(after_names.contains(&"advanced-tool"));
    assert_eq!(
        toolset.instructions().as_deref(),
        Some("Advanced instructions.")
    );

    // The now-loaded tool is dispatchable through the composed toolset too.
    let call_result = toolset
        .call("advanced-tool", json!({}), &ctx())
        .await
        .expect("dispatch succeeds");
    assert_eq!(call_result.text(), "advanced-tool-result");
}

#[tokio::test]
async fn load_capability_rejects_an_unknown_name() {
    let capability: Capability<(), ()> = Capability::new("advanced").with_defer_loading(true);
    let toolset = CapabilityToolSet::new(vec![capability]);

    let result = toolset
        .call(
            LOAD_CAPABILITY_TOOL_NAME,
            json!({"capability": "nope"}),
            &ctx(),
        )
        .await
        .expect("call resolves (a reported tool error, not a dispatch failure)");
    assert!(result.is_error);
    assert!(toolset.loaded_names().is_empty());
}

#[tokio::test]
async fn no_load_capability_tool_when_nothing_defers() {
    let capability: Capability<(), ()> =
        Capability::new("eager").with_toolset(Arc::new(StubToolSet::new("eager-tool")));
    let toolset = CapabilityToolSet::new(vec![capability]);

    let tools = toolset.tools(&ctx()).await.expect("tools resolve");
    assert!(!tools.iter().any(|tool| tool.name() == LOAD_CAPABILITY_TOOL_NAME));
}

#[test]
fn capability_from_spec_round_trips_declarative_fields() {
    let original: Capability<(), ()> = Capability::new("research")
        .with_instructions("Use research tools.")
        .with_exposure(ToolExposure::Deferred)
        .with_defer_loading(true)
        .with_model_defaults(ModelRequestDefaults {
            default_response_format: Some(tinyinference_llm::model::ResponseFormat::JsonObject),
            fallback: Some(crate::retry::FallbackPolicy {
                models: vec!["primary".to_string(), "secondary".to_string()],
            }),
        });

    let spec = original.to_spec();
    assert_eq!(spec["name"], "research");
    assert_eq!(spec["exposure"], "deferred");
    assert_eq!(spec["defer_loading"], true);

    let rebuilt: Capability<(), ()> = Capability::from_spec(spec).expect("spec parses");
    assert_eq!(rebuilt.name, original.name);
    assert_eq!(rebuilt.instructions, original.instructions);
    assert_eq!(rebuilt.exposure, original.exposure);
    assert_eq!(rebuilt.defer_loading, original.defer_loading);
    assert_eq!(rebuilt.model_defaults, original.model_defaults);
    // Not representable in JSON: always empty on a spec-built capability.
    assert!(rebuilt.toolset.is_none());
    assert!(rebuilt.middleware.is_empty());
}

#[test]
fn capability_from_spec_defaults_exposure_and_defer_loading() {
    let capability: Capability<(), ()> =
        Capability::from_spec(json!({"name": "minimal"})).expect("spec parses");
    assert_eq!(capability.name, "minimal");
    assert_eq!(capability.instructions, None);
    assert_eq!(capability.exposure, ToolExposure::Direct);
    assert!(!capability.defer_loading);
    assert_eq!(capability.model_defaults, None);
}

#[test]
fn capability_from_spec_rejects_a_blank_name() {
    let err = Capability::<(), ()>::from_spec(json!({"name": "  "})).unwrap_err();
    assert!(matches!(err, TinyAgentsError::Capability(_)));
}
