//! Tests for the [`AgentHarness`] builder and [`RunPolicy`].

use std::sync::Arc;

use crate::context::{RunConfig, RunContext};
use crate::host::{
    AgentMemory, AllowAllSecurityGate, ErrorFieldClassifier, FixedModelResolver, GateDecision,
    InMemoryAgentMemory, InMemoryExperienceStore, NoopLearningSink, RecordingProgressSink,
    ScreenOutcome, SecurityGate, StaticContextComposer, ToolCallRequest, UnlimitedBudgetGate,
};
use crate::limits::RunLimits;
use crate::middleware::LoggingMiddleware;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, AgentTurnRequest, RunPolicy};
use crate::testkit::ScriptedModel;
use tinyagents_definition::{AgentDefinition, InMemoryDefinitionRegistry};
use tinyinference_llm::providers::MockModel;
use tinytools::{Tool, ToolResult};

use async_trait::async_trait;
use serde_json::json;

struct NoopTool;

struct DenyToolGate;

#[async_trait]
impl SecurityGate for DenyToolGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        Ok(GateDecision::deny("host denied this tool"))
    }

    async fn screen_input(
        &self,
        _text: &str,
        _origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        Ok(ScreenOutcome::Pass)
    }
}

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

#[test]
fn new_harness_is_empty_with_default_policy() {
    let harness: AgentHarness<()> = AgentHarness::new();
    assert!(harness.models().default_name().is_none());
    assert_eq!(harness.tools().names().len(), 0);
    assert!(harness.middleware().is_empty());
    assert_eq!(harness.policy(), &RunPolicy::default());
}

#[test]
fn register_first_model_becomes_default() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("a", Arc::new(MockModel::constant("a")))
        .register_model("b", Arc::new(MockModel::constant("b")));
    assert_eq!(harness.models().default_name(), Some("a"));
    harness.set_default_model("b");
    assert_eq!(harness.models().default_name(), Some("b"));
}

#[test]
fn register_tool_and_push_middleware() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.push_middleware(Arc::new(LoggingMiddleware::new()));
    assert_eq!(harness.tools().names(), vec!["noop".to_string()]);
    assert_eq!(harness.middleware().len(), 1);
}

#[test]
fn with_policy_replaces_policy() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    let policy = RunPolicy {
        limits: RunLimits::default().with_max_model_calls(3),
        retry: RetryPolicy::default().with_max_attempts(1),
        fallback: Some(FallbackPolicy::new(["a", "b"])),
        default_response_format: None,
        ..RunPolicy::default()
    };
    harness.with_policy(policy.clone());
    assert_eq!(harness.policy(), &policy);
    assert_eq!(harness.policy().limits.max_model_calls, 3);
}

#[test]
fn default_matches_new() {
    let harness: AgentHarness<()> = AgentHarness::default();
    assert!(harness.models().default_name().is_none());
}

#[tokio::test]
async fn host_driven_turn_resolves_and_composes_without_touching_explicit_sdk_defaults() {
    let model = Arc::new(ScriptedModel::replies(vec!["host reply"]));
    let memory = Arc::new(InMemoryAgentMemory::default());
    memory
        .remember(crate::host::NewMemory::new("remembered preference").with_agent("helper"))
        .await
        .expect("seed memory");
    let progress = Arc::new(RecordingProgressSink::new());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::new("host system")),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_memory(memory)
    .with_budget(Arc::new(UnlimitedBudgetGate))
    .with_progress(progress.clone())
    .with_learning(Arc::new(NoopLearningSink))
    .with_tool_outcomes(Arc::new(ErrorFieldClassifier))
    .with_experience(Arc::new(InMemoryExperienceStore::default()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);

    let run = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("preference")],
            ),
            RunContext::new(RunConfig::new("host-run").with_thread("thread"), ()),
            &(),
        )
        .await
        .expect("host turn succeeds");

    assert_eq!(run.text().as_deref(), Some("host reply"));
    let request = model.requests().pop().expect("model receives a request");
    assert_eq!(request.messages[0].text(), "host system");
    assert!(
        request
            .messages
            .iter()
            .any(|message| message.text() == "remembered preference")
    );
    assert_eq!(
        progress.len(),
        2,
        "started and finished progress are projected"
    );
    assert!(
        harness.models().default_name().is_none(),
        "host resolution does not mutate SDK model defaults"
    );
}

#[tokio::test]
async fn host_driven_turn_requires_an_installed_bundle_before_model_resolution() {
    let harness: AgentHarness<()> = AgentHarness::new();
    let error = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("hello")],
            ),
            RunContext::new(RunConfig::new("missing-host"), ()),
            &(),
        )
        .await
        .expect_err("host entry point rejects missing configuration");
    assert!(error.to_string().contains("with_host_capabilities"));
}

#[tokio::test]
async fn host_security_denial_returns_a_tool_message_without_executing_the_tool() {
    let mut tool_response = tinyinference_llm::model::ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        tool_response,
        tinyinference_llm::model::ModelResponse::assistant("recovered"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(DenyToolGate),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.with_host_capabilities(host);

    let run = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("denied"), ()),
            &(),
        )
        .await
        .expect("the model receives the denial and can finish");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("host denied this tool"))
    );
}
