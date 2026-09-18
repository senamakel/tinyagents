//! Tests for the [`AgentHarness`] builder and [`RunPolicy`].

use std::sync::{Arc, Mutex};

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
use futures::StreamExt;
use tinyagents_definition::{AgentDefinition, InMemoryDefinitionRegistry};
use tinyinference_llm::providers::MockModel;
use tinytools::{Tool, ToolResult};

use async_trait::async_trait;
use serde_json::json;

struct NoopTool;

struct DenyToolGate;

struct RetryableClassifier;

impl crate::host::ToolOutcomeClassifier for RetryableClassifier {
    fn classify(&self, _name: &str, _result: &ToolResult) -> crate::host::OutcomeClass {
        crate::host::OutcomeClass::RetryableFailure
    }
}

#[derive(Default)]
struct RecordingLearning {
    summaries: Mutex<Vec<crate::host::TurnSummary>>,
}

#[async_trait]
impl crate::host::LearningSink for RecordingLearning {
    async fn on_turn_complete(
        &self,
        summary: &crate::host::TurnSummary,
    ) -> crate::error::Result<()> {
        self.summaries
            .lock()
            .expect("learning lock")
            .push(summary.clone());
        Ok(())
    }
}

#[derive(Default)]
struct RecordingExperience {
    records: Mutex<Vec<crate::host::Experience>>,
}

#[async_trait]
impl crate::host::ExperienceStore for RecordingExperience {
    async fn record(&self, exp: &crate::host::Experience) -> crate::error::Result<()> {
        self.records
            .lock()
            .expect("experience lock")
            .push(exp.clone());
        Ok(())
    }

    async fn recall_for(
        &self,
        _agent_id: &str,
        _task: &str,
    ) -> crate::error::Result<Vec<crate::host::Experience>> {
        Ok(Vec::new())
    }
}

async fn yield_until(mut predicate: impl FnMut() -> bool) {
    for _ in 0..64 {
        if predicate() {
            return;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        predicate(),
        "background terminal finalizer did not complete"
    );
}

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
    yield_until(|| progress.len() == 2).await;
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

#[tokio::test]
async fn hosted_streams_finalize_success_failure_and_drop_with_terminal_host_records() {
    async fn run_terminal_case(
        model: Arc<ScriptedModel>,
        run_id: &str,
        drain: bool,
    ) -> (
        Arc<RecordingLearning>,
        Arc<RecordingExperience>,
        Arc<RecordingProgressSink>,
    ) {
        let learning = Arc::new(RecordingLearning::default());
        let experience = Arc::new(RecordingExperience::default());
        let progress = Arc::new(RecordingProgressSink::new());
        let host = crate::host::HostCapabilities::new(
            Arc::new(StaticContextComposer::empty()),
            Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
                "helper",
                "Helper",
                "test helper",
            )])),
            Arc::new(AllowAllSecurityGate),
            Arc::new(FixedModelResolver::new(model)),
        )
        .with_progress(progress.clone())
        .with_learning(learning.clone())
        .with_experience(experience.clone());
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.with_host_capabilities(host);
        let context = RunContext::new(RunConfig::new(run_id), ());
        let context_id = context.instance_id();
        let mut stream = harness
            .invoke_agent_stream(
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("go")],
                ),
                context,
                &(),
            )
            .await
            .expect("stream starts");
        if drain {
            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if !matches!(item, crate::agent_loop::AgentStreamItem::Event(_)) {
                    terminal = Some(item);
                    break;
                }
            }
            assert!(terminal.is_some(), "stream reaches terminal item");
        }
        drop(stream);
        assert!(
            harness
                .host_run_binding(context_id)
                .expect("binding lock")
                .is_none(),
            "terminal observation and Drop both remove the exact run binding"
        );
        yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
        yield_until(|| experience.records.lock().expect("experience lock").len() == 1).await;
        yield_until(|| {
            progress
                .events()
                .iter()
                .any(crate::host::ProgressEvent::is_terminal)
        })
        .await;
        (learning, experience, progress)
    }

    let (learning, experience, progress) = run_terminal_case(
        Arc::new(ScriptedModel::replies(vec!["ok"])),
        "stream-success",
        true,
    )
    .await;
    assert!(experience.records.lock().expect("experience lock")[0].success);
    assert_eq!(learning.summaries.lock().expect("learning lock").len(), 1);
    assert!(matches!(
        progress.events().last(),
        Some(crate::host::ProgressEvent::Finished { .. })
    ));

    let (_learning, experience, progress) =
        run_terminal_case(Arc::new(ScriptedModel::new(vec![])), "stream-error", true).await;
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert!(
        progress
            .events()
            .iter()
            .any(|event| matches!(event, crate::host::ProgressEvent::Error { .. }))
    );

    let (_learning, experience, progress) = run_terminal_case(
        Arc::new(ScriptedModel::replies(vec!["unused"])),
        "stream-cancel",
        false,
    )
    .await;
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert!(
        progress
            .events()
            .iter()
            .any(|event| matches!(event, crate::host::ProgressEvent::Error { .. }))
    );
}

#[tokio::test]
async fn denied_tool_calls_do_not_enter_terminal_executed_tool_summary() {
    let mut tool_response = tinyinference_llm::model::ModelResponse::assistant("");
    tool_response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "call-1",
            "noop",
            json!({}),
        ));
    let learning = Arc::new(RecordingLearning::default());
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(DenyToolGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::new(vec![
            tool_response,
            tinyinference_llm::model::ModelResponse::assistant("recovered"),
        ])))),
    )
    .with_learning(learning.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.with_host_capabilities(host);
    harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("denied-summary"), ()),
            &(),
        )
        .await
        .expect("denial is recoverable");
    let summaries = learning.summaries.lock().expect("learning lock");
    assert!(
        summaries[0].tools_invoked.is_empty(),
        "denied calls never reached a tool executor"
    );
}

#[tokio::test]
async fn retryable_classifier_changes_the_model_visible_result_without_redispatching() {
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
        tinyinference_llm::model::ModelResponse::assistant("model chose to continue"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_tool_outcomes(Arc::new(RetryableClassifier));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.with_host_capabilities(host);
    let run = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("retryable-result"), ()),
            &(),
        )
        .await
        .expect("retryable result is recoverable");
    assert_eq!(
        run.tool_calls, 1,
        "the runtime never silently repeats an action"
    );
    assert_eq!(
        model.requests().len(),
        2,
        "the model, not the runtime, selected the next step"
    );
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("retryable tool failure"))
    );
}

#[tokio::test]
async fn poisoned_host_binding_fails_closed_before_any_model_fallback() {
    let model = Arc::new(ScriptedModel::replies(vec!["must not be used"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = harness.host_runs.lock().expect("fresh lock");
        panic!("poison host binding map");
    }));
    let error = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("poisoned-binding"), ()),
            &(),
        )
        .await
        .expect_err("poison must not fall back to an unbound model");
    assert!(error.to_string().contains("host run binding lock poisoned"));
    assert!(
        model.requests().is_empty(),
        "no provider request escaped host policy"
    );
}

#[tokio::test]
async fn same_user_run_id_concurrent_host_turns_keep_distinct_bindings() {
    let model = Arc::new(ScriptedModel::replies(vec!["first", "second"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model)),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);
    let first = harness.invoke_agent(
        AgentTurnRequest::new(
            "helper",
            vec![tinyinference_llm::message::Message::user("first")],
        ),
        RunContext::new(RunConfig::new("shared-id"), ()),
        &(),
    );
    let second = harness.invoke_agent(
        AgentTurnRequest::new(
            "helper",
            vec![tinyinference_llm::message::Message::user("second")],
        ),
        RunContext::new(RunConfig::new("shared-id"), ()),
        &(),
    );
    let (first, second) = tokio::join!(first, second);
    assert!(first.is_ok());
    assert!(second.is_ok());
    assert!(harness.host_runs.lock().expect("binding lock").is_empty());
}
