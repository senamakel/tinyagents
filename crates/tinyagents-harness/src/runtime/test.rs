//! Tests for the [`AgentHarness`] builder and [`RunPolicy`].

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use crate::context::{RunConfig, RunContext};
use crate::host::{
    AgentMemory, AllowAllSecurityGate, BudgetGate, CallEstimate, CompressionHint, ContextState,
    ErrorFieldClassifier, FixedModelResolver, GateDecision, InMemoryAgentMemory,
    InMemoryExperienceStore, NoopLearningSink, Permit, RecordingProgressSink, ScreenOutcome,
    SecurityGate, StaticContextComposer, ToolCallRequest, UnlimitedBudgetGate,
};
use crate::limits::RunLimits;
use crate::middleware::LoggingMiddleware;
use crate::retry::{FallbackPolicy, RetryPolicy};
use crate::runtime::{AgentHarness, AgentTurnRequest, RunPolicy};
use crate::subagent::{ChildDataPolicy, SubAgent, SubAgentTool};
use crate::testkit::ScriptedModel;
use futures::StreamExt;
use tinyagents_definition::{AgentDefinition, InMemoryDefinitionRegistry};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::{
    model::{ChatModel, ModelRequest, ModelResponse},
    usage::Usage,
};
use tinytools::{Tool, ToolResult};

use async_trait::async_trait;
use serde_json::json;

struct NoopTool;

struct BlockedTool;

struct DenyToolGate;

struct DenyThenAllowGate {
    denials_remaining: AtomicUsize,
}

struct RedactJsonUserGate;

struct RedactDeltaMiddleware;

struct RetryableClassifier;

struct LeadRecordingResolver {
    model: Arc<dyn ChatModel<()>>,
    team_lead_flags: Mutex<Vec<bool>>,
}

struct RecordingBudget {
    hint: CompressionHint,
    records: Mutex<Vec<Usage>>,
}

impl RecordingBudget {
    fn hard() -> Self {
        Self {
            hint: CompressionHint::Hard,
            records: Mutex::new(Vec::new()),
        }
    }

    fn soft() -> Self {
        Self {
            hint: CompressionHint::Soft,
            records: Mutex::new(Vec::new()),
        }
    }

    fn permissive() -> Self {
        Self {
            hint: CompressionHint::None,
            records: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl BudgetGate for RecordingBudget {
    async fn acquire(&self, _estimate: &CallEstimate) -> crate::error::Result<Permit> {
        Ok(Permit::unlimited())
    }

    async fn record(&self, usage: &Usage) -> crate::error::Result<()> {
        self.records.lock().expect("budget lock").push(*usage);
        Ok(())
    }

    fn compression_hint(&self, _state: &ContextState) -> CompressionHint {
        self.hint
    }
}

#[derive(Default)]
struct PartialThenPendingModel {
    calls: AtomicUsize,
}

#[async_trait]
impl ChatModel<()> for PartialThenPendingModel {
    async fn invoke(
        &self,
        _state: &(),
        _request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            let mut response = ModelResponse::assistant("");
            response
                .message
                .tool_calls
                .push(tinyinference_llm::tool::ToolCall::new(
                    "partial-tool",
                    "noop",
                    json!({}),
                ));
            response.usage = Some(Usage {
                input_tokens: 3,
                output_tokens: 2,
                total_tokens: 5,
                ..Usage::default()
            });
            Ok(response)
        } else {
            std::future::pending().await
        }
    }
}

impl crate::host::ToolOutcomeClassifier for RetryableClassifier {
    fn classify(&self, _name: &str, _result: &ToolResult) -> crate::host::OutcomeClass {
        crate::host::OutcomeClass::RetryableFailure
    }
}

#[async_trait]
impl crate::host::ModelResolver<()> for LeadRecordingResolver {
    async fn resolve(
        &self,
        request: &crate::host::ModelResolveRequest,
    ) -> crate::error::Result<Arc<dyn ChatModel<()>>> {
        self.team_lead_flags
            .lock()
            .expect("resolver lock")
            .push(request.is_team_lead);
        Ok(Arc::clone(&self.model))
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

#[derive(Default)]
struct RecordingMemory {
    items: Mutex<Vec<crate::host::NewMemory>>,
}

#[async_trait]
impl AgentMemory for RecordingMemory {
    async fn recall(
        &self,
        _req: crate::host::RecallRequest,
    ) -> crate::error::Result<Vec<crate::host::MemoryItem>> {
        Ok(Vec::new())
    }

    async fn remember(
        &self,
        item: crate::host::NewMemory,
    ) -> crate::error::Result<crate::host::MemoryId> {
        self.items.lock().expect("memory lock").push(item);
        Ok(crate::host::MemoryId::new("recorded"))
    }

    async fn thread_summary(
        &self,
        _thread: &crate::ids::ThreadId,
    ) -> crate::error::Result<Option<String>> {
        Ok(None)
    }
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
impl SecurityGate for DenyThenAllowGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        if self
            .denials_remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            Ok(GateDecision::deny("approval declined"))
        } else {
            Ok(GateDecision::Allow)
        }
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
impl SecurityGate for RedactJsonUserGate {
    async fn authorize_tool(&self, _call: &ToolCallRequest) -> crate::error::Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    async fn screen_input(
        &self,
        text: &str,
        origin: crate::host::ContentOrigin,
    ) -> crate::error::Result<ScreenOutcome> {
        if origin == crate::host::ContentOrigin::User && text.contains("secret") {
            Ok(ScreenOutcome::Redacted(r#"{"safe":true}"#.to_string()))
        } else {
            Ok(ScreenOutcome::Pass)
        }
    }
}

#[async_trait]
impl crate::middleware::Middleware<(), ()> for RedactDeltaMiddleware {
    fn name(&self) -> &str {
        "redact-delta"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut tinyinference_llm::model::ModelDelta,
    ) -> crate::error::Result<()> {
        delta.content = "[redacted]".to_string();
        Ok(())
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

#[async_trait]
impl Tool for BlockedTool {
    fn name(&self) -> &str {
        "blocked"
    }

    fn description(&self) -> &str {
        "must not be available to this agent"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        panic!("a definition-disallowed tool must never execute")
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
async fn hosted_turn_screens_and_redacts_json_user_blocks_before_model_submission() {
    let model = Arc::new(ScriptedModel::replies(vec!["ok"]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(RedactJsonUserGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);
    let message =
        tinyinference_llm::message::Message::User(tinyinference_llm::message::UserMessage {
            content: vec![tinyinference_llm::message::ContentBlock::Json(
                json!({"secret": "do not forward"}),
            )],
        });

    harness
        .invoke_agent(
            AgentTurnRequest::new("helper", vec![message]),
            RunContext::new(RunConfig::new("json-screen"), ()),
            &(),
        )
        .await
        .expect("hosted turn succeeds");

    let request = model.requests().pop().expect("model request");
    let user = request
        .messages
        .iter()
        .find_map(|message| match message {
            tinyinference_llm::message::Message::User(user) => Some(user),
            _ => None,
        })
        .expect("user message retained");
    assert_eq!(
        user.content,
        vec![tinyinference_llm::message::ContentBlock::Json(
            json!({"safe": true})
        )]
    );
}

#[tokio::test]
async fn hosted_model_resolution_marks_only_root_contexts_as_team_leads() {
    let model = Arc::new(ScriptedModel::replies(vec!["root", "child"]));
    let resolver = Arc::new(LeadRecordingResolver {
        model: model.clone(),
        team_lead_flags: Mutex::new(Vec::new()),
    });
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(AllowAllSecurityGate),
        resolver.clone(),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);

    harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("root")],
            ),
            RunContext::new(RunConfig::new("root").with_max_depth(2), ()),
            &(),
        )
        .await
        .expect("root turn succeeds");

    let parent: RunContext<()> = RunContext::new(RunConfig::new("parent").with_max_depth(2), ());
    let child = parent
        .child(RunConfig::new("child"), ())
        .expect("child context is valid");
    harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("child")],
            ),
            child,
            &(),
        )
        .await
        .expect("child turn succeeds");

    assert_eq!(
        *resolver.team_lead_flags.lock().expect("resolver lock"),
        vec![true, false]
    );
}

#[tokio::test]
async fn hosted_definition_tool_allowlist_filters_schemas_and_rejects_fabricated_calls() {
    let mut blocked_call = ModelResponse::assistant("");
    blocked_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "blocked-call",
            "blocked",
            json!({}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        blocked_call,
        ModelResponse::assistant("recovered"),
    ]));
    let definition = AgentDefinition::new("helper", "Helper", "test helper").with_tools(["noop"]);
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![definition])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_tool(Arc::new(NoopTool));
    harness.register_tool(Arc::new(BlockedTool));
    harness.with_host_capabilities(host);

    let run = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("go")],
            ),
            RunContext::new(RunConfig::new("allowlist"), ()),
            &(),
        )
        .await
        .expect("the model recovers after its denied call");

    assert_eq!(run.text().as_deref(), Some("recovered"));
    assert!(
        run.messages
            .iter()
            .any(|message| message.text().contains("unknown tool `blocked`"))
    );
    let requests = model.requests();
    assert_eq!(
        requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["noop"]
    );
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
async fn denied_tool_calls_release_their_reserved_limit_for_a_later_approval() {
    fn call(id: &str) -> ModelResponse {
        let mut response = ModelResponse::assistant("");
        response
            .message
            .tool_calls
            .push(tinyinference_llm::tool::ToolCall::new(
                id,
                "noop",
                json!({}),
            ));
        response
    }

    let model = Arc::new(ScriptedModel::new(vec![
        call("denied-one"),
        call("denied-two"),
        call("allowed"),
        ModelResponse::assistant("completed after approval"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "helper",
            "Helper",
            "test helper",
        )])),
        Arc::new(DenyThenAllowGate {
            denials_remaining: AtomicUsize::new(2),
        }),
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
            RunContext::new(
                RunConfig::new("denial-limit-release").with_max_tool_calls(1),
                (),
            ),
            &(),
        )
        .await
        .expect("two denials do not spend the only executable tool slot");

    assert_eq!(run.text().as_deref(), Some("completed after approval"));
    assert_eq!(run.executed_tools, vec!["noop"]);
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
        yield_until(|| {
            harness
                .host_run_binding(context_id)
                .expect("binding lock")
                .is_none()
        })
        .await;
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
    assert_eq!(
        progress
            .events()
            .iter()
            .filter(|event| event.is_terminal())
            .count(),
        1,
        "a failed turn has one terminal progress event"
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
    assert_eq!(
        progress
            .events()
            .iter()
            .filter(|event| event.is_terminal())
            .count(),
        1,
        "a cancelled turn has one terminal progress event"
    );
}

#[tokio::test]
async fn dropped_host_invocations_finalize_the_actual_partial_run_once() {
    async fn host_with_partial_model(
        model: Arc<PartialThenPendingModel>,
    ) -> (
        Arc<AgentHarness<()>>,
        Arc<RecordingLearning>,
        Arc<RecordingExperience>,
        Arc<RecordingMemory>,
        Arc<RecordingProgressSink>,
    ) {
        let learning = Arc::new(RecordingLearning::default());
        let experience = Arc::new(RecordingExperience::default());
        let memory = Arc::new(RecordingMemory::default());
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
        .with_learning(learning.clone())
        .with_experience(experience.clone())
        .with_memory(memory.clone())
        .with_progress(progress.clone());
        let mut harness = AgentHarness::new();
        harness.register_tool(Arc::new(NoopTool));
        harness.with_host_capabilities(host);
        (Arc::new(harness), learning, experience, memory, progress)
    }

    async fn wait_for_second_call(model: &PartialThenPendingModel) {
        yield_until(|| model.calls.load(Ordering::SeqCst) >= 2).await;
    }

    let model = Arc::new(PartialThenPendingModel::default());
    let (harness, learning, experience, memory, progress) =
        host_with_partial_model(model.clone()).await;
    let task_harness = harness.clone();
    let task = tokio::spawn(async move {
        task_harness
            .invoke_agent(
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("partial")],
                ),
                RunContext::new(RunConfig::new("unary-drop"), ()),
                &(),
            )
            .await
    });
    wait_for_second_call(&model).await;
    task.abort();
    let _ = task.await;
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0]
            .usage
            .total_tokens,
        5
    );
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0].tools_invoked,
        ["noop"]
    );
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    assert_eq!(memory.items.lock().expect("memory lock").len(), 1);
    let terminals: Vec<_> = progress
        .events()
        .into_iter()
        .filter(|event| event.is_terminal())
        .collect();
    assert!(matches!(
        terminals.as_slice(),
        [crate::host::ProgressEvent::Error { .. }]
    ));

    let model = Arc::new(PartialThenPendingModel::default());
    let (harness, learning, experience, _memory, progress) =
        host_with_partial_model(model.clone()).await;
    let mut stream = harness
        .invoke_agent_stream(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("partial")],
            ),
            RunContext::new(RunConfig::new("stream-drop"), ()),
            &(),
        )
        .await
        .expect("stream starts");
    while model.calls.load(Ordering::SeqCst) < 2 {
        let _ = tokio::time::timeout(std::time::Duration::from_millis(10), stream.next()).await;
    }
    drop(stream);
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0]
            .usage
            .total_tokens,
        5
    );
    assert_eq!(
        learning.summaries.lock().expect("learning lock")[0].tools_invoked,
        ["noop"]
    );
    assert!(!experience.records.lock().expect("experience lock")[0].success);
    let terminals: Vec<_> = progress
        .events()
        .into_iter()
        .filter(|event| event.is_terminal())
        .collect();
    assert!(matches!(
        terminals.as_slice(),
        [crate::host::ProgressEvent::Error { .. }]
    ));
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
    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
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

#[tokio::test]
async fn hard_budget_compression_hint_reduces_context_before_the_provider_call() {
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant("reduced").with_usage(Usage {
            input_tokens: 7,
            output_tokens: 3,
            total_tokens: 10,
            ..Usage::default()
        }),
    ]));
    let budget = Arc::new(RecordingBudget::hard());
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
    .with_budget(budget.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);
    let mut prior_tool_call = ModelResponse::assistant("");
    prior_tool_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "prior-lookup",
            "lookup",
            json!({"query": "old context"}),
        ));
    let original_messages = vec![
        tinyinference_llm::message::Message::system("system instruction one: preserve exactly"),
        tinyinference_llm::message::Message::system("system instruction two: preserve exactly"),
        tinyinference_llm::message::Message::user("old context ".repeat(80)),
        tinyinference_llm::message::Message::Assistant(prior_tool_call.message),
        tinyinference_llm::message::Message::tool("prior-lookup", "old lookup result ".repeat(4)),
        tinyinference_llm::message::Message::user("current task ".repeat(20)),
    ];
    let run = harness
        .invoke_agent(
            AgentTurnRequest::new("helper", original_messages.clone()),
            RunContext::new(RunConfig::new("hard-compression"), ()),
            &(),
        )
        .await
        .expect("hard compression reduces a multi-turn request before calling the provider");
    assert_eq!(run.text().as_deref(), Some("reduced"));
    let request = model.requests().pop().expect("provider was called once");
    assert!(
        request.messages.len() < original_messages.len(),
        "hard compression sent fewer messages to the provider"
    );
    let preserved_system: Vec<_> = request
        .messages
        .iter()
        .filter(|message| matches!(message, tinyinference_llm::message::Message::System(_)))
        .cloned()
        .collect();
    assert_eq!(
        preserved_system,
        original_messages[..2],
        "every system instruction survives byte-for-byte"
    );
    assert!(
        crate::summarization::tool_pairing_is_intact(&request.messages),
        "hard budget trimming leaves a provider-valid tool transcript"
    );
    assert!(
        request.messages.iter().any(|message| {
            matches!(message, tinyinference_llm::message::Message::Assistant(assistant) if !assistant.tool_calls.is_empty())
        }) && request
            .messages
            .iter()
            .any(|message| matches!(message, tinyinference_llm::message::Message::Tool(_))),
        "the reduced request retains a complete user/tool conversational payload"
    );
    assert_eq!(budget.records.lock().expect("budget lock").len(), 1);
}

#[tokio::test]
async fn hard_budget_compression_fails_closed_when_only_system_instructions_remain() {
    let model = Arc::new(ScriptedModel::replies(vec!["must not run"]));
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
    .with_budget(Arc::new(RecordingBudget::hard()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);

    let error = harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![
                    tinyinference_llm::message::Message::system("do not remove this instruction"),
                    tinyinference_llm::message::Message::system("nor this instruction"),
                ],
            ),
            RunContext::new(RunConfig::new("hard-system-only"), ()),
            &(),
        )
        .await
        .expect_err("hard pressure cannot discard sole system instructions");
    assert!(
        error
            .to_string()
            .contains("reducible conversational context")
    );
    assert!(model.requests().is_empty(), "provider was never called");
}

#[tokio::test]
async fn soft_budget_compression_hint_reduces_multiturn_context_without_blocking() {
    let model = Arc::new(ScriptedModel::replies(vec!["soft reduced"]));
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
    .with_budget(Arc::new(RecordingBudget::soft()));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_host_capabilities(host);

    harness
        .invoke_agent(
            AgentTurnRequest::new(
                "helper",
                vec![
                    tinyinference_llm::message::Message::user("one ".repeat(20)),
                    tinyinference_llm::message::Message::assistant("two ".repeat(20)),
                    tinyinference_llm::message::Message::user("three ".repeat(20)),
                    tinyinference_llm::message::Message::assistant("four ".repeat(20)),
                ],
            ),
            RunContext::new(RunConfig::new("soft-compression"), ()),
            &(),
        )
        .await
        .expect("soft compression still invokes the provider");

    assert!(
        model.requests()[0].messages.len() < 4,
        "soft compression reduced the provider request"
    );
}

#[tokio::test]
async fn cached_streaming_deltas_reach_events_and_progress_after_middleware() {
    let model = Arc::new(ScriptedModel::replies(vec!["secret"]));
    let progress = Arc::new(RecordingProgressSink::new());
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
    .with_progress(progress.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_response_cache(Arc::new(crate::cache::InMemoryResponseCache::new()));
    harness.with_host_capabilities(host);
    harness.push_middleware(Arc::new(RedactDeltaMiddleware));

    let mut seeded = harness
        .invoke_agent_stream(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("same")],
            ),
            RunContext::new(RunConfig::new("cache-seed"), ()),
            &(),
        )
        .await
        .expect("cache seed starts");
    while seeded.next().await.is_some() {}

    let mut replay = harness
        .invoke_agent_stream(
            AgentTurnRequest::new(
                "helper",
                vec![tinyinference_llm::message::Message::user("same")],
            ),
            RunContext::new(RunConfig::new("cache-replay"), ()),
            &(),
        )
        .await
        .expect("cache replay starts");
    let mut replayed_events = Vec::new();
    while let Some(item) = replay.next().await {
        if let crate::agent_loop::AgentStreamItem::Event(event) = item {
            replayed_events.push(event);
        }
    }

    assert_eq!(
        model.requests().len(),
        1,
        "replay does not invoke the model"
    );
    assert!(replayed_events.iter().any(
        |record| matches!(&record.event, crate::events::AgentEvent::ModelDelta { delta, .. } if delta.text == "[redacted]")
    ));
    assert!(!replayed_events.iter().any(
        |record| matches!(&record.event, crate::events::AgentEvent::ModelDelta { delta, .. } if delta.text == "secret")
    ));
    yield_until(|| {
        progress.events().iter().any(
            |event| matches!(event, crate::host::ProgressEvent::Token { text, .. } if text == "[redacted]")
        )
    })
    .await;
    assert!(!progress.events().iter().any(
        |event| matches!(event, crate::host::ProgressEvent::Token { text, .. } if text == "secret")
    ));
}

#[tokio::test]
async fn cached_host_response_does_not_re_record_provider_usage() {
    let model = Arc::new(ScriptedModel::new(vec![
        ModelResponse::assistant("cached").with_usage(Usage {
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
            ..Usage::default()
        }),
    ]));
    let budget = Arc::new(RecordingBudget::permissive());
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
    .with_budget(budget.clone());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.with_response_cache(Arc::new(crate::cache::InMemoryResponseCache::new()));
    harness.with_host_capabilities(host);
    for run_id in ["cache-one", "cache-two"] {
        harness
            .invoke_agent(
                AgentTurnRequest::new(
                    "helper",
                    vec![tinyinference_llm::message::Message::user("same")],
                ),
                RunContext::new(RunConfig::new(run_id), ()),
                &(),
            )
            .await
            .expect("cached run succeeds");
    }
    assert_eq!(model.requests().len(), 1, "second call is cache-served");
    assert_eq!(budget.records.lock().expect("budget lock").len(), 1);
}

#[tokio::test]
async fn host_delegate_registry_authorizes_recursive_children() {
    let mut parent_tool_call = ModelResponse::assistant("");
    parent_tool_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "delegate",
            "worker",
            json!({"input": "child task"}),
        ));
    let model = Arc::new(ScriptedModel::new(vec![
        parent_tool_call,
        ModelResponse::assistant("child answer"),
        ModelResponse::assistant("parent answer"),
    ]));
    let mut parent = AgentDefinition::new("parent", "Parent", "delegates");
    parent.subagents.push("worker".into());
    let definitions = Arc::new(InMemoryDefinitionRegistry::new(vec![
        parent,
        AgentDefinition::new("worker", "Worker", "child"),
    ]));
    let host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        definitions,
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(model)),
    );
    // The child deliberately has no host installed. A hosted parent must
    // propagate its own authority and bundle rather than falling back to this
    // child harness's configuration.
    let child_harness = AgentHarness::new();
    let child = Arc::new(SubAgent::new("worker", "child", Arc::new(child_harness)));
    let mut parent_harness = AgentHarness::new();
    parent_harness.register_tool_dispatch(Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(|_: &()| ()),
    )));
    parent_harness.with_host_capabilities(host);
    let run = parent_harness
        .invoke_agent(
            AgentTurnRequest::new(
                "parent",
                vec![tinyinference_llm::message::Message::user("delegate")],
            ),
            RunContext::new(RunConfig::new("authorized-child"), ()),
            &(),
        )
        .await
        .expect("registered delegate runs through the parent's hosted child entry point");
    assert_eq!(run.text().as_deref(), Some("parent answer"));
}

#[tokio::test]
async fn hosted_parent_denial_cannot_be_bypassed_by_a_differently_hosted_child() {
    let mut parent_tool_call = ModelResponse::assistant("");
    parent_tool_call
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "delegate",
            "worker",
            json!({"input": "child task"}),
        ));
    let parent_model = Arc::new(ScriptedModel::new(vec![
        parent_tool_call,
        ModelResponse::assistant("parent recovered from denied delegation"),
    ]));
    let child_model = Arc::new(ScriptedModel::replies(vec!["must never run"]));
    let parent_definitions = Arc::new(InMemoryDefinitionRegistry::new(vec![
        AgentDefinition::new("parent", "Parent", "does not delegate"),
        AgentDefinition::new("worker", "Worker", "child"),
    ]));
    let parent_host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        parent_definitions,
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(parent_model)),
    );
    let child_host = crate::host::HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
            "worker",
            "Worker",
            "permissive child",
        )])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(FixedModelResolver::new(child_model.clone())),
    );
    let mut child_harness = AgentHarness::new();
    child_harness.with_host_capabilities(child_host);
    let child = Arc::new(SubAgent::new("worker", "child", Arc::new(child_harness)));
    let mut parent_harness = AgentHarness::new();
    parent_harness.register_tool_dispatch(Arc::new(SubAgentTool::new(
        child,
        ChildDataPolicy::new(|_: &()| ()),
    )));
    parent_harness.with_host_capabilities(parent_host);

    let error = parent_harness
        .invoke_agent(
            AgentTurnRequest::new(
                "parent",
                vec![tinyinference_llm::message::Message::user("delegate")],
            ),
            RunContext::new(RunConfig::new("denied-mismatched-child"), ()),
            &(),
        )
        .await
        .expect_err("parent policy denies the child before its host can run");
    assert_eq!(error.to_string(), "tool error: tool dispatch failed");
    assert!(
        child_model.requests().is_empty(),
        "the differently-hosted child was never allowed to select its own policy"
    );
}
