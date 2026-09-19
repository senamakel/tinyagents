//! Public hosted-harness invocation contracts.
//!
//! These tests deliberately exercise the product-host boundary from a separate
//! crate. They use only public capability traits and deterministic models, so
//! they cover the same integration path an embedding host uses without making
//! a provider request.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use tinyagents_definition::{AgentDefinition, InMemoryDefinitionRegistry};
use tinyagents_harness::Result;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::host::{
    ContentOrigin, FixedModelResolver, GateDecision, HostCapabilities, LearningSink, ProgressEvent,
    RecordingProgressSink, ScreenOutcome, SecurityGate, TurnContextRequest, TurnSummary,
    UnlimitedBudgetGate,
};
use tinyagents_harness::runtime::{AgentHarness, AgentInvocation, AgentTurnRequest};
use tinyagents_harness::testkit::ScriptedModel;
use tinyinference_llm::message::Message;

#[derive(Default)]
struct RecordingComposer {
    requests: Mutex<Vec<TurnContextRequest>>,
}

#[async_trait]
impl tinyagents_harness::host::ContextComposer for RecordingComposer {
    async fn compose_system_prompt(&self, request: &TurnContextRequest) -> Result<String> {
        self.requests
            .lock()
            .expect("composer lock")
            .push(request.clone());
        Ok(format!("system for {}", request.agent_id))
    }

    async fn preamble(&self, request: &TurnContextRequest) -> Result<Vec<Message>> {
        Ok(vec![Message::system(format!(
            "preamble for {}",
            request.thread_id.as_str()
        ))])
    }
}

struct RedactingGate;

#[async_trait]
impl SecurityGate for RedactingGate {
    async fn authorize_tool(
        &self,
        _call: &tinyagents_harness::host::ToolCallRequest,
    ) -> Result<GateDecision> {
        Ok(GateDecision::Allow)
    }

    async fn screen_input(&self, text: &str, origin: ContentOrigin) -> Result<ScreenOutcome> {
        assert_eq!(
            origin,
            ContentOrigin::User,
            "this fixture only submits user text"
        );
        Ok(ScreenOutcome::Redacted(
            text.replace("api-key-123", "[redacted]"),
        ))
    }
}

#[derive(Default)]
struct RecordingLearning {
    summaries: Mutex<Vec<TurnSummary>>,
}

#[async_trait]
impl LearningSink for RecordingLearning {
    async fn on_turn_complete(&self, summary: &TurnSummary) -> Result<()> {
        self.summaries
            .lock()
            .expect("learning lock")
            .push(summary.clone());
        Ok(())
    }
}

fn definition_registry() -> Arc<InMemoryDefinitionRegistry> {
    Arc::new(InMemoryDefinitionRegistry::new(vec![AgentDefinition::new(
        "helper",
        "Helper",
        "A deterministic hosted test agent",
    )]))
}

async fn yield_until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("host terminal finalizer completed within one second");
}

#[tokio::test]
async fn hosted_invocation_composes_screened_input_and_attributes_host_observers() {
    let model = Arc::new(ScriptedModel::replies(vec!["safe reply"]));
    let composer = Arc::new(RecordingComposer::default());
    let progress = Arc::new(RecordingProgressSink::new());
    let learning = Arc::new(RecordingLearning::default());
    let host = HostCapabilities::new(
        composer.clone(),
        definition_registry(),
        Arc::new(RedactingGate),
        Arc::new(FixedModelResolver::new(model.clone())),
    )
    .with_budget(Arc::new(UnlimitedBudgetGate))
    .with_progress(progress.clone())
    .with_learning(learning.clone());
    let harness: AgentHarness<()> = AgentHarness::new();

    let run = harness
        .invoke_agent(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![Message::user("my api-key-123")]),
                RunContext::new(
                    RunConfig::new("hosted-public-contract").with_thread("thread-42"),
                    (),
                ),
            ),
            &(),
        )
        .await
        .expect("host-authorized turn succeeds");

    assert_eq!(run.text().as_deref(), Some("safe reply"));
    let composed = composer.requests.lock().expect("composer lock").clone();
    assert_eq!(composed.len(), 1);
    assert_eq!(composed[0].agent_id, "helper");
    assert_eq!(composed[0].thread_id.as_str(), "thread-42");
    assert_eq!(composed[0].user_text, "my [redacted]");

    let request = model
        .requests()
        .pop()
        .expect("model received hosted request");
    let submitted: Vec<_> = request.messages.iter().map(Message::text).collect();
    assert_eq!(submitted[0], "system for helper");
    assert_eq!(submitted[1], "preamble for thread-42");
    assert!(submitted.iter().any(|text| *text == "my [redacted]"));
    assert!(submitted.iter().all(|text| !text.contains("api-key-123")));

    yield_until(|| learning.summaries.lock().expect("learning lock").len() == 1).await;
    let summary = learning.summaries.lock().expect("learning lock")[0].clone();
    assert_eq!(summary.agent_id, "helper");
    assert_eq!(summary.thread_id.as_str(), "thread-42");
    assert_eq!(summary.input, "my [redacted]");
    assert_eq!(summary.output, "safe reply");

    yield_until(|| progress.events().iter().any(ProgressEvent::is_terminal)).await;
    let events = progress.events();
    assert!(matches!(
        events.first(),
        Some(ProgressEvent::Started { agent, thread, .. })
            if agent == "helper" && thread.as_ref().map(|id| id.as_str()) == Some("thread-42")
    ));
    assert!(matches!(
        events.last(),
        Some(ProgressEvent::Finished { .. })
    ));
}

#[tokio::test]
async fn hosted_stream_projects_internal_model_failure_to_a_safe_terminal_error() {
    let progress = Arc::new(RecordingProgressSink::new());
    let host = HostCapabilities::new(
        Arc::new(RecordingComposer::default()),
        definition_registry(),
        Arc::new(RedactingGate),
        Arc::new(FixedModelResolver::new(Arc::new(ScriptedModel::new(
            vec![],
        )))),
    )
    .with_progress(progress.clone());
    let harness: AgentHarness<()> = AgentHarness::new();

    let mut stream = harness
        .invoke_agent_stream(
            AgentInvocation::new(
                host,
                AgentTurnRequest::new("helper", vec![Message::user("go")]),
                RunContext::new(RunConfig::new("hosted-safe-stream"), ()),
            ),
            &(),
        )
        .await
        .expect("hosted stream starts before model execution");
    let items: Vec<_> = stream.by_ref().collect().await;

    assert!(items.iter().any(|item| matches!(
        item,
        tinyagents_harness::agent_loop::AgentStreamItem::Failed { error, .. }
            if error == "hosted agent invocation failed"
    )));
    assert!(
        !format!("{items:#?}").contains("response queue is exhausted"),
        "the public hosted stream must not disclose model diagnostics"
    );
    yield_until(|| progress.events().iter().any(ProgressEvent::is_terminal)).await;
    assert!(matches!(
        progress.events().last(),
        Some(ProgressEvent::Error { .. })
    ));
}
