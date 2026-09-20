//! Offline end-to-end coverage for the harness's composable agent features.
//!
//! Every test drives the public [`AgentHarness`] API with deterministic mock
//! providers.  Together they cover response caching, prompt-prefix protection,
//! tool loops, child agents, transcript compaction, lifecycle hooks, reused
//! multi-turn sessions, and a small deterministic combination matrix.

use std::sync::Arc;

use serde_json::json;

use tinyagents_harness::cache::{InMemoryResponseCache, PROMPT_CACHE_KEY_OPTION, ResponseCache};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::middleware::{LoggingMiddleware, MicrocompactMiddleware, Middleware};
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::subagent::ChildDataPolicy;
use tinyagents_harness::testkit::{FakeTool, ScriptedModel};
use tinyagents_harness::{SubAgent, SubAgentSession, SubAgentTool};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn tool_turn(calls: Vec<ToolCall>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some("mock-tool-turn".into()),
            content: Vec::new(),
            tool_calls: calls,
            usage: Some(Usage::new(8, 3)),
            origin: None,
        },
        usage: Some(Usage::new(8, 3)),
        finish_reason: Some("tool_calls".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn text_turn(text: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text.into())],
            tool_calls: Vec::new(),
            usage: Some(Usage::new(5, 2)),
            origin: None,
        },
        usage: Some(Usage::new(5, 2)),
        finish_reason: Some("stop".into()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

/// A probe that observes the request after all earlier `before_model` hooks
/// have had a chance to transform it.
#[derive(Default)]
struct RequestProbe(std::sync::Mutex<Vec<ModelRequest>>);

impl RequestProbe {
    fn requests(&self) -> Vec<ModelRequest> {
        self.0.lock().expect("request probe lock poisoned").clone()
    }
}

#[async_trait::async_trait]
impl Middleware<(), ()> for RequestProbe {
    fn name(&self) -> &str {
        "request_probe"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents_harness::Result<()> {
        self.0
            .lock()
            .expect("request probe lock poisoned")
            .push(request.clone());
        Ok(())
    }
}

#[tokio::test]
async fn mock_provider_reuses_a_cached_single_turn_response() {
    let model = Arc::new(ScriptedModel::replies(vec!["first provider answer"]));
    let cache = Arc::new(InMemoryResponseCache::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .with_response_cache(cache.clone());

    let input = vec![Message::system("Be concise."), Message::user("hello")];
    let first = harness
        .invoke_default(&(), input.clone())
        .await
        .expect("cache miss succeeds");
    let second = harness
        .invoke_default(&(), input)
        .await
        .expect("cache hit succeeds");

    assert_eq!(first.text(), second.text());
    assert_eq!(
        model.requests().len(),
        1,
        "the second turn never reaches the provider"
    );
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.hits, stats.writes), (1, 1, 1));
}

#[tokio::test]
async fn mock_provider_receives_one_stable_prompt_prefix_key_across_a_tool_loop() {
    let model = Arc::new(ScriptedModel::new(vec![
        tool_turn(vec![ToolCall::new(
            "lookup-call",
            "lookup",
            json!({"q": "x"}),
        )]),
        text_turn("complete"),
    ]));
    let mut policy = RunPolicy::default();
    policy.cache.protect_prompt_prefix = true;
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "result")))
        .with_policy(policy);

    harness
        .invoke_default(
            &(),
            vec![
                Message::system("You are stable."),
                Message::user("look up x"),
            ],
        )
        .await
        .expect("tool loop succeeds");

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let keys: Vec<&str> = requests
        .iter()
        .map(|request| {
            request.provider_options[PROMPT_CACHE_KEY_OPTION]
                .as_str()
                .expect("protected request includes a provider cache key")
        })
        .collect();
    assert_eq!(
        keys[0], keys[1],
        "tool results must not perturb the stable prefix"
    );
    assert!(requests.iter().all(|request| {
        request
            .cache_segments
            .iter()
            .any(|segment| segment.id == "system" && segment.cacheable)
            && request
                .cache_segments
                .iter()
                .any(|segment| segment.id == "tools" && segment.cacheable)
    }));
}

#[tokio::test]
async fn tool_loop_compacts_old_results_and_runs_all_lifecycle_hooks() {
    let probe = Arc::new(RequestProbe::default());
    let hooks = Arc::new(LoggingMiddleware::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(ScriptedModel::new(vec![
                tool_turn(vec![ToolCall::new("one", "lookup", json!({"n": 1}))]),
                tool_turn(vec![ToolCall::new("two", "lookup", json!({"n": 2}))]),
                text_turn("complete"),
            ])),
        )
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "large result")))
        .push_middleware(Arc::new(MicrocompactMiddleware::new(1, "[compacted]")))
        .push_middleware(probe.clone())
        .push_middleware(hooks.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("look up two things")])
        .await
        .expect("loop succeeds");

    assert_eq!((run.model_calls, run.tool_calls), (3, 2));
    assert_eq!(run.text(), Some("complete".to_string()));
    let third_request = &probe.requests()[2];
    assert!(
        third_request
            .messages
            .iter()
            .any(|message| message.text() == "[compacted]"),
        "only the outgoing third request is compacted; the full audit transcript remains intact"
    );
    let counts = hooks.counts();
    assert_eq!(counts.before_agent, 1);
    assert_eq!(counts.after_agent, 1);
    assert_eq!((counts.before_model, counts.after_model), (3, 3));
    assert_eq!((counts.before_tool, counts.after_tool), (2, 2));
    assert_eq!(counts.on_error, 0);
}

#[tokio::test]
async fn subagent_session_keeps_context_across_multiple_turns() {
    let child_model = Arc::new(ScriptedModel::replies(vec![
        "first answer",
        "second answer",
    ]));
    let mut child_harness: AgentHarness<()> = AgentHarness::new();
    child_harness.register_model("child", child_model.clone());
    let child = Arc::new(SubAgent::new(
        "researcher",
        "researches",
        Arc::new(child_harness),
    ));

    let mut session = SubAgentSession::new(child);
    session
        .send(&(), (), vec![Message::user("first question")])
        .await
        .expect("first child turn");
    let second = session
        .send(&(), (), vec![Message::user("follow up")])
        .await
        .expect("reused child turn");

    assert_eq!(session.turns(), 2);
    assert_eq!(second.text(), Some("second answer".to_string()));
    let second_request = &child_model.requests()[1];
    let texts: Vec<String> = second_request.messages.iter().map(Message::text).collect();
    assert!(texts.contains(&"first question".to_string()));
    assert!(texts.contains(&"first answer".to_string()));
    assert!(texts.contains(&"follow up".to_string()));
}

/// This is intentionally deterministic rather than random: every combination
/// is reproducible, prints its flags on failure, and still exercises the same
/// branching surface a seed-based fuzz test would explore.
#[tokio::test]
async fn mock_fuzz_matrix_composes_regular_and_subagent_tools() {
    for use_lookup in [false, true] {
        for use_subagent in [false, true] {
            let label = format!("lookup={use_lookup}, subagent={use_subagent}");
            let mut harness: AgentHarness<()> = AgentHarness::new();
            let mut calls = Vec::new();
            if use_lookup {
                harness.register_tool(Arc::new(FakeTool::returning("lookup", "lookup result")));
                calls.push(ToolCall::new("lookup-call", "lookup", json!({"q": "x"})));
            }
            if use_subagent {
                let mut child_harness: AgentHarness<()> = AgentHarness::new();
                child_harness.register_model(
                    "child",
                    Arc::new(ScriptedModel::replies(vec!["child result"])),
                );
                let child = Arc::new(SubAgent::new(
                    "delegate",
                    "delegates",
                    Arc::new(child_harness),
                ));
                harness.register_tool_dispatch(Arc::new(SubAgentTool::new(
                    child,
                    ChildDataPolicy::new(|state: &()| *state),
                )));
                calls.push(ToolCall::new(
                    "child-call",
                    "delegate",
                    json!({"input": "work"}),
                ));
            }
            let responses = if calls.is_empty() {
                vec![text_turn("final")]
            } else {
                vec![tool_turn(calls), text_turn("final")]
            };
            harness
                .register_model("parent", Arc::new(ScriptedModel::new(responses)))
                .set_default_model("parent")
                .with_policy(RunPolicy::default());

            let run = harness
                .invoke_default(&(), vec![Message::user(format!("run {label}"))])
                .await
                .unwrap_or_else(|error| panic!("scenario {label} failed: {error:?}"));
            let expected_tools = usize::from(use_lookup) + usize::from(use_subagent);
            assert_eq!(run.tool_calls, expected_tools, "scenario {label}");
            assert_eq!(
                run.model_calls,
                if expected_tools == 0 { 1 } else { 2 },
                "scenario {label}"
            );
            assert_eq!(run.text(), Some("final".to_string()), "scenario {label}");
        }
    }
}
