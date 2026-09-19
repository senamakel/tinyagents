//! End-to-end coverage for tool dialects through the harness.
//!
//! The protocol crate (`tinytools-agent`) owns how a call is rendered and
//! parsed; these tests pin the *host* half: a native model that narrates a
//! call as text — in any grammar — still dispatches it with a harness-minted
//! id; a forced text dialect strips the schemas off the wire and renders the
//! protocol instead; streamed text never shows tool-call markup to a
//! consumer; and a P-Format run parses positional calls.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::config::ToolDispatcher;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::{AgentEvent, RecordingListener};
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::{FakeTool, ScriptedModel, StreamingMock};
use tinyinference_llm::message::{Message, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelDelta, ModelRequest, ModelResponse, ModelStreamItem, ToolChoice,
};
use tinyinference_llm::providers::MockModel;
use tinytools::{Tool, ToolResult};

struct CaptureMiddleware {
    listener: Arc<RecordingListener>,
}

#[async_trait]
impl Middleware<(), ()> for CaptureMiddleware {
    fn name(&self) -> &str {
        "capture"
    }

    async fn before_agent(
        &self,
        ctx: &mut RunContext<()>,
        _state: &(),
    ) -> tinyagents_harness::Result<()> {
        ctx.events.subscribe(self.listener.clone());
        Ok(())
    }
}

/// A tool with a real parameter, so P-Format has a slot to render.
struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "Looks something up."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "q": { "type": "string" } },
            "required": ["q"]
        })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("tool-output"))
    }
}

/// Every tool-call id the run dispatched, from the tool-started events.
fn dispatched_ids(listener: &RecordingListener) -> Vec<String> {
    listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { call_id, .. } => Some(call_id.to_string()),
            _ => None,
        })
        .collect()
}

/// A native-profile model that answers with text only, once, then finishes.
fn narrating_model(text: &str) -> MockModel {
    MockModel::with_responses(vec![
        ModelResponse::assistant(text),
        ModelResponse::assistant("done"),
    ])
}

fn harness_with(
    model: Arc<dyn ChatModel<()>>,
    listener: &Arc<RecordingListener>,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }));
    harness
}

#[tokio::test]
async fn a_native_model_narrating_a_call_in_any_grammar_dispatches_it() {
    for text in [
        "Let me check. <tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "<｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">{\"q\":\"x\"}</｜DSML｜invoke></｜DSML｜tool_calls>",
        "<｜tool▁call▁begin｜>lookup<｜tool▁sep｜>{\"q\":\"x\"}<｜tool▁call▁end｜>",
        "<|channel|>commentary to=functions.lookup<|message|>{\"q\":\"x\"}<|call|>",
        "<tool_call>{\"name\":\"functions.lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
    ] {
        let listener = Arc::new(RecordingListener::new());
        let harness = harness_with(Arc::new(narrating_model(text)), &listener);
        let run = harness
            .invoke_default(&(), vec![Message::user("go")])
            .await
            .expect("run succeeds");
        assert_eq!(run.tool_calls, 1, "{text}");
        let ids = dispatched_ids(&listener);
        assert_eq!(ids.len(), 1, "{text}");
        assert!(
            ids[0].ends_with("-tool-1"),
            "harness-minted id, got {}: {text}",
            ids[0]
        );
    }
}

#[tokio::test]
async fn an_unknown_narrated_tool_is_not_invented_into_a_known_one() {
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(
        Arc::new(narrating_model(
            "<tool_call>{\"name\":\"launch_missiles\",\"arguments\":{}}</tool_call>",
        )),
        &listener,
    );
    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run completes under the default unknown-tool policy");
    // The name reaches the unknown-tool policy as written; it is never
    // fuzzed onto the one registered tool.
    let started: Vec<String> = listener
        .events()
        .into_iter()
        .filter_map(|record| match record.event {
            AgentEvent::ToolStarted { tool_name, .. } => Some(tool_name),
            _ => None,
        })
        .collect();
    assert!(!started.iter().any(|name| name == "lookup"), "{started:?}");
    assert!(run.model_calls >= 1);
}

#[tokio::test]
async fn a_forced_xml_dialect_renders_the_protocol_and_sends_no_schemas() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model.clone(), &listener);
    harness.with_policy(RunPolicy {
        tool_dialect: ToolDispatcher::Xml,
        ..RunPolicy::default()
    });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert!(request.tools.is_empty(), "no schema goes on the wire");
        let system = request
            .messages
            .iter()
            .find(|m| matches!(m, Message::System(_)))
            .expect("a system turn carries the protocol")
            .text();
        assert!(system.contains("## Tool Use Protocol"));
        assert!(system.contains("**lookup**"));
    }
    // The second request replays the call as text and folds the result.
    let replay = &requests[1];
    let assistant = replay
        .messages
        .iter()
        .find(|m| matches!(m, Message::Assistant(_)))
        .expect("assistant turn replayed")
        .text();
    assert!(assistant.contains("<tool_call>"), "{assistant}");
    assert!(
        replay
            .messages
            .iter()
            .any(|m| m.text().contains("<tool_result id=")),
        "results folded into the text envelope"
    );
    assert!(
        !replay
            .messages
            .iter()
            .any(|m| matches!(m, Message::Tool(_)))
    );
}

#[tokio::test]
async fn a_forced_pformat_dialect_parses_positional_calls() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);
    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(system.contains("P-Format"), "{system}");
    assert!(system.contains("lookup[0|<q>]"), "{system}");
}

/// Middleware that forces `tool_choice` before the dialect rewrite runs, the
/// same shape a caller or another middleware forcing a specific tool would
/// produce.
struct ForceToolChoice(ToolChoice);

#[async_trait]
impl Middleware<(), ()> for ForceToolChoice {
    fn name(&self) -> &str {
        "force-tool-choice"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        request: &mut ModelRequest,
    ) -> tinyagents_harness::Result<()> {
        request.tool_choice = self.0.clone();
        Ok(())
    }
}

#[tokio::test]
async fn a_forced_pformat_dialect_preserves_a_forced_required_tool_choice() {
    // Unlike the XML branch (`prompt_tools::with_tool_instructions`, which
    // renders `tool_choice` into its instructions), P-Format has no schema on
    // the wire either — a forced choice has to survive as plain English in
    // the rendered prompt or it silently loses its meaning once the wire
    // `tool_choice` is reset to `Auto`.
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(ForceToolChoice(ToolChoice::Required)))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(
        system.contains("You must emit at least one tool call."),
        "{system}"
    );
    // The wire choice is reset to `Auto` (no schema is on the wire for a
    // text dialect), so this asserts the prompt carries the constraint
    // instead, not that the wire field kept it.
    assert_eq!(model.requests()[0].tool_choice, ToolChoice::Auto);
}

#[tokio::test]
async fn a_forced_pformat_dialect_preserves_a_forced_named_tool_choice() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<tool_call>lookup[0|needle]</tool_call>",
        "done",
    ]));
    let listener = Arc::new(RecordingListener::new());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(Lookup))
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(ForceToolChoice(ToolChoice::Tool(
            "lookup".to_string(),
        ))))
        .with_policy(RunPolicy {
            tool_dialect: ToolDispatcher::Pformat,
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(run.tool_calls, 1);

    let system = model.requests()[0]
        .messages
        .iter()
        .find(|m| matches!(m, Message::System(_)))
        .expect("system")
        .text();
    assert!(system.contains("You must call the `lookup` tool."), "{system}");
}

/// Middleware recording every visible text delta the harness emits.
struct DeltaRecorder {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for DeltaRecorder {
    fn name(&self) -> &str {
        "deltas"
    }

    async fn on_model_delta(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        delta: &mut ModelDelta,
    ) -> tinyagents_harness::Result<()> {
        self.seen.lock().unwrap().push(delta.content.clone());
        Ok(())
    }
}

#[tokio::test]
async fn streamed_tool_call_markup_never_reaches_consumers() {
    let chunks = [
        "Sure, ",
        "<tool_",
        "call>{\"name\":\"lookup\",",
        "\"arguments\":{\"q\":\"x\"}}</tool_call>",
        " checking.",
    ];
    let full: String = chunks.concat();
    let mut items = vec![ModelStreamItem::Started];
    items.extend(
        chunks
            .iter()
            .map(|chunk| ModelStreamItem::MessageDelta(MessageDelta::text(*chunk))),
    );
    items.push(ModelStreamItem::Completed(ModelResponse::assistant(full)));
    // The scripted stream replays the same call every turn; one model call is
    // enough to observe the dispatch and the scrubbed deltas.
    let model = Arc::new(StreamingMock::new(items));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_middleware(Arc::new(DeltaRecorder { seen: seen.clone() }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let _ = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await;

    let deltas = seen.lock().unwrap().clone();
    let joined = deltas.concat();
    assert!(!joined.contains("<tool_call"), "markup leaked: {deltas:?}");
    assert!(
        !joined.contains("</tool_call>"),
        "markup leaked: {deltas:?}"
    );
    assert!(joined.contains("Sure, "), "{deltas:?}");
    assert!(joined.contains(" checking."), "{deltas:?}");
    let ids = dispatched_ids(&listener);
    assert_eq!(ids.len(), 1, "the scrubbed call still dispatches once");
    assert!(ids[0].ends_with("-tool-1"));
}

#[tokio::test]
async fn a_pure_tool_call_stream_leaves_no_raw_markup_in_the_terminal_response() {
    // A response that is *only* tool-call markup, with no ordinary text
    // around it, suppresses every delta (the scrubber holds all of it back),
    // so terminal-content reconciliation must not depend on having seen any
    // *ordinary* streamed text — only on the scrubber having recovered a
    // call. Otherwise the raw `<tool_call>` text produced by the provider
    // (not the scrubbed one) survives in the terminal response's content
    // block, gets persisted into the transcript, and is replayed to the
    // model on the very next turn alongside the structured call.
    let markup = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>";
    let items = vec![
        ModelStreamItem::Started,
        ModelStreamItem::MessageDelta(MessageDelta::text(markup)),
        ModelStreamItem::Completed(ModelResponse::assistant(markup)),
    ];
    let model = Arc::new(StreamingMock::new(items));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = Arc::new(RecordingListener::new());
    let mut harness = harness_with(model, &listener);
    harness
        .push_middleware(Arc::new(DeltaRecorder { seen: seen.clone() }))
        .with_policy(RunPolicy {
            limits: tinyagents_harness::limits::RunLimits {
                max_model_calls: 1,
                behavior: tinyagents_harness::limits::LimitBehavior::StopWithPartial,
                ..tinyagents_harness::limits::RunLimits::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_streaming_default(&(), vec![Message::user("go")])
        .await
        .expect("run stops cleanly with the partial transcript at the call cap");

    let deltas = seen.lock().unwrap().clone();
    let joined = deltas.concat();
    assert!(joined.is_empty(), "no ordinary text streamed: {deltas:?}");

    let ids = dispatched_ids(&listener);
    assert_eq!(ids.len(), 1, "the scrubbed call still dispatches once");

    // The persisted transcript must not carry the raw markup anywhere,
    // including on the assistant turn the terminal response became.
    for message in &run.messages {
        assert!(
            !message.text().contains("<tool_call"),
            "raw markup leaked into the transcript: {:?}",
            run.messages
        );
    }
}

#[tokio::test]
async fn a_signalled_but_missing_tool_call_is_re_prompted_then_recovered() {
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());
    let model = Arc::new(ScriptedModel::new(vec![
        promised,
        ModelResponse::assistant(
            "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>",
        ),
        ModelResponse::assistant("done"),
    ]));
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(model.clone(), &listener);

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(run.model_calls, 3, "one nudge, one call, one final");
    assert_eq!(run.tool_calls, 1);
    let second = &model.requests()[1];
    let last = second.messages.last().expect("nudge appended").text();
    assert!(last.contains("issue the actual tool call now"), "{last}");
}

#[tokio::test]
async fn dropped_tool_call_nudges_are_bounded() {
    let mut promised = ModelResponse::assistant("");
    promised.finish_reason = Some("tool_calls".into());
    let model = Arc::new(ScriptedModel::new(vec![
        promised.clone(),
        promised.clone(),
        promised.clone(),
        promised,
    ]));
    let listener = Arc::new(RecordingListener::new());
    let harness = harness_with(model, &listener);

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("run ends instead of looping");
    assert_eq!(
        run.model_calls, 4,
        "three nudges, then the answer is taken as final"
    );
    assert_eq!(run.tool_calls, 0);
}
