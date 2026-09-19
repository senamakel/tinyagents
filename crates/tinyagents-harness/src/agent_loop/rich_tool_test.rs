//! Tests for B1/B2 consumption in the agent loop: what a tool sees on its
//! [`ToolExecutionContext`] during a real invocation (call id, store, typed
//! state view, `custom` events) and what the loop does with a rich
//! [`tinytools::ToolResult`] (`follow_up` becomes a user message after the
//! batch's tool rows; `metadata` reaches events and the run but never the
//! transcript).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::runtime::AgentHarness;
use crate::store::namespaced::{InMemoryNamespacedStore, Namespace, NamespacedStore};
use crate::testkit::{EventRecorder, ScriptedModel};
use crate::tool::ToolExecutionContext;
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{FileData, ImageData, Tool, ToolCallOptions, ToolContent, ToolResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
struct AppState {
    tenant: &'static str,
}

/// What a tool observed on its harness context during one invocation.
#[derive(Clone, Debug, Default)]
struct Observed {
    call_id: Option<CallId>,
    had_store: bool,
    tenant: Option<&'static str>,
}

/// A tool that downcasts its run context to the harness type, records what it
/// saw, writes to the store, emits a custom event, and returns `result`.
struct ContextTool {
    name: &'static str,
    result: ToolResult,
    observed: Arc<Mutex<Vec<Observed>>>,
}

impl ContextTool {
    fn new(name: &'static str, result: ToolResult) -> (Arc<Self>, Arc<Mutex<Vec<Observed>>>) {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let tool = Arc::new(Self {
            name,
            result,
            observed: Arc::clone(&observed),
        });
        (tool, observed)
    }
}

#[async_trait]
impl Tool for ContextTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "context-aware tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        unreachable!("the harness always dispatches through execute_with_context")
    }
    async fn execute_with_context(
        &self,
        _arguments: serde_json::Value,
        _options: ToolCallOptions,
        context: Option<&dyn tinytools::ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let harness = context
            .and_then(tinytools::ToolRunContext::host_extension)
            .and_then(|any| any.downcast_ref::<ToolExecutionContext>());
        let mut seen = Observed::default();
        if let Some(harness) = harness {
            seen.call_id = Some(harness.call_id.clone());
            seen.had_store = harness.store.is_some();
            seen.tenant = harness.state::<AppState>().map(|s| s.tenant);
            if let Some(store) = &harness.store {
                store
                    .put(
                        &Namespace::from("calls"),
                        harness.call_id.as_str(),
                        json!({"tool": self.name}),
                    )
                    .await?;
            }
            harness.custom(json!({"stage": "working"}));
        }
        self.observed.lock().unwrap().push(seen);
        Ok(self.result.clone())
    }
}

fn response(tool_calls: Vec<ToolCall>, text: &str) -> ModelResponse {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(text.to_string())]
    };
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage: Some(Usage::new(1, 1)),
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn tool_turn(calls: &[(&str, &str)]) -> ModelResponse {
    response(
        calls
            .iter()
            .map(|(id, name)| ToolCall::new(*id, *name, json!({})))
            .collect(),
        "",
    )
}

fn final_turn(text: &str) -> ModelResponse {
    response(Vec::new(), text)
}

/// A compact role rendering of a transcript for ordering assertions.
fn shape(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| match message {
            Message::System(_) => "system".to_string(),
            Message::User(_) => format!("user:{}", message.text()),
            Message::Assistant(a) if !a.tool_calls.is_empty() => {
                format!("assistant:tools[{}]", a.tool_calls.len())
            }
            Message::Assistant(_) => "assistant".to_string(),
            Message::Tool(t) => format!("tool:{}", t.tool_call_id),
        })
        .collect()
}

struct Fixture {
    harness: AgentHarness<()>,
    model: Arc<ScriptedModel>,
    recorder: EventRecorder,
}

fn fixture(responses: Vec<ModelResponse>) -> Fixture {
    let model = Arc::new(ScriptedModel::new(responses));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    Fixture {
        harness,
        model,
        recorder: EventRecorder::new(),
    }
}

impl Fixture {
    fn ctx(&self, run_id: &str) -> RunContext<()> {
        RunContext::new(RunConfig::new(run_id), ()).with_events(self.recorder.sink())
    }
}

// ── Context parity (B1) ─────────────────────────────────────────────────────

#[tokio::test]
async fn a_tool_sees_the_call_id_the_model_used() {
    let mut fx = fixture(vec![
        tool_turn(&[("call-xyz", "ctx")]),
        final_turn("done"),
    ]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);

    let run = fx
        .harness
        .invoke_in_context(&(), fx.ctx("call-id"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let seen = observed.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].call_id, Some(CallId::new("call-xyz")));
    // The transcript answers the same id.
    assert!(run.messages.iter().any(
        |message| matches!(message, Message::Tool(t) if t.tool_call_id == "call-xyz")
    ));
}

#[tokio::test]
async fn concurrent_calls_each_see_their_own_call_id() {
    let mut fx = fixture(vec![
        tool_turn(&[("call-a", "ctx"), ("call-b", "ctx")]),
        final_turn("done"),
    ]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);

    fx.harness
        .invoke_in_context(&(), fx.ctx("concurrent"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let mut ids: Vec<_> = observed
        .lock()
        .unwrap()
        .iter()
        .map(|seen| seen.call_id.clone().unwrap().as_str().to_string())
        .collect();
    ids.sort();
    assert_eq!(ids, ["call-a", "call-b"]);
}

#[tokio::test]
async fn store_is_none_without_one_and_the_run_store_with_one() {
    let mut fx = fixture(vec![tool_turn(&[("c1", "ctx")]), final_turn("done")]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);
    fx.harness
        .invoke_in_context(&(), fx.ctx("no-store"), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert!(!observed.lock().unwrap()[0].had_store);

    let mut fx = fixture(vec![tool_turn(&[("c2", "ctx")]), final_turn("done")]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);
    let store = Arc::new(InMemoryNamespacedStore::new());
    let ctx = fx
        .ctx("with-store")
        .with_namespaced_store(Arc::clone(&store) as Arc<dyn NamespacedStore>);
    fx.harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert!(observed.lock().unwrap()[0].had_store);
    // The tool's write landed on the host's store, keyed by the call id.
    let item = store
        .get(&Namespace::from("calls"), "c2")
        .await
        .unwrap()
        .expect("tool wrote through the run store");
    assert_eq!(item.value, json!({"tool": "ctx"}));
}

#[tokio::test]
async fn state_view_is_typed_when_attached_and_absent_otherwise() {
    let mut fx = fixture(vec![tool_turn(&[("c1", "ctx")]), final_turn("done")]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);
    fx.harness
        .invoke_in_context(&(), fx.ctx("no-state"), vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(observed.lock().unwrap()[0].tenant, None);

    let mut fx = fixture(vec![tool_turn(&[("c2", "ctx")]), final_turn("done")]);
    let (tool, observed) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);
    let ctx = fx
        .ctx("with-state")
        .with_state_view(Arc::new(AppState { tenant: "acme" }));
    fx.harness
        .invoke_in_context(&(), ctx, vec![Message::user("go")])
        .await
        .expect("run succeeds");
    assert_eq!(observed.lock().unwrap()[0].tenant, Some("acme"));
}

#[tokio::test]
async fn custom_events_land_on_the_run_stream_inside_the_call() {
    let mut fx = fixture(vec![tool_turn(&[("c1", "ctx")]), final_turn("done")]);
    let (tool, _) = ContextTool::new("ctx", ToolResult::success("ok"));
    fx.harness.register_tool(tool);
    fx.harness
        .invoke_in_context(&(), fx.ctx("custom"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let events = fx.recorder.events();
    let position = |pred: &dyn Fn(&AgentEvent) -> bool| events.iter().position(pred).unwrap();
    let started = position(&|e| matches!(e, AgentEvent::ToolStarted { .. }));
    let custom = position(&|e| {
        matches!(
            e,
            AgentEvent::Custom { call_id: Some(id), payload }
                if id.as_str() == "c1" && payload == &json!({"stage": "working"})
        )
    });
    let completed = position(&|e| matches!(e, AgentEvent::ToolCompleted { .. }));
    assert!(
        started < custom && custom < completed,
        "custom event is ordered inside its call: {:?}",
        fx.recorder.kinds()
    );
}

// ── Rich returns (B2) ───────────────────────────────────────────────────────

#[tokio::test]
async fn follow_up_becomes_a_user_message_after_the_tool_result() {
    let mut fx = fixture(vec![tool_turn(&[("c1", "shot")]), final_turn("done")]);
    let (tool, _) = ContextTool::new(
        "shot",
        ToolResult::success("clicked").with_follow_up([ToolContent::Text {
            text: "Here is the page after the click.".into(),
        }]),
    );
    fx.harness.register_tool(tool);

    let run = fx
        .harness
        .invoke_in_context(&(), fx.ctx("follow-up"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    assert_eq!(
        shape(&run.messages),
        [
            "user:go",
            "assistant:tools[1]",
            "tool:c1",
            "user:Here is the page after the click.",
            "assistant",
        ]
    );
    // The next model call saw it.
    let requests = fx.model.requests();
    assert_eq!(
        shape(&requests[1].messages)[3],
        "user:Here is the page after the click."
    );
    // The tool-result row itself does not carry the follow-up.
    let Message::Tool(row) = &run.messages[2] else {
        panic!("tool row");
    };
    assert_eq!(row.content, vec![ContentBlock::Text("clicked".into())]);
}

#[tokio::test]
async fn follow_ups_in_a_batch_come_after_every_tool_row_in_call_order() {
    let mut fx = fixture(vec![
        tool_turn(&[("c1", "first"), ("c2", "second")]),
        final_turn("done"),
    ]);
    let (first, _) = ContextTool::new(
        "first",
        ToolResult::success("1").with_follow_up([ToolContent::Text {
            text: "after first".into(),
        }]),
    );
    let (second, _) = ContextTool::new(
        "second",
        ToolResult::success("2").with_follow_up([ToolContent::Text {
            text: "after second".into(),
        }]),
    );
    fx.harness.register_tool(first);
    fx.harness.register_tool(second);

    let run = fx
        .harness
        .invoke_in_context(&(), fx.ctx("batch"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // Both tool rows stay adjacent to the assistant row (a provider
    // requirement); the follow-ups trail the batch in call order.
    assert_eq!(
        shape(&run.messages),
        [
            "user:go",
            "assistant:tools[2]",
            "tool:c1",
            "tool:c2",
            "user:after first",
            "user:after second",
            "assistant",
        ]
    );
}

#[tokio::test]
async fn follow_up_image_becomes_an_image_block_and_a_file_a_placeholder() {
    let mut fx = fixture(vec![tool_turn(&[("c1", "shot")]), final_turn("done")]);
    let (tool, _) = ContextTool::new(
        "shot",
        ToolResult::success("ok").with_follow_up([
            ToolContent::Image {
                media_type: "image/png".into(),
                data: ImageData::Url("https://example.test/shot.png".into()),
            },
            ToolContent::Image {
                media_type: "image/jpeg".into(),
                data: ImageData::Base64("AA==".into()),
            },
            ToolContent::File {
                name: "report.pdf".into(),
                media_type: "application/pdf".into(),
                data: FileData::Path("/tmp/report.pdf".into()),
            },
            ToolContent::Json {
                data: json!({"k": 1}),
            },
        ]),
    );
    fx.harness.register_tool(tool);

    let run = fx
        .harness
        .invoke_in_context(&(), fx.ctx("blocks"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    let Message::User(follow_up) = &run.messages[3] else {
        panic!("expected the follow-up user message, got {:?}", run.messages[3]);
    };
    assert_eq!(
        follow_up.content,
        vec![
            ContentBlock::Image(tinyinference_llm::message::ImageRef {
                url: "https://example.test/shot.png".into(),
                mime_type: Some("image/png".into()),
            }),
            ContentBlock::Image(tinyinference_llm::message::ImageRef {
                url: "data:image/jpeg;base64,AA==".into(),
                mime_type: Some("image/jpeg".into()),
            }),
            ContentBlock::Text("[file report.pdf (application/pdf)]".into()),
            ContentBlock::Json(json!({"k": 1})),
        ]
    );
}

#[tokio::test]
async fn metadata_reaches_the_event_and_the_run_but_never_the_transcript() {
    const MARKER: &str = "host-only-marker-7f3a";
    let mut fx = fixture(vec![tool_turn(&[("c1", "meta")]), final_turn("done")]);
    let (tool, _) = ContextTool::new(
        "meta",
        ToolResult::success("visible").with_metadata(json!({"marker": MARKER})),
    );
    fx.harness.register_tool(tool);

    let run = fx
        .harness
        .invoke_in_context(&(), fx.ctx("metadata"), vec![Message::user("go")])
        .await
        .expect("run succeeds");

    // On the event.
    let completed = fx
        .recorder
        .events()
        .into_iter()
        .find_map(|event| match event {
            AgentEvent::ToolCompleted {
                call_id, metadata, ..
            } => Some((call_id, metadata)),
            _ => None,
        })
        .expect("ToolCompleted emitted");
    assert_eq!(completed.0, CallId::new("c1"));
    assert_eq!(completed.1, Some(json!({"marker": MARKER})));

    // On the run.
    assert_eq!(run.executed_tools, ["meta"]);
    assert_eq!(run.tool_metadata.len(), 1);
    assert_eq!(run.tool_metadata[0].call_id, CallId::new("c1"));
    assert_eq!(run.tool_metadata[0].tool_name, "meta");
    assert_eq!(run.tool_metadata[0].metadata, json!({"marker": MARKER}));

    // Nowhere in what the model was sent, nor in the run transcript — not
    // even in the tool row's host-side artifact.
    for request in fx.model.requests() {
        let wire = serde_json::to_string(&request.messages).unwrap();
        assert!(!wire.contains(MARKER), "metadata leaked to the model: {wire}");
    }
    let transcript = serde_json::to_string(&run.messages).unwrap();
    assert!(
        !transcript.contains(MARKER),
        "metadata leaked into the transcript: {transcript}"
    );
}
