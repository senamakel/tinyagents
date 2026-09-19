//! End-to-end coverage for deferred tool discovery in the agent loop.
//!
//! A `Deferred` tool must never appear in a request's `tools` array; instead
//! the loop advertises the `tool_search` / `tool_call` bridge, answers
//! `tool_search` from its own catalogue, unwraps `tool_call` to the real tool
//! before admission, and keeps the `tools` array byte-identical across every
//! model call of the run — including after a search → call round-trip — so
//! a provider prompt cache is never invalidated by discovery.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::{AgentEvent, RecordingListener};
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyagents_harness::testkit::FakeTool;
use tinyagents_harness::tool::discover::{TOOL_CALL_NAME, TOOL_SEARCH_NAME, ToolDiscoveryPolicy};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolExposure, ToolResult};

/// A tool with a chosen exposure that records what it was called with.
struct ExposedTool {
    name: &'static str,
    description: &'static str,
    exposure: ToolExposure,
    calls: Mutex<Vec<Value>>,
}

impl ExposedTool {
    fn new(name: &'static str, description: &'static str, exposure: ToolExposure) -> Arc<Self> {
        Arc::new(Self {
            name,
            description,
            exposure,
            calls: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Tool for ExposedTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker symbol."}},
            "required": ["symbol"]
        })
    }

    fn exposure(&self) -> ToolExposure {
        self.exposure
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.calls.lock().unwrap().push(args.clone());
        Ok(ToolResult::success(format!(
            "{} → {}",
            self.name,
            args["symbol"].as_str().unwrap_or("?")
        )))
    }
}

/// A scripted model that records the `tools` array of every request.
struct RecordingModel {
    responses: Mutex<Vec<ModelResponse>>,
    tools_seen: Mutex<Vec<String>>,
}

impl RecordingModel {
    fn new(responses: Vec<ModelResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into_iter().rev().collect()),
            tools_seen: Mutex::new(Vec::new()),
        })
    }

    fn tools_seen(&self) -> Vec<String> {
        self.tools_seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.tools_seen
            .lock()
            .unwrap()
            .push(serde_json::to_string(&request.tools).expect("tools serialise"));
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| text("out of script")))
    }
}

fn tool_call(id: &str, name: &str, arguments: Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(1, 1)),
        },
        usage: Some(Usage::new(1, 1)),
        finish_reason: Some("tool_calls".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn text(body: &str) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(body.to_string())],
            tool_calls: Vec::new(),
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

/// A `before_tool` hook that records the tool names it is asked about.
struct BeforeToolSpy {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<(), ()> for BeforeToolSpy {
    fn name(&self) -> &str {
        "before_tool_spy"
    }

    async fn before_tool(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        call: &mut ToolCall,
    ) -> tinyagents_harness::Result<()> {
        self.seen.lock().unwrap().push(call.name.clone());
        Ok(())
    }
}

fn tool_names(tools_json: &str) -> Vec<String> {
    serde_json::from_str::<Vec<Value>>(tools_json)
        .expect("tools array")
        .into_iter()
        .map(|tool| tool["name"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn deferred_tool_is_found_called_and_never_on_the_wire() {
    let listener = Arc::new(RecordingListener::new());
    let before_tool = Arc::new(Mutex::new(Vec::new()));
    let deferred = ExposedTool::new(
        "stock_quote",
        "Fetch the latest price for a ticker symbol.",
        ToolExposure::Deferred,
    );
    let hidden = ExposedTool::new("internal_step", "Host-only step.", ToolExposure::Hidden);
    let model = RecordingModel::new(vec![
        tool_call(
            "c1",
            TOOL_SEARCH_NAME,
            json!({"query": "price of a ticker"}),
        ),
        tool_call(
            "c2",
            TOOL_CALL_NAME,
            json!({"name": "stock_quote", "arguments": {"symbol": "ACME"}}),
        ),
        // A revealed tool is also callable by its own name.
        tool_call("c3", "stock_quote", json!({"symbol": "XYZ"})),
        // A hidden tool is unknown to the model even by name.
        tool_call("c4", "internal_step", json!({"symbol": "no"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(deferred.clone())
        .register_tool(hidden.clone())
        .push_middleware(Arc::new(CaptureMiddleware {
            listener: listener.clone(),
        }))
        .push_middleware(Arc::new(BeforeToolSpy {
            seen: before_tool.clone(),
        }))
        .with_policy(RunPolicy {
            // `ToolSearched.query` follows the same `capture.tool_io` gate as
            // a normal tool call's arguments; enable it so this test's
            // assertion on the recorded query is meaningful.
            capture: tinyagents_harness::runtime::PayloadCapture {
                tool_io: true,
                ..Default::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("what is ACME trading at?")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("done".to_string()));

    // The wire carries the direct tool plus the two bridge tools, in a
    // deterministic order, and is byte-identical on every model call.
    let seen = model.tools_seen();
    assert_eq!(seen.len(), 5);
    assert!(
        seen.iter().all(|tools| tools == &seen[0]),
        "tools array drifted: {seen:#?}"
    );
    assert_eq!(
        tool_names(&seen[0]),
        vec!["read_file", TOOL_SEARCH_NAME, TOOL_CALL_NAME]
    );
    assert!(!seen[0].contains("Fetch the latest price for a ticker symbol."));
    // The manifest names the deferred tool without its schema.
    assert!(seen[0].contains("- stock_quote: Fetch the latest price for a ticker symbol"));
    assert!(!seen[0].contains("internal_step"));

    // Both the bridged and the direct-by-name call reached the real tool.
    let calls = deferred.calls.lock().unwrap().clone();
    assert_eq!(
        calls,
        vec![json!({"symbol": "ACME"}), json!({"symbol": "XYZ"})]
    );
    assert!(hidden.calls.lock().unwrap().is_empty());

    // `before_tool` saw the real name for the bridged call, never `tool_call`;
    // `tool_search` is answered intrinsically before any hook runs.
    let hooks = before_tool.lock().unwrap().clone();
    assert_eq!(hooks, vec!["stock_quote", "stock_quote", "internal_step"]);

    // Transcript: the search answer carries the full schema; the hidden call
    // was answered as unknown, listing only model-callable names.
    let tool_messages: Vec<String> = run
        .messages
        .iter()
        .filter(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .collect();
    assert!(tool_messages[0].contains("\"name\": \"stock_quote\""));
    assert!(tool_messages[0].contains("\"symbol\""));
    assert!(tool_messages[1].contains("stock_quote → ACME"));
    assert!(tool_messages[3].contains("unknown tool `internal_step`"));
    assert!(tool_messages[3].contains("stock_quote"));
    let listed = tool_messages[3]
        .split("valid tools: [")
        .nth(1)
        .expect("valid tool listing");
    assert!(!listed.contains("internal_step"));

    // Events make the surface auditable.
    let events: Vec<AgentEvent> = listener.events().into_iter().map(|r| r.event).collect();
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolsAdvertised { direct: 3, deferred: 1, schema_bytes } if *schema_bytes > 0
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolSearched { matched: 1, query, .. } if query == "price of a ticker"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::DeferredToolCall { tool_name, .. } if tool_name == "stock_quote"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolStarted { tool_name, .. } if tool_name == "stock_quote"
    )));
    assert!(events.iter().all(|event| !matches!(
        event,
        AgentEvent::ToolStarted { tool_name, .. } if tool_name == TOOL_CALL_NAME
    )));
}

#[tokio::test]
async fn disabled_discovery_drops_the_bridge_and_keeps_direct_calls_working() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"query": "quote"})),
        tool_call("c2", "stock_quote", json!({"symbol": "ACME"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")))
        .register_tool(deferred.clone())
        .with_policy(RunPolicy {
            discovery: ToolDiscoveryPolicy {
                enabled: false,
                ..ToolDiscoveryPolicy::default()
            },
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    assert_eq!(tool_names(&model.tools_seen()[0]), vec!["read_file"]);
    // `tool_search` is unknown when nothing advertised it …
    let first_tool_message = run
        .messages
        .iter()
        .find(|message| matches!(message, Message::Tool(_)))
        .map(Message::text)
        .unwrap();
    assert!(first_tool_message.contains("unknown tool `tool_search`"));
    // … but a deferred tool called by name still runs: deferral only subtracts
    // from the wire, never from what the host registered.
    assert_eq!(deferred.calls.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn no_deferred_tools_means_no_bridge() {
    let model = RecordingModel::new(vec![text("done")]);
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("read_file", "contents")));
    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");
    assert_eq!(tool_names(&model.tools_seen()[0]), vec!["read_file"]);
}

#[tokio::test]
async fn host_registered_tool_search_wins_over_the_intrinsic_bridge() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let host_search = ExposedTool::new(
        TOOL_SEARCH_NAME,
        "The host's own search tool.",
        ToolExposure::Direct,
    );
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"symbol": "anything"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(deferred.clone())
        .register_tool(host_search.clone());

    harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // The host's `tool_search` keeps its slot and its description; only the
    // intrinsic `tool_call` half of the bridge is added.
    let tools = model.tools_seen()[0].clone();
    assert_eq!(tool_names(&tools), vec![TOOL_SEARCH_NAME, TOOL_CALL_NAME]);
    assert!(tools.contains("The host's own search tool."));
    assert!(!tools.contains("deferred tool(s) are searchable"));
    // …and the call went to the host's tool, not the intrinsic answer.
    assert_eq!(host_search.calls.lock().unwrap().len(), 1);
}

/// Regression: the collision check that decides whether to advertise an
/// intrinsic bridge schema used to look only at `tool_schemas` (the `Direct`
/// set), while admission's own collision check (`self.tools.dispatch`) sees
/// every exposure. A `Hidden` tool registered as `tool_search` therefore used
/// to be missed here: the loop still advertised the intrinsic `tool_search`
/// schema, but admission suppressed the intrinsic handler for a name it
/// recognized as registered, so a model that used the advertised bridge got
/// an unknown-tool answer instead of a search result. Both paths must use the
/// same collision rule.
#[tokio::test]
async fn hidden_registration_suppresses_the_matching_bridge_schema() {
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let hidden_search = ExposedTool::new(
        TOOL_SEARCH_NAME,
        "Host-internal, never model-visible.",
        ToolExposure::Hidden,
    );
    let model = RecordingModel::new(vec![
        tool_call(
            "c1",
            TOOL_CALL_NAME,
            json!({"name": "stock_quote", "arguments": {"symbol": "ACME"}}),
        ),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(deferred.clone())
        .register_tool(hidden_search.clone());

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");
    assert_eq!(run.text(), Some("done".to_string()));

    // The intrinsic `tool_search` schema is suppressed (the registry already
    // owns that name, even though it is Hidden and unreachable itself); only
    // `tool_call` is advertised.
    let tools = model.tools_seen()[0].clone();
    assert_eq!(tool_names(&tools), vec![TOOL_CALL_NAME]);
    // The deferred tool is still reachable directly through `tool_call`, and
    // the hidden registration never ran.
    assert_eq!(deferred.calls.lock().unwrap().len(), 1);
    assert!(hidden_search.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn tool_schemas_projection_applies_to_wire_and_catalog() {
    use tinyagents_harness::tool::{SchemaCompaction, SchemaPreparation};

    let long = "d".repeat(400);
    let direct = Arc::new(FakeTool::returning("read_file", "contents"));
    let deferred = ExposedTool::new("stock_quote", "Quote.", ToolExposure::Deferred);
    let model = RecordingModel::new(vec![
        tool_call("c1", TOOL_SEARCH_NAME, json!({"query": "quote"})),
        text("done"),
    ]);

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model.clone())
        .set_default_model("mock")
        .register_tool(direct)
        .register_tool(deferred)
        .register_tool(ExposedTool::new(
            "verbose_direct",
            Box::leak(long.clone().into_boxed_str()),
            ToolExposure::Direct,
        ))
        .with_policy(RunPolicy {
            tool_schemas: Some(
                SchemaPreparation::openai().with_compaction(SchemaCompaction {
                    max_description_bytes: Some(40),
                    max_schema_bytes: None,
                }),
            ),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    // The verbose direct description was clipped on the wire…
    let tools: Vec<Value> = serde_json::from_str(&model.tools_seen()[0]).unwrap();
    let verbose = tools
        .iter()
        .find(|t| t["name"] == "verbose_direct")
        .unwrap();
    assert!(verbose["description"].as_str().unwrap().len() <= 40);
    assert!(!model.tools_seen()[0].contains(&long));
    // …and the search answer for the deferred tool went through the same
    // projection (its short description is untouched but present).
    let answer = run
        .messages
        .iter()
        .find(|m| matches!(m, Message::Tool(_)))
        .map(Message::text)
        .unwrap();
    assert!(answer.contains("\"name\": \"stock_quote\""));

    // Regression: the intrinsic `tool_search`/`tool_call` bridge schemas used
    // to be appended *after* provider preparation ran, so they reached the
    // wire unprojected — a `max_description_bytes` budget (or a Gemini
    // `minimum`/`maximum` strip) never applied to them even though the same
    // policy is configured for the whole run. `tool_search`'s description
    // embeds the deferred-tool manifest, which is comfortably over 40 bytes
    // unprojected, so this is a real assertion, not a vacuous one.
    let bridge_search = tools
        .iter()
        .find(|t| t["name"] == TOOL_SEARCH_NAME)
        .unwrap();
    assert!(
        bridge_search["description"].as_str().unwrap().len() <= 40,
        "bridge schema description was not projected through the run's SchemaPreparation"
    );
}
