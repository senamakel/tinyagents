//! LIVE before/after proof for deferred tool discovery.
//!
//! Opt-in (`TOOL_DEFERRAL_LIVE=1`) and network-gated on `OPENROUTER_API_KEY`
//! (model overridable with `TOOL_DEFERRAL_MODEL`, default `openai/gpt-4.1-mini`).
//! It runs the same task twice against the same real model with the same
//! registry of 41 tools: once with every tool `Direct` (the historical
//! behaviour), once with the 40 long-tail tools `Deferred`. Both runs must
//! reach the one tool the task needs; the deferred run must do it with fewer
//! prompt tokens on its first call and a byte-identical `tools` array on every
//! call. No credential is logged.
//!
//! ```text
//! TOOL_DEFERRAL_LIVE=1 cargo test -p tinyagents-integration-tests \
//!     --test live_tool_deferral -- --nocapture
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::{AgentEvent, RecordingListener};
use tinyagents_harness::middleware::Middleware;
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::providers::openai::OpenAiModel;
use tinytools::{Tool, ToolExposure, ToolResult};

/// The long tail: 40 plausible integration tools a host might register. Only
/// `stock_quote` answers the task.
const LONG_TAIL: &[(&str, &str)] = &[
    ("stock_quote", "Fetch the latest trading price for a stock ticker symbol. Returns price, bid, ask and volume."),
    ("calendar_invite", "Send a calendar invite to one or more attendees for a given time window."),
    ("pdf_read", "Extract the text of a PDF document stored in the workspace."),
    ("slack_post", "Post a message to a Slack channel the user is a member of."),
    ("gmail_search", "Search the user's Gmail inbox with a query string."),
    ("gmail_send", "Send an email from the user's Gmail account."),
    ("github_create_issue", "Open an issue in a GitHub repository."),
    ("github_list_pulls", "List open pull requests for a GitHub repository."),
    ("github_merge_pull", "Merge a pull request in a GitHub repository."),
    ("jira_create_ticket", "Create a Jira ticket in a project."),
    ("jira_transition", "Move a Jira ticket to another workflow state."),
    ("notion_search", "Search pages in the user's Notion workspace."),
    ("notion_append", "Append blocks to a Notion page."),
    ("linear_create_issue", "Create an issue in a Linear team."),
    ("weather_forecast", "Get the weather forecast for a city over the next days."),
    ("currency_convert", "Convert an amount between two currencies at today's rate."),
    ("flight_search", "Search flights between two airports on a date."),
    ("hotel_search", "Search hotels in a city for a date range."),
    ("maps_directions", "Get driving directions between two places."),
    ("maps_geocode", "Turn an address into latitude and longitude."),
    ("image_generate", "Generate an image from a text prompt."),
    ("image_describe", "Describe the contents of an image file."),
    ("audio_transcribe", "Transcribe an audio file to text."),
    ("text_to_speech", "Synthesise speech audio from text."),
    ("translate_text", "Translate text between two languages."),
    ("spreadsheet_read", "Read a range of cells from a spreadsheet."),
    ("spreadsheet_write", "Write values into a range of cells in a spreadsheet."),
    ("database_query", "Run a read-only SQL query against the analytics database."),
    ("cron_schedule", "Schedule a recurring job with a cron expression."),
    ("cron_cancel", "Cancel a scheduled recurring job."),
    ("webhook_call", "Call an outbound webhook URL with a JSON payload."),
    ("browser_open", "Open a URL in the headless browser and return the page text."),
    ("browser_click", "Click an element on the current headless browser page."),
    ("crypto_price", "Fetch the current price of a cryptocurrency."),
    ("news_search", "Search recent news articles for a topic."),
    ("wikipedia_summary", "Fetch the summary of a Wikipedia article."),
    ("contacts_lookup", "Look up a person in the user's address book."),
    ("todo_add", "Add an item to the user's to-do list."),
    ("todo_complete", "Mark a to-do item as complete."),
    ("timer_set", "Set a countdown timer that notifies the user when it ends."),
];

struct LiveTool {
    name: &'static str,
    description: &'static str,
    exposure: ToolExposure,
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

#[async_trait]
impl Tool for LiveTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn parameters_schema(&self) -> Value {
        // Every long-tail tool takes a realistic argument surface so the
        // "before" run pays the schema cost a real integration catalogue does.
        json!({
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Primary subject of the call (symbol, id, address, query…)."},
                "options": {
                    "type": "object",
                    "description": "Optional request settings.",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum number of results.", "minimum": 1, "maximum": 100},
                        "dry_run": {"type": "boolean", "description": "Validate the request without performing it."},
                        "format": {"type": "string", "enum": ["json", "text"], "description": "Response format."}
                    }
                }
            },
            "required": ["target"]
        })
    }

    fn exposure(&self) -> ToolExposure {
        self.exposure
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        self.calls
            .lock()
            .unwrap()
            .push((self.name.to_string(), args.clone()));
        Ok(ToolResult::success(match self.name {
            "stock_quote" => format!(
                "{}: last 142.17 USD, bid 142.10, ask 142.25, volume 3.1M",
                args["target"].as_str().unwrap_or("?")
            ),
            other => format!("{other}: ok"),
        }))
    }
}

struct ReadNote;

#[async_trait]
impl Tool for ReadNote {
    fn name(&self) -> &str {
        "read_note"
    }

    fn description(&self) -> &str {
        "Read one of the user's saved notes by title."
    }

    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]})
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("(empty note)"))
    }
}

/// Records every request's serialised `tools` array and delegates to the
/// real provider.
struct Observed {
    inner: OpenAiModel,
    tools_seen: Mutex<Vec<String>>,
}

#[async_trait]
impl ChatModel<()> for Observed {
    fn profile(&self) -> Option<&tinyinference_llm::model::ModelProfile> {
        ChatModel::<()>::profile(&self.inner)
    }

    async fn invoke(
        &self,
        state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        self.tools_seen
            .lock()
            .unwrap()
            .push(serde_json::to_string(&request.tools).expect("tools serialise"));
        self.inner.invoke(state, request).await
    }
}

struct Capture {
    listener: Arc<RecordingListener>,
}

#[async_trait]
impl Middleware<(), ()> for Capture {
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

struct Outcome {
    label: &'static str,
    model_calls: usize,
    first_call_input_tokens: u64,
    total_input_tokens: u64,
    advertised: usize,
    deferred: usize,
    schema_bytes: usize,
    tools_byte_stable: bool,
    searched: bool,
    quoted: bool,
}

async fn run_once(label: &'static str, api_key: &str, model_name: &str, long_tail: ToolExposure) -> Outcome {
    let listener = Arc::new(RecordingListener::new());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let model = Arc::new(Observed {
        inner: OpenAiModel::openrouter(api_key.to_string()).with_model(model_name),
        tools_seen: Mutex::new(Vec::new()),
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("live", model.clone())
        .set_default_model("live")
        .register_tool(Arc::new(ReadNote))
        .push_middleware(Arc::new(Capture {
            listener: listener.clone(),
        }));
    for (name, description) in LONG_TAIL {
        harness.register_tool(Arc::new(LiveTool {
            name,
            description,
            exposure: long_tail,
            calls: calls.clone(),
        }));
    }

    let run = harness
        .invoke_default(
            &(),
            vec![
                Message::system(
                    "You are a terse assistant. Use tools when they help; if a needed tool \
                     is not in your list, look for it before giving up. Answer in one line.",
                ),
                Message::user("What is ACME stock trading at right now?"),
            ],
        )
        .await
        .expect("live run succeeds");

    let events: Vec<AgentEvent> = listener.events().into_iter().map(|r| r.event).collect();
    let mut inputs = events.iter().filter_map(|event| match event {
        AgentEvent::ModelCompleted { usage, .. } => usage.as_ref().map(|u| u.input_tokens),
        _ => None,
    });
    let first_call_input_tokens = inputs.next().unwrap_or(0);
    let (advertised, deferred, schema_bytes) = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolsAdvertised {
                direct,
                deferred,
                schema_bytes,
            } => Some((*direct, *deferred, *schema_bytes)),
            _ => None,
        })
        .expect("ToolsAdvertised emitted");
    let seen = model.tools_seen.lock().unwrap().clone();
    let quoted = calls
        .lock()
        .unwrap()
        .iter()
        .any(|(name, _)| name == "stock_quote");

    Outcome {
        label,
        model_calls: run.model_calls,
        first_call_input_tokens,
        total_input_tokens: run.usage.usage.input_tokens,
        advertised,
        deferred,
        schema_bytes,
        tools_byte_stable: seen.iter().all(|tools| tools == &seen[0]),
        searched: events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolSearched { .. })),
        quoted,
    }
}

#[tokio::test]
async fn live_deferral_reaches_the_same_tool_with_fewer_prompt_tokens() {
    if std::env::var("TOOL_DEFERRAL_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping live tool-deferral check: set TOOL_DEFERRAL_LIVE=1");
        return;
    }
    let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") else {
        eprintln!("skipping live tool-deferral check: OPENROUTER_API_KEY is not set");
        return;
    };
    let model_name =
        std::env::var("TOOL_DEFERRAL_MODEL").unwrap_or_else(|_| "openai/gpt-4.1-mini".to_string());

    let before = run_once("before (all Direct)", &api_key, &model_name, ToolExposure::Direct).await;
    let after = run_once("after (long tail Deferred)", &api_key, &model_name, ToolExposure::Deferred).await;

    eprintln!("\nmodel: {model_name}");
    eprintln!(
        "{:<28} {:>6} {:>10} {:>10} {:>12} {:>9} {:>8} {:>7}",
        "run", "calls", "1st input", "Σ input", "schema bytes", "on wire", "search", "quoted"
    );
    for o in [&before, &after] {
        eprintln!(
            "{:<28} {:>6} {:>10} {:>10} {:>12} {:>4}+{:<4} {:>8} {:>7}",
            o.label,
            o.model_calls,
            o.first_call_input_tokens,
            o.total_input_tokens,
            o.schema_bytes,
            o.advertised,
            o.deferred,
            o.searched,
            o.quoted
        );
    }
    eprintln!(
        "first-call prompt tokens: {} -> {} ({:.0}% fewer); tools byte-stable across the deferred run: {}\n",
        before.first_call_input_tokens,
        after.first_call_input_tokens,
        100.0 * (1.0 - after.first_call_input_tokens as f64 / before.first_call_input_tokens.max(1) as f64),
        after.tools_byte_stable
    );

    assert!(before.quoted, "the direct run should have called stock_quote");
    assert!(after.quoted, "the deferred run should have discovered and called stock_quote");
    assert!(after.searched, "the deferred run should have gone through tool_search");
    assert_eq!(after.deferred, LONG_TAIL.len());
    assert!(after.schema_bytes < before.schema_bytes / 3);
    assert!(
        after.first_call_input_tokens < before.first_call_input_tokens,
        "deferral should cut the first call's prompt"
    );
    assert!(after.tools_byte_stable, "the tools array must not change within a run");
}
