//! What a turn observer learns about one streamed model call, end to end.
//!
//! A gateway such as OpenRouter names the upstream provider that served a
//! call in a top-level `provider` field, and gives the response an `id`
//! (`gen-…`) the call can be looked up by afterwards. These tests stream such
//! a response from a loopback server through the real OpenAI-compatible
//! provider and the agent loop, and check that both observations of the call
//! (the logical `Model` and the physical `ModelDispatch`) carry who served
//! it, its response id, the reasoning tokens and the time to the first
//! streamed output.
//!
//! The server is hand-rolled on `std::net` for the same reason as
//! `e2e_stream_error_code.rs`: the dev-dependency `tokio` has no `net`
//! feature.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use tinyagents_harness::events::EventSink;
use tinyagents_harness::observability::turn_observer::{
    self, ObservedUsage, Observer, ObserverScope, TraceContent, TurnObservation,
};
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::openai::OpenAiModel;

/// An OpenRouter-shaped stream: reasoning first, then text, then a usage
/// chunk with reasoning tokens and the gateway's cost.
const OPENROUTER_STREAM: &str = concat!(
    "data: {\"id\":\"gen-1761500000-abc\",\"provider\":\"Wafer\",\"model\":\"moonshotai/kimi-k2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning\":\"thinking\"}}]}\n\n",
    "data: {\"id\":\"gen-1761500000-abc\",\"provider\":\"Wafer\",\"model\":\"moonshotai/kimi-k2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
    "data: {\"id\":\"gen-1761500000-abc\",\"provider\":\"Wafer\",\"model\":\"moonshotai/kimi-k2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"gen-1761500000-abc\",\"provider\":\"Wafer\",\"model\":\"moonshotai/kimi-k2\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"total_tokens\":120,\"prompt_tokens_details\":{\"cached_tokens\":60},\"completion_tokens_details\":{\"reasoning_tokens\":7},\"cost\":0.0012}}\n\n",
    "data: [DONE]\n\n",
);

/// A plain OpenAI-compatible stream: no `provider`, no `id`.
const PLAIN_STREAM: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n",
);

/// Answers every request with `sse_body` as a 200 event stream.
fn start_server(sse_body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind a loopback port");
    let port = listener.local_addr().expect("local addr").port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            serve(stream, sse_body);
        }
    });
    format!("http://127.0.0.1:{port}")
}

/// Reads one request (headers and body), then writes the stream.
fn serve(mut stream: TcpStream, sse_body: &str) {
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(value) = line
            .to_ascii_lowercase()
            .strip_prefix("content-length:")
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            content_length = value;
        }
    }
    let mut body = vec![0u8; content_length];
    let _ = reader.read_exact(&mut body);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        sse_body.len(),
        sse_body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

#[derive(Default)]
struct Recorder(Mutex<Vec<TurnObservation>>);

impl Observer for Recorder {
    fn on_event(&self, event: &TurnObservation) {
        self.0.lock().unwrap().push(event.clone());
    }
}

/// Runs one streamed turn against `sse_body` under an observer scope and
/// returns what the observer saw.
async fn observe_streamed_turn(sse_body: &'static str) -> Vec<TurnObservation> {
    let base_url = start_server(sse_body);
    let model = OpenAiModel::new("test-key")
        .with_base_url(&base_url)
        .with_model("moonshotai/kimi-k2");
    let model = turn_observer::observe_physical_dispatches(Arc::new(model));
    let recorder = Arc::new(Recorder::default());
    let scope = ObserverScope::new(recorder.clone(), TraceContent::MetadataOnly);
    turn_observer::with_observer(scope, async {
        let mut harness: AgentHarness<()> = AgentHarness::new();
        harness.with_model_resolver(Arc::new(
            tinyagents_harness::host::FixedModelResolver::named("remote", model),
        ));
        turn_observer::install(&mut harness, &EventSink::new(), true);
        harness
            .invoke_streaming_default(&(), vec![Message::user("go")])
            .await
            .expect("the streamed run succeeds");
    })
    .await;
    recorder.0.lock().unwrap().clone()
}

#[tokio::test]
async fn a_streamed_call_reports_who_served_it_its_response_id_and_first_token_time() {
    let events = observe_streamed_turn(OPENROUTER_STREAM).await;

    let model = events
        .iter()
        .find(|event| matches!(event, TurnObservation::Model { .. }))
        .expect("one logical model observation");
    let TurnObservation::Model {
        served_by,
        response_id,
        first_token_ms,
        duration_ms,
        usage,
        ..
    } = model
    else {
        unreachable!()
    };
    assert_eq!(served_by.as_deref(), Some("Wafer"));
    assert_eq!(response_id.as_deref(), Some("gen-1761500000-abc"));
    let first_token_ms = first_token_ms.expect("a streamed call has a first-token time");
    assert!(first_token_ms <= *duration_ms);
    let usage: &ObservedUsage = usage.as_ref().expect("usage");
    assert_eq!(usage.reasoning_tokens, 7);
    assert_eq!(usage.cached_tokens, 60);

    let dispatch = events
        .iter()
        .find(|event| matches!(event, TurnObservation::ModelDispatch { .. }))
        .expect("one physical dispatch observation");
    let TurnObservation::ModelDispatch {
        served_by,
        response_id,
        first_token_ms,
        duration_ms,
        usage,
        ..
    } = dispatch
    else {
        unreachable!()
    };
    assert_eq!(served_by.as_deref(), Some("Wafer"));
    assert_eq!(response_id.as_deref(), Some("gen-1761500000-abc"));
    let first_token_ms = first_token_ms.expect("a streamed dispatch has a first-token time");
    assert!(first_token_ms <= *duration_ms);
    assert_eq!(usage.as_ref().map(|usage| usage.reasoning_tokens), Some(7));
}

#[tokio::test]
async fn a_response_without_provider_or_id_reports_neither() {
    let events = observe_streamed_turn(PLAIN_STREAM).await;

    let (served_by, response_id, first_token_ms) = events
        .iter()
        .find_map(|event| match event {
            TurnObservation::Model {
                served_by,
                response_id,
                first_token_ms,
                ..
            } => Some((served_by, response_id, first_token_ms)),
            _ => None,
        })
        .expect("one logical model observation");
    assert_eq!(served_by, &None);
    assert_eq!(response_id, &None);
    assert!(first_token_ms.is_some(), "the text still streamed");
}
