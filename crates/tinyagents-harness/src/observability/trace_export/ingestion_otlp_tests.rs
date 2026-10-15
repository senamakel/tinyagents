use serde_json::{Value, json};

use super::*;

fn batch() -> Value {
    json!({ "batch": [
        { "id": "e0", "type": "trace-create", "body": {
            "id": "flow:demo:run-1",
            "name": "flow.run:demo",
            "userId": "user-7",
            "sessionId": "flow:demo",
            "release": "1.2.3",
            "environment": "staging",
            "tags": ["run:flow", "trigger:manual"],
            "input": { "goal": "ship" },
            "output": { "ok": true },
            "metadata": { "flow_id": "demo", "health": { "failed": 0 } },
        } },
        { "id": "e1", "type": "span-create", "body": {
            "id": "flow:demo:run-1:run:r1",
            "traceId": "flow:demo:run-1",
            "name": "demo",
            "startTime": "2026-10-15T10:00:00.000Z",
            "endTime": "2026-10-15T10:00:02.000Z",
            "metadata": { "graph_id": "demo" },
        } },
        { "id": "e2", "type": "span-create", "body": {
            "id": "flow:demo:run-1:node:fetch:0",
            "traceId": "flow:demo:run-1",
            "parentObservationId": "flow:demo:run-1:run:r1",
            "name": "fetch",
            "startTime": "2026-10-15T10:00:00.500Z",
            "endTime": "2026-10-15T10:00:01.000Z",
            "level": "ERROR",
            "statusMessage": "boom",
            "metadata": { "langgraph_node": "fetch", "langgraph_step": 0 },
        } },
        { "id": "e3", "type": "event-create", "body": {
            "id": "evt-1",
            "traceId": "flow:demo:run-1",
            "parentObservationId": "flow:demo:run-1:run:r1",
            "name": "route",
            "startTime": "2026-10-15T10:00:01.500Z",
        } },
        { "id": "e4", "type": "generation-create", "body": {
            "id": "gen-1",
            "traceId": "flow:demo:run-1",
            "parentObservationId": "flow:demo:run-1:node:fetch:0",
            "name": "model",
            "startTime": "2026-10-15T10:00:00.600Z",
            "endTime": "2026-10-15T10:00:00.900Z",
            "model": "gpt-x",
            "usage": { "input": 10, "output": 5 },
            "costDetails": { "total": 0.01 },
        } },
        { "id": "e5", "type": "score-create", "body": { "id": "s1", "name": "q", "value": 1 } },
    ] })
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|item| item["key"] == key)
        .map(|item| &item["value"])
}

fn text<'a>(span: &'a Value, key: &str) -> Option<&'a str> {
    attr(span, key)?["stringValue"].as_str()
}

#[test]
fn every_observation_carries_the_trace_attributes_v4_filters_on() {
    let spans = ingestion_batch_to_otlp_spans(&batch()).unwrap();
    assert_eq!(
        spans.len(),
        4,
        "trace-create and score-create are not spans"
    );
    for span in &spans {
        assert_eq!(text(span, "langfuse.trace.name"), Some("flow.run:demo"));
        assert_eq!(text(span, "langfuse.user.id"), Some("user-7"));
        assert_eq!(text(span, "langfuse.session.id"), Some("flow:demo"));
        assert_eq!(text(span, "langfuse.release"), Some("1.2.3"));
        assert_eq!(text(span, "langfuse.environment"), Some("staging"));
        assert_eq!(text(span, "langfuse.trace.metadata.flow_id"), Some("demo"));
        assert_eq!(
            text(span, "langfuse.trace.metadata.health"),
            Some(r#"{"failed":0}"#)
        );
        let tags = &attr(span, "langfuse.trace.tags").unwrap()["arrayValue"]["values"];
        assert_eq!(tags[1]["stringValue"], "trigger:manual");
        assert_eq!(span["traceId"], spans[0]["traceId"]);
    }
}

#[test]
fn trace_io_moves_to_the_root_observation_only() {
    let spans = ingestion_batch_to_otlp_spans(&batch()).unwrap();
    let root = &spans[0];
    assert!(root.get("parentSpanId").is_none());
    assert_eq!(text(root, "langfuse.observation.type"), Some("span"));
    assert_eq!(
        text(root, "langfuse.observation.input"),
        Some(r#"{"goal":"ship"}"#)
    );
    assert_eq!(
        text(root, "langfuse.observation.output"),
        Some(r#"{"ok":true}"#)
    );
    for child in &spans[1..] {
        assert!(text(child, "langfuse.observation.input").is_none());
        assert!(attr(child, "langfuse.trace.input").is_none());
    }
}

#[test]
fn parentage_levels_and_generation_fields_map_to_otel() {
    let spans = ingestion_batch_to_otlp_spans(&batch()).unwrap();
    let (root, node, event, generation) = (&spans[0], &spans[1], &spans[2], &spans[3]);
    assert_eq!(node["parentSpanId"], root["spanId"]);
    assert_eq!(generation["parentSpanId"], node["spanId"]);
    assert_eq!(text(node, "langfuse.observation.level"), Some("ERROR"));
    assert_eq!(
        text(node, "langfuse.observation.status_message"),
        Some("boom")
    );
    assert_eq!(node["status"]["code"], 2);
    assert_eq!(
        text(node, "langfuse.observation.metadata.langgraph_node"),
        Some("fetch")
    );
    assert_eq!(
        text(node, "langfuse.observation.metadata.langgraph_step"),
        Some("0")
    );
    assert_eq!(text(event, "langfuse.observation.type"), Some("event"));
    assert_eq!(event["startTimeUnixNano"], event["endTimeUnixNano"]);
    assert_eq!(
        text(generation, "langfuse.observation.type"),
        Some("generation")
    );
    assert_eq!(
        text(generation, "langfuse.observation.model.name"),
        Some("gpt-x")
    );
    assert_eq!(
        text(generation, "langfuse.observation.usage_details"),
        Some(r#"{"input":10,"output":5}"#)
    );
    assert_eq!(
        text(generation, "langfuse.observation.cost_details"),
        Some(r#"{"total":0.01}"#)
    );
    assert_eq!(root["startTimeUnixNano"], "1792058400000000000");
    assert_eq!(root["endTimeUnixNano"], "1792058402000000000");
}

#[test]
fn span_ids_are_deterministic_across_exports() {
    let first = ingestion_batch_to_otlp_spans(&batch()).unwrap();
    let second = ingestion_batch_to_otlp_spans(&batch()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first[0]["spanId"].as_str().unwrap().len(), 16);
    assert_eq!(first[0]["traceId"].as_str().unwrap().len(), 32);
}

#[test]
fn an_open_span_is_closed_at_its_start_and_flagged() {
    let mut payload = batch();
    payload["batch"][2]["body"]
        .as_object_mut()
        .unwrap()
        .remove("endTime");
    let spans = ingestion_batch_to_otlp_spans(&payload).unwrap();
    assert_eq!(spans[1]["startTimeUnixNano"], spans[1]["endTimeUnixNano"]);
    assert_eq!(
        text(&spans[1], "langfuse.observation.metadata.unfinished"),
        Some("true")
    );
}

#[test]
fn requests_wrap_spans_under_one_resource() {
    let requests = ingestion_batch_otlp_requests(&batch(), "openhuman", "openhuman.flows").unwrap();
    assert_eq!(requests.len(), 1);
    let resource = &requests[0]["resourceSpans"][0];
    assert_eq!(
        resource["resource"]["attributes"][0]["value"]["stringValue"],
        "openhuman"
    );
    assert_eq!(
        resource["scopeSpans"][0]["scope"]["name"],
        "openhuman.flows"
    );
    assert_eq!(
        resource["scopeSpans"][0]["spans"].as_array().unwrap().len(),
        4
    );
}

#[test]
fn a_batch_without_a_trace_is_refused() {
    assert!(ingestion_batch_to_otlp_spans(&json!({ "batch": [] })).is_err());
    assert!(ingestion_batch_to_otlp_spans(&json!({})).is_err());
}
