//! Langfuse v4 OTLP projection of a locally built ingestion batch.
//!
//! Langfuse v4 drops the legacy `/api/public/ingestion` trace path: a trace is
//! no longer a record of its own, only the OTEL trace id its observations
//! share, and every observation is written once to an immutable table. This
//! module converts a `{"batch": [...]}` payload that an exporter assembled in
//! memory (one `trace-create` plus completed `span-create` /
//! `event-create` / `generation-create` events, as the graph and harness
//! exporters build it) into OTLP spans that follow the v4 attribute rules:
//!
//! - trace name, user, session, tags, metadata, environment, release and
//!   version are copied onto **every** span, because v4 filters observations
//!   directly and a root-only value is invisible on a child;
//! - trace input/output move to the root observation
//!   (`langfuse.observation.input` / `langfuse.observation.output`);
//! - parentage rides the OTEL parent span context, and span ids are derived
//!   deterministically from the batch's observation ids, so a run exported in
//!   another request still nests under the same parent.
//!
//! The conversion is pure. Events of any other type (`score-create`,
//! `*-update`, `sdk-log`) are not traces in v4 and are skipped; scores keep
//! using the Langfuse scores API. Each observation in the batch must already be
//! final: v4 rejects nothing on re-export, it duplicates, so a caller exports
//! a span once.

use serde_json::{Map, Value, json};

use super::otlp::{attribute, chunk_otlp_spans, hex_id, json_attribute};
use crate::error::{Result, TinyAgentsError};

/// The `langfuse.observation.type` for a batch event type, or `None` when the
/// event is not an observation (trace, score, update, log).
fn observation_type(event_type: &str) -> Option<&'static str> {
    match event_type {
        "span-create" => Some("span"),
        "event-create" => Some("event"),
        "generation-create" => Some("generation"),
        _ => None,
    }
}

/// A metadata key as an OTEL attribute suffix: dots would read as nesting.
fn metadata_key(key: &str) -> String {
    key.replace('.', "_")
}

/// A scalar string stays as written; anything else is serialized as JSON,
/// which is what Langfuse expects for structured attribute values.
fn value_attribute(key: &str, value: &Value) -> Value {
    match value {
        Value::String(text) => attribute(key, text.clone()),
        other => json_attribute(key, other),
    }
}

/// `key = value` attributes for each entry of a metadata object, under
/// `prefix.<key>`. A non-object value lands under `prefix` itself.
fn metadata_attributes(prefix: &str, metadata: Option<&Value>, out: &mut Vec<Value>) {
    match metadata {
        Some(Value::Object(map)) => {
            for (key, value) in map {
                if !value.is_null() {
                    out.push(value_attribute(
                        &format!("{prefix}.{}", metadata_key(key)),
                        value,
                    ));
                }
            }
        }
        Some(Value::Null) | None => {}
        Some(other) => out.push(value_attribute(prefix, other)),
    }
}

/// An ISO-8601 batch timestamp as OTLP unix nanoseconds.
fn iso_to_nanos(value: Option<&Value>) -> Option<u64> {
    let text = value?.as_str()?;
    let parsed = chrono::DateTime::parse_from_rfc3339(text).ok()?;
    u64::try_from(parsed.timestamp_nanos_opt()?).ok()
}

/// The trace-level attributes v4 needs on every observation.
fn trace_attributes(trace: &Map<String, Value>) -> Vec<Value> {
    let mut attrs = Vec::new();
    let text = |key: &str| trace.get(key).and_then(Value::as_str);
    if let Some(name) = text("name") {
        attrs.push(attribute("langfuse.trace.name", name));
    }
    if let Some(user) = text("userId") {
        attrs.push(attribute("langfuse.user.id", user));
    }
    if let Some(session) = text("sessionId") {
        attrs.push(attribute("langfuse.session.id", session));
    }
    for (field, key) in [
        ("environment", "langfuse.environment"),
        ("release", "langfuse.release"),
        ("version", "langfuse.version"),
    ] {
        if let Some(value) = text(field) {
            attrs.push(attribute(key, value));
        }
    }
    if let Some(tags) = trace.get("tags").and_then(Value::as_array) {
        let values: Vec<Value> = tags
            .iter()
            .filter_map(Value::as_str)
            .map(|tag| json!({ "stringValue": tag }))
            .collect();
        if !values.is_empty() {
            attrs.push(json!({
                "key": "langfuse.trace.tags",
                "value": { "arrayValue": { "values": values } },
            }));
        }
    }
    metadata_attributes("langfuse.trace.metadata", trace.get("metadata"), &mut attrs);
    attrs
}

/// One observation event as an OTLP span. `root_io` carries the trace's
/// input/output onto the root observation when that span has none of its own.
fn observation_span(
    trace_hex: &str,
    kind: &str,
    body: &Map<String, Value>,
    trace_attrs: &[Value],
    root_io: Option<(&Value, &Value)>,
) -> Option<Value> {
    let id = body.get("id")?.as_str()?;
    let start = iso_to_nanos(body.get("startTime")).or_else(|| iso_to_nanos(body.get("timestamp")));
    let start = start?;
    let end = iso_to_nanos(body.get("endTime"));

    let mut attrs = vec![attribute("langfuse.observation.type", kind)];
    attrs.extend(trace_attrs.iter().cloned());

    let present = |key: &str| body.get(key).filter(|value| !value.is_null());
    let input = present("input").or(root_io.map(|(input, _)| input).filter(|v| !v.is_null()));
    let output = present("output").or(root_io.map(|(_, output)| output).filter(|v| !v.is_null()));
    if let Some(input) = input {
        attrs.push(json_attribute("langfuse.observation.input", input));
    }
    if let Some(output) = output {
        attrs.push(json_attribute("langfuse.observation.output", output));
    }
    let level = present("level").and_then(Value::as_str);
    let status_message = present("statusMessage").and_then(Value::as_str);
    if let Some(level) = level {
        attrs.push(attribute("langfuse.observation.level", level));
    }
    if let Some(message) = status_message {
        attrs.push(attribute("langfuse.observation.status_message", message));
    }
    if let Some(version) = present("version").and_then(Value::as_str) {
        attrs.push(attribute("langfuse.version", version));
    }
    if let Some(model) = present("model").and_then(Value::as_str) {
        attrs.push(attribute("langfuse.observation.model.name", model));
    }
    for (field, key) in [
        ("modelParameters", "langfuse.observation.model.parameters"),
        ("usageDetails", "langfuse.observation.usage_details"),
        ("costDetails", "langfuse.observation.cost_details"),
    ] {
        if let Some(value) = present(field) {
            attrs.push(json_attribute(key, value));
        }
    }
    if present("usageDetails").is_none()
        && let Some(usage) = present("usage")
    {
        attrs.push(json_attribute("langfuse.observation.usage_details", usage));
    }
    if let Some(at) = present("completionStartTime").and_then(Value::as_str) {
        attrs.push(attribute("langfuse.observation.completion_start_time", at));
    }
    metadata_attributes(
        "langfuse.observation.metadata",
        present("metadata"),
        &mut attrs,
    );
    if end.is_none() && kind != "event" {
        // A span still open when the batch was built: v4 exports a span once,
        // so it goes out closed at its start and is flagged for readers.
        attrs.push(attribute("langfuse.observation.metadata.unfinished", "true"));
    }

    let name = body.get("name").and_then(Value::as_str).unwrap_or(kind);
    let mut span = json!({
        "traceId": trace_hex,
        "spanId": hex_id(id, 8),
        "name": name,
        "startTimeUnixNano": start.to_string(),
        "endTimeUnixNano": end.unwrap_or(start).max(start).to_string(),
        "attributes": attrs,
    });
    if let Some(parent) = present("parentObservationId").and_then(Value::as_str) {
        span["parentSpanId"] = json!(hex_id(parent, 8));
    }
    if level == Some("ERROR") {
        span["status"] = json!({ "code": 2, "message": status_message.unwrap_or_default() });
    }
    Some(span)
}

/// Convert a locally built Langfuse ingestion batch into Langfuse v4 OTLP
/// spans (see the module docs for the attribute rules).
///
/// The OTEL trace id is derived from the `trace-create` id, the same way
/// [`super::otlp::otlp_requests`] derives it from a span's trace id, so two
/// exporters that agree on a trace id land in one Langfuse trace. The first
/// observation without a parent is the root and receives the trace's
/// input/output when it has none of its own.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Validation`] when the payload has no `batch`
/// array or no `trace-create` event with an id.
pub fn ingestion_batch_to_otlp_spans(payload: &Value) -> Result<Vec<Value>> {
    let events = payload
        .get("batch")
        .and_then(Value::as_array)
        .ok_or_else(|| TinyAgentsError::Validation("ingestion payload has no batch".into()))?;
    let trace = events
        .iter()
        .find(|event| event["type"] == "trace-create")
        .and_then(|event| event["body"].as_object())
        .ok_or_else(|| TinyAgentsError::Validation("ingestion batch has no trace-create".into()))?;
    let trace_id = trace
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| TinyAgentsError::Validation("trace-create has no id".into()))?;
    let trace_hex = hex_id(trace_id, 16);
    let trace_attrs = trace_attributes(trace);
    let null = Value::Null;
    let trace_io = (
        trace.get("input").unwrap_or(&null),
        trace.get("output").unwrap_or(&null),
    );

    let mut root_assigned = false;
    let mut spans = Vec::with_capacity(events.len());
    for event in events {
        let Some(kind) = event["type"].as_str().and_then(observation_type) else {
            continue;
        };
        let Some(body) = event["body"].as_object() else {
            continue;
        };
        let is_root = !root_assigned
            && body
                .get("parentObservationId")
                .is_none_or(Value::is_null);
        let root_io = is_root.then_some((trace_io.0, trace_io.1));
        if let Some(span) = observation_span(&trace_hex, kind, body, &trace_attrs, root_io) {
            root_assigned |= is_root;
            spans.push(span);
        }
    }
    Ok(spans)
}

/// [`ingestion_batch_to_otlp_spans`] wrapped into OTLP/HTTP JSON request
/// bodies (resource `service.name = service`, scope `scope`), chunked like
/// [`super::otlp::otlp_requests`].
///
/// # Errors
///
/// As [`ingestion_batch_to_otlp_spans`].
pub fn ingestion_batch_otlp_requests(
    payload: &Value,
    service: &str,
    scope: &str,
) -> Result<Vec<Value>> {
    Ok(chunk_otlp_spans(
        ingestion_batch_to_otlp_spans(payload)?,
        service,
        scope,
    ))
}

#[cfg(test)]
#[path = "ingestion_otlp_tests.rs"]
mod tests;
