//! The stored form of a [`TurnState`] in `session.turn_states`.
//!
//! [`TurnState`] is a wire type: camelCase, with RFC 3339 `startedAt` and
//! `updatedAt`, mirrored 1:1 by the app's chat slice. The stored form is its
//! snake_case storage DTO, built from the wire form by a fixed mapping:
//!
//! | wire | stored |
//! | --- | --- |
//! | every camelCase field name, at any depth | snake_case |
//! | `startedAt`, `updatedAt` (RFC 3339) | `started_ms`, `updated_ms` (epoch ms) |
//! | `transcript[].startedAt` / `endedAt` of a `thinking` item (epoch ms) | `started_ms` / `ended_ms` |
//! | a sub-agent tool call's `args` | verbatim: it is the tool's own JSON, not a field of ours |
//!
//! The mapping is bijective on the fields a [`TurnState`] has, so a turn
//! written and read back is equal to the one written; the wire format is not
//! touched. Times keep millisecond precision and come back as UTC (`Z`).

use serde_json::{Map, Value};
use tinystoragedrivers_types::StorageError;

use crate::turn_state::TurnState;

/// The free-form payload key whose value is passed through untouched.
const OPAQUE: &str = "args";

/// The top-level time fields: `(wire, stored)`.
const TIMES: [(&str, &str); 2] = [("startedAt", "started_ms"), ("updatedAt", "updated_ms")];

/// The epoch-ms fields of a `thinking` transcript item: `(stored before
/// the rename, stored)`.
const THINKING_TIMES: [(&str, &str); 2] = [("started_at", "started_ms"), ("ended_at", "ended_ms")];

fn invalid(message: impl Into<String>) -> StorageError {
    StorageError::serialization(message.into())
}

/// The stored body of `state`.
///
/// # Errors
///
/// When `state` does not serialize, or a time is not RFC 3339.
pub(in crate::port::drivers) fn encode(state: &TurnState) -> Result<Value, StorageError> {
    let Value::Object(mut wire) =
        serde_json::to_value(state).map_err(|error| invalid(error.to_string()))?
    else {
        return Err(invalid("a turn state serializes as an object"));
    };
    let mut stored = Map::new();
    for (wire_name, stored_name) in TIMES {
        let text = wire
            .remove(wire_name)
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| invalid(format!("turn state has no {wire_name}")))?;
        let parsed = chrono::DateTime::parse_from_rfc3339(&text).map_err(|error| {
            invalid(format!(
                "turn state {wire_name} {text:?} is not RFC 3339: {error}"
            ))
        })?;
        stored.insert(stored_name.to_owned(), parsed.timestamp_millis().into());
    }
    for (name, value) in wire {
        let key = snake_case(&name);
        let value = if key == "transcript" {
            thinking_times_to_ms(snake_keys(value))
        } else {
            snake_keys(value)
        };
        stored.insert(key, value);
    }
    Ok(Value::Object(stored))
}

/// The [`TurnState`] stored as `body`.
///
/// # Errors
///
/// When `body` is missing a time or is not a turn state.
pub(in crate::port::drivers) fn decode(body: &Value) -> Result<TurnState, StorageError> {
    let Value::Object(stored) = body else {
        return Err(invalid("a stored turn state is an object"));
    };
    let mut wire = Map::new();
    let mut times = Vec::new();
    for (name, value) in stored {
        if let Some((wire_name, _)) = TIMES.iter().find(|(_, stored_name)| name == stored_name) {
            let ms = value
                .as_i64()
                .ok_or_else(|| invalid(format!("turn state {name} is not an integer")))?;
            let instant = chrono::DateTime::from_timestamp_millis(ms)
                .ok_or_else(|| invalid(format!("turn state {name} {ms} is out of range")))?;
            times.push((
                (*wire_name).to_owned(),
                Value::String(instant.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)),
            ));
            continue;
        }
        let value = if name == "transcript" {
            camel_keys(thinking_times_from_ms(value.clone()))
        } else {
            camel_keys(value.clone())
        };
        wire.insert(camel_case(name), value);
    }
    wire.extend(times);
    serde_json::from_value(Value::Object(wire))
        .map_err(|error| invalid(format!("stored turn state is unreadable: {error}")))
}

/// Renames the keys of every object in `value` to snake_case, except under
/// [`OPAQUE`].
fn snake_keys(value: Value) -> Value {
    map_keys(value, snake_case)
}

/// The inverse of [`snake_keys`].
fn camel_keys(value: Value) -> Value {
    map_keys(value, camel_case)
}

fn map_keys(value: Value, rename: fn(&str) -> String) -> Value {
    match value {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(name, value)| {
                    let opaque = name == OPAQUE;
                    let value = if opaque {
                        value
                    } else {
                        map_keys(value, rename)
                    };
                    (rename(&name), value)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| map_keys(item, rename))
                .collect(),
        ),
        other => other,
    }
}

/// `started_at` / `ended_at` of a `thinking` item are epoch ms: store them as
/// `started_ms` / `ended_ms`.
fn thinking_times_to_ms(transcript: Value) -> Value {
    rename_thinking_times(transcript, THINKING_TIMES)
}

fn thinking_times_from_ms(transcript: Value) -> Value {
    rename_thinking_times(transcript, THINKING_TIMES.map(|(from, to)| (to, from)))
}

fn rename_thinking_times(transcript: Value, renames: [(&str, &str); 2]) -> Value {
    let Value::Array(items) = transcript else {
        return transcript;
    };
    Value::Array(
        items
            .into_iter()
            .map(|item| match item {
                Value::Object(mut fields)
                    if fields.get("kind").and_then(Value::as_str) == Some("thinking") =>
                {
                    for (from, to) in renames {
                        if let Some(value) = fields.remove(from) {
                            fields.insert(to.to_owned(), value);
                        }
                    }
                    Value::Object(fields)
                }
                other => other,
            })
            .collect(),
    )
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 2);
    for ch in name.chars() {
        if ch.is_ascii_uppercase() {
            out.push('_');
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn camel_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut upper = false;
    for ch in name.chars() {
        if ch == '_' {
            upper = true;
        } else if upper {
            out.push(ch.to_ascii_uppercase());
            upper = false;
        } else {
            out.push(ch);
        }
    }
    out
}
