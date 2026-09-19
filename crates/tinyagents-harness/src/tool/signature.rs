//! Compact, TypeScript-style renderings of a tool's argument schema.
//!
//! A JSON Schema is the right wire format for native tool calling but a poor
//! thing to paste into a prompt: `{"type":"object","properties":{"path":{"type":
//! "string","description":"…"}},"required":["path"]}` spends most of its bytes
//! on scaffolding. `{path: string, limit?: integer}` says the same to a model
//! in a fifth of the tokens, and it is the shape OpenClaw's tool-search hints
//! and Codex's Code Mode declarations both settled on.
//!
//! The rendering is lossy on purpose — constraints (`minimum`, `pattern`),
//! nested descriptions, and anything past [`MAX_DEPTH`] are dropped — which is
//! why the native path never uses it. Top-level property descriptions are
//! rendered separately by [`argument_notes`] so a prompt can still explain
//! what each argument means.

use std::fmt::Write as _;

use serde_json::Value;

use super::discover::first_sentence;

/// Nesting past this depth renders as a bare `object` / `unknown[]`.
pub const MAX_DEPTH: usize = 4;
/// Properties past this count in one object render as `…`.
pub const MAX_PROPERTIES: usize = 16;
/// A rendered signature is clipped to this many characters.
pub const MAX_SIGNATURE_CHARS: usize = 300;
/// Longest per-argument note kept by [`argument_notes`].
pub const MAX_NOTE_CHARS: usize = 100;

/// Renders `schema` as a TypeScript-style type, clipped to
/// [`MAX_SIGNATURE_CHARS`].
#[must_use]
pub fn type_signature(schema: &Value) -> String {
    let rendered = render(schema, 0);
    if rendered.chars().count() <= MAX_SIGNATURE_CHARS {
        return rendered;
    }
    let mut clipped: String = rendered.chars().take(MAX_SIGNATURE_CHARS).collect();
    clipped.push('…');
    clipped
}

/// One `name: first sentence` line per top-level property that carries a
/// description, in `serde_json`'s (sorted) key order. Empty when none does.
#[must_use]
pub fn argument_notes(schema: &Value) -> Vec<String> {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    properties
        .iter()
        .filter_map(|(name, property)| {
            let description = property.get("description")?.as_str()?.trim();
            (!description.is_empty())
                .then(|| format!("{name}: {}", first_sentence(description, MAX_NOTE_CHARS)))
        })
        .collect()
}

fn render(schema: &Value, depth: usize) -> String {
    let Some(object) = schema.as_object() else {
        return "unknown".to_string();
    };
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        return values
            .iter()
            .map(|value| match value {
                Value::String(s) => format!("{s:?}"),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(value) = object.get("const") {
        return match value {
            Value::String(s) => format!("{s:?}"),
            other => other.to_string(),
        };
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = object.get(key).and_then(Value::as_array)
            && !variants.is_empty()
        {
            let mut seen = Vec::new();
            for variant in variants {
                let rendered = render(variant, depth);
                if !seen.contains(&rendered) {
                    seen.push(rendered);
                }
            }
            return seen.join(" | ");
        }
    }
    if let Some(variants) = object.get("allOf").and_then(Value::as_array)
        && let Some(first) = variants.first()
    {
        return render(first, depth);
    }

    let kind = match object.get("type") {
        Some(Value::String(kind)) => kind.clone(),
        Some(Value::Array(kinds)) => {
            return kinds
                .iter()
                .filter_map(Value::as_str)
                .map(|kind| {
                    let mut single = object.clone();
                    single.insert("type".to_string(), Value::String(kind.to_string()));
                    render(&Value::Object(single), depth)
                })
                .collect::<Vec<_>>()
                .join(" | ");
        }
        _ if object.contains_key("properties") => "object".to_string(),
        _ if object.contains_key("items") => "array".to_string(),
        _ => return "unknown".to_string(),
    };

    match kind.as_str() {
        "object" => render_object(object, depth),
        "array" => {
            let items = object.get("items").map_or_else(
                || "unknown".to_string(),
                |items| {
                    if depth >= MAX_DEPTH {
                        "unknown".to_string()
                    } else {
                        render(items, depth + 1)
                    }
                },
            );
            if items.contains(' ') || items.contains('|') {
                format!("Array<{items}>")
            } else {
                format!("{items}[]")
            }
        }
        other => other.to_string(),
    }
}

fn render_object(object: &serde_json::Map<String, Value>, depth: usize) -> String {
    let Some(properties) = object.get("properties").and_then(Value::as_object) else {
        return "object".to_string();
    };
    if properties.is_empty() {
        return "{}".to_string();
    }
    if depth >= MAX_DEPTH {
        return "object".to_string();
    }
    let required: Vec<&str> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut out = String::from("{");
    for (index, (name, property)) in properties.iter().enumerate() {
        if index == MAX_PROPERTIES {
            out.push_str(", …");
            break;
        }
        if index > 0 {
            out.push_str(", ");
        }
        let optional = if required.contains(&name.as_str()) {
            ""
        } else {
            "?"
        };
        // Infallible: writing to a String never errors.
        let _ = write!(
            out,
            "{}{optional}: {}",
            render_property_name(name),
            render(property, depth + 1)
        );
    }
    out.push('}');
    out
}

/// Renders a JSON Schema property name as a TypeScript member key.
///
/// JSON Schema property names are arbitrary strings, not restricted to valid
/// JavaScript identifiers: a schema can legally declare `"file-path"` or
/// `"2fa_code"`. Rendered bare, `file-path: string` reads as a subtraction
/// expression rather than a member, and a model can misparse it. A name that
/// is a valid identifier (starts with a letter/`_`/`$`, and is otherwise
/// alphanumeric/`_`/`$`) still renders bare, matching the common case and
/// keeping the compact rendering's whole point of being terse; anything else
/// is JSON-quoted, matching how TypeScript itself would need to write it as
/// an object type member (`{"file-path": string}`).
fn render_property_name(name: &str) -> String {
    let is_identifier = name
        .chars()
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_' || first == '$')
        && name
            .chars()
            .all(|ch| ch.is_alphanumeric() || ch == '_' || ch == '$');
    if is_identifier {
        name.to_string()
    } else {
        serde_json::to_string(name).unwrap_or_else(|_| format!("{name:?}"))
    }
}

#[cfg(test)]
#[path = "signature_test.rs"]
mod test;
