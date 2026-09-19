//! Generates `model-catalog.snapshot.json` from the [models.dev] API.
//!
//! Run with `cargo run -p tinyagents-registry --bin catalog_gen` from the
//! workspace root. Fetches `https://models.dev/api.json` (via `curl`, to
//! avoid adding an HTTP client dependency to this crate for a dev-only tool),
//! maps a curated set of providers/models into the
//! [`tinyagents_registry::catalog::ModelCatalogEntry`] shape — including
//! tiered pricing, context windows, modalities, and reasoning flags — and
//! writes a validated snapshot.
//!
//! # Usage
//!
//! ```text
//! cargo run -p tinyagents-registry --bin catalog_gen [-- --output PATH] [--input PATH]
//! ```
//!
//! `--input PATH` reads the models.dev payload from a local file instead of
//! fetching it (useful offline or in a network-restricted environment).
//! `--output PATH` defaults to `crates/tinyagents-registry/model-catalog.snapshot.json`
//! relative to the current directory (the workspace root, when run the usual
//! way).
//!
//! The generated snapshot is validated with
//! [`ModelCatalogSnapshot::validate`] before it is written; a snapshot that
//! fails validation is never written, so a bad `models.dev` payload cannot
//! silently corrupt the checked-in seed.
//!
//! [models.dev]: https://models.dev/api.json

use std::process::Command;

use serde_json::Value;
use tinyagents_harness::cost::{ModelPricing, PriceTier};
use tinyagents_registry::catalog::{
    ModelCapabilities, ModelCatalogEntry, ModelCatalogSnapshot, ModelCatalogSource,
};

/// `(models.dev provider id, our catalog provider id, max models to keep)`.
///
/// Curated rather than exhaustive: models.dev lists 200+ providers, many
/// thin re-exports of the same underlying models through a proxy. This list
/// covers the providers the harness's own adapters (`tinyinference-llm`)
/// speak natively, plus the other major hosted labs, each capped so the
/// snapshot stays a small, reviewable seed rather than a mirror of the
/// entire upstream catalog.
const PROVIDERS: &[(&str, &str, usize)] = &[
    ("openai", "openai", 10),
    ("anthropic", "anthropic", 8),
    ("google", "gemini", 8),
    ("mistral", "mistral", 6),
    ("groq", "groq", 4),
    ("deepseek", "deepseek", 4),
    ("xai", "xai", 4),
];

const MODELS_DEV_URL: &str = "https://models.dev/api.json";
const DEFAULT_OUTPUT: &str = "crates/tinyagents-registry/model-catalog.snapshot.json";

fn main() {
    if let Err(error) = run() {
        eprintln!("catalog_gen: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut output = DEFAULT_OUTPUT.to_string();
    let mut input: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => output = args.next().ok_or("--output needs a value")?,
            "--input" => input = Some(args.next().ok_or("--input needs a value")?),
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    let payload = match input {
        Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("reading {path}: {e}"))?,
        None => fetch_via_curl(MODELS_DEV_URL)?,
    };
    let root: Value = serde_json::from_str(&payload).map_err(|e| format!("parsing JSON: {e}"))?;

    let retrieved_at = chrono::Utc::now().to_rfc3339();
    let mut models = Vec::new();
    for (source_provider, catalog_provider, limit) in PROVIDERS {
        let Some(provider) = root.get(*source_provider) else {
            continue;
        };
        let Some(model_map) = provider.get("models").and_then(Value::as_object) else {
            continue;
        };
        let mut entries: Vec<ModelCatalogEntry> = model_map
            .iter()
            .filter_map(|(model_id, model)| to_entry(catalog_provider, model_id, model))
            .collect();
        // Prefer models with known pricing first (more useful in a small seed),
        // then fall back to declaration order; cap at this provider's limit.
        entries.sort_by_key(|entry| entry.pricing.input_per_token.is_none());
        entries.truncate(*limit);
        models.extend(entries);
    }

    if models.is_empty() {
        return Err(
            "no models matched the curated provider list; upstream shape may have changed"
                .to_string(),
        );
    }

    let snapshot = ModelCatalogSnapshot {
        schema_version: 1,
        snapshot_id: format!("models-dev-{}", chrono::Utc::now().format("%Y%m%d")),
        created_at: retrieved_at.clone(),
        currency: "USD".to_string(),
        unit: "token".to_string(),
        description: Some(
            "Generated from models.dev by crates/tinyagents-registry/src/bin/catalog_gen.rs"
                .to_string(),
        ),
        sources: vec![ModelCatalogSource {
            name: "models.dev".to_string(),
            url: MODELS_DEV_URL.to_string(),
            retrieved_at,
        }],
        models,
    };

    snapshot
        .validate()
        .map_err(|e| format!("generated snapshot failed validation: {e}"))?;

    let json = serde_json::to_string_pretty(&snapshot).map_err(|e| e.to_string())?;
    std::fs::write(&output, json + "\n").map_err(|e| format!("writing {output}: {e}"))?;
    eprintln!(
        "catalog_gen: wrote {} models to {output}",
        snapshot.models.len()
    );
    Ok(())
}

fn fetch_via_curl(url: &str) -> Result<String, String> {
    let output = Command::new("curl")
        .args(["-sS", "--fail", url])
        .output()
        .map_err(|e| format!("running curl: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "curl exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("curl output was not UTF-8: {e}"))
}

/// Converts one models.dev model record into a [`ModelCatalogEntry`].
/// Returns `None` for a record too sparse to be useful (no `id`).
fn to_entry(catalog_provider: &str, model_id: &str, model: &Value) -> Option<ModelCatalogEntry> {
    let id = model.get("id").and_then(Value::as_str).unwrap_or(model_id);

    let limit = model.get("limit");
    let max_input_tokens = limit
        .and_then(|l| l.get("input"))
        .or_else(|| limit.and_then(|l| l.get("context")))
        .and_then(Value::as_u64);
    let max_output_tokens = limit.and_then(|l| l.get("output")).and_then(Value::as_u64);

    let modalities_in: Vec<&str> = model
        .get("modalities")
        .and_then(|m| m.get("input"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let modalities_out: Vec<&str> = model
        .get("modalities")
        .and_then(|m| m.get("output"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    // This is a chat-model catalog: an image/audio/video-only-output model
    // (text-to-speech, image generation, video generation) has no text
    // completion to serve as a `ChatModel` and often has an output-token
    // limit that legitimately exceeds its input limit (an audio-duration
    // budget, not a context overflow) — the wrong shape for this catalog.
    if !modalities_out.contains(&"text") {
        return None;
    }

    let cost = model.get("cost");
    let per_token = |value: Option<&Value>| value.and_then(Value::as_f64).map(|v| v / 1_000_000.0);
    let base_input = per_token(cost.and_then(|c| c.get("input")));
    let base_output = per_token(cost.and_then(|c| c.get("output")));
    let base_cache_read = per_token(cost.and_then(|c| c.get("cache_read")));
    let base_cache_write = per_token(cost.and_then(|c| c.get("cache_write")));

    let mut tiers = Vec::new();
    if let Some(extra_tiers) = cost.and_then(|c| c.get("tiers")).and_then(Value::as_array) {
        // Sort ascending by threshold so the base rate's implicit cap is the
        // first declared tier's size, and later tiers extend upward.
        let mut thresholds: Vec<(u64, &Value)> = extra_tiers
            .iter()
            .filter_map(|tier| {
                let size = tier.get("tier")?.get("size")?.as_u64()?;
                Some((size, tier))
            })
            .collect();
        thresholds.sort_by_key(|(size, _)| *size);
        if let Some((first_cap, _)) = thresholds.first() {
            tiers.push(PriceTier {
                up_to_tokens: Some(*first_cap),
                input: base_input,
                output: base_output,
                cache_read: base_cache_read,
                cache_write: base_cache_write,
            });
        }
        for (index, (_, tier)) in thresholds.iter().enumerate() {
            let up_to_tokens = thresholds.get(index + 1).map(|(size, _)| *size);
            tiers.push(PriceTier {
                up_to_tokens,
                input: per_token(tier.get("input")),
                output: per_token(tier.get("output")),
                cache_read: per_token(tier.get("cache_read")),
                cache_write: per_token(tier.get("cache_write")),
            });
        }
    }

    let capabilities = ModelCapabilities {
        streaming: true,
        tool_calling: model
            .get("tool_call")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        parallel_tool_calling: false,
        json_schema: model
            .get("structured_output")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        system_messages: true,
        vision: modalities_in.contains(&"image"),
        audio_input: modalities_in.contains(&"audio"),
        audio_output: modalities_out.contains(&"audio"),
        pdf_input: modalities_in.contains(&"pdf"),
        prompt_caching: base_cache_read.is_some(),
        reasoning: model
            .get("reasoning")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };

    Some(ModelCatalogEntry {
        provider: catalog_provider.to_string(),
        model_id: id.to_string(),
        aliases: Vec::new(),
        mode: "chat".to_string(),
        max_input_tokens,
        max_output_tokens,
        deprecation_date: None,
        release_date: model
            .get("release_date")
            .and_then(Value::as_str)
            .map(str::to_string),
        pricing: ModelPricing {
            input_per_token: base_input,
            output_per_token: base_output,
            cache_read_input_per_token: base_cache_read,
            cache_creation_input_per_token: base_cache_write,
            input_audio_per_token: None,
            output_reasoning_per_token: None,
            tiers,
        },
        capabilities,
        source: "models.dev".to_string(),
        source_url: model.get("doc").and_then(Value::as_str).map(str::to_string),
        raw: model.clone(),
    })
}
