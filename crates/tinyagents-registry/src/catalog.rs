//! Deterministic, offline model catalog.
//!
//! Where [`CapabilityRegistry`](crate::CapabilityRegistry) resolves
//! *executable* capabilities by name, this module resolves *facts about
//! models* by name: a checked-in snapshot of provider model prices, context
//! windows, and capability flags. Recursive runs lean on it for the decisions
//! that surround a model call — estimating roll-up cost across parent/child
//! runs, choosing a cheaper sub-model for a delegated step, or gating a feature
//! (tool calling, JSON schema, vision) before a sub-agent is dispatched — all
//! without a network round-trip, so the default offline build stays
//! deterministic.
//!
//! The snapshot is embedded at compile time from
//! `docs/modules/registry/model-catalog.snapshot.json` and loaded via
//! [`ModelCatalog::seed`]; alternative snapshots can be supplied with
//! [`ModelCatalog::from_json`] or [`ModelCatalog::from_snapshot`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use tinyagents_harness::cost::ModelPricing;
use tinyagents_harness::error::TinyAgentsError;

/// Provider ids the catalog accepts without an explicit allowlist override.
/// Kept intentionally small: an entry naming anything else fails validation
/// (see [`ModelCatalogSnapshot::validate`]) rather than being silently
/// accepted, since an unrecognized provider id is the most common way a bad
/// snapshot generator run slips through review.
const KNOWN_PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "gemini",
    "google",
    "mistral",
    "cohere",
    "groq",
    "deepseek",
    "xai",
    "meta",
    "together",
    "fireworks",
    "openrouter",
    "ollama",
    "azure",
    "bedrock",
    "vertex",
    "perplexity",
    "tinyhumans",
];

const SEED_SNAPSHOT: &str = include_str!("../model-catalog.snapshot.json");

/// An in-memory, immutable view over a [`ModelCatalogSnapshot`].
///
/// Construct it from the embedded seed ([`ModelCatalog::seed`]) or from custom
/// JSON, then look entries up by `(provider, model_id)` or by model id alone.
/// Lookups also match an entry's [`aliases`](ModelCatalogEntry::aliases).
#[derive(Clone, Debug)]
pub struct ModelCatalog {
    snapshot: ModelCatalogSnapshot,
}

impl ModelCatalog {
    /// Wraps an already-parsed, already-valid [`ModelCatalogSnapshot`].
    ///
    /// Does not itself validate `snapshot`; prefer
    /// [`try_from_snapshot`](Self::try_from_snapshot) (or [`from_json`](Self::from_json),
    /// which calls it) unless the snapshot is already known-good (for example,
    /// round-tripped from an existing, already-validated [`ModelCatalog`]).
    pub fn from_snapshot(snapshot: ModelCatalogSnapshot) -> Self {
        Self { snapshot }
    }

    /// Wraps `snapshot` after validating it with
    /// [`ModelCatalogSnapshot::validate`].
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Validation`] describing the first validation
    /// failure found. See [`ModelCatalogSnapshot::validate`] for the checks
    /// performed.
    pub fn try_from_snapshot(snapshot: ModelCatalogSnapshot) -> Result<Self> {
        snapshot.validate()?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Parses and validates a catalog from a JSON snapshot string.
    ///
    /// # Errors
    ///
    /// Returns an error if `source` is not valid JSON, does not match the
    /// [`ModelCatalogSnapshot`] shape, or fails
    /// [`ModelCatalogSnapshot::validate`] (duplicate `(provider, model_id)`
    /// pairs, negative prices, a missing `source`, an output limit exceeding
    /// the input context, an alias collision, an invalid date, or an unknown
    /// provider id).
    pub fn from_json(source: &str) -> Result<Self> {
        let snapshot: ModelCatalogSnapshot = serde_json::from_str(source)?;
        Self::try_from_snapshot(snapshot)
    }

    /// Loads the catalog from the snapshot embedded in the crate at build time.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded snapshot fails to parse (which would
    /// indicate a corrupted checked-in file).
    pub fn seed() -> Result<Self> {
        Self::from_json(SEED_SNAPSHOT)
    }

    /// Returns the underlying snapshot, including its metadata and sources.
    pub fn snapshot(&self) -> &ModelCatalogSnapshot {
        &self.snapshot
    }

    /// Returns all catalog entries.
    pub fn models(&self) -> &[ModelCatalogEntry] {
        &self.snapshot.models
    }

    /// Looks up an entry by `provider` and `model_id`, matching either the
    /// canonical [`model_id`](ModelCatalogEntry::model_id) or any of its
    /// [`aliases`](ModelCatalogEntry::aliases). Returns `None` if no entry for
    /// that provider matches.
    pub fn get(&self, provider: &str, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot.models.iter().find(|entry| {
            entry.provider == provider
                && (entry.model_id == model_id
                    || entry.aliases.iter().any(|alias| alias == model_id))
        })
    }

    /// Looks up an entry by model id (or alias) across all providers, returning
    /// the first match. Use [`get`](Self::get) when the provider is known and
    /// the same id might appear under more than one provider.
    pub fn get_by_model_id(&self, model_id: &str) -> Option<&ModelCatalogEntry> {
        self.snapshot.models.iter().find(|entry| {
            entry.model_id == model_id || entry.aliases.iter().any(|alias| alias == model_id)
        })
    }

    /// Hydrates a runtime
    /// [`ModelProfile`][tinyinference_llm::model::ModelProfile] from the catalog
    /// entry for `provider`/`model_id`, bridging offline catalog facts into the
    /// capability profile resolution and fallback consume. Returns `None` when
    /// no entry matches.
    pub fn profile(
        &self,
        provider: &str,
        model_id: &str,
    ) -> Option<tinyinference_llm::model::ModelProfile> {
        self.get(provider, model_id).map(profile_from_entry)
    }
}

/// The deserialized form of a model-catalog snapshot file.
///
/// Carries provenance metadata (schema version, snapshot id, creation time,
/// pricing currency/unit, and the [`sources`](Self::sources) it was derived
/// from) alongside the list of [`models`](Self::models).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSnapshot {
    /// Version of the snapshot schema this file conforms to.
    pub schema_version: u32,
    /// Unique identifier for this snapshot revision.
    pub snapshot_id: String,
    /// ISO-8601 timestamp recording when the snapshot was generated.
    pub created_at: String,
    /// Currency that all pricing fields are denominated in (e.g. `"USD"`).
    pub currency: String,
    /// The unit prices are quoted per (e.g. `"token"`).
    pub unit: String,
    /// Optional human-readable description of the snapshot.
    #[serde(default)]
    pub description: Option<String>,
    /// Provenance entries describing where the snapshot data came from.
    #[serde(default)]
    pub sources: Vec<ModelCatalogSource>,
    /// The catalog entries themselves, one per model.
    #[serde(default)]
    pub models: Vec<ModelCatalogEntry>,
}

impl ModelCatalogSnapshot {
    /// Validates the snapshot, returning the first failure found (see
    /// `docs/modules/registry/model-catalog.md`'s "Refresh Workflow" for the
    /// checks this enforces). Checked, in order:
    ///
    /// 1. no duplicate `(provider, model_id)` pair
    /// 2. no negative price (flat or tiered)
    /// 3. every entry has a non-empty `source`
    /// 4. `max_output_tokens` never exceeds `max_input_tokens` when both are known
    /// 5. no alias collides with another entry's canonical id or alias
    /// 6. every date field (`created_at`, `retrieved_at`, `deprecation_date`)
    ///    parses as `YYYY-MM-DD` or a full ISO-8601 timestamp
    /// 7. every `provider` is a recognized id (see `KNOWN_PROVIDERS`)
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Validation`] with a message identifying the
    /// offending entry and rule.
    pub fn validate(&self) -> Result<()> {
        let mut seen_ids = std::collections::HashSet::new();
        let mut seen_names = std::collections::HashSet::new();

        for source in &self.sources {
            validate_date(&source.retrieved_at, "source.retrieved_at")?;
        }
        validate_date(&self.created_at, "created_at")?;

        for entry in &self.models {
            let id = (entry.provider.clone(), entry.model_id.clone());
            if !seen_ids.insert(id) {
                return Err(fail(format!(
                    "duplicate (provider, model_id) pair: ({}, {})",
                    entry.provider, entry.model_id
                )));
            }

            if entry.source.trim().is_empty() {
                return Err(fail(format!(
                    "entry {}/{} is missing a source",
                    entry.provider, entry.model_id
                )));
            }

            if let (Some(max_in), Some(max_out)) =
                (entry.max_input_tokens, entry.max_output_tokens)
                && max_out > max_in
            {
                return Err(fail(format!(
                    "entry {}/{} has max_output_tokens ({max_out}) greater than \
                     max_input_tokens ({max_in})",
                    entry.provider, entry.model_id
                )));
            }

            if !KNOWN_PROVIDERS.contains(&entry.provider.as_str()) {
                return Err(fail(format!(
                    "entry {}/{} names an unrecognized provider id",
                    entry.provider, entry.model_id
                )));
            }

            if let Some(date) = &entry.deprecation_date {
                validate_date(date, "deprecation_date")?;
            }
            if let Some(date) = &entry.release_date {
                validate_date(date, "release_date")?;
            }

            validate_pricing(&entry.provider, &entry.model_id, &entry.pricing)?;

            // The model's own canonical id and every alias must be globally
            // unique across the snapshot (an alias colliding with another
            // entry's id, or two entries sharing an alias, both make lookup
            // ambiguous).
            for name in std::iter::once(entry.model_id.clone()).chain(entry.aliases.clone()) {
                if !seen_names.insert((entry.provider.clone(), name.clone())) {
                    return Err(fail(format!(
                        "alias or id `{name}` collides with another entry under provider `{}`",
                        entry.provider
                    )));
                }
            }
        }
        Ok(())
    }
}

fn fail(message: String) -> TinyAgentsError {
    TinyAgentsError::Validation(format!("model catalog validation failed: {message}"))
}

fn validate_pricing(provider: &str, model_id: &str, pricing: &ModelPricing) -> Result<()> {
    let negative = |rate: Option<f64>| rate.is_some_and(|r| r < 0.0);
    if negative(pricing.input_per_token)
        || negative(pricing.output_per_token)
        || negative(pricing.cache_read_input_per_token)
        || negative(pricing.cache_creation_input_per_token)
        || negative(pricing.input_audio_per_token)
        || negative(pricing.output_reasoning_per_token)
    {
        return Err(fail(format!(
            "entry {provider}/{model_id} has a negative flat price"
        )));
    }
    for tier in &pricing.tiers {
        if negative(tier.input)
            || negative(tier.output)
            || negative(tier.cache_read)
            || negative(tier.cache_write)
        {
            return Err(fail(format!(
                "entry {provider}/{model_id} has a negative tiered price"
            )));
        }
    }
    Ok(())
}

/// Accepts `YYYY-MM-DD` or a full ISO-8601/RFC-3339 timestamp
/// (`YYYY-MM-DDTHH:MM:SSZ`, with or without fractional seconds/offset).
fn validate_date(value: &str, field: &str) -> Result<()> {
    let plain_date = value.len() == 10
        && value.as_bytes().get(4) == Some(&b'-')
        && value.as_bytes().get(7) == Some(&b'-')
        && value.split('-').count() == 3
        && value.split('-').all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
    let timestamp = chrono::DateTime::parse_from_rfc3339(value).is_ok();
    if plain_date || timestamp {
        Ok(())
    } else {
        Err(fail(format!("{field} `{value}` is not a valid date")))
    }
}

/// One provenance record for a [`ModelCatalogSnapshot`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogSource {
    /// Human-readable name of the source.
    pub name: String,
    /// URL the data was retrieved from.
    pub url: String,
    /// ISO-8601 timestamp recording when the source was retrieved.
    pub retrieved_at: String,
}

/// A single model's catalog record: identity, limits, pricing, and capability
/// flags.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelCatalogEntry {
    /// Provider that serves this model (e.g. `"openai"`, `"anthropic"`).
    pub provider: String,
    /// Canonical model identifier within the provider.
    pub model_id: String,
    /// Alternate identifiers that also resolve to this entry in lookups.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Serving mode for the model (e.g. `"chat"`, `"embedding"`).
    pub mode: String,
    /// Maximum number of input (context) tokens, when known.
    #[serde(default)]
    pub max_input_tokens: Option<u64>,
    /// Maximum number of output tokens, when known.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Announced deprecation date, when the provider has published one.
    #[serde(default)]
    pub deprecation_date: Option<String>,
    /// Date the model was released, when known.
    #[serde(default)]
    pub release_date: Option<String>,
    /// Per-token pricing for the model.
    #[serde(default)]
    pub pricing: ModelPricing,
    /// Capability flags advertised for the model.
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    /// Identifier of the upstream source this entry was derived from.
    pub source: String,
    /// Optional URL pointing at the source documentation for this entry.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Raw provider payload preserved verbatim for fields not modeled above.
    #[serde(default)]
    pub raw: Value,
}

/// Boolean capability flags advertised for a model.
///
/// These let recursive runs gate behavior before dispatch — for example,
/// refusing to hand tools to a sub-agent backed by a model whose
/// [`tool_calling`](Self::tool_calling) flag is `false`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelCapabilities {
    /// The model supports token streaming.
    #[serde(default)]
    pub streaming: bool,
    /// The model supports tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// The model can request multiple tool calls in a single turn.
    #[serde(default)]
    pub parallel_tool_calling: bool,
    /// The model supports JSON-schema-constrained structured output.
    #[serde(default)]
    pub json_schema: bool,
    /// The model accepts system messages.
    #[serde(default)]
    pub system_messages: bool,
    /// The model accepts image input.
    #[serde(default)]
    pub vision: bool,
    /// The model accepts audio input.
    #[serde(default)]
    pub audio_input: bool,
    /// The model can produce audio output.
    #[serde(default)]
    pub audio_output: bool,
    /// The model accepts PDF input.
    #[serde(default)]
    pub pdf_input: bool,
    /// The model supports prompt caching.
    #[serde(default)]
    pub prompt_caching: bool,
    /// The model exposes explicit reasoning/thinking.
    #[serde(default)]
    pub reasoning: bool,
}

fn profile_from_entry(entry: &ModelCatalogEntry) -> tinyinference_llm::model::ModelProfile {
    use tinyinference_llm::model::{Modalities, ModelProfile, ModelStatus};

    let caps = &entry.capabilities;
    ModelProfile {
        provider: Some(entry.provider.clone()),
        model: Some(entry.model_id.clone()),
        status: if entry.deprecation_date.is_some() {
            ModelStatus::Deprecated
        } else {
            ModelStatus::Stable
        },
        modalities: Modalities {
            text_in: true,
            text_out: true,
            image_in: caps.vision,
            audio_in: caps.audio_input,
            audio_out: caps.audio_output,
            ..Modalities::default()
        },
        tool_calling: caps.tool_calling,
        parallel_tool_calls: caps.parallel_tool_calling,
        streaming: caps.streaming,
        streaming_tool_chunks: caps.streaming && caps.tool_calling,
        native_structured_output: caps.json_schema,
        json_schema: caps.json_schema,
        reasoning: caps.reasoning,
        max_input_tokens: entry.max_input_tokens,
        max_output_tokens: entry.max_output_tokens,
        ..ModelProfile::default()
    }
}

/// Tests that the embedded seed snapshot loads and that entries resolve by
/// canonical id and by alias.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_seed_model_catalog_snapshot() {
        let catalog = ModelCatalog::seed().unwrap();

        assert_eq!(catalog.snapshot().schema_version, 1);
        assert!(catalog.get("openai", "gpt-4.1").is_some());
        assert!(catalog.get("anthropic", "claude-sonnet-4").is_some());
        assert!(catalog.get("gemini", "gemini-2.5-flash").is_some());
    }

    #[test]
    fn looks_up_model_by_alias_or_id() {
        let catalog = ModelCatalog::seed().unwrap();

        let by_id = catalog.get_by_model_id("gemini/gemini-2.5-pro").unwrap();
        let by_alias = catalog.get_by_model_id("gemini-2.5-pro").unwrap();

        assert_eq!(by_id.model_id, by_alias.model_id);
    }

    #[test]
    fn bridges_catalog_entry_into_runtime_profile() {
        let catalog = ModelCatalog::seed().unwrap();
        let entry = catalog.get("openai", "gpt-4.1").unwrap();

        let profile = profile_from_entry(entry);
        assert_eq!(profile.provider.as_deref(), Some("openai"));
        assert_eq!(profile.model.as_deref(), Some("gpt-4.1"));
        // The catalog's advertised capability flags carry across the bridge.
        assert_eq!(profile.tool_calling, entry.capabilities.tool_calling);
        assert_eq!(profile.max_input_tokens, entry.max_input_tokens);

        // The convenience accessor returns the same bridged profile.
        let via_catalog = catalog.profile("openai", "gpt-4.1").unwrap();
        assert_eq!(via_catalog, profile);
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    fn base_entry() -> ModelCatalogEntry {
        ModelCatalogEntry {
            provider: "openai".to_string(),
            model_id: "gpt-test".to_string(),
            aliases: Vec::new(),
            mode: "chat".to_string(),
            max_input_tokens: Some(100_000),
            max_output_tokens: Some(4_096),
            deprecation_date: None,
            pricing: ModelPricing::default(),
            capabilities: ModelCapabilities::default(),
            source: "manual".to_string(),
            source_url: None,
            raw: Value::Null,
        }
    }

    fn base_snapshot(models: Vec<ModelCatalogEntry>) -> ModelCatalogSnapshot {
        ModelCatalogSnapshot {
            schema_version: 1,
            snapshot_id: "test".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            currency: "USD".to_string(),
            unit: "token".to_string(),
            description: None,
            sources: Vec::new(),
            models,
        }
    }

    #[test]
    fn valid_snapshot_passes() {
        base_snapshot(vec![base_entry()]).validate().unwrap();
    }

    #[test]
    fn rejects_duplicate_provider_model_id_pairs() {
        let snapshot = base_snapshot(vec![base_entry(), base_entry()]);
        let error = snapshot.validate().unwrap_err().to_string();
        assert!(error.contains("duplicate"), "got: {error}");
    }

    #[test]
    fn rejects_negative_flat_price() {
        let mut entry = base_entry();
        entry.pricing.input_per_token = Some(-0.01);
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("negative"), "got: {error}");
    }

    #[test]
    fn rejects_negative_tiered_price() {
        let mut entry = base_entry();
        entry.pricing.tiers.push(tinyagents_harness::cost::PriceTier {
            up_to_tokens: None,
            input: Some(-1.0),
            output: None,
            cache_read: None,
            cache_write: None,
        });
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("negative"), "got: {error}");
    }

    #[test]
    fn rejects_missing_source() {
        let mut entry = base_entry();
        entry.source = String::new();
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("missing a source"), "got: {error}");
    }

    #[test]
    fn rejects_output_limit_exceeding_input_context() {
        let mut entry = base_entry();
        entry.max_input_tokens = Some(1_000);
        entry.max_output_tokens = Some(2_000);
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("max_output_tokens"), "got: {error}");
    }

    #[test]
    fn rejects_alias_collision() {
        let mut aliased = base_entry();
        aliased.model_id = "gpt-other".to_string();
        aliased.aliases = vec!["gpt-test".to_string()]; // collides with base_entry's id
        let error = base_snapshot(vec![base_entry(), aliased])
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("collides"), "got: {error}");
    }

    #[test]
    fn rejects_invalid_date() {
        let mut entry = base_entry();
        entry.deprecation_date = Some("not-a-date".to_string());
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("not a valid date"), "got: {error}");
    }

    #[test]
    fn accepts_plain_and_rfc3339_dates() {
        let mut entry = base_entry();
        entry.deprecation_date = Some("2026-01-01".to_string());
        base_snapshot(vec![entry.clone()]).validate().unwrap();
        entry.deprecation_date = Some("2026-01-01T00:00:00Z".to_string());
        base_snapshot(vec![entry]).validate().unwrap();
    }

    #[test]
    fn rejects_unknown_provider() {
        let mut entry = base_entry();
        entry.provider = "totally-unknown-vendor".to_string();
        let error = base_snapshot(vec![entry]).validate().unwrap_err().to_string();
        assert!(error.contains("unrecognized provider"), "got: {error}");
    }

    #[test]
    fn from_json_rejects_an_invalid_snapshot() {
        let mut snapshot = base_snapshot(vec![base_entry()]);
        snapshot.models[0].pricing.input_per_token = Some(-1.0);
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(ModelCatalog::from_json(&json).is_err());
    }
}
