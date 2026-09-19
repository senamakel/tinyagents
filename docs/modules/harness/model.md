# Harness Model And Provider Feature

The model feature owns provider-neutral chat model traits, request/response
shapes, streaming model chunks, provider capability profiles, provider-specific
options, and adapter conformance requirements.

## Source Inspiration

LangChain separates chat model interfaces, message types, provider integrations,
and model capability metadata:

- `BaseChatModel` and model streams:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/language_models>
- provider adapters such as OpenAI, Anthropic, Ollama, Mistral, Fireworks, Groq,
  Perplexity, and xAI:
  <https://github.com/langchain-ai/langchain/tree/master/libs/partners>
- model profiles:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/model_profile.py>
- model profile generation package:
  <https://github.com/langchain-ai/langchain/tree/master/libs/model-profiles>
- standard chat model tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

TinyAgents should adapt this into a strongly typed Rust surface with
feature-gated provider crates/modules and a shared conformance suite.

## Responsibilities

- Define provider-neutral chat model traits.
- Normalize request options across providers without hiding provider escape
  hatches.
- Expose provider capability profiles.
- Validate requests against profiles before network calls.
- Convert provider responses into canonical `Message` and `ContentBlock` values.
- Preserve provider response ids, raw metadata, safety/refusal metadata, and
  usage details.
- Support invoke and stream paths with equivalent final semantics.
- Support model aliases, default models, dynamic model selection, and reusable
  resolved-model state.
- Support feature-gated provider adapters.
- Provide fake and replay models for tests.

## Non-Responsibilities

- It does not execute tools.
- It does not choose graph routes.
- It does not own provider API keys globally.
- It does not hardcode time-sensitive model prices.
- It does not assume all models support tool calling, structured output,
  multimodal input, or streaming.

## Core Types

```rust
#[async_trait]
pub trait ChatModel<State, Ctx = ()>: Send + Sync {
    fn profile(&self) -> Option<&ModelProfile>;

    async fn invoke(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelResponse>;

    async fn stream(
        &self,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        request: ModelRequest,
    ) -> Result<ModelStream>;
}

pub struct ModelRequest {
    pub model: Option<ModelName>,
    pub model_hints: Vec<ModelHint>,
    pub reuse_resolved_model: bool,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDeclaration>,
    pub tool_choice: ToolChoice,
    pub response_format: Option<ResponseFormat>,
    pub model_settings: ModelSettings,
    pub provider_options: serde_json::Value,
    pub cache_policy: Option<CachePolicy>,
    pub required_capabilities: CapabilitySet,
    pub metadata: serde_json::Value,
}

pub struct ModelResponse {
    pub message: AssistantMessage,
    pub resolved_model: ResolvedModel,
    pub usage: Option<UsageRecord>,
    pub finish_reason: Option<FinishReason>,
    pub structured: Option<serde_json::Value>,
    pub provider: ProviderMetadata,
    pub timings: TimingBreakdown,
    pub cache: Option<CacheDecision>,
}
```

## Model Resolution

Agents and model calls should be able to define their own model preferences
without hardcoding one provider everywhere. The harness resolves a model at the
start of a model call, records the selected model, and makes that resolved model
available to later calls, child agents, graph nodes, event streams, and persisted
state.

The design mirrors OpenHuman-style smart model resolution from hints: callers
can provide candidate models and capability requirements, while the registry and
harness choose the best registered executable model under policy.

Resolution inputs:

- explicit request override such as `model "anthropic/claude-sonnet"`
- agent default model declared in agent registration or `.rag`
- model hints from an orchestrator, task, middleware, or runtime context
- required capabilities such as tool calling, streaming, native JSON schema,
  vision, long context, reasoning, or prompt caching
- cost, latency, locality, privacy, and provider-family preferences
- previously resolved model stored in state
- registry default model

Resolution order should be explicit and configurable. A conservative default is:

1. request-level explicit model override
2. reusable model stored in the current agent/thread state, when policy allows
3. highest-priority valid model hint
4. agent default model
5. registry default model
6. configured fallback chain that satisfies required capabilities

If a candidate fails capability, budget, tenant, provider-availability, or
policy checks, the resolver should skip it and emit a diagnostic event. In
particular, when the request-level explicit override is skipped (unregistered
name, missing capability, or provider-retired) and resolution falls through to
a lower-priority candidate, the agent loop emits
`AgentEvent::ModelOverrideSkipped { requested, resolved }`
(`model.override_skipped`) so the substitution is observable. If no candidate
remains, the model call fails before contacting a provider. A model
with no advertised profile is treated as unknown and does not satisfy non-empty
required capabilities; callers that need capability filtering should register
profiles for every candidate they expect the resolver to consider.

## Resolution Types

```rust
pub struct ModelHint {
    pub model: ModelRef,
    pub priority: i32,
    pub reason: Option<String>,
    pub requirements: CapabilitySet,
    pub preferences: ModelPreferences,
}

pub struct ModelPreferences {
    pub prefer_local: bool,
    pub prefer_low_latency: bool,
    pub max_estimated_cost: Option<Money>,
    pub provider_family: Option<String>,
    pub privacy_tier: Option<String>,
}

pub struct ModelSelection {
    pub requested: Option<ModelRef>,
    pub hints: Vec<ModelHint>,
    pub agent_default: Option<ModelRef>,
    pub previous: Option<ResolvedModel>,
    pub reuse_previous: bool,
    pub required_capabilities: CapabilitySet,
}

pub struct ResolvedModel {
    pub registry_name: ModelName,
    pub provider: ProviderName,
    pub provider_model_id: String,
    pub catalog_entry_id: Option<String>,
    pub source: ModelResolutionSource,
    pub selected_at: SystemTime,
    pub requirements_satisfied: CapabilitySet,
    pub fallback_from: Vec<ModelRef>,
    pub resolver_version: String,
}

pub enum ModelResolutionSource {
    RequestOverride,
    StateReuse,
    Hint,
    AgentDefault,
    RegistryDefault,
    Fallback,
}
```

The `ResolvedModel` record is more than metadata. It is the durable binding that
answers "which model did this agent actually use?" and "can the next step reuse
the same model?".

## Persisting Resolved Models In State

Every model call should attach `ResolvedModel` to:

- `ModelResponse`
- assistant message/provider metadata
- model lifecycle events
- usage and cost records
- agent run status
- graph task metadata when graph-backed
- checkpoint state when the graph or harness state is durable

Agent state should have a standard place for reusable model resolution:

```rust
pub struct AgentModelState {
    pub current: Option<ResolvedModel>,
    pub history: Vec<ResolvedModelUse>,
}

pub struct ResolvedModelUse {
    pub resolved: ResolvedModel,
    pub run_id: RunId,
    pub call_id: CallId,
    pub node_id: Option<NodeId>,
    pub usage: Option<UsageRecord>,
}
```

The harness should update this state after successful model selection and before
provider invocation is recorded. If the provider call fails because the selected
model is unavailable, the failure and attempted `ResolvedModel` must still be
observable so fallback and debugging are explainable.

Reuse policy should be opt-in per agent/run. Reuse is useful for consistency,
cost estimation, provider prompt-cache stability, and sub-agent continuity, but
it should not override a later explicit request or a stronger policy constraint.

## Agent Defaults And Per-Agent Resolution

Each registered agent may declare:

- default model
- fallback models
- model hints
- required capabilities
- whether to reuse the prior resolved model
- whether children inherit its resolved model
- whether humans or parent orchestrators may steer model choice

Sub-agents may inherit a parent orchestrator's resolved model only when both
parent and child policies allow it. Otherwise the child resolves independently
and records its own `ResolvedModel`.

Model steering should use the same steering policy as other run controls. A
human or parent orchestrator may narrow a model set, request a cheaper/faster
model, or force a model override only when policy grants that authority.

## Model Profiles

Every provider adapter should expose a `ModelProfile` when data is known.

```rust
pub struct ModelProfile {
    pub provider: ProviderName,
    pub model: ModelName,
    pub display_name: Option<String>,
    pub status: ModelStatus,
    pub release_date: Option<Date>,
    pub last_updated: Option<Date>,
    pub max_input_tokens: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub modalities: Modalities,
    pub supports_tool_calling: bool,
    pub supports_tool_choice: bool,
    pub supports_tool_call_streaming: bool,
    pub supports_native_structured_output: bool,
    pub supports_reasoning_output: bool,
    pub supports_temperature: bool,
    pub supports_attachments: bool,
    pub provider_extras: serde_json::Value,
}
```

Profiles are used to:

- reject impossible requests before making a provider call
- choose native structured output versus tool-based structured output
- decide whether tool-call chunks are streamable or must be reconstructed from
  a final response
- estimate context pressure
- expose capability information in docs and UIs
- select fallbacks that satisfy required capabilities

Profiles are not a pricing table. Prices belong to the cost feature and should
be updateable independently.

### `ModelProfile` as behaviour (implemented)

The struct above is aspirational; the real `tinyinference_llm::model::ModelProfile`
(`vendor/tinyinference/crates/tinyinference-llm/src/model/types.rs`) now also
carries fields that let an adapter's *behaviour* — not just its advertised
capabilities — be driven by data:

- `schema_transform: Option<SchemaTransform>` — a serializable, named JSON-schema
  transform (`StripDefs`, `InlineRefs`, `NoAdditionalProperties`, `GeminiCompat`,
  `OpenAiStrict`, or `Chain(Vec<SchemaTransform>)`) with a real `apply(&self,
  &Value) -> Value`. `StripDefs`/`InlineRefs` resolve `$ref`s into `$defs`;
  `NoAdditionalProperties` recursively forces `additionalProperties: false`;
  `GeminiCompat` strips keywords Gemini's schema dialect rejects;
  `OpenAiStrict` composes inlining + `additionalProperties: false` + requiring
  every property, matching OpenAI's strict JSON-schema mode.
- `default_structured_mode: Option<StructuredMode>` (`Tool` / `Native` /
  `Prompted`) and `prompted_output_template: Option<String>` for the prompted
  fallback.
- `thinking_tags: Option<(String, String)>` — an open/close tag pair (for
  example `("<think>", "</think>")`) a model wraps chain-of-thought in.
- `ignore_streamed_leading_whitespace: bool` and
  `thinking_level_map: BTreeMap<String, ReasoningConfig>` (a named reasoning
  level, e.g. `"low"`, mapped to the `ReasoningConfig` it expands to).
- `compat: ProviderCompat` — provider-family quirks that do not fit the
  capability model: `mid_conversation_system_messages`, `strict_tools`,
  `cache_retention`, `session_affinity`, `max_tool_name_length`,
  `tool_id_pattern`.

All fields are additive and `serde(default)`, so existing serialized profiles
deserialize unchanged. See `model/test.rs` in the vendor crate for the
transform and round-trip tests. Consuming these fields from harness
`tool/schema_prepare.rs`, `structured/`, and `agent_loop/model_call.rs` is not
yet wired up — the data model exists but the harness does not yet read it.

### Deferred (async/batch) responses (implemented)

`ModelStreamItem::Deferred(DeferredHandle)` is a new terminal stream item for
a provider that accepts a request but finishes it asynchronously (an OpenAI
batch or background response). `DeferredHandle { provider, id, kind,
metadata }` is a serializable, provider-neutral handle a host can persist and
resume polling with. `ChatModel::fetch_deferred(&self, &DeferredHandle) ->
Result<DeferredStatus>` (`DeferredStatus::{Pending, Completed(Box<ModelResponse>),
Failed(String)}`) resolves it later; the default implementation returns
`Error::Unsupported`, so only adapters that can actually defer need to
override it. No provider adapter overrides it yet (OpenAI batch/background
mapping was scoped out of this pass) — the shape exists for a future adapter
to fill in. `StreamAccumulator::deferred()` surfaces a folded-in handle, and
`StreamAccumulator::finish()` returns `Error::Unsupported` rather than
silently discarding it when one was seen.

### `ProviderRequestOptions` (implemented)

`tinyinference_llm::providers::ProviderRequestOptions { on_payload, on_response,
http }` lets a host hook the OpenAI and Anthropic adapters' Chat
Completions/Messages transport without the harness knowing about the hook:
`on_payload: Option<Arc<dyn Fn(&mut Value) + Send + Sync>>` mutates the wire
body immediately before it is sent, `on_response: Option<Arc<dyn Fn(&Value) +
Send + Sync>>` observes the raw response JSON after a successful call, and
`http: Option<reqwest::Client>` overrides the adapter's own client. Set it at
construction with `AnthropicModel::with_request_options` /
`OpenAiModel::with_request_options`. On the OpenAI adapter it currently covers
the Chat Completions path only (not the Responses/Codex path).

### Multimodal content blocks (implemented)

`ContentBlock` gained `Audio(MediaRef)`, `Video(MediaRef)`, and
`Document(MediaRef)`, where `MediaRef` is `Url { url, media_type }` / `Base64
{ data, media_type }` / `Path { path, media_type }`. The harness never
fetches a `Url` or reads a `Path` itself — resolving those into bytes is a
host concern. `Modalities` gained `video_in`, `video_out`, and
`document_in`. Adapter support follows each provider's actual wire format:
OpenAI's Chat Completions adapter maps `Audio(MediaRef::Base64{..})` to an
`input_audio` content part (and fails closed, rather than silently dropping,
for a `Url`/`Path` audio reference or any `Document`/`Video` block, which
Chat Completions cannot represent); Anthropic's Messages adapter maps
`Document` to a native `document` content block (`base64`/`url` source, or a
placeholder text block for `Path`) and renders `Audio`/`Video` as a
placeholder text block noting the omitted attachment, since neither has a
Messages API wire form. See `providers/openai/test.rs` and
`providers/anthropic/test.rs` for the serialization fixtures. An
SSRF-guarded downloader for resolving `Url`/`Path` references into bytes was
scoped out of this pass (that resolution is a host concern).

## Model Lifecycle Gating

`ModelProfile::status` (`ModelStatus::Stable` / `Preview` / `Deprecated` /
`Retired`) drives lifecycle gating during resolution. Two helpers on
`ModelProfile` classify a profile:

- `is_usable()` — `true` for any status **except** `Retired` (still callable, so
  `Deprecated` is usable but flagged).
- `is_deprecated()` — `true` for both `Deprecated` and `Retired`.

```rust
use tinyagents_harness::model::{ModelProfile, ModelStatus};

let retired = ModelProfile { status: ModelStatus::Retired, ..ModelProfile::default() };
assert!(!retired.is_usable());
assert!(retired.is_deprecated());

let deprecated = ModelProfile { status: ModelStatus::Deprecated, ..ModelProfile::default() };
assert!(deprecated.is_usable());   // still callable, but flagged
assert!(deprecated.is_deprecated());
```

`ModelRegistry::resolve` skips any candidate whose profile reports `Retired`
across **every** resolution path — explicit override, state reuse, hints,
agent default, and registry default — so a provider-retired model is never
selected accidentally. A higher-priority retired hint is passed over for a live
one, and resolution falls through to (or fails on) live candidates rather than a
retired registry default. Opt back in with `ModelSelection { allow_retired:
true, .. }` (for example when replaying historical runs). `Deprecated` models
remain selectable.

```rust
use tinyagents_harness::model::ModelSelection;

// A retired model resolves only when the caller opts in.
let allowed = registry.resolve(ModelSelection {
    requested: Some("retired_override".into()),
    allow_retired: true,
    ..ModelSelection::default()
});
assert_eq!(allowed.unwrap().resolved.name, "retired_override");
```

A model with **no** advertised profile is never lifecycle-excluded (it is only
excluded when it fails a non-empty `required_capabilities` set).

## Provider Options

`ModelSettings` should contain normalized settings:

- temperature
- max output tokens
- stop sequences
- top-p or equivalent sampling controls
- timeout
- seed
- reasoning effort or budget
- response format

`provider_options` should preserve provider-specific knobs such as OpenAI
Responses API options, Anthropic thinking config, Ollama local options, or
future provider extensions. Provider adapters should document which normalized
fields they honor and which provider options they pass through.

For OpenAI-compatible providers, TinyAgents sends normalized fields for common
controls (`temperature`, `top_p`, `max_tokens`, `stop`, and `seed`) and flattens
object-shaped `provider_options` into the request body. Reserved core fields
such as `model`, `messages`, `tools`, `temperature`, `stream`, and
`stream_options` are ignored from `provider_options` so the normalized request
remains authoritative. Local providers can still receive arbitrary typed knobs
through distinct provider fields, for example:

```rust
let request = ModelRequest::new(messages)
    .with_temperature(0.2)
    .with_top_p(0.9)
    .with_provider_options(json!({
        "options": {
            "num_ctx": 8192,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "mirostat": 2
        },
        "keep_alive": "10m",
        "hotness": "spicy"
    }));
```

## Adapter Requirements

Provider adapters must:

- map TinyAgents messages to provider request payloads
- map provider response payloads back to TinyAgents messages
- preserve provider message ids and response ids
- preserve tool call ids and arguments
- produce `invalid_tool_call` data when tool calls cannot be parsed
- surface usage metadata when available
- mark estimated usage when usage is locally estimated
- emit model lifecycle events through `RunContext`
- respect cancellation and timeouts
- distinguish transport, authentication, rate-limit, provider, safety,
  validation, and parsing errors
- expose deterministic fake/replay fixtures for tests

Adapters should avoid global mutable provider clients unless those clients are
thread-safe and explicitly configured.

## Streaming

Streaming model responses should produce typed chunks:

```rust
pub enum ModelStreamItem {
    Started(ModelCallStarted),
    MessageDelta(MessageDelta),
    ToolCallDelta(ToolCallDelta),
    UsageDelta(UsageDelta),
    ProviderEvent(serde_json::Value),
    Completed(ModelResponse),
    Failed(TinyAgentsError),
}
```

`MessageDelta` carries separate visible text and reasoning/thinking fields.
Provider adapters must map exposed thinking fragments, such as
OpenAI-compatible `reasoning_content` or `reasoning` deltas, onto the reasoning
field so UIs can render them without appending them to final assistant text.
Middleware receives the same side channel through `ModelDelta.reasoning`.

The final merged stream must be equivalent to `invoke` for the same provider
where the provider supports both paths. If a provider emits cumulative usage
rather than delta usage, the adapter must normalize it before sending usage
events.

## Conformance Tests

Each provider adapter should pass standard tests for:

- simple text invocation
- system message handling
- Unicode input/output
- tool declaration binding
- tool call id preservation
- no-argument tool calls
- structured output by native provider mode when supported
- structured output by tool strategy when supported
- streaming text chunks
- streaming tool-call chunks when supported
- multimodal input where supported
- usage metadata normalization
- provider options passthrough
- callback/event lifecycle
- cancellation and timeout behavior
- retryable versus non-retryable error classification
