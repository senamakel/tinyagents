# Harness Structured Output Feature

Structured output lets callers request a typed response instead of free-form
assistant text. The harness should support both provider-native structured
output and tool-call-based structured output.

## Source Inspiration

LangChain v1 implements structured output with provider and tool strategies:

- <https://github.com/langchain-ai/langchain/blob/master/libs/langchain_v1/langchain/agents/structured_output.py>
- provider implementations such as OpenAI:
  <https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/chat_models/base.py>
- standard structured-output tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/standard-tests>

## Responsibilities

- Accept Rust types and JSON schema as response schemas.
- Support provider-native response-format APIs.
- Support artificial tool-call schemas when provider-native mode is unavailable.
- Validate responses into typed values.
- Preserve raw assistant messages alongside parsed output.
- Handle validation errors according to policy.
- Support union/oneOf variants.
- Support strict and non-strict schema modes.
- Emit structured-output events.

## Strategies

`StructuredStrategy` (`crates/tinyagents-harness/src/structured/types.rs`) has
four variants:

```rust
pub enum StructuredStrategy {
    ProviderSchema,
    ToolCall,
    Prompted { template: Option<String> },
    ToolCallUnion,
}
```

`StructuredStrategy::for_profile` resolves `ResponseFormat::Auto` to
`ProviderSchema` (native structured output + JSON Schema support, or no
profile) or `ToolCall` (tool-calling model without native structured output);
it never returns `Prompted`/`ToolCallUnion` — those two are reached only
through `RunPolicy::structured_strategy_override`
(`StructuredStrategyOverride::{Prompted { template }, ToolCallUnion {
variants }}`), which bypasses the profile-based heuristic outright when set.

- **`Prompted`** (A6) — for a model with no native schema or tool-calling
  support to lean on: the schema is injected into a leading system message
  instead of a provider API field (`default_prompted_template()`'s wording, or
  a caller-supplied `template`), and extraction reuses the same
  `ProviderSchema` code path (parse response text through the repair ladder).
  Mirrors Pydantic AI's `PromptedOutput`.
- **`ToolCallUnion`** (A6) — one synthetic tool per `(name, schema)` variant
  instead of a single schema tool; extraction
  (`StructuredExtractor::extract_tool_call_union`) scans the response's tool
  calls for the first one matching *any* variant, validates its arguments
  against *that variant's* schema, and records which one matched on
  `StructuredOutput::variant` / `AgentRun::structured_variant`. Build the
  extractor directly with `StructuredExtractor::new_union(label, variants)`
  when driving extraction outside the agent loop.

## Provider Strategy

Provider strategy sends the schema to the model provider through a native
response-format API. It is preferred when:

- the provider validates server-side
- the model profile declares structured output support
- the schema is accepted by that provider
- streaming behavior is understood

Provider strategy must still validate the returned value locally. Provider
validation errors should be surfaced as provider errors; local parsing errors
should be surfaced as structured-output validation errors.

## Tool Strategy

Tool strategy exposes an artificial output tool to the model. The final
structured output is parsed from the tool-call arguments.

Tool strategy must handle:

- multiple structured-output tool calls when only one is expected
- missing structured-output tool calls
- invalid JSON arguments
- schema validation errors
- union variant selection
- optional repair messages back into the agent loop

The artificial tool should not execute application side effects. It is a parse
carrier only.

## `EndStrategy`: output tool + function tools in one turn (A6)

A single turn can both answer (call the structured-output schema tool, under
`StructuredStrategy::ToolCall`/`ToolCallUnion`) *and* ask to run further
tools. `RunPolicy::end_strategy: EndStrategy` decides what happens to the two,
mirroring Pydantic AI's `end_strategy`:

- **`Graceful`** (default) — run the accompanying function-tool calls (their
  side effects and results are never silently dropped), then finish the run
  with the structured output already recorded. Never spends an extra model
  call once the model has already answered.
- **`Early`** — finish immediately on the output-tool call. The accompanying
  function-tool calls are **not** executed; their `tool_calls` entries are
  closed with a synthetic "run stopped before this tool call was executed"
  result so the transcript stays replayable for a future turn.
- **`Exhaustive`** — ignore the output-tool call this turn entirely (never
  recorded): run the function-tool calls and give the model another turn,
  exactly as if the output tool had not been called. The run only finishes
  once a later turn's output-tool call has no accompanying function-tool
  calls.

This replaces the pre-A6 behavior, which always recorded the structured value
and then unconditionally continued the loop (equivalent to neither `Graceful`
nor `Exhaustive` — it recorded early like `Graceful` but kept going like
`Exhaustive`, discarding the recorded value the moment a later turn produced a
different one).

## Error Policy: the output-validation retry loop (A3)

Extraction itself is `StructuredExtractor::extract(&response)`
(`crates/tinyagents-harness/src/structured/mod.rs`): it parses and validates a
single completed `ModelResponse` into `Result<StructuredOutput>`, climbing a
local repair ladder (code fence, prose slice, relaxed JSON, truncation close)
and validating against the declared schema.
`StructuredExtractor::extract_outcome` is the non-fatal sibling — it returns a
`StructuredOutcome { value, raw, error, variant }` recording a failure as data
instead of an `Err`.

The agent loop's **final turn** wraps that non-fatal extraction in a retry
loop instead of propagating the first failure. Two things can trigger a retry:

- **Extraction failure** — `extract_outcome`'s `error` is `Some` (schema-
  invalid or unparseable text).
- **Validator rejection** — a registered `OutputValidator<State, Ctx>`
  (`AgentHarness::with_output_validator`) is called once extraction *did*
  succeed, and returns `Err(TinyAgentsError::ModelRetry(message))`. Any other
  `Err` variant fails the run immediately, exactly like an error from any
  other fallible call in the loop.

```rust
#[async_trait]
pub trait OutputValidator<State: Send + Sync, Ctx: Send + Sync = ()>: Send + Sync {
    async fn validate(
        &self,
        ctx: &mut RunContext<Ctx>,
        state: &State,
        output: &serde_json::Value,
    ) -> Result<()>;
}
```

Either failure pushes `RunPolicy::output_retry.message_template` (default
`"{error}\n\nFix the errors and try again."`, with `{error}` substituted) onto
the transcript as a `Message::user` turn, emits
`AgentEvent::OutputRetry { attempt, error }`, and `continue`s the loop — so the
retry costs one more model call and counts against
`RunLimits::max_model_calls` like any other. `RunPolicy::output_retry` is an
`OutputRetryPolicy { max_attempts: u8, message_template: String }`, default
`max_attempts = 1` (one retry, two attempts total); `max_attempts = 0`
disables the loop entirely, reproducing the pre-A3 one-shot behavior.
Exhausting the budget fails the run with
`TinyAgentsError::StructuredOutput`, same as before A3 existed.

Tool-level vocabulary mirrors this: a tool that wants "ask the model to try
again" returns `Err(TinyAgentsError::ModelRetry(msg).into())` from
`Tool::execute` instead of `Ok(ToolResult::error(..))`; the harness folds it
into a recoverable `ToolResult::retry(msg)` instead of aborting the run (see
`agent_loop/tools.rs::execute_tool_recovering_model_retry`).
`TinyAgentsError::ToolFailed` is the permanent counterpart
(`ToolResult::failed(msg)`) and `crate::retry::is_retryable` treats it as
non-retryable, so `RetryMiddleware` does not re-attempt a call a tool has
explicitly marked permanent.

`AgentRun::structured_as::<T: DeserializeOwned>()` is the typed convenience
over `run.structured` (`Result<T>`, erroring with
`TinyAgentsError::StructuredOutput` when the run produced no structured value
or `T` does not deserialize).

## Return Shape

```rust
pub struct StructuredRun<T> {
    pub parsed: T,
    pub raw: AssistantMessage,
    pub validation: ValidationRecord,
}
```

When callers request `include_raw`, the harness should return both raw and
parsed values. When they do not, the raw value should still be available through
events and stores if recording is enabled.
