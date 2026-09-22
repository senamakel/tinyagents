//! Model-family execution guidance.
//!
//! Some model families (DeepSeek, GLM, Qwen, GPT, Grok, ...) tend to stop
//! after announcing a plan, answer from memory what a tool could look up, or
//! ask a clarifying question where the default interpretation is obvious.
//! Others (Claude, Gemini) already behave that way by default and only lose
//! prompt budget to the reminder. A host renders
//! [`execution_discipline_for`] into a stable tier of its system prompt for
//! the families that need it and leaves it out for the rest, the same way
//! Hermes gates its "execution discipline" block.
//!
//! The matcher is a substring test over the lowercased model id (which may
//! carry a provider prefix such as `openrouter/deepseek/deepseek-v4-flash`),
//! so a host needs nothing but the resolved model name.

use tinyinference_llm::model::ModelProfile;

/// Guidance rendered for the model families in [`NEEDS_EXECUTION_DISCIPLINE`].
pub const EXECUTION_DISCIPLINE: &str = "\
## Execution discipline

- When you say you will do something, make the tool call in the same response. Never end a turn on an announcement or a plan.
- Keep going until the task is done or you have a concrete result to hand back; a summary of what you would do next is not a result.
- Batch independent tool calls into one response instead of one call per turn.
- Use a tool for anything a tool can check: current facts, file contents, system state, arithmetic, dates.
- When a request has an obvious default interpretation, act on it; ask only when the ambiguity changes which tool you would call.
";

/// Lowercase substrings of model ids whose families benefit from
/// [`EXECUTION_DISCIPLINE`].
pub const NEEDS_EXECUTION_DISCIPLINE: &[&str] = &[
    "deepseek", "glm", "qwen", "gpt-", "o1-", "o3-", "o4-", "grok", "kimi", "minimax", "mistral",
    "llama",
];

/// Lowercase substrings of model ids that never receive the block, even when
/// a provider prefix or alias happens to contain one of the family markers.
const NEVER: &[&str] = &["claude", "gemini"];

/// Whether `model` names a family that should carry [`EXECUTION_DISCIPLINE`].
///
/// `model` may carry a provider prefix (`openrouter/deepseek/...`) or a
/// managed alias; the test is a case-insensitive substring match. An empty or
/// unknown id returns `false` so an unrecognised model pays nothing.
#[must_use]
pub fn needs_execution_discipline(model: &str) -> bool {
    let lower = model.trim().to_ascii_lowercase();
    if lower.is_empty() || NEVER.iter().any(|family| lower.contains(family)) {
        return false;
    }
    NEEDS_EXECUTION_DISCIPLINE
        .iter()
        .any(|family| lower.contains(family))
}

/// [`EXECUTION_DISCIPLINE`] when `model` needs it, `None` otherwise.
#[must_use]
pub fn execution_discipline_for(model: &str) -> Option<&'static str> {
    needs_execution_discipline(model).then_some(EXECUTION_DISCIPLINE)
}

/// [`execution_discipline_for`] over a resolved [`ModelProfile`], checking the
/// profile's `model` id first and its `provider` as a fallback.
#[must_use]
pub fn execution_discipline_for_profile(profile: &ModelProfile) -> Option<&'static str> {
    let model = profile.model.as_deref().unwrap_or_default().trim();
    let lower = model.to_ascii_lowercase();
    if NEVER.iter().any(|family| lower.contains(family)) {
        return None;
    }
    if let Some(guidance) = execution_discipline_for(model) {
        return Some(guidance);
    }
    execution_discipline_for(profile.provider.as_deref().unwrap_or_default())
}
