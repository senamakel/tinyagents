//! Resolve an `ANTHROPIC_API_KEY` for the spawned `claude` CLI.
//!
//! v1 resolution order:
//!   1. Process env `ANTHROPIC_API_KEY` (highest precedence).
//!   2. `~/.claude/.credentials.json` — only used if the CLI is already
//!      logged in via `claude login`. We pass it through transparently by
//!      *not* setting `ANTHROPIC_API_KEY`; the CLI then reads its own
//!      credentials file.
//!
//! v1.1 will wire a host-provided auth-profile store so an Anthropic key
//! saved in the embedding application's settings is picked up automatically.
//! Subscription / OAuth auth (Claude Pro/Max) is deferred to v2. See
//! `auth_status::probe` for the complementary read-only status check used by
//! settings UIs.

/// Where the resolved Anthropic credential came from, for logging and UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthSource {
    /// Explicit API key — pass via `ANTHROPIC_API_KEY` env var.
    EnvApiKey,
    /// No explicit key resolved. Defer to whatever the CLI finds in
    /// `~/.claude/.credentials.json`.
    CliCredentials,
}

/// Probe sources in priority order. Returns the resolved API key plus the
/// origin label (for logging) when found. The returned key is only the
/// key value — call-sites set env on spawn, never log it.
pub fn resolve() -> (AuthSource, Option<String>) {
    if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
        let k = k.trim();
        if !k.is_empty() {
            return (AuthSource::EnvApiKey, Some(k.to_string()));
        }
    }
    (AuthSource::CliCredentials, None)
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
