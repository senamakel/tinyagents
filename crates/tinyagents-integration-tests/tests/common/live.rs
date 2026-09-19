//! Opt-in gating for tests that make real, billable network calls.
//!
//! Every `live_*.rs` test is also marked `#[ignore]`, so it never runs on a
//! bare `cargo test`. To actually run one, opt in explicitly:
//!
//! ```text
//! TINYAGENTS_LIVE=1 cargo test -p tinyagents-integration-tests --test live_streaming \
//!     -- --ignored --nocapture
//! ```
//!
//! [`require_live`] is the single gate every live test calls. It:
//!
//! 1. Requires `TINYAGENTS_LIVE=1` (or, kept as a backwards-compatible alias
//!    for `live_prompt_cache.rs`, `PROMPT_CACHE_LIVE=1`) to already be set in
//!    the *process* environment, checked before `.env` is touched at all — so
//!    a `.env` file alone, with no explicit opt-in, can never make a live
//!    test dial out.
//! 2. Only once that flag is confirmed, loads `.env` via `dotenvy` so local
//!    credentials can live there instead of the shell environment.
//! 3. Checks that every environment variable named in `keys` is present and
//!    non-empty, printing a one-line skip reason that names whatever is
//!    missing.
//!
//! This replaces the copy-pasted
//! `let _ = dotenvy::dotenv(); if std::env::var("OPENAI_API_KEY").is_err() { return; }`
//! gate that used to open every live test: that pattern loaded `.env`
//! unconditionally, so any dev box with a `.env` file ran (and paid for) real
//! API calls on a bare `cargo test`, while CI only stayed green because it
//! happened not to have a `.env` file lying around. `#[ignore]` on every live
//! test, combined with this explicit double opt-in (an env flag *and*
//! `--ignored`), makes both failure modes impossible.

/// Returns `true` when live tests are enabled (`TINYAGENTS_LIVE=1`, or the
/// `PROMPT_CACHE_LIVE=1` alias) and every variable named in `keys` is set to a
/// non-empty value. Loads `.env` (via `dotenvy`) only after confirming the
/// opt-in flag. Prints a one-line reason and returns `false` when the test
/// should skip.
pub fn require_live(keys: &[&str]) -> bool {
    if !live_flag_set() {
        eprintln!(
            "skipping live test: set TINYAGENTS_LIVE=1 and run with --ignored to enable it"
        );
        return false;
    }

    // Only load `.env` once the caller has explicitly opted in, so a stray
    // `.env` file can never by itself cause a live test to run.
    let _ = dotenvy::dotenv();

    let missing: Vec<&str> = keys
        .iter()
        .copied()
        .filter(|key| {
            std::env::var(key)
                .map(|value| value.trim().is_empty())
                .unwrap_or(true)
        })
        .collect();

    if !missing.is_empty() {
        eprintln!(
            "skipping live test: missing required env var(s): {}",
            missing.join(", ")
        );
        return false;
    }

    true
}

/// `true` when the live opt-in flag is set in the process environment.
///
/// Checked with `std::env::var` directly (not `.env`) so the opt-in itself
/// must come from the shell or CI environment rather than a file that could
/// silently be sitting on a dev box.
fn live_flag_set() -> bool {
    let is_on = |name: &str| std::env::var(name).map(|v| v == "1").unwrap_or(false);
    is_on("TINYAGENTS_LIVE") || is_on("PROMPT_CACHE_LIVE")
}
