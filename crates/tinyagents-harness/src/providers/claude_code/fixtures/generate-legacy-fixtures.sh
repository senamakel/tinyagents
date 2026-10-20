#!/usr/bin/env bash
set -euo pipefail

old_sha=d5ef63ce7ece369e8faaba69ab32a881540dde78
fixture_dir=crates/tinyagents-harness/src/providers/claude_code/fixtures
generator_dir=target/legacy-fixture-generator

mkdir -p "$generator_dir/src" "$generator_dir/old-release" "$generator_dir/output"
git show "$old_sha:crates/tinyagents-harness/src/providers/claude_code/session_store.rs" \
  > "$generator_dir/old-release/session_store.rs"
git show "$old_sha:crates/tinyagents-harness/src/providers/claude_code/settings.rs" \
  > "$generator_dir/old-release/settings.rs"
cp "$fixture_dir/generate-legacy-fixtures.rs" "$generator_dir/src/main.rs"

cat > "$generator_dir/Cargo.toml" <<'MANIFEST'
[package]
name = "claude-code-legacy-fixture-generator"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }
MANIFEST

cargo run --manifest-path "$generator_dir/Cargo.toml" -- "$generator_dir/output"
cp "$generator_dir/output/sessions/claude-code-sessions.json" "$fixture_dir/claude-code-sessions.json"
cp "$generator_dir/output/settings/claude_code_settings.json" "$fixture_dir/claude_code_settings.json"
