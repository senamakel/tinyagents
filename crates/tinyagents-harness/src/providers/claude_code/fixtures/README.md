# Legacy Claude Code persistence fixtures

These files were written by the pre-migration file-backed public APIs from
TinyAgents commit `d5ef63ce7ece369e8faaba69ab32a881540dde78`. The generator
inputs `session_store.rs` and `settings.rs` are byte-identical to their
pre-epic versions at `31cc57180f67b2f539a3c98d03388251084f0da0`. The files are
not attributed to a release tag. `claude-code-sessions.json` comes from
`session_store::SessionStore::open`, `set`, and `record_delivered`; it includes
an orphan delivered-turn entry because the old API allowed recording turns
before a session UUID existed. `claude_code_settings.json` comes from
`settings::save`.

Regenerate them from the TinyAgents repository root with:

```sh
bash crates/tinyagents-harness/src/providers/claude_code/fixtures/generate-legacy-fixtures.sh
```

The script extracts those exact source files from the archived commit, builds
the small generator in `target/`, and copies the resulting old-writer bytes
back here. The tests import these captured bytes directly; they do not rebuild
the old schema locally.

Fixture SHA-256 values:

```text
742e9251ce13e2efc1ec19315b1d717aeed50e62e61a3fea92a48616108cd520  claude-code-sessions.json
42c3736530ed37f3a0072b863b864e44bca3fe251deaa95c09954c9b2ecf97a6  claude_code_settings.json
```
