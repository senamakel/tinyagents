# tinyagents-language::src

The declarative `.rag` blueprint format: the safe boundary that lets a model
author the very workflow it is standing inside. This crate is flat (no
submodule directories besides `test/`), so this README serves as the
file-by-file map for the whole crate.

## Design

`.rag` source describes an agent graph — state channels, nodes, routes,
capability references — without ever embedding executable code. It compiles
into the same `Blueprint` shape whether written by a human or emitted by a
model running inside the harness, and every capability reference (model,
tool, agent, subgraph, router, reducer) is bound *by name* against a live
registry before the plan can run. That binding gate is what makes
self-authoring safe: a generated topology cannot smuggle in a capability the
host never sanctioned.

The pipeline is fixed and one-directional:

```text
source -> lexer -> tokens -> parser -> AST -> compiler -> Blueprint -> resolver/capability binding
```

This crate owns everything up to and including the validated `Blueprint`. It
deliberately does **not** depend on `tinyagents-graph` or
`tinyagents-registry`: materialising a `Blueprint` into a runnable
`CompiledGraph` (via a Rust-side `NodeFactory`) is the downstream
`tinyagents-graph` crate's job, and a live capability catalog is supplied by
any caller implementing the local `CapabilitySource` trait — typically
`tinyagents-registry`'s `CapabilityRegistry`. Depending only on
`tinyagents-harness` (for `TinyAgentsError`/`Result`) keeps the language
surface small and auditable, which matters because it is the input path for
agent-authored source.

## Public surface (file by file)

| File | Owns |
| --- | --- |
| `lib.rs` | Crate root; re-exports the public surface and documents the pipeline. |
| `span.rs` | `Span` — a byte range plus 1-based line/column anchor, the location unit every token/AST node/diagnostic carries. |
| `source.rs` | `SourceFile` / `SourceMap` / `SourceId` — owns source text and line-offset indexing so a byte offset can be turned back into `(line, column)` and a snippet. |
| `diagnostic.rs` | `Diagnostic` / `Label` / `Severity` — structured errors plus the caret-underline renderer; folds into `TinyAgentsError::Parse` via `Diagnostic::into_parse_error`. |
| `ast.rs` | The source AST (`Program`, `GraphDecl`, `NodeDecl`, `RouteDecl`, `EdgeDecl`, `CommandDecl`, `SendDecl`, `JoinDecl`, `SteeringDecl`, `Literal`, …) produced by the parser. Re-exported from `types.rs` for back-compat. |
| `types.rs` | `Token`/`SpannedToken` (lexer output) and the compiled `Blueprint` artifact (`ChannelSpec`, `NodeSpec`, `EdgeSpec`, `JoinSpec`, `Routing`, `BlueprintProvenance`, `Origin`, …). Re-exports `ast::*` and `span::Span` so legacy `crate::types::{Program, Span, …}` paths keep resolving. Holds *only* type definitions — see `lexer.rs`/`parser.rs`/`compiler.rs` for the logic that produces and consumes them. |
| `lexer.rs` | `tokenize` — source text into a `Vec<SpannedToken>`. Small, hand-written, side-effect free by design. |
| `parser.rs` | `parse_str` / `parse` — a hand-written recursive-descent parser turning tokens into a `Program` AST. Structural validation only; semantic validation is the compiler's job. |
| `compiler.rs` | `compile` / `compile_with_provenance` — semantic validation (duplicate names, unknown targets, conflicting routing sources, …) and lowering of a `Program` into one `Blueprint` per graph. `compile_source` chains parse → compile → registry-bind in one call. |
| `capability_resolver.rs` | `CapabilityResolver` / `CapabilitySource` / `CapabilityKind` / `ReferenceClass` — the allowlist-based binding gate. `bind_capabilities` (legacy model/tool-only) and `bind_capabilities_with_registry` (full registry-backed: node kinds, subgraphs, routers, reducers, agents, scripts) validate a compiled `Blueprint`. |
| `resolver.rs` | `Resolver` / `resolve_source` — the spanned-diagnostic counterpart to `capability_resolver.rs`. Walks the AST directly (or a compiled `Blueprint`) and reports every unresolved reference as a source-anchored `Diagnostic`, collecting all of them rather than failing fast. Wraps a `CapabilityResolver` internally so both binding paths share one policy. |
| `diff.rs` | `blueprint_diff` / `BlueprintDiff` / `NodeDiff` / `ChannelDiff` / `FieldChange` — structured, human-readable diffs between two compiled `Blueprint`s, backing generated-workflow review. |
| `testkit.rs` | Deterministic `parse -> compile` helpers (`try_compile`, `compile_all`, `blueprint`, `blueprint_with_provenance`, `node`, `assert_kind`, `assert_next`, `assert_terminal`, `assert_route`) for asserting on lowered topology without a registry. Re-exported from `lib.rs` as `language_testkit`. |
| `test/` | Unit test suite, split by pipeline phase (see below). Not a public module (`#[cfg(test)] mod test;`). |

### Capability binding: two paths, one policy

Both binding paths exist because they serve different callers, but they route
through the *same* classification policy
(`CapabilityResolver::classify_reference` /
`secondary_model_reference`) so they cannot drift apart:

- `capability_resolver.rs`'s `CapabilityResolver::bind_blueprint` /
  `bind_capabilities_with_registry` — span-less, used when only a compiled
  `Blueprint` is available (e.g. loaded from storage).
- `resolver.rs`'s `Resolver` — spans-aware, used on the AST right after
  parsing so failures render a caret-underline diagnostic pointing at the
  offending `.rag` source. `resolve_source` is the recommended single entry
  point: parse, resolve against a registry with full source spans, then
  lower to validated blueprints.

### Provenance

`compile_with_provenance` (used by `testkit.rs` and any caller that needs
traceability) attaches a `BlueprintProvenance` to the `Blueprint`, recording
the source `Span` of every node/channel/edge and the blueprint's `Origin`
(`File` vs. `Generated`). The plain `compile` path leaves `provenance: None`
so its output is unaffected. `diff.rs` ignores provenance entirely — a
`blueprint_diff` is purely a function of compiled topology and bindings.

## `test/` layout

Tests are split by pipeline phase rather than kept in one file:

| File | Covers |
| --- | --- |
| `mod.rs` | Shared fixtures (e.g. `SUPPORT_AGENT`) and `mod` wiring for the test submodules. |
| `lexer.rs` | Tokenization, escapes, spans, literal formatting. |
| `parser.rs` | Grammar productions into the AST. |
| `extended_grammar.rs` | Extended grammar: channel policy args, `command`, `send`/`join`, `subgraph`, `subagent`, `repl_agent`, `interrupt`, IO shape, checkpoint/interrupt policy. |
| `compiler.rs` | AST → `Blueprint` semantic validation and lowering. |
| `capability_binding.rs` | The legacy `bind_capabilities` allowlist gate. |
| `registry_binding.rs` | Registry-backed binding (`bind_capabilities_with_registry` / `compile_source`). |
| `resolver.rs` | The spanned-diagnostic `Resolver` binding gate. |
| `diagnostics.rs` | Span, source-map, and diagnostic-rendering behaviour. |
| `provenance_diff_testkit.rs` | Provenance tagging, `blueprint_diff`, and `testkit.rs` helpers. |
| `graph_materialisation.rs` | Lowering a `Blueprint` into a runnable graph via a `NodeFactory` and executing it, exercising the downstream `tinyagents-graph` integration surface. |

**Known gap (not fixed here — docs-only pass):** `graph_materialisation.rs`,
`registry_binding.rs`, and `resolver.rs` are present in this directory but are
*not* declared as `mod` items in `test/mod.rs`. They are therefore excluded
from compilation and their tests never run under `cargo test`. This looks
like an accidental omission (likely from splitting `test/mod.rs` by pipeline
phase) rather than intentional; wiring them back in is a behavior change
outside the scope of this documentation pass.

## Operational constraints

- The lexer/parser/compiler pipeline never executes or evaluates anything in
  the source — the grammar only admits declarations and capability-by-name
  references, which is what makes it safe to run on model-generated text.
- Every diagnostic-producing path (`Diagnostic::into_parse_error`,
  `Resolver::check_program`, `fold_diagnostic`) renders a caret-underline
  presentation when a `SourceFile` is available and falls back to a
  source-free `line:column` rendering otherwise; callers that only hold a
  token stream (no original source) get the latter.
- `compile` rejects any node carrying a `steering { … }` block outright
  rather than silently dropping it: no faithful lowering onto the runtime's
  `SteeringPolicy` exists yet (no `parent`/`human` actor separation, no
  delivery policy). The grammar still parses it so tooling can read the
  declaration.
- This crate intentionally has no dependency on `tinyagents-registry` or
  `tinyagents-graph`. Registry-shaped capability sources are supplied through
  the local `CapabilitySource` trait; runnable graph materialisation happens
  entirely downstream, in `tinyagents-graph`.
