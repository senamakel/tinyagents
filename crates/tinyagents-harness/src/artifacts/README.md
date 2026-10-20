# harness::artifacts

Blob-store offload for oversized worker results on long-horizon runs, plus the
fail-closed key resolution and host policy hooks it depends on.

## Why this exists

For minutes-to-hours runs, keeping every worker's full result inline in
context accumulates without bound, and a summary can never restore full
fidelity. The convention instead writes an oversized result to disk under the
agent's artifact root and hands the parent a short pointer (path + abstract)
instead of the payload. Context stays lean; the full artifact is recoverable
with an ordinary file read.

Two directories under the artifact root:

| Directory    | Holds                                                   |
| ------------ | -------------------------------------------------------- |
| `outputs/`   | Deliverables, handed between steps **by path**.          |
| `workspace/` | Scratch, not meant to be handed back.                     |

The convention has two halves, deliberately split: a host-owned **prompt**
half that tells workers to offload on purpose (not in this crate — it names
host tools), and the **harness** half here ([`offload_oversized_result`]),
which fires on every worker outcome so an oversized result is offloaded even
when the worker inlined it anyway.

Every failure mode in this module is soft: the caller falls back to its
original inline payload and whatever summarisation/truncation backstop it
already had.

## Public surface

- [`ArtifactOffload`] (`ops.rs`) — per-run writer holding a
  `BlobStore`, host policies, and the ids used to name artifacts and tag
  logs. `write` / `write_returning_stored` perform a redacted write; `read`
  fetches one back by key; `resolve` validates a name without writing.
- [`offload_oversized_result`] (`ops.rs`) — the deterministic entry point:
  offloads `output` when it exceeds a threshold, returning the text the
  parent should receive plus the artifact when one was written.
- [`should_offload`], [`effective_offload_threshold`], [`build_abstract`],
  [`render_artifact_pointer`], [`extract_artifact_paths`],
  [`note_artifact_handoff`] (`ops.rs`) — the supporting free functions:
  threshold decision, cap-aware threshold, abstract truncation, pointer
  rendering, pointer parsing, handoff logging, and safe filesystem
  materialization. The filesystem variant checks the host policy against its
  resolved target before storing the blob.
- [`resolve_artifact_key`], [`sanitize_component`] (`paths.rs`) — fail-closed
  key resolution under the two convention namespaces and the helper it builds on.
- [`policy`] module (`policy.rs`) — the two host-supplied gates:
  [`ArtifactPathPolicy`] (which paths are off limits) and
  [`ArtifactRedactor`] (what is scrubbed before bytes are stored), plus the
  permissive [`OpenPathPolicy`] / [`NoRedaction`] defaults used in tests and
  by hosts with nothing to restrict.
- [`ArtifactKind`], [`OffloadedArtifact`], [`OffloadError`] and the
  convention constants ([`OUTPUTS_DIR`], [`SCRATCH_DIR`],
  [`DEFAULT_OFFLOAD_THRESHOLD_BYTES`], [`ABSTRACT_BUDGET_CHARS`],
  [`ARTIFACT_POINTER_PREFIX`]) (`types.rs`) — the inert value types, kept
  dependency-free (std + `thiserror` only) so a host can build against them
  without pulling in the rest of the engine.

## Files

| File         | Role                                                                 |
| ------------ | --------------------------------------------------------------------- |
| `mod.rs`     | Module overview, re-exports.                                          |
| `types.rs`   | `ArtifactKind`, `OffloadedArtifact`, `OffloadError`, constants.        |
| `paths.rs`   | Fail-closed key resolution (`resolve_artifact_key`, `sanitize_component`). |
| `policy.rs`  | `ArtifactPathPolicy`, `ArtifactRedactor` traits and their defaults.    |
| `tool_results.rs` | Per-tool-result persistence: `ToolResultArtifactStore` (over a host-supplied tinystoragedrivers `BlobStore`; pointers are `tool-results/<session>/<tool>/<call>.txt` blob keys; `read`, `purge_session`, `with_retention`), `apply_per_result_persistence`, `spill_aggregate_tool_results`, paged artifact reads (`read_target`). Host supplies redactor, read/wrapper tool names, read limit, and a read tool that serves `tool-results/` keys from the blob store. |
| `tool_results_tests.rs` | Tests for the above, incl. a byte-stable envelope fixture. |
| `ops.rs`     | `ArtifactOffload` writer over a `BlobStore`, `offload_oversized_result`, pointer/handoff plumbing. |
| `mod_tests.rs`, `contract_tests.rs`, `policy_tests.rs`, `request_truncation_tests.rs` | Unit tests: happy path, fallback path, fail-closed hardening. |

## Operational constraints

- Artifacts are blobs on a tinystoragedrivers `BlobStore` (in memory unless
  the host passes its scoped handle to `ArtifactOffload::new`). A pointer
  carries the blob key (`outputs/<name>`), not a filesystem path: read it back
  with `ArtifactOffload::read`, or have the host expose the store through its
  own read tool or materialise it. `resolve_artifact_key` is fail-closed on
  `..`, absolute names, drive prefixes and control characters; there are no
  symlinks to escape through.
- A `None` redactor stores bytes verbatim. That is a legitimate choice for a
  host with nothing to scrub, but it is a security decision the host must make
  explicitly — this crate cannot supply credential/PII patterns for it.
- Any preview surfaced back into a model's context must be built from the
  **stored** (post-redaction) body, never from the caller's original input.
  `write_returning_stored` and `offload_oversized_result` exist specifically
  so callers have that stored value at hand.
- Every write and every handoff emits a structured `[artifact]` log line, so a
  run journal shows both ends of a pointer (see `mod.rs` for the exact
  messages).
