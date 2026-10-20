//! Persist oversized tool outputs as artifacts the model can read back.
//!
//! Tool results enter the model context before the provider has seen them, so
//! this is the last cheap point to replace large raw output with a bounded
//! preview. The full, redacted body is written to a tinystoragedrivers
//! [`BlobStore`] under `harness/tool_results/<session>/<tool>/<call>.txt`, and the
//! pointer the model receives is that blob key.
//!
//! The store takes a blob handle, never a filesystem path:
//!
//! * [`ToolResultArtifactStore::new`] writes to a host-supplied scoped blob
//!   store. The library has no implicit storage fallback.
//!
//! Nothing lands in the agent's working tree. A host that lets the model open
//! the pointer must serve the key from the blob store in its read tool (see
//! [`ToolResultArtifactStore::read`]); that wiring is policy and therefore the
//! host's, not this crate's.
//!
//! Distinct from [`offload_oversized_result`](super::offload_oversized_result),
//! which offloads a *worker's final result* under `outputs/`: this handles each
//! *tool result* mid-turn under `tool-results/<session>/<tool>/`.
//!
//! What the host supplies, because the crate cannot decide it: the
//! [`ArtifactRedactor`] every body passes through before it is stored, the name
//! of its reading tool, the wrapper tool a read may arrive inside, and the
//! largest body its reader will open.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tinyagents_storage_kit::{BlobStore, Capability};

use super::policy::{ArtifactRedactor, Redacted};

/// The blob namespace every tool-result artifact lives under.
const ARTIFACT_ROOT: &str = "harness/tool_results";
/// Pointers written before the v2 schema (`tool-results/...`) or before the
/// store moved onto blobs (`artifacts/tool-results/...`) still page: a
/// transcript outlives the store. [`canonical_key`] maps both onto
/// [`ARTIFACT_ROOT`], where the storage step moved the blobs.
const LEGACY_ARTIFACT_ROOTS: [&str; 2] = ["tool-results", "artifacts/tool-results"];

/// A read of a persisted artifact, recognised from a tool call's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRead {
    pub path: String,
    pub offset: usize,
}

/// The artifact a tool call reads, if any.
///
/// `read_tool` and `wrapper_tool` are the host's vocabulary, passed in because
/// tool names are not the crate's to decide (OpenHuman passes `file_read` and
/// `use_skill`).
///
/// Only a `read_tool` call counts, because only its result *is* the stored body:
/// `file_write`, `glob`, `list` or `apply_patch` can name an artifact path too,
/// and their output must still take the normal ladder. The read may arrive
/// wrapped in `wrapper_tool`, reported under that name
/// (`use_skill {"skill":"files","tool":"file_read","args":{"path":…}}`), so
/// `wrapper_tool` — and only it, the one tool whose result *is* the
/// wrapped tool's result — is followed into the tool it runs (#6284). Any
/// other tool that happens to carry `tool`/`args` fields is not a wrapper.
///
/// This recognises the `tool-results/…` blob key a [`ToolResultArtifactStore`]
/// hands out (and the legacy `artifacts/tool-results/…` form).
pub fn artifact_read_target(
    tool_name: &str,
    args: &Value,
    read_tool: &str,
    wrapper_tool: &str,
) -> Option<ArtifactRead> {
    read_target_matching(tool_name, args, read_tool, wrapper_tool, &is_artifact_key)
}

/// A path component match: `tool-results-backup/…` shares the prefix but is
/// not the artifact namespace. The legacy `artifacts/tool-results/…` form is
/// recognised too, so a transcript written before the move keeps paging.
fn is_artifact_key(path: &str) -> bool {
    let path = path.trim_start_matches("./");
    std::iter::once(&ARTIFACT_ROOT)
        .chain(&LEGACY_ARTIFACT_ROOTS)
        .any(|root| {
            path.strip_prefix(root)
                .is_some_and(|rest| rest.starts_with('/'))
        })
}

fn read_target_matching(
    tool_name: &str,
    args: &Value,
    read_tool: &str,
    wrapper_tool: &str,
    is_artifact: &dyn Fn(&str) -> bool,
) -> Option<ArtifactRead> {
    if tool_name == wrapper_tool {
        let inner_tool = args.get("tool").and_then(Value::as_str)?;
        return read_target_matching(
            inner_tool,
            args.get("args")?,
            read_tool,
            wrapper_tool,
            is_artifact,
        );
    }
    if tool_name != read_tool {
        return None;
    }
    let path = args.get("path").and_then(Value::as_str)?;
    let under_root = is_artifact(path);
    // An absent or null offset starts at 0. A present one that is not a
    // non-negative integer that fits `usize` is not a read `file_read` serves
    // (it rejects it), so it is not an artifact read either; never reinterpret
    // it as 0.
    let offset = match args.get("offset") {
        None | Some(Value::Null) => 0,
        Some(value) => usize::try_from(value.as_u64()?).ok()?,
    };
    under_root.then(|| ArtifactRead {
        path: path.to_string(),
        offset,
    })
}

/// Bound one page of an artifact read to `budget_bytes`, naming the exact
/// `offset` the next read continues from. Never persists: re-persisting a read
/// of an artifact creates a new artifact whose preview is the same bounded
/// head, and the model can loop between previews without ever reaching the
/// body.
pub fn page_artifact_read(
    content: String,
    read: &ArtifactRead,
    budget_bytes: usize,
    read_tool: &str,
) -> String {
    if budget_bytes == 0 {
        return content;
    }
    // A page is useless without its continuation, so a budget too small to
    // carry one is raised to the floor a persisted envelope already takes for
    // the same reason (`MIN_ENVELOPE_ALLOWANCE_BYTES`). Every page therefore
    // fits `max(budget_bytes, MIN_ENVELOPE_ALLOWANCE_BYTES)` and advances.
    let budget_bytes = budget_bytes.max(MIN_ENVELOPE_ALLOWANCE_BYTES);
    if content.len() <= budget_bytes {
        return content;
    }
    let start = read.offset;
    let Some(total) = start.checked_add(content.len()) else {
        // Only reachable with an offset no real read carries (`file_read`
        // rejects offsets past its at-most-10-MiB file). Bound the result but
        // advertise no continuation, since none could advance.
        let cut = floor_char_boundary(&content, budget_bytes);
        return content[..cut].to_string();
    };
    let escaped_path = serde_json::to_string(&read.path).expect("serializing a string cannot fail");
    let with_path = |next: usize| {
        format!(
            "\n\n[artifact page: bytes {start}..{next} of {total}. Continue with {read_tool} {{\"path\":{escaped_path},\"offset\":{next}}}]"
        )
    };
    // Without the path (the caller already has it). At most ~100 bytes, so it
    // always leaves body room under the floor.
    let without_path = |next: usize| {
        format!(
            "\n\n[artifact page: bytes {start}..{next} of {total}. Continue with {read_tool} at \"offset\":{next}]"
        )
    };
    // Sized from the trailer this page will actually carry, not a fixed
    // reservation. `next` never has more digits than `total`, so a trailer
    // rendered with `total` is its longest form.
    let use_path = with_path(total).len() + 4 <= budget_bytes;
    let longest = if use_path {
        with_path(total).len()
    } else {
        without_path(total).len()
    };
    let cut = floor_char_boundary(&content, budget_bytes - longest);
    let trailer = if use_path {
        with_path(start + cut)
    } else {
        without_path(start + cut)
    };
    format!("{}{trailer}", &content[..cut])
}
const AGGREGATE_PREVIEW_BUDGET_BYTES: usize = 512;
/// #4469 item 6: floor for how tightly a persisted `[tool_result_preview]`
/// envelope may be bounded during aggregate spill. `allowed_len` can saturate to
/// `0` (or a handful of bytes) once earlier-spilled results have already consumed
/// the aggregate budget; bounding the envelope to that would return `""` — or a
/// header cut mid-line — discarding the `artifact_path` pointer the model needs
/// to `file_read` the full output. This floor keeps the envelope header (through
/// the `artifact_path` / `read_with` lines) intact even when the raw budget math
/// says zero; `apply_tool_result_budget` retains the head, so the pointer always
/// survives. Slightly overshooting the aggregate budget here is the correct
/// trade — a valid pointer is worth a few hundred bytes.
const MIN_ENVELOPE_ALLOWANCE_BYTES: usize = 512;
pub(super) const TRAILER_RESERVED: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BudgetOutcome {
    original_bytes: usize,
    final_bytes: usize,
    truncated: bool,
}

impl BudgetOutcome {
    fn unchanged(len: usize) -> Self {
        Self {
            original_bytes: len,
            final_bytes: len,
            truncated: false,
        }
    }
}

pub(super) fn apply_tool_result_budget(
    content: String,
    budget_bytes: usize,
) -> (String, BudgetOutcome) {
    let original_bytes = content.len();
    if budget_bytes == 0 || original_bytes <= budget_bytes {
        return (content, BudgetOutcome::unchanged(original_bytes));
    }

    let head_capacity = budget_bytes.saturating_sub(TRAILER_RESERVED).max(1);
    let mut cut = floor_char_boundary(&content, head_capacity);
    if cut == 0 {
        cut = content
            .char_indices()
            .next()
            .map(|(_, c)| c.len_utf8())
            .unwrap_or(0);
    }

    let dropped_bytes = original_bytes.saturating_sub(cut);
    let mut out = String::with_capacity(cut + TRAILER_RESERVED);
    out.push_str(&content[..cut]);
    // Say what was lost, and say what NOT to do about it (#6408).
    //
    // The previous wording ended "re-run with a narrower query to see the
    // rest", which is an instruction to repeat the call. For a listing tool
    // with no narrowing argument in reach — `GITHUB_LIST_PULL_REQUESTS` is the
    // reported case — the model has nothing to narrow, so it re-issues the
    // identical call, gets the identical truncation, and repeats until the
    // successful-repeat tracker halts the run: 8-14 calls per question, the
    // token and quota burn, and an "Incomplete" the user cannot explain.
    //
    // Truncation here is deterministic: the same call returns the same bytes
    // and the same cut. Saying so is what makes the retry stop, and giving the
    // totals lets the model judge whether the head it kept is enough to answer
    // from. Keep the `truncated by tool_result_budget` phrase — it is the
    // grep handle several suites and the runbooks match on.
    out.push_str(&format!(
        "\n\n[… {dropped_bytes} of {original_bytes} bytes truncated by tool_result_budget. \
         Repeating this call returns the same truncation — narrow the request \
         (filter, paginate, or request a smaller range) or answer from the {cut} bytes above …]"
    ));

    let final_bytes = out.len();
    (
        out,
        BudgetOutcome {
            original_bytes,
            final_bytes,
            truncated: true,
        },
    )
}

/// Writes oversized tool results to a [`BlobStore`].
///
/// Each body is stored under `harness/tool_results/<session>/<tool>/<call>.txt` and the
/// model is handed that key as `artifact_path`.
///
/// The host supplies what the crate cannot decide: the redactor every body
/// passes through before it is stored (an artifact is exactly as readable as
/// the result it replaces), the name of its reading tool (quoted in the
/// envelope so the model knows how to open the artifact), and the largest body
/// that tool will open.
#[derive(Clone)]
pub struct ToolResultArtifactStore {
    blobs: Arc<dyn BlobStore>,
    session_key: String,
    redactor: Arc<dyn ArtifactRedactor>,
    read_tool: String,
    max_readable_bytes: u64,
    retention: Option<Duration>,
}

impl std::fmt::Debug for ToolResultArtifactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolResultArtifactStore")
            .field("blobs", &self.blobs)
            .field("session_key", &self.session_key)
            .field("read_tool", &self.read_tool)
            .field("max_readable_bytes", &self.max_readable_bytes)
            .field("retention", &self.retention)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedToolResult {
    pub output: String,
    pub path: String,
    pub original_bytes: usize,
    pub stored_bytes: usize,
    /// Whether the redactor rewrote anything before the body was stored.
    pub redacted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultArtifactOutcome {
    pub original_bytes: usize,
    pub final_bytes: usize,
    pub persisted: bool,
    pub artifact_path: Option<String>,
}

impl ToolResultArtifactOutcome {
    pub fn unchanged(len: usize) -> Self {
        Self {
            original_bytes: len,
            final_bytes: len,
            persisted: false,
            artifact_path: None,
        }
    }
}

impl ToolResultArtifactStore {
    /// A store over `blobs` (a scoped tinystoragedrivers blob handle).
    ///
    /// `read_tool` is the host's reading tool and `max_readable_bytes` the
    /// largest body it will open; a body whose redacted form exceeds it is
    /// never stored, because the model could not read it back.
    pub fn new(
        blobs: Arc<dyn BlobStore>,
        session_key: impl Into<String>,
        redactor: Arc<dyn ArtifactRedactor>,
        read_tool: impl Into<String>,
        max_readable_bytes: u64,
    ) -> Self {
        Self {
            blobs,
            session_key: sanitize_component(&session_key.into()),
            redactor,
            read_tool: read_tool.into(),
            max_readable_bytes,
            retention: None,
        }
    }

    /// Give every artifact a lifetime. Blobs carry no modification time, so
    /// age-based retention is the driver's: once `retention` has passed since
    /// the write, the blob reads as absent and the driver reclaims it.
    ///
    /// A driver without [`Capability::BlobTtl`] stores the artifact without a
    /// lifetime (logged); such a host reclaims with [`Self::purge_session`].
    pub fn with_retention(mut self, retention: Duration) -> Self {
        self.retention = Some(retention);
        self
    }

    /// The blob store artifacts are written to.
    pub fn blobs(&self) -> &Arc<dyn BlobStore> {
        &self.blobs
    }

    /// The artifact a tool call reads, if it reads one of this store's
    /// artifacts: a `tool-results/…` key (or the legacy
    /// `artifacts/tool-results/…` form) passed to the read tool.
    pub fn read_target(
        &self,
        tool_name: &str,
        args: &Value,
        wrapper_tool: &str,
    ) -> Option<ArtifactRead> {
        artifact_read_target(tool_name, args, &self.read_tool, wrapper_tool)
    }

    /// The host's reading tool, as quoted in every envelope.
    pub fn read_tool(&self) -> &str {
        &self.read_tool
    }

    /// Read an artifact back by the key a pointer carried. `Ok(None)` means no
    /// such artifact (never written, expired or purged).
    ///
    /// The key is checked fail-closed: it must sit under `tool-results/` (the
    /// legacy `artifacts/tool-results/` prefix is accepted and mapped), carry
    /// no `..`, empty or control-character component, and so a pointer a model
    /// rewrote by hand cannot address any other blob of the scope.
    pub async fn read(&self, key: &str) -> anyhow::Result<Option<String>> {
        let key = canonical_key(key)?;
        let blob = self.blobs.get(&key).await?;
        Ok(blob.map(|blob| String::from_utf8_lossy(&blob.bytes).into_owned()))
    }

    /// Delete every artifact of this store's session and return how many went.
    pub async fn purge_session(&self) -> anyhow::Result<u64> {
        let dir = format!("{ARTIFACT_ROOT}/{}", self.session_key);
        Ok(self.blobs.delete_tree(&dir).await?)
    }

    /// The pointer the model is given for one artifact: its blob key.
    pub fn path_for_read_tool(&self, tool_name: &str, call_id: Option<&str>) -> String {
        format!(
            "{ARTIFACT_ROOT}/{}",
            self.session_relative_path(tool_name, call_id)
        )
    }

    /// `<session>/<tool>/<call>.txt`, the part of an artifact's key below the
    /// namespace.
    fn session_relative_path(&self, tool_name: &str, call_id: Option<&str>) -> String {
        let call = call_id
            .map(sanitize_component)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(random_call_id);
        format!(
            "{}/{}/{}.txt",
            self.session_key,
            sanitize_component(tool_name),
            call
        )
    }

    /// Store `content` and return an envelope previewing it. When `content`
    /// would be unreadable once sanitized (see [`readable_body`]) and a
    /// `fallback` is given, the fallback is stored instead; when neither fits,
    /// this errors and the caller truncates inline rather than writing an
    /// artifact nobody can read.
    async fn persist(
        &self,
        tool_name: &str,
        call_id: Option<&str>,
        content: &str,
        fallback: Option<&str>,
        preview_budget_bytes: usize,
        reason: &str,
    ) -> anyhow::Result<PersistedToolResult> {
        let (content, sanitized) = readable_body(
            content,
            fallback,
            self.max_readable_bytes,
            self.redactor.as_ref(),
            &self.read_tool,
        )?;
        let read_tool = self.read_tool.as_str();
        let pointer = self.path_for_read_tool(tool_name, call_id);
        let body = sanitized.text.clone().into_bytes();
        let content_type = Some("text/plain; charset=utf-8");
        match self.retention {
            Some(ttl) if self.blobs.capabilities().contains(Capability::BlobTtl) => {
                self.blobs
                    .put_expiring(&pointer, body, content_type, ttl)
                    .await?;
            }
            Some(_) => {
                tracing::debug!(
                    key = %pointer,
                    "[tool-result-artifact] blob driver cannot expire blobs; stored without a lifetime"
                );
                self.blobs.put(&pointer, body, content_type).await?;
            }
            None => {
                self.blobs.put(&pointer, body, content_type).await?;
            }
        }
        tracing::debug!(
            key = %pointer,
            stored_bytes = sanitized.text.len(),
            "[tool-result-artifact] stored tool result on the blob store"
        );
        let relative_path = pointer;
        let location_note = "Full scrubbed output was persisted on the artifact store at the artifact_path above (not part of the project).";

        let (preview, preview_outcome) =
            apply_tool_result_budget(sanitized.text.clone(), preview_budget_bytes);
        let redaction_note = if sanitized.changed {
            " Credential/PII redaction was applied before storage and preview exposure."
        } else {
            ""
        };
        let truncation_note = if preview_outcome.truncated {
            format!(
                " Preview is bounded; {} stored bytes are available via {read_tool}.",
                sanitized.text.len()
            )
        } else {
            String::new()
        };

        let escaped_pointer =
            serde_json::to_string(&relative_path).expect("serializing a string cannot fail");
        let envelope = format!(
            "[tool_result_preview]\n\
             tool: {tool_name}\n\
             reason: {reason}\n\
             original_bytes: {}\n\
             stored_bytes: {}\n\
             artifact_path: {relative_path}\n\
             read_with: {read_tool} {{\"path\":{escaped_pointer}}} (a long read returns one page and names the \"offset\" to continue from)\n\
             notes: {location_note}{redaction_note}{truncation_note}\n\n\
             [preview]\n{preview}",
            content.len(),
            sanitized.text.len(),
        );

        Ok(PersistedToolResult {
            output: envelope,
            path: relative_path,
            original_bytes: content.len(),
            stored_bytes: sanitized.text.len(),
            redacted: sanitized.changed,
        })
    }
}

/// The body to store: `primary` if its sanitized form fits `limit`, else
/// `fallback` if *its* sanitized form does, else an error. Both checks are on
/// the sanitized size, the bytes actually written, because redaction can grow a
/// body (`+15551234567` becomes `[REDACTED_PII_PHONE]`): a raw body under the
/// limit can still produce an artifact the read tool refuses to open.
fn readable_body<'a>(
    primary: &'a str,
    fallback: Option<&'a str>,
    limit: u64,
    redactor: &dyn ArtifactRedactor,
    read_tool: &str,
) -> anyhow::Result<(&'a str, Redacted)> {
    let sanitized = redactor.redact(primary);
    if sanitized.text.len() as u64 <= limit {
        return Ok((primary, sanitized));
    }
    if let Some(fallback) = fallback {
        let sanitized_fallback = redactor.redact(fallback);
        if sanitized_fallback.text.len() as u64 <= limit {
            return Ok((fallback, sanitized_fallback));
        }
    }
    anyhow::bail!(
        "tool result would not be readable once stored: {} sanitized bytes exceed the {limit}-byte {read_tool} limit",
        sanitized.text.len()
    )
}

/// Persist an over-budget result and return its envelope.
///
/// `full_output` is the tool's output before any earlier stage rewrote it
/// (summarizer, TokenJuice). When given, *that* is what gets stored, so the
/// artifact holds what the tool returned rather than a compacted copy of it;
/// `content` still decides whether the budget was exceeded and is what the
/// model would otherwise have seen.
pub async fn apply_per_result_persistence(
    content: String,
    full_output: Option<String>,
    store: Option<&ToolResultArtifactStore>,
    tool_name: &str,
    call_id: Option<&str>,
    budget_bytes: usize,
) -> (String, ToolResultArtifactOutcome) {
    let original_bytes = content.len();
    if budget_bytes == 0 || original_bytes <= budget_bytes {
        return (
            content,
            ToolResultArtifactOutcome::unchanged(original_bytes),
        );
    }

    if let Some(store) = store {
        match store
            .persist(
                tool_name,
                call_id,
                full_output.as_deref().unwrap_or(&content),
                full_output.as_ref().map(|_| content.as_str()),
                budget_bytes,
                "per-result budget exceeded",
            )
            .await
        {
            Ok(persisted) => {
                let envelope_allowance = budget_bytes.max(envelope_budget_floor(&persisted.output));
                let (output, final_bytes) =
                    bound_text_to_budget(persisted.output, envelope_allowance);
                if final_bytes >= original_bytes {
                    // #4469 item 9: this branch does NOT fall back to inline
                    // truncation — the envelope is returned regardless, because it
                    // carries the `artifact_path` pointer to the full stored output
                    // (worth keeping even when the preview text nets no byte saving
                    // vs. the raw result). Log it as an observation only.
                    tracing::debug!(
                        "[agent][tool-result-artifacts] persisted envelope not smaller than raw result tool={} original_bytes={} final_bytes={} budget_bytes={} -- keeping envelope for its artifact_path pointer",
                        tool_name,
                        original_bytes,
                        final_bytes,
                        budget_bytes
                    );
                }
                tracing::info!(
                    "[agent][tool-result-artifacts] persisted oversized tool result tool={} original_bytes={} stored_bytes={} path={} redacted={}",
                    tool_name,
                    persisted.original_bytes,
                    persisted.stored_bytes,
                    persisted.path,
                    persisted.redacted
                );
                return (
                    output,
                    ToolResultArtifactOutcome {
                        // The size of what was stored, which `full_output` can
                        // make larger than `content`; the artifact index and its
                        // contents list read this number.
                        original_bytes: persisted.original_bytes,
                        final_bytes,
                        persisted: true,
                        artifact_path: Some(persisted.path),
                    },
                );
            }
            Err(err) => {
                tracing::warn!(
                    "[agent][tool-result-artifacts] persist failed tool={} original_bytes={} err={} — falling back to inline truncation",
                    tool_name,
                    original_bytes,
                    err
                );
            }
        }
    }

    // Reached two ways, and only one of them was audible: a persist that FAILED
    // warns just above, but a run with no artifact store configured falls
    // through to here silently — the oversized tail is discarded with nothing
    // recording that it happened. Say so, so "where did the rest of my search
    // result go" is answerable from the logs rather than by reading this
    // function.
    if store.is_none() {
        tracing::info!(
            "[agent][tool-result-artifacts] no artifact store configured; truncating oversized tool result inline tool={} original_bytes={} budget_bytes={} — the tail is discarded, not recoverable",
            tool_name,
            original_bytes,
            budget_bytes
        );
    }
    let (output, BudgetOutcome { final_bytes, .. }) =
        apply_tool_result_budget(content, budget_bytes);
    (
        output,
        ToolResultArtifactOutcome {
            original_bytes,
            final_bytes,
            persisted: false,
            artifact_path: None,
        },
    )
}

pub async fn spill_aggregate_tool_results(
    results: &mut [tinytools_agent::dialect::ToolOutcome],
    store: Option<&ToolResultArtifactStore>,
    budget_bytes: usize,
) {
    if budget_bytes == 0 {
        return;
    }
    let Some(store) = store else {
        return;
    };

    let mut total: usize = results.iter().map(|result| result.output.len()).sum();
    if total <= budget_bytes {
        return;
    }

    let mut indexes: Vec<usize> = (0..results.len()).collect();
    indexes.sort_by_key(|idx| std::cmp::Reverse(results[*idx].output.len()));

    for idx in indexes {
        if total <= budget_bytes {
            break;
        }
        let original = results[idx].output.clone();
        let original_len = original.len();
        let allowed_len = budget_bytes.saturating_sub(total.saturating_sub(original_len));
        let persisted_output = if looks_like_preview_envelope(&original) {
            Ok(PersistedToolResult {
                output: original.clone(),
                path: "<existing-preview>".to_string(),
                original_bytes: original_len,
                stored_bytes: original_len,
                redacted: false,
            })
        } else {
            store
                .persist(
                    &results[idx].name,
                    results[idx].tool_call_id.as_deref(),
                    &original,
                    None,
                    allowed_len.min(AGGREGATE_PREVIEW_BUDGET_BYTES),
                    "aggregate tool-result budget exceeded",
                )
                .await
        };
        match persisted_output {
            Ok(persisted) => {
                // #4469 item 6: never bound the preview envelope below the minimum
                // that preserves its `[tool_result_preview]` header + artifact
                // pointer — `allowed_len` can be 0 here, which would blank the
                // result and strip the `artifact_path` the model reads to recover
                // the full output.
                // `apply_tool_result_budget` keeps a head of budget minus its
                // trailer reserve. Include the complete rendered pointer/header
                // in that head, even when the path is an unusually long absolute
                // detached pointer.
                let envelope_allowance = allowed_len.max(envelope_budget_floor(&persisted.output));
                let (output, final_bytes) =
                    bound_text_to_budget(persisted.output, envelope_allowance);
                total = total
                    .saturating_sub(original_len)
                    .saturating_add(final_bytes);
                tracing::info!(
                    "[agent][tool-result-artifacts] aggregate spill tool={} original_bytes={} final_bytes={} total_bytes={} path={}",
                    results[idx].name,
                    original_len,
                    final_bytes,
                    total,
                    persisted.path
                );
                results[idx].output = output;
            }
            Err(err) => {
                tracing::warn!(
                    "[agent][tool-result-artifacts] aggregate spill failed tool={} bytes={} err={} -- falling back to inline budget trim",
                    results[idx].name,
                    original_len,
                    err
                );
                let (output, final_bytes) = bound_text_to_budget(original, allowed_len);
                total = total
                    .saturating_sub(original_len)
                    .saturating_add(final_bytes);
                results[idx].output = output;
            }
        }
    }
}

fn looks_like_preview_envelope(value: &str) -> bool {
    value.starts_with("[tool_result_preview]\n")
}

fn bound_text_to_budget(content: String, budget_bytes: usize) -> (String, usize) {
    if budget_bytes == 0 {
        return (String::new(), 0);
    }
    let (mut output, BudgetOutcome { final_bytes, .. }) =
        apply_tool_result_budget(content, budget_bytes);
    if final_bytes <= budget_bytes {
        return (output, final_bytes);
    }
    let cut = floor_char_boundary(&output, budget_bytes);
    output.truncate(cut);
    let final_bytes = output.len();
    (output, final_bytes)
}

fn envelope_budget_floor(envelope: &str) -> usize {
    let header_len = envelope
        .find("\nnotes:")
        .map(|end| end + 1)
        .unwrap_or(envelope.len());
    MIN_ENVELOPE_ALLOWANCE_BYTES.max(header_len.saturating_add(TRAILER_RESERVED + 1))
}

/// Round a byte index DOWN to the nearest UTF-8 character boundary.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut end = index;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// 32 lowercase hex characters, unique per call within and across processes for
/// naming an artifact whose tool call carried no id.
fn random_call_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let salt = COUNTER.fetch_add(1, Ordering::Relaxed);
    let half = |extra: u64| {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(nanos);
        hasher.write_u64(salt);
        hasher.write_u64(extra);
        hasher.finish()
    };
    format!("{:016x}{:016x}", half(1), half(2))
}

fn sanitize_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().min(80));
    for ch in value.chars().take(80) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

/// Normalise a pointer to the blob key it names, refusing anything that is not
/// a plain `harness/tool_results/<…>` key. A pre-v2 `tool-results/<…>` or
/// `artifacts/tool-results/<…>` pointer maps onto the current root.
fn canonical_key(key: &str) -> anyhow::Result<String> {
    let trimmed = key.trim_start_matches("./");
    let canonical = LEGACY_ARTIFACT_ROOTS
        .iter()
        .find_map(|root| {
            trimmed
                .strip_prefix(root)
                .filter(|rest| rest.starts_with('/'))
        })
        .map_or_else(
            || trimmed.to_owned(),
            |rest| format!("{ARTIFACT_ROOT}{rest}"),
        );
    let under_namespace = canonical
        .strip_prefix(ARTIFACT_ROOT)
        .is_some_and(|rest| rest.starts_with('/'));
    let clean = canonical
        .split('/')
        .all(|part| !part.is_empty() && part != "." && part != "..")
        && !canonical.chars().any(char::is_control)
        && !canonical.contains('\\');
    if !under_namespace || !clean {
        anyhow::bail!("not a tool-result artifact key: {key}");
    }
    Ok(canonical)
}

#[cfg(test)]
#[path = "tool_results_tests.rs"]
mod test;
