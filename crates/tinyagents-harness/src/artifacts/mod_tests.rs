//! Tests for the artifact-offload convention.
//!
//! Covers the happy path (oversized result lands in `outputs/`, the parent gets
//! a path + abstract), the fallback path (offload refused, inline payload
//! survives for the host's backstop), and the fail-closed path hardening that
//! keeps offload inside its two namespaces and out of host-internal state.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::*;
use tinyagents_storage_kit::Scope;
use tinystoragedrivers_testkit::MemoryBackend;
use tinystoragedrivers_testkit::memory_storage;

// ── Test policies ─────────────────────────────────────────────────────────────

/// Stands in for a host security policy: an internal state root plus a rule
/// naming specific internal state locations.
///
/// Mirrors the real shape — a host has both a root it owns wholesale and
/// individual state paths that may sit elsewhere — because the two are checked
/// separately and produce different errors.
#[derive(Debug)]
struct TestPolicy {
    internal_root: PathBuf,
    /// Path suffixes treated as internal state wherever they appear.
    internal_suffixes: Vec<String>,
}

impl TestPolicy {
    fn rooted_at(internal_root: PathBuf) -> Self {
        Self {
            internal_root,
            internal_suffixes: Vec::new(),
        }
    }

    fn with_internal_suffix(mut self, suffix: &str) -> Self {
        self.internal_suffixes.push(suffix.to_string());
        self
    }
}

impl ArtifactPathPolicy for TestPolicy {
    fn is_internal_state(&self, path: &Path) -> bool {
        let shown = path.to_string_lossy().replace('\\', "/");
        self.internal_suffixes
            .iter()
            .any(|suffix| shown.contains(suffix))
    }

    fn internal_root(&self) -> Option<&Path> {
        Some(&self.internal_root)
    }
}

/// Stands in for a host credential scrubber.
#[derive(Debug)]
struct TestRedactor;

const SECRET: &str = "sk-live-0123456789abcdef";
const REDACTION: &str = "[redacted]";

impl ArtifactRedactor for TestRedactor {
    fn redact(&self, content: &str) -> Redacted {
        if content.contains(SECRET) {
            Redacted::rewritten(content.replace(SECRET, REDACTION))
        } else {
            Redacted::unchanged(content)
        }
    }
}

const READ_TOOL: &str = "file_read";

fn offload() -> ArtifactOffload {
    ArtifactOffload::new(test_blobs(), "worker-agent", "task-1")
        .with_path_policy(Arc::new(TestPolicy::rooted_at(PathBuf::from("internal"))))
        .with_redactor(Arc::new(TestRedactor))
}

fn test_blobs() -> Arc<dyn tinyagents_storage_kit::BlobStore> {
    let scope = Scope::new("local").expect("valid scope");
    memory_storage(&scope).blobs().clone()
}

/// Everything stored, as `(key, body)` pairs.
async fn stored_bodies(offload: &ArtifactOffload) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for meta in offload.blobs().list("").await.expect("list") {
        let blob = offload
            .blobs()
            .get(&meta.key)
            .await
            .expect("get")
            .expect("listed blob exists");
        out.push((meta.key, String::from_utf8(blob.bytes).expect("utf8")));
    }
    out
}

#[tokio::test]
async fn filesystem_policy_denial_precedes_blob_or_file_writes() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("action");
    let workspace = temp.path().join("private");
    std::fs::create_dir_all(root.join(OUTPUTS_DIR)).expect("create output root");
    std::fs::create_dir_all(&workspace).expect("create workspace");
    let target = root
        .join(OUTPUTS_DIR)
        .join("worker")
        .join("task-1-result.md");
    std::fs::create_dir_all(target.parent().expect("target parent")).expect("create target parent");
    std::fs::write(&target, "keep the existing content").expect("seed previous output");
    let offload =
        ArtifactOffload::new(test_blobs(), "worker", "task-1").with_path_policy(Arc::new(
            TestPolicy::rooted_at(workspace)
                .with_internal_suffix(&format!("/action/{OUTPUTS_DIR}/")),
        ));

    let output = "sensitive result".repeat(100);
    let (text, artifact) =
        offload_oversized_result_to_fs(output.clone(), &offload, 1, READ_TOOL, &root).await;

    assert_eq!(text, output);
    assert!(artifact.is_none());
    assert!(
        stored_bodies(&offload).await.is_empty(),
        "denied targets must not create blob records"
    );
    assert_eq!(
        std::fs::read_to_string(target).expect("read existing target"),
        "keep the existing content"
    );
}

#[tokio::test]
async fn filesystem_offload_materializes_only_the_redacted_stored_body() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("action");
    std::fs::create_dir_all(&root).expect("create root");
    let offload = ArtifactOffload::new(test_blobs(), "worker", "materialized")
        .with_redactor(Arc::new(TestRedactor));
    let output = format!("before {SECRET} after");

    let (pointer, artifact) =
        offload_oversized_result_to_fs(output, &offload, 1, READ_TOOL, &root).await;

    let artifact = artifact.expect("offloaded artifact");
    assert!(pointer.contains(&artifact.relative_path));
    let materialized = std::fs::read_to_string(root.join(&artifact.relative_path))
        .expect("read materialized file");
    assert_eq!(materialized, format!("before {REDACTION} after"));
    assert!(!materialized.contains(SECRET));
}

#[cfg(unix)]
#[tokio::test]
async fn filesystem_offload_refuses_symlinked_output_directory_before_storage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("action");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&root).expect("create root");
    std::fs::create_dir_all(&outside).expect("create outside");
    std::fs::create_dir_all(root.join(OUTPUTS_DIR).parent().expect("outputs parent"))
        .expect("create namespaced parent");
    std::os::unix::fs::symlink(&outside, root.join(OUTPUTS_DIR)).expect("symlink outputs");
    let offload = ArtifactOffload::new(test_blobs(), "worker", "symlink");

    let output = "must remain inline".repeat(100);
    let (text, artifact) =
        offload_oversized_result_to_fs(output.clone(), &offload, 1, READ_TOOL, &root).await;

    assert_eq!(text, output);
    assert!(artifact.is_none());
    assert!(stored_bodies(&offload).await.is_empty());
    assert_eq!(
        std::fs::read_dir(&outside).expect("read outside").count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn atomic_materialization_does_not_create_descendants_through_a_symlink() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("action");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&root).expect("create root");
    std::fs::create_dir_all(&outside).expect("create outside");
    std::fs::create_dir_all(root.join(OUTPUTS_DIR).parent().expect("outputs parent"))
        .expect("create namespaced parent");
    std::os::unix::fs::symlink(&outside, root.join(OUTPUTS_DIR)).expect("symlink outputs");

    let result = super::ops::atomic_write_under_dir(
        &root,
        &Path::new(OUTPUTS_DIR).join("worker"),
        std::ffi::OsStr::new("result.md"),
        b"body",
    );

    assert!(result.is_err(), "symlinked parent must fail closed");
    assert!(
        !outside.join("worker").exists(),
        "no directory is created outside the root"
    );
}

#[test]
fn capability_materialization_writes_atomically_beneath_the_open_root() {
    let temp = tempfile::tempdir().expect("tempdir");
    let root = temp.path().join("action");
    std::fs::create_dir_all(&root).expect("create root");

    super::ops::atomic_write_with_capability(
        &root,
        Path::new("outputs/worker"),
        std::ffi::OsStr::new("result.md"),
        b"body",
    )
    .expect("capability-relative atomic write");

    assert_eq!(
        std::fs::read(root.join("outputs/worker/result.md")).expect("read materialized body"),
        b"body"
    );
}

// ── Kinds and directories ─────────────────────────────────────────────────────

#[test]
fn kinds_map_to_the_documented_directories() {
    assert_eq!(ArtifactKind::Output.subdir(), OUTPUTS_DIR);
    assert_eq!(ArtifactKind::Scratch.subdir(), SCRATCH_DIR);
    assert_eq!(ArtifactKind::Output.as_str(), "output");
    assert_eq!(ArtifactKind::Scratch.as_str(), "scratch");
}

// ── Path resolution ───────────────────────────────────────────────────────────

#[test]
fn resolves_under_the_convention_namespace() {
    let key = resolve_artifact_key(None, ArtifactKind::Output, "report.md").expect("resolve");
    assert_eq!(key, "harness/outputs/report.md");
}

#[test]
fn scratch_resolves_under_its_own_namespace_not_the_internal_root() {
    // The `workspace` namespace of the artifact store and the host's internal
    // root are different places that happen to share a word. Conflating them
    // would route every scratch write into host state.
    let policy = TestPolicy::rooted_at(PathBuf::from("internal/workspace"));
    let key =
        resolve_artifact_key(Some(&policy), ArtifactKind::Scratch, "notes.md").expect("resolve");
    assert_eq!(key, "harness/workspace/notes.md");
}

#[test]
fn rejects_parent_traversal() {
    for name in ["../../etc/passwd", "a/../../b", "a\\..\\b"] {
        let err = resolve_artifact_key(None, ArtifactKind::Output, name)
            .expect_err("traversal must be refused");
        assert!(
            matches!(err, OffloadError::PathEscape { .. }),
            "{name}: got {err:?}"
        );
    }
}

#[test]
fn rejects_absolute_names() {
    for name in ["/etc/passwd", "\\share\\x", "C:\\Windows\\x", "c:/x"] {
        let err = resolve_artifact_key(None, ArtifactKind::Output, name)
            .expect_err("absolute name must be refused");
        assert!(
            matches!(err, OffloadError::AbsolutePath { .. }),
            "{name}: got {err:?}"
        );
    }
}

#[test]
fn rejects_control_characters() {
    let err =
        resolve_artifact_key(None, ArtifactKind::Output, "a\0b").expect_err("NUL must be refused");
    assert!(
        matches!(err, OffloadError::PathEscape { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_empty_and_whitespace_names() {
    for name in ["", "   ", "\t\n", "./", ".//."] {
        let err = resolve_artifact_key(None, ArtifactKind::Output, name)
            .expect_err("blank name must be refused");
        assert!(matches!(err, OffloadError::EmptyName), "{name:?}: {err:?}");
    }
}

#[test]
fn accepts_leading_current_dir_segments_and_collapses_empty_ones() {
    // `./report.md` is how a model commonly writes a relative path; refusing it
    // would reject a correct request on a formatting technicality.
    let key = resolve_artifact_key(None, ArtifactKind::Output, "./a//report.md").expect("resolve");
    assert_eq!(key, "harness/outputs/a/report.md");
}

#[test]
fn rejects_keys_inside_the_internal_root_fail_closed() {
    let policy = TestPolicy::rooted_at(PathBuf::from("harness/outputs/internal"));
    let err = resolve_artifact_key(Some(&policy), ArtifactKind::Output, "internal/leak.md")
        .expect_err("internal root must be refused");
    assert!(
        matches!(err, OffloadError::InternalRoot { .. }),
        "got {err:?}"
    );
}

#[test]
fn rejects_host_internal_state_keys_with_the_specific_error() {
    // The specific check runs first so its more useful message survives when a
    // key trips both rules.
    let policy = TestPolicy::rooted_at(PathBuf::from("outputs")).with_internal_suffix("outputs");
    let err = resolve_artifact_key(Some(&policy), ArtifactKind::Output, "leak.md")
        .expect_err("internal state must be refused");
    assert!(
        matches!(err, OffloadError::InternalState { .. }),
        "the specific rule must win over blanket containment, got {err:?}"
    );
}

#[test]
fn sanitize_component_strips_separators_and_never_returns_empty() {
    assert_eq!(sanitize_component("sub-1a/2b"), "sub-1a_2b");
    assert_eq!(sanitize_component("../etc"), "___etc");
    assert_eq!(sanitize_component("ok_name-1"), "ok_name-1");
    // A component that sanitizes to nothing must still be a usable directory
    // name, or the resulting path is malformed rather than merely odd.
    assert_eq!(sanitize_component(""), "unknown");
    assert_eq!(sanitize_component("///"), "___");
    assert!(sanitize_component(&"x".repeat(500)).len() <= 80);
}

// ── Thresholds ────────────────────────────────────────────────────────────────

#[test]
fn should_offload_respects_threshold_and_the_zero_disable() {
    assert!(should_offload(100, 50));
    assert!(!should_offload(50, 50), "the threshold is exclusive");
    assert!(!should_offload(10, 50));
    // Zero is the documented opt-out, not "offload everything".
    assert!(!should_offload(usize::MAX, 0));
}

#[test]
fn offload_threshold_tightens_to_an_agents_own_result_cap() {
    // A cap below the default would truncate the result before offload fired,
    // so the artifact would never reach disk at all.
    assert_eq!(effective_offload_threshold(8_192, Some(4_000)), 4_000);
    assert_eq!(effective_offload_threshold(8_192, Some(16_000)), 8_192);
    assert_eq!(effective_offload_threshold(8_192, None), 8_192);
    // A zero cap means "no cap", not "offload nothing".
    assert_eq!(effective_offload_threshold(8_192, Some(0)), 8_192);
}

// ── Abstracts ─────────────────────────────────────────────────────────────────

#[test]
fn build_abstract_returns_short_content_unchanged() {
    assert_eq!(build_abstract("  short  ", 100), "short");
}

#[test]
fn build_abstract_cuts_at_a_line_boundary_when_one_is_available() {
    let content = format!("{}\n{}", "a".repeat(60), "b".repeat(60));
    let out = build_abstract(&content, 100);
    assert!(out.ends_with("..."));
    assert!(!out.contains('b'), "should have cut at the newline: {out}");
}

#[test]
fn build_abstract_cuts_at_a_word_boundary_when_there_is_no_line_break() {
    let content = format!("{} {}", "a".repeat(60), "b".repeat(60));
    let out = build_abstract(&content, 100);
    assert!(out.ends_with("..."));
    assert!(!out.contains('b'), "should have cut at the space: {out}");
}

#[test]
fn build_abstract_handles_a_zero_budget_and_boundary_free_text() {
    assert_eq!(build_abstract("anything", 0), "");
    // No line or word break in the back half — a hard cut is the only option.
    let out = build_abstract(&"a".repeat(200), 100);
    assert!(out.ends_with("..."));
}

#[test]
fn build_abstract_never_splits_a_multibyte_character() {
    // Budget counted in chars, truncation done on a String: a byte-indexed cut
    // here would panic rather than merely misformat.
    let content = "é".repeat(200);
    let out = build_abstract(&content, 50);
    assert!(out.ends_with("..."));
    assert!(out.is_char_boundary(out.len()));
}

// ── Pointers ──────────────────────────────────────────────────────────────────

fn sample_artifact(redacted: bool) -> OffloadedArtifact {
    OffloadedArtifact {
        kind: ArtifactKind::Output,
        relative_path: "outputs/agent/task-result.md".to_string(),
        stored_bytes: 1234,
        original_bytes: 1300,
        redacted,
    }
}

#[test]
fn pointer_carries_path_size_and_a_read_call() {
    let rendered = render_artifact_pointer(&sample_artifact(false), "the abstract", READ_TOOL);
    assert!(rendered.starts_with(ARTIFACT_POINTER_PREFIX));
    assert!(rendered.contains("path=outputs/agent/task-result.md"));
    assert!(rendered.contains("bytes=1234"));
    assert!(rendered.contains("kind=output"));
    assert!(rendered.contains(READ_TOOL));
    assert!(rendered.contains("the abstract"));
}

#[test]
fn pointer_names_the_read_tool_it_was_given() {
    // The tool name is host vocabulary. Hard-coding one here would put a tool
    // the host may not have into its prompts.
    let rendered = render_artifact_pointer(&sample_artifact(false), "x", "read_file");
    assert!(rendered.contains("read_with: read_file"));
    assert!(!rendered.contains("file_read"));
}

#[test]
fn pointer_discloses_redaction_when_it_happened() {
    let clean = render_artifact_pointer(&sample_artifact(false), "x", READ_TOOL);
    assert!(!clean.contains("redaction"));
    let redacted = render_artifact_pointer(&sample_artifact(true), "x", READ_TOOL);
    assert!(redacted.contains("redaction was applied"));
}

#[test]
fn extract_artifact_paths_reads_pointers_out_of_a_handoff() {
    let handoff =
        format!("{ARTIFACT_POINTER_PREFIX} kind=output path=outputs/a.md bytes=10\nprose");
    assert_eq!(extract_artifact_paths(&handoff), vec!["outputs/a.md"]);
}

#[test]
fn extract_artifact_paths_dedupes_and_keeps_encounter_order() {
    let handoff = format!(
        "{ARTIFACT_POINTER_PREFIX} path=b.md bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=a.md bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=b.md bytes=1"
    );
    assert_eq!(extract_artifact_paths(&handoff), vec!["b.md", "a.md"]);
}

#[test]
fn extract_artifact_paths_ignores_non_pointer_and_malformed_lines() {
    let handoff = format!(
        "ordinary prose\n\
         {ARTIFACT_POINTER_PREFIX} no path field here\n\
         {ARTIFACT_POINTER_PREFIX} path= bytes=1\n\
         {ARTIFACT_POINTER_PREFIX} path=good.md bytes=1"
    );
    // The empty `path=` case is the subtle one: splitting on whitespace rather
    // than the FIRST whitespace would yield `bytes=1` as the path.
    assert_eq!(extract_artifact_paths(&handoff), vec!["good.md"]);
}

#[test]
fn note_artifact_handoff_reports_how_many_paths_crossed() {
    let paths = vec!["a.md".to_string(), "b.md".to_string()];
    assert_eq!(
        note_artifact_handoff(HANDOFF_STAGE_RECORDED, "agent", "task", &paths),
        2
    );
    assert_eq!(
        note_artifact_handoff(HANDOFF_STAGE_CONSUMED, "agent", "task", &[]),
        0
    );
}

#[test]
fn handoff_stages_are_distinct() {
    // They exist to tell the two ends of one pointer apart in a journal; equal
    // values would render the same line twice.
    assert_ne!(HANDOFF_STAGE_RECORDED, HANDOFF_STAGE_CONSUMED);
}

// ── Writing ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn write_stores_under_outputs_and_reports_the_key() {
    let offload = offload();

    let artifact = offload
        .write(ArtifactKind::Output, "report.md", "hello")
        .await
        .expect("write");

    assert_eq!(artifact.relative_path, "harness/outputs/report.md");
    assert_eq!(artifact.stored_bytes, 5);
    assert_eq!(artifact.original_bytes, 5);
    assert!(!artifact.redacted);
    assert_eq!(
        offload.read(&artifact.relative_path).await.expect("read"),
        Some("hello".to_string())
    );
}

#[tokio::test]
async fn write_redacts_credentials_before_they_are_stored() {
    let offload = offload();
    let body = format!("token: {SECRET}\n");

    let artifact = offload
        .write(ArtifactKind::Output, "creds.md", &body)
        .await
        .expect("write");

    assert!(artifact.redacted, "the redactor rewrote the body");
    let stored = stored_bodies(&offload).await;
    assert_eq!(stored.len(), 1);
    assert!(
        !stored[0].1.contains(SECRET),
        "the secret reached the store: {}",
        stored[0].1
    );
    assert!(stored[0].1.contains(REDACTION));
    // Original bytes describe the caller's payload, stored bytes the artifact.
    assert_eq!(artifact.original_bytes, body.len());
    assert_eq!(artifact.stored_bytes, stored[0].1.len());
}

#[tokio::test]
async fn write_without_a_redactor_stores_bytes_verbatim() {
    // The documented consequence of omitting a redactor. Pinned so the default
    // cannot quietly become "scrub something" and give false assurance.
    let offload = ArtifactOffload::new(test_blobs(), "agent", "task");
    let artifact = offload
        .write(ArtifactKind::Output, "raw.md", SECRET)
        .await
        .expect("write");
    assert!(!artifact.redacted);
    assert_eq!(
        offload.read(&artifact.relative_path).await.expect("read"),
        Some(SECRET.to_string())
    );
}

#[tokio::test]
async fn write_refuses_a_traversal_target_without_storing_anything() {
    let offload = offload();

    let err = offload
        .write(ArtifactKind::Output, "../escape.md", "x")
        .await
        .expect_err("traversal must be refused");

    assert!(
        matches!(err, OffloadError::PathEscape { .. }),
        "got {err:?}"
    );
    assert!(
        stored_bodies(&offload).await.is_empty(),
        "nothing may be stored on a refused key"
    );
}

#[tokio::test]
async fn default_result_name_sanitizes_both_identifiers() {
    let offload = ArtifactOffload::new(test_blobs(), "agent/../x", "task id/1");
    // Each of `/`, `.`, `.`, `/` becomes its own underscore — the separators are
    // replaced, never collapsed, so no identifier can smuggle in a key level.
    assert_eq!(
        offload.default_result_name(),
        "agent____x/task_id_1-result.md"
    );
}

#[tokio::test]
async fn scratch_and_outputs_are_separate_namespaces() {
    let offload = offload();
    offload
        .write(ArtifactKind::Output, "a.md", "out")
        .await
        .expect("write output");
    offload
        .write(ArtifactKind::Scratch, "a.md", "scratch")
        .await
        .expect("write scratch");

    assert_eq!(
        offload.read("harness/outputs/a.md").await.expect("read"),
        Some("out".to_string())
    );
    assert_eq!(
        offload.read("harness/workspace/a.md").await.expect("read"),
        Some("scratch".to_string())
    );
}

/// A path a transcript quoted before the keys moved under `harness/` still reads
/// the (migrated) artifact.
#[tokio::test]
async fn read_accepts_the_pre_v2_directory_names() {
    let offload = offload();
    offload
        .write(ArtifactKind::Output, "a.md", "out")
        .await
        .expect("write output");
    offload
        .write(ArtifactKind::Scratch, "b.md", "scratch")
        .await
        .expect("write scratch");
    assert_eq!(
        offload.read("outputs/a.md").await.unwrap().as_deref(),
        Some("out")
    );
    assert_eq!(
        offload.read("workspace/b.md").await.unwrap().as_deref(),
        Some("scratch")
    );
}

#[tokio::test]
async fn read_returns_none_for_a_missing_artifact() {
    assert_eq!(
        offload()
            .read("harness/outputs/nope.md")
            .await
            .expect("read"),
        None
    );
}

#[tokio::test]
async fn read_refuses_keys_outside_the_two_namespaces() {
    let offload = offload();
    for key in [
        "internal/state",
        "etc/passwd",
        "outputs",
        "harness/outputs/../workspace/x",
    ] {
        let err = offload.read(key).await.expect_err("must be refused");
        assert!(
            matches!(err, OffloadError::PathEscape { .. }),
            "{key}: {err:?}"
        );
    }
}

#[tokio::test]
async fn artifacts_land_on_a_host_supplied_blob_store() {
    use tinyagents_storage_kit::{Scope, StorageBackend};

    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("host").expect("scope"))
        .expect("scoped");
    let blobs = scoped.blobs().clone();
    let offload = ArtifactOffload::new(blobs.clone(), "agent", "task");

    offload
        .write(ArtifactKind::Output, "report.md", "body")
        .await
        .expect("write");

    let blob = blobs
        .get("harness/outputs/report.md")
        .await
        .expect("get")
        .expect("present on the host store");
    assert_eq!(blob.bytes, b"body");
    assert_eq!(
        blob.meta.content_type.as_deref(),
        Some("text/markdown; charset=utf-8")
    );
}

#[tokio::test]
async fn a_failing_store_surfaces_a_storage_error() {
    let offload = offload();
    // A key longer than any driver's limit makes the port refuse the write.
    let name = "x".repeat(4096);
    let err = offload
        .write(ArtifactKind::Output, &name, "x")
        .await
        .expect_err("the store must refuse an oversized key");
    assert!(matches!(err, OffloadError::Storage(_)), "got {err:?}");
}

// ── The offload entry point ───────────────────────────────────────────────────

#[tokio::test]
async fn oversized_result_is_offloaded_and_the_parent_gets_a_key_plus_abstract() {
    let offload = offload();
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 8_192, READ_TOOL).await;
    let artifact = artifact.expect("an artifact was written");

    assert!(text.starts_with(ARTIFACT_POINTER_PREFIX));
    assert!(
        text.len() < big.len(),
        "the pointer must be smaller than the payload it replaced"
    );
    assert_eq!(
        extract_artifact_paths(&text),
        vec![artifact.relative_path.clone()]
    );
    assert_eq!(
        offload.read(&artifact.relative_path).await.expect("read"),
        Some(big),
        "full fidelity is preserved in the store"
    );
}

#[tokio::test]
async fn small_result_stays_inline() {
    let offload = offload();

    let (text, artifact) =
        offload_oversized_result("small".to_string(), &offload, 8_192, READ_TOOL).await;

    assert_eq!(text, "small");
    assert!(artifact.is_none());
}

#[tokio::test]
async fn offload_is_disabled_by_a_zero_threshold() {
    let offload = offload();
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 0, READ_TOOL).await;

    assert_eq!(text, big);
    assert!(artifact.is_none());
}

#[tokio::test]
async fn offload_failure_keeps_the_inline_payload_for_the_host_backstop() {
    // The load-bearing soft-failure contract: a refused offload must never cost
    // the caller its content, or a storage problem turns into data loss.
    // Every key under `outputs` is host-internal state here, so the write is
    // refused by policy.
    let offload = ArtifactOffload::new(test_blobs(), "agent", "task").with_path_policy(Arc::new(
        TestPolicy::rooted_at(PathBuf::from("internal")).with_internal_suffix("outputs"),
    ));
    let big = "x".repeat(10_000);

    let (text, artifact) = offload_oversized_result(big.clone(), &offload, 8_192, READ_TOOL).await;

    assert_eq!(
        text, big,
        "the inline payload must survive a failed offload"
    );
    assert!(artifact.is_none());
}

#[tokio::test]
async fn abstract_is_built_from_the_redacted_body_not_the_raw_output() {
    // The pointer goes straight into the parent's context. Building the
    // abstract from the raw text would re-expose the credential that was just
    // scrubbed out of the artifact — and the artifact would still look correct.
    let offload = offload();
    let body = format!("{SECRET} {}", "x".repeat(10_000));

    let (text, artifact) = offload_oversized_result(body, &offload, 8_192, READ_TOOL).await;
    let artifact = artifact.expect("an artifact was written");

    assert!(artifact.redacted);
    assert!(
        !text.contains(SECRET),
        "the secret leaked into the pointer: {text}"
    );
    assert!(text.contains(REDACTION));
}
