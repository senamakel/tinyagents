use super::*;
use crate::artifacts::policy::{ArtifactRedactor, Redacted};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tinyagents_storage_kit::Scope;
use tinyagents_storage_kit::StorageBackend;
use tinystoragedrivers_testkit::{MemoryBackend, memory_storage};

/// Stand-in for a host's credential/PII pass: rewrites a GitHub token and
/// phone numbers, and (like the real one) can grow a body.
#[derive(Debug)]
struct TestRedactor;

impl ArtifactRedactor for TestRedactor {
    fn redact(&self, content: &str) -> Redacted {
        let mut out = content.replace(&test_github_token(), "[REDACTED_SECRET]");
        out = out.replace("+1555", "[REDACTED_PII_PHONE]");
        if out == content {
            Redacted::unchanged(out)
        } else {
            Redacted::rewritten(out)
        }
    }
}

fn test_github_token() -> String {
    "TEST_GITHUB_TOKEN_DO_NOT_USE".to_owned()
}

fn store(session: &str) -> ToolResultArtifactStore {
    let scope = Scope::new("local").expect("valid scope");
    ToolResultArtifactStore::new(
        memory_storage(&scope).blobs().clone(),
        session,
        Arc::new(TestRedactor),
        "file_read",
        10 * 1024 * 1024,
    )
}

fn read_target(tool: &str, args: &serde_json::Value) -> Option<ArtifactRead> {
    artifact_read_target(tool, args, "file_read", "use_skill")
}

#[tokio::test]
async fn threshold_persists_redacted_preview_and_file() {
    let store = store("session/one");
    let raw = format!("{} {}", "x".repeat(4096), test_github_token());

    let (out, outcome) =
        apply_per_result_persistence(raw, None, Some(&store), "shell", Some("call-1"), 1024).await;

    assert!(outcome.persisted);
    assert!(out.contains("artifact_path: harness/tool_results/session_one/shell/call-1.txt"));
    assert!(out.contains(
        "read_with: file_read {\"path\":\"harness/tool_results/session_one/shell/call-1.txt\"}"
    ));
    assert!(out.contains("original_bytes:"));
    assert!(out.contains("[preview]"));
    assert!(out.contains("Credential/PII redaction was applied"));
    assert!(!out.contains(&test_github_token()));

    let stored = store
        .read("harness/tool_results/session_one/shell/call-1.txt")
        .await
        .unwrap()
        .expect("artifact stored");
    assert!(stored.contains("xxxx"));
    assert!(!stored.contains(&test_github_token()));
}

/// A call with no id still gets a unique, well-formed file name.
#[tokio::test]
async fn a_call_without_an_id_gets_a_random_hex_name() {
    let store = store("s");
    let a = store.path_for_read_tool("shell", None);
    let b = store.path_for_read_tool("shell", None);
    assert_ne!(a, b);
    let stem = a
        .strip_prefix("harness/tool_results/s/shell/")
        .and_then(|rest| rest.strip_suffix(".txt"))
        .expect("path shape");
    assert_eq!(stem.len(), 32);
    assert!(stem.chars().all(|c| c.is_ascii_hexdigit()));
}

#[tokio::test]
async fn fallback_truncates_when_store_missing() {
    let raw = "z".repeat(4096);
    let (out, outcome) =
        apply_per_result_persistence(raw, None, None, "shell", Some("call"), 512).await;
    assert!(!outcome.persisted);
    assert!(out.contains("truncated by tool_result_budget"));
    assert!(out.len() < 4096);
}

/// The truncation trailer must not tell the model to re-run the call (#6408).
///
/// The old wording ended "re-run with a narrower query to see the rest". A
/// listing tool with no narrowing argument in reach leaves the model only the
/// identical call, which returns the identical truncation, until the
/// successful-repeat tracker halts the run. Assert the retry instruction is
/// gone and the deterministic-truncation statement that replaces it is
/// present, so a revert to the old trailer fails here.
#[tokio::test]
async fn truncation_trailer_does_not_instruct_a_retry() {
    let raw = "z".repeat(4096);
    let (out, outcome) =
        apply_per_result_persistence(raw, None, None, "GITHUB_LIST_PULL_REQUESTS", None, 512).await;

    assert!(
        !outcome.persisted,
        "fixture must truncate inline, not persist"
    );
    assert!(
        !out.contains("re-run"),
        "trailer must not instruct a re-run; got: {out}"
    );
    assert!(
        out.contains("Repeating this call returns the same truncation"),
        "trailer must say the truncation is deterministic; got: {out}"
    );
    // Both totals, so the model can judge whether the retained head suffices.
    assert!(
        out.contains("of 4096 bytes truncated by tool_result_budget"),
        "trailer must report dropped-of-original bytes; got: {out}"
    );
}

#[tokio::test]
async fn persisted_preview_is_bounded_for_small_budget() {
    let store = store("session");
    let raw = "x".repeat(800);

    let (out, outcome) =
        apply_per_result_persistence(raw, None, Some(&store), "shell", Some("call"), 320).await;

    assert!(outcome.persisted);
    assert!(outcome.final_bytes <= envelope_budget_floor(&out));
    assert_eq!(out.len(), outcome.final_bytes);
    assert!(out.contains("[tool_result_preview]"));
    assert!(
        store
            .read("harness/tool_results/session/shell/call.txt")
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn tiny_per_result_budget_keeps_the_recovery_pointer() {
    let store = store("session");
    let (out, outcome) = apply_per_result_persistence(
        "x".repeat(800),
        None,
        Some(&store),
        "shell",
        Some("call"),
        8,
    )
    .await;
    assert!(outcome.persisted);
    assert!(out.contains("artifact_path: harness/tool_results/session/shell/call.txt"));
    assert!(out.contains("read_with: file_read"));
}

#[tokio::test]
async fn long_pointer_survives_a_small_result_budget() {
    let store = store(&"r".repeat(220));
    let (out, outcome) = apply_per_result_persistence(
        "x".repeat(2_000),
        None,
        Some(&store),
        "shell",
        Some("call"),
        20,
    )
    .await;
    assert!(outcome.persisted);
    let pointer = outcome.artifact_path.unwrap();
    assert!(out.contains(&format!("artifact_path: {pointer}\n")));
    let read_line = out
        .lines()
        .find(|line| line.starts_with("read_with:"))
        .unwrap();
    let call = read_line
        .split_once("file_read ")
        .unwrap()
        .1
        .split(" (a long")
        .next()
        .unwrap();
    let parsed: Value = serde_json::from_str(call).unwrap();
    assert_eq!(parsed["path"], pointer);
}

#[tokio::test]
async fn aggregate_spills_largest_until_under_budget() {
    let store = store("session");
    let mut results = vec![
        tinytools_agent::dialect::ToolOutcome {
            name: "small".into(),
            output: "a".repeat(100),
            success: true,
            tool_call_id: Some("small".into()),
            trusted_verbatim: false,
        },
        tinytools_agent::dialect::ToolOutcome {
            name: "largest".into(),
            output: "b".repeat(2000),
            success: true,
            tool_call_id: Some("largest".into()),
            trusted_verbatim: false,
        },
        tinytools_agent::dialect::ToolOutcome {
            name: "medium".into(),
            output: "c".repeat(900),
            success: true,
            tool_call_id: Some("medium".into()),
            trusted_verbatim: false,
        },
    ];

    spill_aggregate_tool_results(&mut results, Some(&store), 1800).await;

    assert!(results[1].output.starts_with("[tool_result_preview]\n"));
    let total: usize = results.iter().map(|result| result.output.len()).sum();
    assert!(total <= 1800, "total={total}");
    assert!(!results[0].output.starts_with("[tool_result_preview]\n"));
    assert!(
        store
            .read("harness/tool_results/session/largest/largest.txt")
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn aggregate_forces_budget_when_envelope_has_no_savings() {
    let store = store("session");
    let mut results = vec![
        tinytools_agent::dialect::ToolOutcome {
            name: "one".into(),
            output: "a".repeat(350),
            success: true,
            tool_call_id: Some("one".into()),
            trusted_verbatim: false,
        },
        tinytools_agent::dialect::ToolOutcome {
            name: "two".into(),
            output: "b".repeat(350),
            success: true,
            tool_call_id: Some("two".into()),
            trusted_verbatim: false,
        },
        tinytools_agent::dialect::ToolOutcome {
            name: "three".into(),
            output: "c".repeat(350),
            success: true,
            tool_call_id: Some("three".into()),
            trusted_verbatim: false,
        },
    ];

    spill_aggregate_tool_results(&mut results, Some(&store), 500).await;

    let total: usize = results.iter().map(|result| result.output.len()).sum();
    // #4469 item 6: the aggregate spill now floors each persisted envelope at
    // Each envelope is floored at the size needed to retain its complete
    // rendered header and the truncation trailer reserve, so the total may
    // slightly overshoot the raw aggregate budget.
    assert!(
        results.iter().all(|result| !result.output.is_empty()),
        "no persisted envelope may be blanked — the artifact pointer must survive"
    );
    let floor_total: usize = results
        .iter()
        .filter(|result| looks_like_preview_envelope(&result.output))
        .map(|result| envelope_budget_floor(&result.output))
        .sum();
    assert!(
        total <= floor_total,
        "total={total} exceeds floor={floor_total}"
    );
    assert!(
        store
            .read("harness/tool_results/session/one/one.txt")
            .await
            .unwrap()
            .is_some()
    );
}

#[test]
fn artifact_read_target_finds_a_path_nested_in_a_wrapper_call() {
    let args = json!({
        "skill": "files",
        "tool": "file_read",
        "args": {"path": "artifacts/tool-results/s/use_skill/c.txt", "offset": 42}
    });
    assert_eq!(
        read_target("use_skill", &args),
        Some(ArtifactRead {
            path: "artifacts/tool-results/s/use_skill/c.txt".to_string(),
            offset: 42,
        })
    );
    assert_eq!(
        read_target("file_read", &json!({"path": "src/main.rs"})),
        None
    );
}

#[tokio::test]
async fn persisted_outcome_reports_the_size_of_the_stored_body() {
    let store = store("session");
    let rewritten = "s".repeat(3_000);
    let full = "r".repeat(8_000);

    let (out, outcome) = apply_per_result_persistence(
        rewritten,
        Some(full),
        Some(&store),
        "shell",
        Some("call"),
        1_000,
    )
    .await;

    assert!(outcome.persisted);
    assert_eq!(
        outcome.original_bytes, 8_000,
        "the outcome feeds the artifact index, so it must report the stored body's size"
    );
    assert!(out.contains("original_bytes: 8000"), "{out}");
}

#[test]
fn artifact_read_target_matches_only_file_read_under_the_artifact_directory() {
    let path = "artifacts/tool-results/s/shell/c.txt";
    assert!(read_target("file_read", &json!({"path": path})).is_some());
    assert_eq!(
        read_target("file_write", &json!({"path": path, "content": "x"})),
        None,
        "a write to an artifact path is not a read of its content"
    );
    assert_eq!(
        read_target(
            "use_skill",
            &json!({"skill": "files", "tool": "glob", "args": {"path": path}})
        ),
        None,
        "a wrapped non-read tool is not a read of its content"
    );
}

#[test]
fn artifact_read_target_matches_the_artifact_directory_as_a_path_component() {
    assert!(
        read_target(
            "file_read",
            &json!({"path": "./artifacts/tool-results/s/c.txt"})
        )
        .is_some()
    );
    assert_eq!(
        read_target(
            "file_read",
            &json!({"path": "artifacts/tool-results-backup/report.txt"})
        ),
        None,
        "a sibling directory sharing the prefix is not the artifact directory"
    );
}

#[test]
fn an_artifact_page_stays_within_the_budget_with_a_long_path() {
    let read = ArtifactRead {
        path: format!(
            "artifacts/tool-results/{}/use_skill/call.txt",
            "s".repeat(240)
        ),
        offset: 1_234_567,
    };

    let page = page_artifact_read("y".repeat(5_000), &read, 1_000, "file_read");
    assert!(
        page.len() <= 1_000,
        "a page must fit the result budget, got {} bytes",
        page.len()
    );
    let body = page
        .find("\n\n[artifact page")
        .expect("continuation marker");
    assert!(
        page.contains(&format!("\"offset\":{}", 1_234_567 + body)),
        "the marker must name the offset right after this page: {page}"
    );
}

#[test]
fn a_page_never_exceeds_the_floored_budget_and_always_advances() {
    // A path too long for its full trailer to leave body room under the floor.
    let read = ArtifactRead {
        path: format!("artifacts/tool-results/{}/c.txt", "p".repeat(600)),
        offset: 7,
    };
    let content = "z".repeat(5_000);

    for budget in [2, 100, MIN_ENVELOPE_ALLOWANCE_BYTES, 700] {
        let page = page_artifact_read(content.clone(), &read, budget, "file_read");
        let limit = budget.max(MIN_ENVELOPE_ALLOWANCE_BYTES);
        assert!(
            page.len() <= limit,
            "budget {budget}: a page must fit max(budget, floor) = {limit}, got {} bytes",
            page.len()
        );
        let body = page
            .find("\n\n[artifact page")
            .expect("continuation marker");
        assert!(
            body > 0,
            "budget {budget}: every page must advance past its offset"
        );
        assert!(
            page.contains(&format!("\"offset\":{}", 7 + body)),
            "budget {budget}: the marker must name the offset right after this page: {page}"
        );
    }
}

#[test]
fn page_continuation_json_escapes_paths() {
    let read = ArtifactRead {
        path: r#"C:\\state\\a\"b.txt"#.to_string(),
        offset: 0,
    };
    let page = page_artifact_read("x".repeat(2_000), &read, 1_000, "file_read");
    let call = page.split("Continue with file_read ").nth(1).unwrap();
    let call = call.split(']').next().unwrap();
    let parsed: Value = serde_json::from_str(call).unwrap();
    assert_eq!(parsed["path"], read.path);
}

#[test]
fn a_body_redaction_grows_past_the_read_limit_falls_back_to_the_processed_copy() {
    let raw = "call +15551234567 or +15557654321";
    // The limit is the raw size: the raw body fits, its redacted form does not.
    let (chosen, stored) = readable_body(
        raw,
        Some("processed copy"),
        raw.len() as u64,
        &TestRedactor,
        "file_read",
    )
    .expect("the processed copy fits");
    assert_eq!(
        chosen, "processed copy",
        "a body whose sanitized form exceeds the read limit must not be stored, got {:?}",
        stored.text
    );

    let (kept, _) = readable_body(
        raw,
        Some("processed copy"),
        10_000,
        &TestRedactor,
        "file_read",
    )
    .expect("fits");
    assert_eq!(
        kept, raw,
        "a body that stays within the limit is stored as returned"
    );
}

#[test]
fn a_body_is_refused_when_neither_candidate_fits_the_read_limit() {
    let raw = "call +15551234567 or +15557654321";
    let fallback = "fallback +15550001111 +15550002222";
    let limit = raw.len().min(fallback.len()) as u64;
    assert!(
        readable_body(raw, Some(fallback), limit, &TestRedactor, "file_read").is_err(),
        "when neither the raw body nor the fallback fits once sanitized, nothing may be stored"
    );
    assert!(
        readable_body(raw, None, limit, &TestRedactor, "file_read").is_err(),
        "a body with no fallback that does not fit must not be stored either"
    );
}

#[test]
fn artifact_read_target_follows_only_use_skill_into_a_wrapped_tool() {
    let nested = json!({"tool": "file_read", "args": {"path": "artifacts/tool-results/s/c.txt"}});
    assert!(
        read_target("use_skill", &nested).is_some(),
        "use_skill forwards file_read's result, so its wrapped read counts"
    );
    for outer in ["glob", "file_write", "shell"] {
        assert_eq!(
            read_target(outer, &nested),
            None,
            "{outer} carrying tool/args fields is not a wrapper; its result is its own"
        );
    }
}

#[test]
fn a_page_near_the_maximum_offset_neither_overflows_nor_advertises_a_stuck_continuation() {
    let read = ArtifactRead {
        path: "artifacts/tool-results/s/shell/c.txt".to_string(),
        offset: usize::MAX - 10,
    };
    // The offset comes straight from the model's arguments, so the page
    // arithmetic must not overflow on an absurd one.
    let page = std::panic::catch_unwind(|| {
        page_artifact_read("q".repeat(5_000), &read, 1_000, "file_read")
    });
    assert!(
        page.is_ok(),
        "an offset near usize::MAX must not overflow the page arithmetic"
    );
    let page = page.unwrap();
    assert!(
        page.len() <= 1_000,
        "the result is still bounded, got {} bytes",
        page.len()
    );
    assert!(
        !page.contains("Continue with"),
        "no continuation may be advertised when the next offset cannot advance"
    );
}

#[test]
fn artifact_read_target_rejects_an_explicit_invalid_offset() {
    let path = "artifacts/tool-results/s/shell/c.txt";
    for bad in [json!(-1), json!(1.5), json!("12")] {
        assert_eq!(
            read_target("file_read", &json!({"path": path, "offset": bad.clone()})),
            None,
            "offset {bad} is not a read file_read serves, so it must not become an artifact read at 0"
        );
    }
    for absent in [json!({"path": path}), json!({"path": path, "offset": null})] {
        assert_eq!(
            read_target("file_read", &absent).map(|read| read.offset),
            Some(0),
            "an absent or null offset reads from the start"
        );
    }
}

/// Literal fixture: the envelope the model reads is a wire format hosts and
/// runbooks match on, so pin it byte for byte.
#[tokio::test]
async fn envelope_text_is_byte_stable() {
    let store = store("s");
    let persisted = store
        .persist("shell", Some("c"), "hello world", None, 1024, "r")
        .await
        .unwrap();
    let expected = "[tool_result_preview]\n\
tool: shell\n\
reason: r\n\
original_bytes: 11\n\
stored_bytes: 11\n\
artifact_path: harness/tool_results/s/shell/c.txt\n\
read_with: file_read {\"path\":\"harness/tool_results/s/shell/c.txt\"} (a long read returns one page and names the \"offset\" to continue from)\n\
notes: Full scrubbed output was persisted on the artifact store at the artifact_path above (not part of the project).\n\n\
[preview]\nhello world";
    assert_eq!(persisted.output, expected);
    assert!(!persisted.redacted);
}

/// Library users with no backend get a private in-memory store; nothing is
/// shared between two of them.
#[tokio::test]
async fn in_memory_stores_do_not_share_artifacts() {
    let a = store("s");
    let b = store("s");
    a.persist("shell", Some("c"), "hello", None, 1024, "r")
        .await
        .unwrap();
    assert!(
        a.read("harness/tool_results/s/shell/c.txt")
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        b.read("harness/tool_results/s/shell/c.txt").await.unwrap(),
        None
    );
}

/// The pointer is a blob key on a host-supplied store, not a filesystem path.
#[tokio::test]
async fn a_host_supplied_blob_store_receives_the_redacted_body() {
    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("tenant").unwrap())
        .unwrap();
    let blobs = scoped.blobs().clone();
    let store = ToolResultArtifactStore::new(
        blobs.clone(),
        "session",
        Arc::new(TestRedactor),
        "file_read",
        1024,
    );
    let body = format!("a {}", test_github_token());
    let persisted = store
        .persist("shell", Some("c"), &body, None, 1024, "r")
        .await
        .unwrap();
    assert_eq!(persisted.path, "harness/tool_results/session/shell/c.txt");
    let blob = blobs
        .get("harness/tool_results/session/shell/c.txt")
        .await
        .unwrap()
        .expect("stored on the host blob store");
    assert_eq!(
        String::from_utf8(blob.bytes).unwrap(),
        "a [REDACTED_SECRET]"
    );
    assert_eq!(
        blob.meta.content_type.as_deref(),
        Some("text/plain; charset=utf-8")
    );
}

#[tokio::test]
async fn read_refuses_keys_outside_the_artifact_namespace() {
    let store = store("s");
    store
        .persist("shell", Some("c"), "hello", None, 1024, "r")
        .await
        .unwrap();
    for key in [
        "harness/outputs/x.md",
        "tool-results",
        "harness/tool_results/../outputs/x.md",
        "harness/tool_results//s/shell/c.txt",
        "harness/tool_results/s/shell/c.txt\n",
        "tool-results\\s\\shell\\c.txt",
        "tool-results-backup/s/shell/c.txt",
        "artifacts/other/s/shell/c.txt",
    ] {
        assert!(store.read(key).await.is_err(), "{key:?} must be refused");
    }
}

/// A pointer from a transcript written before the move still resolves.
#[tokio::test]
async fn read_accepts_the_legacy_relative_pointer() {
    let store = store("s");
    store
        .persist("shell", Some("c"), "hello", None, 1024, "r")
        .await
        .unwrap();
    for key in [
        "artifacts/tool-results/s/shell/c.txt",
        "./tool-results/s/shell/c.txt",
    ] {
        assert_eq!(store.read(key).await.unwrap().as_deref(), Some("hello"));
    }
    assert_eq!(
        store
            .read("harness/tool_results/s/shell/missing.txt")
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn purge_session_removes_only_this_sessions_artifacts() {
    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    let blobs = scoped.blobs().clone();
    let mk = |session: &str| {
        ToolResultArtifactStore::new(
            blobs.clone(),
            session,
            Arc::new(TestRedactor),
            "file_read",
            1024,
        )
    };
    let (live, old) = (mk("live"), mk("old"));
    live.persist("shell", Some("a"), "x", None, 1024, "r")
        .await
        .unwrap();
    old.persist("shell", Some("a"), "y", None, 1024, "r")
        .await
        .unwrap();
    old.persist("grep", Some("b"), "z", None, 1024, "r")
        .await
        .unwrap();
    blobs
        .put("harness/outputs/keep.md", b"k".to_vec(), None)
        .await
        .unwrap();

    assert_eq!(old.purge_session().await.unwrap(), 2);
    assert_eq!(
        old.read("harness/tool_results/old/shell/a.txt")
            .await
            .unwrap(),
        None
    );
    assert!(
        live.read("harness/tool_results/live/shell/a.txt")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        blobs
            .head("harness/outputs/keep.md")
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(old.purge_session().await.unwrap(), 0, "idempotent");
}

#[tokio::test]
async fn retention_lets_a_ttl_capable_driver_expire_the_artifact() {
    let expired = store("s").with_retention(Duration::ZERO);
    assert!(
        expired.blobs().capabilities().contains(Capability::BlobTtl),
        "the memory driver is the reference TTL implementation"
    );
    expired
        .persist("shell", Some("c"), "hello", None, 1024, "r")
        .await
        .unwrap();
    assert_eq!(
        expired
            .read("harness/tool_results/s/shell/c.txt")
            .await
            .unwrap(),
        None,
        "a zero lifetime is already expired"
    );

    let kept = store("s").with_retention(Duration::from_secs(3600));
    kept.persist("shell", Some("c"), "hello", None, 1024, "r")
        .await
        .unwrap();
    assert_eq!(
        kept.read("harness/tool_results/s/shell/c.txt")
            .await
            .unwrap()
            .as_deref(),
        Some("hello")
    );
}

#[test]
fn read_target_recognises_blob_keys_and_the_legacy_form() {
    let store = store("s");
    let pointer = store.path_for_read_tool("shell", Some("c"));
    assert_eq!(
        store.read_target(
            "file_read",
            &json!({"path": pointer, "offset": 7}),
            "use_skill"
        ),
        Some(ArtifactRead {
            path: pointer.clone(),
            offset: 7
        })
    );
    let wrapped = store.read_target(
        "use_skill",
        &json!({"skill": "files", "tool": "file_read", "args": {"path": pointer}}),
        "use_skill",
    );
    assert_eq!(wrapped.map(|read| read.offset), Some(0));
    assert!(
        store
            .read_target(
                "file_read",
                &json!({"path": "artifacts/tool-results/s/shell/c.txt"}),
                "use_skill"
            )
            .is_some()
    );
    // Absolute paths and sibling namespaces are not this store's artifacts.
    for path in [
        "/etc/hosts",
        "/tmp/tool-results/s/c.txt",
        "tool-results-backup/s/c.txt",
    ] {
        assert_eq!(
            store.read_target("file_read", &json!({"path": path}), "use_skill"),
            None,
            "{path}"
        );
    }
    assert_eq!(
        store.read_target("file_write", &json!({"path": pointer}), "use_skill"),
        None
    );
}
