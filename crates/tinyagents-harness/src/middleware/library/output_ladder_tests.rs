use super::*;
use crate::middleware::library::ListedOutputCapPolicy;

struct Upper {
    compacted: usize,
}

#[async_trait]
impl OutputProcessor for Upper {
    async fn compact(&mut self, _t: &str, content: String, _f: Option<String>) -> CompactedOutput {
        self.compacted += 1;
        CompactedOutput {
            summarized_from_bytes: Some(content.len()),
            notice: Some("note".into()),
            text: content.to_uppercase(),
        }
    }
}

fn policy() -> ListedOutputCapPolicy {
    ListedOutputCapPolicy {
        default_budget_bytes: 16_000,
        tool_caps: [("web_fetch".to_string(), 50_000)].into(),
        compaction_exempt: vec!["propose".into()],
        truncation_exempt: vec!["propose".into()],
        max_persistable_bytes: 0,
    }
}

fn input<'a>(tool: &'a str, content: String, wants: bool) -> OutputLadderInput<'a> {
    OutputLadderInput {
        tool,
        call_id: "c1",
        content,
        focus: None,
        wants_compaction: wants,
        store: None,
        summary_label: "test",
    }
}

#[tokio::test]
async fn compacts_then_discloses_without_a_cap_hit() {
    let mut p = Upper { compacted: 0 };
    let out = run_output_ladder(&policy(), &mut p, input("shell", "abc".into(), true)).await;
    assert_eq!(p.compacted, 1);
    assert!(out.starts_with("[test: summary of 3 bytes of tool output, complete]"));
    assert!(out.ends_with("note\n\nABC"));
}

#[tokio::test]
async fn exempt_tool_is_neither_compacted_nor_capped() {
    let mut p = Upper { compacted: 0 };
    let big = "x".repeat(40_000);
    let out = run_output_ladder(&policy(), &mut p, input("propose", big.clone(), true)).await;
    assert_eq!(p.compacted, 0);
    assert_eq!(out, big);
}

#[tokio::test]
async fn declared_cap_skips_compaction_without_focus() {
    let mut p = Upper { compacted: 0 };
    run_output_ladder(&policy(), &mut p, input("web_fetch", "abc".into(), true)).await;
    assert_eq!(p.compacted, 0);
    let mut i = input("web_fetch", "abc".into(), true);
    i.focus = Some("q".into());
    run_output_ladder(&policy(), &mut p, i).await;
    assert_eq!(p.compacted, 1);
}

#[tokio::test]
async fn oversized_result_is_bounded_to_the_shared_budget() {
    let mut p = Upper { compacted: 0 };
    let out = run_output_ladder(&policy(), &mut p, input("shell", "y".repeat(40_000), false)).await;
    assert!(out.len() < 40_000, "got {}", out.len());
}

#[tokio::test]
async fn body_past_persistable_limit_is_bounded_inline_not_spilled() {
    use crate::artifacts::policy::NoRedaction;
    use std::sync::Arc;
    let scope = tinyagents_storage_kit::Scope::new("local").unwrap();
    let blobs = tinystoragedrivers_testkit::memory_storage(&scope)
        .blobs()
        .clone();
    let store = ToolResultArtifactStore::new(
        blobs,
        "s",
        Arc::new(NoRedaction),
        "file_read",
        10 * 1024 * 1024,
    );
    let mut pol = policy();
    pol.max_persistable_bytes = 20_000;
    let mut p = Upper { compacted: 0 };
    let mut i = input("shell", "z".repeat(30_000), false);
    i.store = Some(&store);
    let out = run_output_ladder(&pol, &mut p, i).await;
    assert!(out.len() < 30_000);
    assert!(
        store.blobs().list("").await.unwrap().is_empty(),
        "nothing may be written for an ineligible body"
    );

    // Within the limit it does spill.
    let mut i = input("shell", "z".repeat(18_000), false);
    i.store = Some(&store);
    let out = run_output_ladder(&pol, &mut p, i).await;
    assert!(out.contains("harness/tool_results"), "{out}");
}
