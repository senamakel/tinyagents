use tempfile::TempDir;

use super::*;
use crate::transcript::TranscriptMessage;

fn workspace() -> TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn message_kind(role: &str, content: &str) -> EntryKind {
    EntryKind::Message(TranscriptMessage::new(role, content))
}

// ── Append / branch / fork / labels ─────────────────────────────────────

#[test]
fn linear_append_defaults_parent_to_head() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let a = tree
        .append_to_head(message_kind("user", "hi"))
        .expect("append a");
    let b = tree
        .append_to_head(message_kind("assistant", "hello"))
        .expect("append b");

    let entry_b = tree.get(&b).expect("get b").expect("b exists");
    assert_eq!(entry_b.parent_id, Some(a.clone()));
    assert_eq!(tree.head().expect("head"), Some(b.clone()));
    assert_eq!(tree.tips().expect("tips"), vec![b]);
}

#[test]
fn branch_fork_shares_parent_pointer_without_copying() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let root = tree.append(None, message_kind("user", "root")).unwrap();
    let main = tree
        .append(Some(&root), message_kind("assistant", "main reply"))
        .unwrap();
    // An existing continuation off `main`, before any forking happens.
    let original_continuation = tree
        .append(Some(&main), message_kind("user", "continue original"))
        .unwrap();

    let fork_point = tree
        .fork(
            &main,
            Fork {
                scope: ForkScope::Branch,
                position: ForkPosition::At,
            },
        )
        .expect("fork");
    // Branch scope never copies: the fork point IS the existing entry.
    assert_eq!(fork_point, main);

    let branch_tip = tree
        .append(Some(&fork_point), message_kind("assistant", "alt reply"))
        .unwrap();

    // `main` now has two children (`original_continuation` and
    // `branch_tip`), both of which are tips; `main` itself is not.
    let mut tips = tree.tips().expect("tips");
    tips.sort();
    let mut expected = vec![original_continuation.clone(), branch_tip.clone()];
    expected.sort();
    assert_eq!(tips, expected);

    // Ancestor chains diverge only at `branch_tip`, sharing everything else.
    let branch_chain = tree.ancestor_chain(&branch_tip).expect("chain");
    assert_eq!(branch_chain.len(), 3);
    assert_eq!(branch_chain[0].id, root);
    assert_eq!(branch_chain[1].id, main);
}

#[test]
fn tree_fork_copies_the_ancestor_path() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let root = tree.append(None, message_kind("user", "root")).unwrap();
    let mid = tree
        .append(Some(&root), message_kind("assistant", "mid"))
        .unwrap();

    let copied_tip = tree
        .fork(
            &mid,
            Fork {
                scope: ForkScope::Tree,
                position: ForkPosition::At,
            },
        )
        .expect("fork");
    assert_ne!(copied_tip, mid, "tree fork must produce a new id");

    let original = tree.get(&mid).unwrap().unwrap();
    let copy = tree.get(&copied_tip).unwrap().unwrap();
    assert_eq!(original.kind, copy.kind);

    let copy_chain = tree.ancestor_chain(&copied_tip).expect("chain");
    assert_eq!(copy_chain.len(), 2);
    assert_ne!(copy_chain[0].id, root, "root was copied too, not shared");
    assert_eq!(copy_chain[0].kind, tree.get(&root).unwrap().unwrap().kind);

    // The original path is untouched.
    let original_chain = tree.ancestor_chain(&mid).expect("chain");
    assert_eq!(original_chain[0].id, root);
}

#[test]
fn fork_before_targets_the_parent_and_drops_the_tip() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let root = tree.append(None, message_kind("user", "root")).unwrap();
    let tip = tree
        .append(Some(&root), message_kind("assistant", "reply"))
        .unwrap();

    let point = tree
        .fork(
            &tip,
            Fork {
                scope: ForkScope::Branch,
                position: ForkPosition::Before,
            },
        )
        .expect("fork");
    assert_eq!(point, root);
}

#[test]
fn fork_before_on_root_errors() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    let root = tree.append(None, message_kind("user", "root")).unwrap();

    let result = tree.fork(
        &root,
        Fork {
            scope: ForkScope::Branch,
            position: ForkPosition::Before,
        },
    );
    assert!(result.is_err());
}

#[test]
fn labels_name_a_tip_and_list_as_branches() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    let root = tree.append(None, message_kind("user", "root")).unwrap();

    let labeled = tree.label(&root, "checkpoint-1").expect("label");
    let branches = tree.branches().expect("branches");
    assert_eq!(branches.len(), 1);
    assert_eq!(branches[0].name, "checkpoint-1");
    assert_eq!(branches[0].tip_id, labeled);

    // Label entries are tree nodes (children of the labeled tip)...
    let labeled_entry = tree.get(&labeled).unwrap().unwrap();
    assert_eq!(labeled_entry.parent_id, Some(root));
    assert!(matches!(labeled_entry.kind, EntryKind::Label(_)));

    // ...but never show up in a projected context.
    let context = tree.build_context(&labeled).expect("context");
    assert_eq!(context.len(), 1); // just "root"
}

// ── build_context ────────────────────────────────────────────────────

#[test]
fn build_context_returns_full_chain_without_compaction() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let a = tree.append(None, message_kind("user", "one")).unwrap();
    let b = tree
        .append(Some(&a), message_kind("assistant", "two"))
        .unwrap();
    let c = tree
        .append(Some(&b), message_kind("user", "three"))
        .unwrap();

    let context = tree.build_context(&c).expect("context");
    assert_eq!(context.len(), 3);
    assert_eq!(context[0].text(), "one");
    assert_eq!(context[1].text(), "two");
    assert_eq!(context[2].text(), "three");
}

#[test]
fn build_context_stops_at_newest_compaction_and_orders_chronologically() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let a = tree.append(None, message_kind("user", "one")).unwrap();
    let b = tree
        .append(Some(&a), message_kind("assistant", "two"))
        .unwrap();
    let kept_start = tree
        .append(Some(&b), message_kind("user", "three"))
        .unwrap();

    let compaction = tree
        .append(
            Some(&kept_start),
            EntryKind::Compaction(CompactionEntry {
                summary: "summary of one/two".to_string(),
                first_kept_entry_id: kept_start.clone(),
                tokens_before: 500,
                usage: None,
                details: serde_json::json!({}),
            }),
        )
        .unwrap();

    let after = tree
        .append(Some(&compaction), message_kind("assistant", "four"))
        .unwrap();

    let context = tree.build_context(&after).expect("context");
    // summary + kept_start ("three") + after ("four") — "one"/"two" dropped.
    assert_eq!(context.len(), 3);
    assert_eq!(context[0].text(), "summary of one/two");
    assert_eq!(context[1].text(), "three");
    assert_eq!(context[2].text(), "four");

    // A second, newer compaction supersedes the first.
    let second_kept = tree
        .append(Some(&after), message_kind("user", "five"))
        .unwrap();
    let second_compaction = tree
        .append(
            Some(&second_kept),
            EntryKind::Compaction(CompactionEntry {
                summary: "summary through four".to_string(),
                first_kept_entry_id: second_kept.clone(),
                tokens_before: 800,
                usage: None,
                details: serde_json::json!({}),
            }),
        )
        .unwrap();

    let context = tree.build_context(&second_compaction).expect("context");
    assert_eq!(context.len(), 2);
    assert_eq!(context[0].text(), "summary through four");
    assert_eq!(context[1].text(), "five");
}

#[test]
fn build_context_skips_labels_and_branch_summaries() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let a = tree.append(None, message_kind("user", "one")).unwrap();
    let labeled = tree.label(&a, "mark").unwrap();
    let summarized = tree
        .append(
            Some(&labeled),
            EntryKind::BranchSummary(BranchSummaryEntry {
                from_id: a.clone(),
                summary: "abandoned path".to_string(),
            }),
        )
        .unwrap();
    let b = tree
        .append(Some(&summarized), message_kind("assistant", "two"))
        .unwrap();

    let context = tree.build_context(&b).expect("context");
    assert_eq!(context.len(), 2);
    assert_eq!(context[0].text(), "one");
    assert_eq!(context[1].text(), "two");
}

#[test]
fn build_context_maps_custom_entries_to_message_custom() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    let a = tree.append(None, message_kind("user", "one")).unwrap();
    let custom = tree
        .append(
            Some(&a),
            EntryKind::Custom(CustomEntry {
                kind: "notification".to_string(),
                payload: serde_json::json!({"level": "info"}),
                display: Some("note".to_string()),
            }),
        )
        .unwrap();

    let context = tree.build_context(&custom).expect("context");
    assert_eq!(context.len(), 2);
    match &context[1] {
        tinyagents_harness::tinyinference_llm::Message::Custom(custom) => {
            assert_eq!(custom.kind, "notification");
            assert_eq!(custom.display.as_deref(), Some("note"));
        }
        other => panic!("expected Message::Custom, got {other:?}"),
    }
}

// ── Legacy import ────────────────────────────────────────────────────

#[test]
fn legacy_jsonl_messages_import_with_derived_linear_parents() {
    let messages = vec![
        TranscriptMessage::new("system", "sys"),
        TranscriptMessage::new("user", "hi"),
        TranscriptMessage::new("assistant", "hello"),
    ];
    let entries = legacy::from_messages("sess-legacy", &messages);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].id, EntryId::derive("sess-legacy", 0));
    assert_eq!(entries[0].parent_id, None);
    assert_eq!(entries[1].parent_id, Some(entries[0].id.clone()));
    assert_eq!(entries[2].parent_id, Some(entries[1].id.clone()));

    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-legacy");
    tree.import_legacy(&entries).expect("import");

    let tip = EntryId::derive("sess-legacy", 2);
    let context = tree.build_context(&tip).expect("context");
    assert_eq!(context.len(), 3);
    assert_eq!(context[1].text(), "hi");

    // Re-importing the same source is a no-op: ids collide and are skipped,
    // so the tree stays exactly as before rather than erroring or duplicating.
    tree.import_legacy(&entries)
        .expect("re-import is idempotent");
    let context_again = tree.build_context(&tip).expect("context");
    assert_eq!(context_again.len(), 3);
}

#[test]
fn legacy_sqlite_messages_import_with_derived_linear_parents() {
    use crate::types::SessionMessage;
    let now = chrono::Utc::now();
    let rows = vec![
        SessionMessage {
            id: 1,
            session_id: "sess-sql".to_string(),
            role: "user".to_string(),
            content: "hi".to_string(),
            reasoning_content: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            cost_usd: None,
            created_at: now,
        },
        SessionMessage {
            id: 2,
            session_id: "sess-sql".to_string(),
            role: "assistant".to_string(),
            content: "hello".to_string(),
            reasoning_content: None,
            model: Some("test-model".to_string()),
            input_tokens: Some(10),
            output_tokens: Some(5),
            cost_usd: Some(0.001),
            created_at: now,
        },
    ];
    let entries = legacy::from_session_messages("sess-sql", &rows);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].parent_id, None);
    assert_eq!(entries[1].parent_id, Some(entries[0].id.clone()));

    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-sql");
    tree.import_legacy(&entries).expect("import");
    let tip = EntryId::derive("sess-sql", 1);
    let context = tree.build_context(&tip).expect("context");
    assert_eq!(context.len(), 2);
    assert_eq!(context[1].text(), "hello");
}

// ── Index rebuild ────────────────────────────────────────────────────

#[test]
fn rebuild_index_matches_incremental_index() {
    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");

    let a = tree.append(None, message_kind("user", "one")).unwrap();
    let b = tree
        .append(Some(&a), message_kind("assistant", "two"))
        .unwrap();
    let branch_point = tree
        .fork(
            &a,
            Fork {
                scope: ForkScope::Branch,
                position: ForkPosition::At,
            },
        )
        .unwrap();
    let c = tree
        .append(Some(&branch_point), message_kind("assistant", "alt"))
        .unwrap();

    // Both `b` and `c` are tips right now; every append/fork keeps the
    // index fresh incrementally via `append`/`fork`'s own transaction —
    // but there is no incremental index write for `append`/`fork`
    // themselves (only labels/rebuild write branch_entries directly), so
    // compare a live walk to a rebuilt index instead of two index reads.
    let before_b = tree.ancestor_chain(&b).expect("chain b");
    let before_c = tree.ancestor_chain(&c).expect("chain c");

    tree.rebuild_index().expect("rebuild");

    let after_b = tree.build_context(&b).expect("context b");
    let after_c = tree.build_context(&c).expect("context c");

    assert_eq!(before_b.len(), 2);
    assert_eq!(before_c.len(), 2);
    assert_eq!(after_b.len(), 2);
    assert_eq!(after_c.len(), 2);
    assert_eq!(after_b[0].text(), "one");
    assert_eq!(after_c[1].text(), "alt");
}

// ── Serde round trip ─────────────────────────────────────────────────

#[test]
fn every_entry_kind_round_trips_through_serde() {
    let kinds = vec![
        message_kind("user", "hi"),
        EntryKind::Compaction(CompactionEntry {
            summary: "sum".to_string(),
            first_kept_entry_id: EntryId::from("sess:2"),
            tokens_before: 100,
            usage: None,
            details: serde_json::json!({"rule": "cut"}),
        }),
        EntryKind::BranchSummary(BranchSummaryEntry {
            from_id: EntryId::from("sess:1"),
            summary: "abandoned".to_string(),
        }),
        EntryKind::Label(LabelEntry {
            name: "checkpoint".to_string(),
        }),
        EntryKind::Custom(CustomEntry {
            kind: "note".to_string(),
            payload: serde_json::json!({"a": 1}),
            display: Some("a note".to_string()),
        }),
    ];
    for kind in kinds {
        let json = serde_json::to_string(&kind).expect("serialize");
        let round_tripped: EntryKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(kind, round_tripped);
    }

    let entry = Entry {
        id: EntryId::from("sess:0"),
        parent_id: None,
        ordinal: 0,
        kind: message_kind("system", "sys"),
        ts: "2026-01-01T00:00:00Z".to_string(),
    };
    let json = serde_json::to_string(&entry).expect("serialize entry");
    let round_tripped: Entry = serde_json::from_str(&json).expect("deserialize entry");
    assert_eq!(entry, round_tripped);
}

// ── SessionCompactionSink ───────────────────────────────────────────────

#[test]
fn compaction_sink_persists_a_record_anchored_at_the_tip() {
    use tinyagents_harness::summarization::{CompactionReason, CompactionRecord, CompactionSink};

    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    tree.append(None, message_kind("user", "one")).unwrap();
    tree.append_to_head(message_kind("assistant", "two"))
        .unwrap();
    let tip = tree
        .append_to_head(message_kind("user", "three"))
        .unwrap();

    let sink = SessionCompactionSink::new(ws.path(), "sess-1").expect("sink");
    assert_eq!(sink.tip(), Some(tip.clone()));

    let record = CompactionRecord {
        summary: "one and two, summarized".to_string(),
        // Skip "one" only; keep "two" and "three" verbatim.
        first_kept_index: 1,
        tokens_before: 300,
        tokens_after: 120,
        usage: None,
        details: serde_json::json!({ "rule": "threshold" }),
        reason: CompactionReason::Threshold,
    };
    sink.persist(&record).expect("persist");

    let new_tip = sink.tip().expect("sink has a new tip after persisting");
    assert_ne!(new_tip, tip);

    let context = tree.build_context(&new_tip).expect("context");
    assert_eq!(context.len(), 2);
    assert_eq!(context[0].text(), "one and two, summarized");
    assert_eq!(context[1].text(), "three");
}

#[test]
fn compaction_sink_advances_its_tip_across_repeated_compactions() {
    use tinyagents_harness::summarization::{CompactionReason, CompactionRecord, CompactionSink};

    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    tree.append(None, message_kind("user", "a")).unwrap();
    tree.append_to_head(message_kind("assistant", "b"))
        .unwrap();
    tree.append_to_head(message_kind("user", "c")).unwrap();

    let sink = SessionCompactionSink::new(ws.path(), "sess-1").expect("sink");

    sink.persist(&CompactionRecord {
        summary: "first summary".to_string(),
        first_kept_index: 1,
        tokens_before: 100,
        tokens_after: 40,
        usage: None,
        details: serde_json::json!({}),
        reason: CompactionReason::Threshold,
    })
    .expect("first persist");
    let after_first = sink.tip().expect("tip after first compaction");

    // Grow the (already-compacted) transcript, then compact again.
    tree.append(Some(&after_first), message_kind("assistant", "d"))
        .unwrap();

    sink.persist(&CompactionRecord {
        summary: "second summary".to_string(),
        first_kept_index: 0,
        tokens_before: 200,
        tokens_after: 20,
        usage: None,
        details: serde_json::json!({}),
        reason: CompactionReason::Overflow,
    })
    .expect("second persist");

    let after_second = sink.tip().expect("tip after second compaction");
    assert_ne!(after_second, after_first);

    let context = tree.build_context(&after_second).expect("context");
    // Only the second (newest) compaction's summary is visible.
    assert_eq!(context[0].text(), "second summary");
}

#[test]
fn compaction_sink_is_a_no_op_on_an_empty_session() {
    use tinyagents_harness::summarization::{CompactionReason, CompactionRecord, CompactionSink};

    let ws = workspace();
    let sink = SessionCompactionSink::new(ws.path(), "sess-empty").expect("sink");
    assert_eq!(sink.tip(), None);

    sink.persist(&CompactionRecord {
        summary: "nothing to compact".to_string(),
        first_kept_index: 0,
        tokens_before: 10,
        tokens_after: 5,
        usage: None,
        details: serde_json::json!({}),
        reason: CompactionReason::Manual,
    })
    .expect("persist on empty session should be a no-op, not an error");

    assert_eq!(sink.tip(), None);
}

#[test]
fn compaction_sink_skips_an_out_of_range_index_rather_than_corrupting_the_tree() {
    use tinyagents_harness::summarization::{CompactionReason, CompactionRecord, CompactionSink};

    let ws = workspace();
    let tree = EntryTree::new(ws.path(), "sess-1");
    let tip = tree.append(None, message_kind("user", "only one")).unwrap();

    let sink = SessionCompactionSink::new(ws.path(), "sess-1").expect("sink");

    sink.persist(&CompactionRecord {
        summary: "bogus".to_string(),
        first_kept_index: 5, // out of range: only one message entry exists
        tokens_before: 10,
        tokens_after: 5,
        usage: None,
        details: serde_json::json!({}),
        reason: CompactionReason::Threshold,
    })
    .expect("out-of-range index is skipped, not an error");

    // No compaction entry was written; the tip is unchanged.
    assert_eq!(sink.tip(), Some(tip));
}
