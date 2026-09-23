use super::{SUBAGENT_SEPARATOR, SessionRef, session_stem};

#[test]
fn a_stem_is_deterministic_and_carries_no_timestamp() {
    let session = SessionRef::scoped("thread-9fa08c44", "orchestrator");
    let first = session_stem(&session);
    let second = session_stem(&SessionRef::scoped("thread-9fa08c44", "orchestrator"));

    assert_eq!(first, second);
    assert!(first.starts_with("thread-9fa08c44-"));
    assert!(first.contains(".orchestrator-"));
    // The whole point: no `{unix_ts}_` prefix, so nothing varies per launch.
    assert!(
        !first
            .split(['_', '.'])
            .next()
            .is_some_and(|head| { head.len() >= 10 && head.chars().all(|c| c.is_ascii_digit()) }),
        "{first} still looks timestamp-prefixed"
    );
}

#[test]
fn two_agents_on_one_key_get_distinct_stems() {
    let left = session_stem(&SessionRef::scoped("thread-1", "orchestrator"));
    let right = session_stem(&SessionRef::scoped("thread-1", "researcher"));

    assert_ne!(left, right);
}

#[test]
fn an_unscoped_root_is_just_the_key_plus_a_digest() {
    let stem = session_stem(&SessionRef::root("thread-1"));
    assert!(stem.starts_with("thread-1-"));
}

#[test]
fn a_scoped_session_with_a_blank_agent_id_is_still_distinct_from_an_unscoped_root() {
    // `scoped` and `root` are two different constructors precisely because
    // they name two different identities; a blank/whitespace agent id must
    // not silently make `scoped` collapse into `root`'s stem (it used to,
    // when the agent component was omitted entirely for a blank value).
    let scoped_blank = SessionRef::scoped("thread-1", "   ");
    let root = SessionRef::root("thread-1");
    assert_ne!(session_stem(&scoped_blank), session_stem(&root));

    // Two distinct blank/empty agent ids must also stay distinct from each
    // other, since the digest covers the raw (pre-sanitization) value.
    let scoped_empty = SessionRef::scoped("thread-1", "");
    assert_ne!(session_stem(&scoped_blank), session_stem(&scoped_empty));
}

#[test]
fn path_traversal_in_a_key_cannot_escape_the_transcript_directory() {
    // `.` no longer survives sanitization — it is reserved for the agent and
    // generation separators — but the traversal characters it used to leave
    // behind must still never produce a path separator.
    let stem = session_stem(&SessionRef::root("../../etc/passwd"));
    assert!(!stem.contains('/'), "{stem}");
    assert!(!stem.contains('\\'), "{stem}");
    assert!(!stem.contains('.'), "{stem} still contains a literal '.'");
}

#[test]
fn generations_are_distinct_and_ordered_by_suffix() {
    let first = SessionRef::scoped("thread-1", "orchestrator");
    let second = first.next_generation();
    let third = second.next_generation();

    let first_stem = session_stem(&first);
    let second_stem = session_stem(&second);
    let third_stem = session_stem(&third);

    assert!(second_stem.starts_with(&format!("{first_stem}.g1")));
    assert!(third_stem.starts_with(&format!("{first_stem}.g2")));
    assert_ne!(first_stem, second_stem);
    assert_ne!(second_stem, third_stem);
}

#[test]
fn a_generation_knows_the_one_it_succeeded() {
    let first = SessionRef::scoped("thread-1", "orchestrator");
    let second = first.next_generation();

    assert_eq!(first.parent_session_id(), None);
    assert_eq!(
        second.parent_session_id().as_deref(),
        Some(session_stem(&first).as_str())
    );
    assert_eq!(second.session_id(), session_stem(&second));
}

#[test]
fn a_subagent_stem_carries_the_separator_every_root_scan_filters_on() {
    let parent = SessionRef::scoped("thread-1", "orchestrator");
    let child = SessionRef::child_of(&parent, "worker-7");

    let stem = session_stem(&child);
    assert!(stem.contains(SUBAGENT_SEPARATOR));
    assert!(child.is_subagent());
    assert!(!parent.is_subagent());
}

#[test]
fn a_root_stem_never_contains_the_subagent_separator() {
    // Root scans treat `__` as "this is a delegated worker", so a root stem
    // that contained it would vanish from thread lookup entirely.
    for key in ["thread-1", "a__b", "with space", "../escape"] {
        let stem = session_stem(&SessionRef::scoped(key, "agent__name"));
        assert!(
            !stem.contains(SUBAGENT_SEPARATOR),
            "root stem {stem:?} from key {key:?} looks like a sub-agent"
        );
    }
}

#[test]
fn nested_delegation_records_the_whole_path_in_one_flat_stem() {
    // Short keys: with a 32-hex-char digest on every component, even
    // moderately descriptive names ("orchestrator", "researcher") push a
    // 2-3 level chain's own resolved stem past `MAX_PARENT_CHAIN_PREFIX` —
    // intentionally, since that bound exists to keep the *filename* safe,
    // not to guarantee unlimited readability. This test is about the flat
    // "__"-joined shape surviving when the chain stays short enough not to
    // collapse; `a_deeply_nested_delegation_chain_stays_bounded` covers the
    // collapse itself.
    let root = SessionRef::scoped("t1", "o");
    let child = SessionRef::child_of(&root, "r1");
    let grandchild = SessionRef::child_of(&child, "r2");

    let stem = session_stem(&grandchild);
    assert_eq!(stem.matches(SUBAGENT_SEPARATOR).count(), 2);
    assert!(stem.starts_with(&session_stem(&root)));
}

// ---- Collision-resistance regressions -----------------------------------
//
// Every case here is a raw input pair that the pre-digest sanitizer mapped
// to the *same* filename, letting two distinct conversations read and
// overwrite each other's transcript.

#[test]
fn collapsing_underscore_runs_no_longer_aliases_distinct_keys() {
    let a = session_stem(&SessionRef::root("a_b"));
    let b = session_stem(&SessionRef::root("a__b"));
    assert_ne!(a, b);
}

#[test]
fn a_literal_dot_in_a_key_no_longer_aliases_the_generation_suffix() {
    let literal_dot = session_stem(&SessionRef::root("thread-1.g1"));
    let real_generation = session_stem(&SessionRef::root("thread-1").next_generation());
    assert_ne!(literal_dot, real_generation);
}

#[test]
fn a_literal_dot_in_a_key_no_longer_aliases_the_agent_separator() {
    let unscoped_with_dot = session_stem(&SessionRef::root("t.a"));
    let scoped = session_stem(&SessionRef::scoped("t", "a"));
    assert_ne!(unscoped_with_dot, scoped);
}

#[test]
fn a_very_long_key_still_produces_a_filesystem_safe_stem() {
    let long_key = "k".repeat(400);
    let stem = session_stem(&SessionRef::scoped(&long_key, "agent"));
    // Comfortably under common filesystem name limits (255 bytes) even after
    // `session_raw/{stem}.jsonl` and an agent id/generation suffix.
    assert!(stem.len() < 200, "{} bytes: {stem}", stem.len());
}

#[test]
fn two_long_keys_that_share_a_bounded_prefix_still_get_distinct_stems() {
    let base = "k".repeat(400);
    let a = session_stem(&SessionRef::root(&base));
    let b = session_stem(&SessionRef::root(format!("{base}-tail"))); // differs past the bound
    assert_ne!(a, b);
}

/// Pins the exact digest algorithm and its output, not just "some digest".
/// A durable filename must "re-derive the same stem forever" — swapping the
/// hash (or its parameters) is exactly the kind of change that must be
/// caught here rather than silently shipped, because it would re-derive a
/// different filename for every session ever written.
#[test]
fn the_digest_algorithm_is_pinned_to_known_fnv1a64_outputs() {
    assert_eq!(
        super::fnv1a64(b"", 0xcbf2_9ce4_8422_2325),
        0xcbf2_9ce4_8422_2325
    );
    assert_eq!(
        super::fnv1a64(b"a", 0xcbf2_9ce4_8422_2325),
        0xaf63_dc4c_8601_ec8c
    );
    assert_eq!(
        super::fnv1a64(b"thread-9fa08c44", 0xcbf2_9ce4_8422_2325),
        0xdf74_ac18_4530_bb17
    );
    // fnv1a128 combines two 64-bit passes with two different fixed seeds —
    // pin the combined output too, since a future change that widened or
    // reseeded only one pass would otherwise slip past the check above.
    assert_eq!(
        super::fnv1a128(b""),
        0xcbf2_9ce4_8422_2325_9e37_79b9_7f4a_7c15
    );
    assert_eq!(
        super::fnv1a128(b"a"),
        0xaf63_dc4c_8601_ec8c_22c0_4a33_4b91_791c
    );
    assert_eq!(
        super::fnv1a128(b"thread-9fa08c44"),
        0xdf74_ac18_4530_bb17_1e90_1a3d_7200_1847
    );
}

/// Pins the exact worst-case arithmetic documented on
/// `MAX_PARENT_CHAIN_PREFIX`: every level of a maximally-sized chain (a
/// maximal `session_key`, a maximal `agent_id` on the root, and an existing
/// compaction at every level) must stay under the 255-byte filesystem name
/// limit, whether or not that level's own resolved stem was itself the
/// trigger for a collapse one level up.
#[test]
fn every_level_of_a_maximal_chain_stays_under_the_filesystem_limit() {
    // The worst single (unparented) level: a maximal session_key, a maximal
    // agent_id, and an existing compaction (`.g{n}`) — see
    // `MAX_PARENT_CHAIN_PREFIX`'s doc for the arithmetic this pins.
    let max_key = "k".repeat(200);
    let root = SessionRef::scoped(&max_key, &max_key).next_generation();
    let root_stem = session_stem(&root);
    assert!(
        root_stem.len() < 255,
        "an unparented root must fit on its own: {} bytes",
        root_stem.len()
    );

    // The worst parented level: a child (never has an agent_id) of that same
    // maximal root, itself also compacted.
    let child = SessionRef::child_of(&root, max_key.clone()).next_generation();
    let child_stem = session_stem(&child);
    assert!(
        child_stem.len() < 255,
        "a child of a maximal root must still fit: {} bytes",
        child_stem.len()
    );

    // And the level after that, to confirm the collapse repeats rather than
    // the bound only holding for one transition.
    let grandchild = SessionRef::child_of(&child, max_key).next_generation();
    let grandchild_stem = session_stem(&grandchild);
    assert!(
        grandchild_stem.len() < 255,
        "a grandchild must still fit: {} bytes",
        grandchild_stem.len()
    );
}

/// A delegation chain several levels deep, each level with a long key, must
/// not grow the final stem without bound — see `MAX_PARENT_CHAIN_PREFIX`.
/// 50 levels of ~113-byte components would be several KB uncollapsed,
/// comfortably past any sane bound; this pins that the collapse actually
/// keeps growth from compounding rather than merely being "small enough in
/// this one example".
#[test]
fn a_deeply_nested_delegation_chain_stays_bounded() {
    let long_key = "k".repeat(80);
    let mut current = SessionRef::scoped(&long_key, "orchestrator");
    for level in 0..50 {
        current = SessionRef::child_of(&current, format!("{long_key}-{level}"));
    }
    let stem = session_stem(&current);
    assert!(
        stem.len() < 700,
        "50 levels of long keys must not grow the stem past the collapse \
         bound: {} bytes",
        stem.len()
    );
    // Still never fabricates or destroys the sub-agent separator.
    assert!(stem.contains(SUBAGENT_SEPARATOR));
}

/// Bounding the chain must never change what a *shallow* delegation's stem
/// looks like — the collapse only kicks in once the accumulated chain
/// actually exceeds the bound.
#[test]
fn a_shallow_delegation_chain_is_unaffected_by_the_bound() {
    let root = SessionRef::scoped("t1", "o");
    let child = SessionRef::child_of(&root, "r1");
    let grandchild = SessionRef::child_of(&child, "r2");

    let stem = session_stem(&grandchild);
    assert!(!stem.contains("chain-"), "{stem} collapsed too early");
    assert_eq!(stem.matches(SUBAGENT_SEPARATOR).count(), 2);
}
