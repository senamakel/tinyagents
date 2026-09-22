use super::{SUBAGENT_SEPARATOR, SessionRef, session_stem};

#[test]
fn a_stem_is_deterministic_and_carries_no_timestamp() {
    let session = SessionRef::scoped("thread-9fa08c44", "orchestrator");
    let first = session_stem(&session);
    let second = session_stem(&SessionRef::scoped("thread-9fa08c44", "orchestrator"));

    assert_eq!(first, second);
    assert_eq!(first, "thread-9fa08c44.orchestrator");
    // The whole point: nothing here varies per process or per launch.
    assert!(!first.chars().any(|c| c.is_ascii_digit() && first.starts_with(c)));
}

#[test]
fn two_agents_on_one_key_get_distinct_stems() {
    let left = session_stem(&SessionRef::scoped("thread-1", "orchestrator"));
    let right = session_stem(&SessionRef::scoped("thread-1", "researcher"));

    assert_ne!(left, right);
}

#[test]
fn an_unscoped_root_is_just_the_key() {
    assert_eq!(session_stem(&SessionRef::root("thread-1")), "thread-1");
}

#[test]
fn a_blank_agent_id_does_not_add_a_separator() {
    let session = SessionRef::scoped("thread-1", "   ");
    assert_eq!(session_stem(&session), "thread-1");
}

#[test]
fn path_traversal_in_a_key_cannot_escape_the_transcript_directory() {
    let stem = session_stem(&SessionRef::root("../../etc/passwd"));
    assert!(!stem.contains('/'));
    assert!(!stem.contains(".."), "{stem}");
}

#[test]
fn generations_are_distinct_and_ordered_by_suffix() {
    let first = SessionRef::scoped("thread-1", "orchestrator");
    let second = first.next_generation();
    let third = second.next_generation();

    assert_eq!(session_stem(&first), "thread-1.orchestrator");
    assert_eq!(session_stem(&second), "thread-1.orchestrator.g1");
    assert_eq!(session_stem(&third), "thread-1.orchestrator.g2");
}

#[test]
fn a_generation_knows_the_one_it_succeeded() {
    let first = SessionRef::scoped("thread-1", "orchestrator");
    let second = first.next_generation();

    assert_eq!(first.parent_session_id(), None);
    assert_eq!(
        second.parent_session_id().as_deref(),
        Some("thread-1.orchestrator")
    );
    assert_eq!(second.session_id(), "thread-1.orchestrator.g1");
}

#[test]
fn a_subagent_stem_carries_the_separator_every_root_scan_filters_on() {
    let parent = SessionRef::scoped("thread-1", "orchestrator");
    let child = SessionRef::child_of(&parent, "worker-7");

    let stem = session_stem(&child);
    assert_eq!(stem, "thread-1.orchestrator__worker-7");
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
    let root = SessionRef::scoped("thread-1", "orchestrator");
    let child = SessionRef::child_of(&root, "researcher");
    let grandchild = SessionRef::child_of(&child, "reader");

    assert_eq!(
        session_stem(&grandchild),
        "thread-1.orchestrator__researcher__reader"
    );
}
