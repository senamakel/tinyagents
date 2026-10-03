use super::*;

#[test]
fn user_placement_builds_a_marked_user_message() {
    let message = checkpoint_message(SummaryPlacement::User, "## Goal\nship it");
    assert!(matches!(message, Message::User(_)));
    assert!(message.text().starts_with(CHECKPOINT_PREFIX));
    assert!(is_checkpoint(&message));
    assert_eq!(
        checkpoint_body(&message).as_deref(),
        Some("## Goal\nship it")
    );
}

#[test]
fn system_placement_builds_a_marked_system_message() {
    let message = checkpoint_message(SummaryPlacement::System, "body");
    assert!(matches!(message, Message::System(_)));
    assert!(is_checkpoint(&message));
    assert_eq!(checkpoint_body(&message).as_deref(), Some("body"));
}

#[test]
fn rewrapping_a_checkpoint_does_not_duplicate_the_marker() {
    let first = checkpoint_message(SummaryPlacement::User, "body");
    let second = checkpoint_message(SummaryPlacement::User, &first.text());
    assert_eq!(first, second);
    assert_eq!(second.text().matches(CHECKPOINT_PREFIX).count(), 1);
}

#[test]
fn ordinary_messages_are_not_checkpoints() {
    assert!(!is_checkpoint(&Message::user("hello")));
    assert!(!is_checkpoint(&Message::system("You are helpful.")));
    assert!(!is_checkpoint(&Message::assistant(CHECKPOINT_PREFIX)));
    assert_eq!(checkpoint_body(&Message::user("hello")), None);
}
