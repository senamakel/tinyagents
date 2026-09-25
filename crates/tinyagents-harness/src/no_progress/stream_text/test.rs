use super::*;

#[test]
fn catches_repeated_process_narration_across_chunks() {
    let mut detector = StreamTextStallDetector::default();
    let text = (0..12)
        .map(|i| format!("Let me check the source number {i} before I answer the user. "))
        .collect::<String>();
    let mut stalled = false;
    for chunk in text.as_bytes().chunks(7) {
        stalled |= detector.observe(std::str::from_utf8(chunk).unwrap());
    }
    assert!(stalled);
}

#[test]
fn varied_prose_and_short_repetitions_continue() {
    let mut detector = StreamTextStallDetector::default();
    for sentence in [
        "Jev accepts typed questions about the supplied state.",
        "TypeSafe describes Choice, Score and Noul answers.",
        "A desktop host observes the screen before asking Jev.",
        "The host executes a bounded action and verifies its result.",
    ] {
        assert!(!detector.observe(sentence));
    }
    let mut detector = StreamTextStallDetector::default();
    for i in 0..7 {
        assert!(!detector.observe(&format!(
            "Let me check one more source number {i} before answering. "
        )));
    }
    let mut detector = StreamTextStallDetector::default();
    for i in 0..12 {
        assert!(!detector.observe(&format!(
            "Jev is a typed decision model used in example number {i}. "
        )));
    }
}

#[test]
fn tool_progress_resets_the_window() {
    let mut detector = StreamTextStallDetector::default();
    for i in 0..7 {
        detector.observe(&format!(
            "Let me inspect the page number {i} before answering. "
        ));
    }
    detector.reset();
    for i in 0..7 {
        assert!(!detector.observe(&format!(
            "Let me inspect the result number {i} before answering. "
        )));
    }
}

#[test]
fn useful_facts_between_planning_sentences_break_the_streak() {
    let mut detector = StreamTextStallDetector::default();
    for i in 0..12 {
        assert!(!detector.observe(&format!(
            "Let me check source {i} before answering the user. Jev returns a typed Choice for state {i}. "
        )));
    }
}
