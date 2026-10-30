use super::*;

#[test]
fn in_memory_transcript_starts_empty() {
    let locator = in_memory_transcript();
    let meta = crate::transcript::TranscriptMeta {
        agent_name: "fixture".into(),
        agent_id: None,
        agent_type: None,
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
        session_id: None,
        parent_session_id: None,
    };

    let transcript = locator.open_stem("fixture", meta).expect("bind transcript");
    assert!(
        transcript
            .read_session()
            .expect("read empty transcript")
            .is_none()
    );
}
