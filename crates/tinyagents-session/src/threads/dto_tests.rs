use super::*;
use serde_json::json;

fn wire_thread() -> ConversationThread {
    ConversationThread {
        id: "thread/1".into(),
        title: "Title".into(),
        chat_id: Some(7),
        is_active: true,
        message_count: 2,
        last_message_at: "2026-01-02T00:00:00.250Z".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        parent_thread_id: Some("parent".into()),
        labels: vec!["tasks".into()],
        personality_id: None,
        working_dir: Some("/work".into()),
    }
}

#[test]
fn times_round_trip_at_millisecond_precision() {
    for text in [
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00.250Z",
        "1970-01-01T00:00:00Z",
    ] {
        assert_eq!(from_ms(to_ms(text).unwrap()).unwrap(), text);
    }
    // Offsets and sub-millisecond digits fold into UTC milliseconds.
    assert_eq!(
        from_ms(to_ms("2026-01-01T01:00:00.123456+01:00").unwrap()).unwrap(),
        "2026-01-01T00:00:00.123Z"
    );
}

#[test]
fn a_time_that_is_not_rfc_3339_is_invalid_input() {
    for text in ["now", "2026-07-01", ""] {
        let error = to_ms(text).unwrap_err();
        assert_eq!(
            error.kind(),
            tinystoragedrivers_types::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn a_legacy_time_that_is_not_rfc_3339_becomes_the_epoch() {
    assert_eq!(to_ms_or_epoch("now"), 0);
    assert_eq!(to_ms_or_epoch("2026-01-01T00:00:00Z"), 1_767_225_600_000);
}

#[test]
fn a_thread_body_is_flat_snake_case_with_ms_times() {
    let dto = ThreadDto::from_wire(
        &wire_thread(),
        ThreadState {
            incarnation: "6f9619ff-8b86-d011-b42d-00c04fc964ff".into(),
            deleted: false,
            visible_end: 2,
            messages_key: None,
        },
        to_ms,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&dto).unwrap(),
        json!({
            "id": "thread/1", "title": "Title", "chat_id": 7, "is_active": true,
            "created_ms": 1_767_225_600_000_i64, "last_message_ms": 1_767_312_000_250_i64,
            "parent_thread_id": "parent", "labels": ["tasks"], "working_dir": "/work",
            "incarnation": "6f9619ff-8b86-d011-b42d-00c04fc964ff",
            "deleted": false, "visible_end": 2
        })
    );
}

#[test]
fn a_thread_body_reads_back_as_the_wire_thread_with_a_derived_count() {
    let state = ThreadState {
        incarnation: "6f9619ff-8b86-d011-b42d-00c04fc964ff".into(),
        deleted: true,
        visible_end: 2,
        messages_key: Some("abc".into()),
    };
    let dto = ThreadDto::from_wire(&wire_thread(), state.clone(), to_ms).unwrap();
    let (thread, read_state) = dto.into_wire().unwrap();
    assert_eq!(thread, wire_thread());
    assert_eq!(read_state, state);
}

#[test]
fn a_thread_body_refuses_fields_it_does_not_know() {
    let mut body = json!({
        "id": "t", "title": "", "is_active": true, "created_ms": 0, "last_message_ms": 0,
        "labels": [], "incarnation": "i", "deleted": false, "visible_end": 0
    });
    serde_json::from_value::<ThreadDto>(body.clone()).unwrap();
    body["createdAt"] = json!("2026-01-01T00:00:00Z");
    serde_json::from_value::<ThreadDto>(body).unwrap_err();
}

#[test]
fn a_message_body_is_snake_case_with_an_ms_time_and_no_wrapper() {
    let message = ThreadMessage {
        id: "m1".into(),
        content: "hello".into(),
        message_type: "text".into(),
        extra_metadata: json!({"camelCase": {"stays": 1}}),
        sender: "user".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    let dto = MessageDto::from_message(&message, to_ms).unwrap();
    let body = serde_json::to_value(&dto).unwrap();
    assert_eq!(
        body,
        json!({
            "id": "m1", "content": "hello", "message_type": "text",
            "extra_metadata": {"camelCase": {"stays": 1}}, "sender": "user",
            "created_ms": 1_767_225_600_000_i64
        })
    );
    assert_eq!(dto.into_message().unwrap(), message);
}

#[test]
fn a_message_body_without_extra_metadata_reads_as_null_like_the_wire_type() {
    let dto: MessageDto = serde_json::from_value(json!({
        "id": "m", "content": "", "message_type": "text", "sender": "user", "created_ms": 0
    }))
    .unwrap();
    assert_eq!(
        dto.into_message().unwrap().extra_metadata,
        serde_json::Value::Null
    );
}
