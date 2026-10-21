use super::*;

fn header() -> &'static str {
    r#"{"_meta":{"agent":"a","dispatcher":"native","created":"x","updated":"x","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0},"header_extra":true}"#
}

#[test]
fn rewrite_targets_final_context_and_preserves_other_records() {
    let source = format!(
        "{}\n{}\n{}\n{}\n{{\"role\":\n",
        header(),
        r#"{"role":"system","content":"old context","id":"old"}"#,
        r#"{"role":"user","content":"before compaction"}"#,
        r#"{"kind":"compaction","ts":"kept","replacement":[{"role":"assistant","content":"new context"},{"role":"system","content":"keep\n\n### PROFILE.md\n\nsecret\n\n### Next\nbody","request_id":"preserved","unknown":{"a":1}}],"custom":"retained"}"#,
    )
    .into_bytes();

    let (rewritten, changed) = rewrite_first_system_message_jsonl(&source, |content| {
        let cleaned = content.replace("### PROFILE.md\n\nsecret\n\n", "");
        if cleaned == *content {
            false
        } else {
            *content = cleaned;
            true
        }
    })
    .unwrap();

    assert!(changed);
    let source_lines: Vec<_> = source.split(|byte| *byte == b'\n').collect();
    let output_lines: Vec<_> = rewritten.split(|byte| *byte == b'\n').collect();
    assert_eq!(output_lines.len(), source_lines.len());
    assert_eq!(output_lines[0], source_lines[0]);
    assert_eq!(output_lines[1], source_lines[1]);
    assert_eq!(output_lines[2], source_lines[2]);
    assert_eq!(output_lines[4], source_lines[4]);

    let original: serde_json::Value = serde_json::from_slice(source_lines[3]).unwrap();
    let rewritten: serde_json::Value = serde_json::from_slice(output_lines[3]).unwrap();
    let mut expected = original;
    expected["replacement"][1]["content"] =
        serde_json::Value::String("keep\n\n### Next\nbody".into());
    assert_eq!(rewritten, expected);
}

#[test]
fn rewrite_uses_only_the_latest_compaction_context() {
    let source = format!(
        "{}\n{}\n{}\n{}\n",
        header(),
        r#"{"kind":"compaction","replacement":[{"role":"system","content":"discarded"}]}"#,
        r#"{"kind":"compaction","replacement":[{"role":"user","content":"current"}]}"#,
        r#"{"role":"system","content":"current system"}"#,
    )
    .into_bytes();

    let (rewritten, changed) = rewrite_first_system_message_jsonl(&source, |content| {
        *content = "updated".into();
        true
    })
    .unwrap();

    assert!(changed);
    let lines: Vec<_> = rewritten.split(|byte| *byte == b'\n').collect();
    assert_eq!(
        lines[1],
        source.split(|byte| *byte == b'\n').nth(1).unwrap()
    );
    assert_eq!(
        lines[2],
        source.split(|byte| *byte == b'\n').nth(2).unwrap()
    );
    let message: serde_json::Value = serde_json::from_slice(lines[3]).unwrap();
    assert_eq!(message["content"], "updated");
}

#[test]
fn rewrite_returns_identical_bytes_when_transform_declines() {
    let source = format!(
        "{}\n{{\"role\":\"system\",\"content\":\"unchanged\",\"x\":7}}\n",
        header()
    )
    .into_bytes();

    let (rewritten, changed) = rewrite_first_system_message_jsonl(&source, |_| false).unwrap();

    assert!(!changed);
    assert_eq!(rewritten, source);
}

#[test]
fn rewrite_rejects_an_invalid_header_but_preserves_later_malformed_lines() {
    assert!(rewrite_first_system_message_jsonl(b"", |_| true).is_err());
    assert!(rewrite_first_system_message_jsonl(b"not json\n", |_| true).is_err());

    let source = format!(
        "{}\n{{\"role\":\"system\",\"content\":\"before\"}}\nnot json\n",
        header()
    )
    .into_bytes();
    let (rewritten, changed) = rewrite_first_system_message_jsonl(&source, |content| {
        *content = "after".into();
        true
    })
    .unwrap();
    assert!(changed);
    assert!(rewritten.ends_with(b"not json\n"));
}

#[test]
fn rewrite_skips_interrupted_system_rows_and_preserves_crlf_and_partial_tail() {
    let source = format!(
        "{}\r\n{{\"role\":\"system\",\"content\":\"partial\",\"interrupted\":true}}\r\n{{\"role\":\"system\",\"content\":\"current\",\"id\":\"kept\"}}",
        header()
    )
    .into_bytes();

    let (rewritten, changed) = rewrite_first_system_message_jsonl(&source, |content| {
        *content = "rewritten".into();
        true
    })
    .unwrap();

    assert!(changed);
    let source_lines: Vec<_> = source.split_inclusive(|byte| *byte == b'\n').collect();
    let output_lines: Vec<_> = rewritten.split_inclusive(|byte| *byte == b'\n').collect();
    assert_eq!(output_lines.len(), 3);
    assert_eq!(output_lines[0], source_lines[0]);
    assert_eq!(output_lines[1], source_lines[1]);
    assert!(!output_lines[2].ends_with(b"\n"));
    let message: serde_json::Value = serde_json::from_slice(output_lines[2]).unwrap();
    assert_eq!(message["content"], "rewritten");
    assert_eq!(message["id"], "kept");
}
