use super::*;
use serde_json::json;
use tinyinference_llm::message::MessageDelta;
use tinyinference_llm::tool::ToolDelta;
use tinyinference_llm::usage::Usage;

fn response_with(raw: Option<Value>, message_id: Option<&str>) -> ModelResponse {
    let mut response = ModelResponse::assistant("ok");
    response.raw = raw;
    response.message.id = message_id.map(str::to_owned);
    response
}

#[test]
fn served_by_reads_a_gateways_top_level_provider() {
    let response = response_with(Some(json!({"provider": "DeepInfra"})), None);
    assert_eq!(served_by(&response).as_deref(), Some("DeepInfra"));
}

#[test]
fn served_by_is_none_without_a_provider_string() {
    assert_eq!(served_by(&response_with(None, None)), None);
    assert_eq!(served_by(&response_with(Some(json!({})), None)), None);
    assert_eq!(
        served_by(&response_with(Some(json!({"provider": ""})), None)),
        None
    );
    assert_eq!(
        served_by(&response_with(
            Some(json!({"provider": {"name": "x"}})),
            None
        )),
        None
    );
}

#[test]
fn response_id_prefers_the_message_id() {
    let response = response_with(Some(json!({"id": "raw-id"})), Some("gen-1"));
    assert_eq!(response_id(&response).as_deref(), Some("gen-1"));
}

#[test]
fn response_id_falls_back_to_the_raw_id() {
    let response = response_with(Some(json!({"id": "chatcmpl-9"})), None);
    assert_eq!(response_id(&response).as_deref(), Some("chatcmpl-9"));
}

#[test]
fn response_id_is_none_when_empty_or_absent() {
    assert_eq!(response_id(&response_with(None, None)), None);
    assert_eq!(response_id(&response_with(None, Some(""))), None);
    assert_eq!(
        response_id(&response_with(Some(json!({"id": 7})), None)),
        None
    );
}

#[test]
fn output_items_are_text_reasoning_and_tool_calls() {
    let text = MessageDelta {
        text: "a".into(),
        ..MessageDelta::default()
    };
    let reasoning = MessageDelta {
        reasoning: "r".into(),
        ..MessageDelta::default()
    };
    let tool = ToolDelta {
        call_id: "c".into(),
        content: String::new(),
        tool_name: None,
        content_index: None,
    };
    let tool_in_message = MessageDelta {
        tool_call: Some(tool.clone()),
        ..MessageDelta::default()
    };
    assert!(is_output(&ModelStreamItem::MessageDelta(text)));
    assert!(is_output(&ModelStreamItem::MessageDelta(reasoning)));
    assert!(is_output(&ModelStreamItem::MessageDelta(tool_in_message)));
    assert!(is_output(&ModelStreamItem::ToolCallDelta(tool)));
}

#[test]
fn markers_usage_and_empty_deltas_are_not_output() {
    assert!(!is_output(&ModelStreamItem::Started));
    assert!(!is_output(&ModelStreamItem::UsageDelta(Usage::new(1, 1))));
    assert!(!is_output(&ModelStreamItem::MessageDelta(
        MessageDelta::default()
    )));
}
