use super::*;
use tinyinference_llm::tool::ToolCall;

#[test]
fn provider_constructs_with_default_config() {
    let config = ClaudeAgentSdkConfig::default();
    let provider = ClaudeAgentSdkProvider::new(config);
    assert_eq!(provider.config.binary, "claude");
    assert_eq!(provider.config.default_model, "claude-sonnet-4-6");
}

#[test]
fn config_default_disabled() {
    let config = ClaudeAgentSdkConfig::default();
    assert!(!config.enabled);
    assert!(config.max_budget_usd.is_none());
}

#[test]
fn large_request_is_delivered_over_stdin_instead_of_argv() {
    let system_prompt = "system instruction\n".repeat(2_500);
    assert!(system_prompt.len() > 32_767);

    let invocation = build_invocation(Some(&system_prompt), "hello", "claude-sonnet-4-6", None);

    assert_eq!(
        invocation.args,
        [
            "-p",
            "--model",
            "claude-sonnet-4-6",
            "--output-format",
            "stream-json",
            "--no-color"
        ]
    );
    assert!(
        !invocation
            .args
            .iter()
            .any(|arg| arg.contains(&system_prompt))
    );
    assert_eq!(
        invocation.stdin,
        format!("[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\nhello")
    );
}

#[test]
fn invocation_preserves_plain_message_and_budget_flags() {
    let invocation = build_invocation(None, "hello", "claude-opus-4-6", Some(1.25));

    assert_eq!(invocation.stdin, "hello");
    assert_eq!(
        &invocation.args[6..],
        ["--max-turns", "10", "--budget", "1.2500"]
    );
}

#[test]
fn spawn_error_message_includes_the_os_source() {
    let source = std::io::Error::from_raw_os_error(206);
    let error = spawn_error(r"C:\Users\test\.local\bin\claude.exe", source);

    assert!(error.to_string().contains("os error 206"));
    assert_eq!(
        error.chain().count(),
        2,
        "io::Error source must be preserved"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn provider_pipes_large_request_to_cli_stdin() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude");
    let staging = dir.path().join("claude.staging");
    std::fs::write(
        &staging,
        r#"#!/bin/sh
cat > "$0.stdin"
printf '%s\n' '{"type":"result","result":"captured","is_error":false}'
"#,
    )
    .expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::new(config);
    let system_prompt = "system instruction\n".repeat(2_500);

    let output = provider
        .invoke(
            &(),
            ModelRequest::new(vec![
                Message::system(&system_prompt),
                Message::user("hello"),
            ])
            .with_model("claude-sonnet-4-6"),
        )
        .await
        .expect("fake claude response")
        .text();

    assert_eq!(output, "captured");
    assert_eq!(
        std::fs::read_to_string(format!("{}.stdin", script.display())).expect("captured stdin"),
        format!("[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\nhello")
    );
}

#[tokio::test]
async fn provider_spawn_error_includes_the_os_source() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClaudeAgentSdkConfig {
        binary: dir.path().join("missing-claude").display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::new(config);

    let error = provider
        .invoke_cli(None, "hello", "claude-sonnet-4-6")
        .await
        .expect_err("missing binary must fail");

    assert!(error.to_string().contains("failed to spawn claude binary"));
    assert!(error.to_string().contains("os error"));
    assert_eq!(
        error.chain().count(),
        2,
        "io::Error source must be preserved"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn chat_model_uses_prompt_guided_protocol_and_model_override() {
    use std::os::unix::fs::PermissionsExt;
    use tinyinference_llm::tool::ToolSchema;

    let dir = tempfile::tempdir().expect("tempdir");
    let script = dir.path().join("claude");
    let staging = dir.path().join("claude.staging");
    std::fs::write(
            &staging,
            r#"#!/bin/sh
cat > "$0.stdin"
printf '%s\n' "$@" > "$0.args"
printf '%s\n' '{"type":"result","result":"Calling.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"needle\"}}</tool_call>","is_error":false}'
"#,
        )
        .expect("write fake claude");
    std::fs::set_permissions(&staging, std::fs::Permissions::from_mode(0o700))
        .expect("make fake claude executable");
    std::fs::rename(&staging, &script).expect("publish fake claude");

    let config = ClaudeAgentSdkConfig {
        binary: script.display().to_string(),
        ..Default::default()
    };
    let provider = ClaudeAgentSdkProvider::for_model(config, "profile-model");
    let mut prior_call = ModelResponse::assistant("calling").message;
    prior_call.tool_calls.push(ToolCall {
        id: "call-1".to_string(),
        name: "lookup".to_string(),
        arguments: serde_json::json!({"query": "needle"}),
        invalid: None,
    });
    let request = ModelRequest {
        messages: vec![
            Message::system("Base system"),
            Message::user("original question"),
            Message::Assistant(prior_call),
            Message::tool("call-1", "first result"),
            Message::tool("call-2", "second result"),
        ],
        tools: vec![ToolSchema::new(
            "lookup",
            "looks up data",
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
        )],
        model: Some("request-model".to_string()),
        ..Default::default()
    };

    let response = provider
        .invoke(&(), request)
        .await
        .expect("fake claude response");

    assert_eq!(response.text(), "Calling.");
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].name, "lookup");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        serde_json::json!({"query": "needle"})
    );
    let stdin =
        std::fs::read_to_string(format!("{}.stdin", script.display())).expect("captured stdin");
    assert!(stdin.contains("Base system"));
    assert!(
        stdin.contains("[USER]\noriginal question\n[/USER]"),
        "unexpected CLI stdin: {stdin:?}"
    );
    assert!(
        stdin.contains("[ASSISTANT]\ncalling\n<tool_call>"),
        "the assistant transcript must retain its text before the prior tool call: {stdin:?}"
    );
    assert!(stdin.contains("## Tool Use Protocol"));
    let prior_call = stdin
        .lines()
        .find_map(|line| {
            line.strip_prefix("<tool_call>")?
                .strip_suffix("</tool_call>")
        })
        .map(|body| serde_json::from_str::<serde_json::Value>(body).expect("tool call JSON"));
    assert_eq!(
        prior_call,
        Some(serde_json::json!({"name": "lookup", "arguments": {"query": "needle"}})),
        "prior structured tool call must survive in CLI stdin: {stdin:?}"
    );
    // Results are replayed under the protocol crate's envelope, keyed by the
    // call id they answer.
    assert!(
        stdin.contains("[Tool results]\n<tool_result id=\"call-1\">\nfirst result\n</tool_result>"),
        "unexpected CLI stdin: {stdin:?}"
    );
    assert!(stdin.contains("<tool_result id=\"call-2\">\nsecond result\n</tool_result>"));
    let args =
        std::fs::read_to_string(format!("{}.args", script.display())).expect("captured args");
    assert!(args.contains("request-model"));
    assert_eq!(
        provider
            .profile()
            .and_then(|profile| profile.model.as_deref()),
        Some("profile-model")
    );
    assert_eq!(
        provider
            .profile()
            .and_then(|profile| profile.provider.as_deref()),
        Some("claude-agent-sdk")
    );
}

#[test]
fn cache_identity_includes_selected_model() {
    let config = ClaudeAgentSdkConfig::default();
    let first = ClaudeAgentSdkProvider::for_model(config.clone(), "model-a");
    let second = ClaudeAgentSdkProvider::for_model(config, "model-b");

    assert_ne!(first.cache_identity(), second.cache_identity());
}

/// The CLI takes one `--system-prompt`, so every system message has to be
/// folded into it. Taking only the first dropped the artifact contents list and
/// the turn-cap wrap-up, which are appended as system messages exactly so the
/// context ladder cannot remove them (codex on #6068).
#[test]
fn every_system_message_reaches_the_cli_prompt() {
    let messages = vec![
        Message::system("you are a helpful agent"),
        Message::user("fetch the issues"),
        Message::system("## Stored results from this turn\n\n- `file_read` → `a.json`"),
    ];

    let system = coalesce_system_prompt(&messages).expect("a system prompt");

    assert!(
        system.contains("you are a helpful agent"),
        "the persona must survive: {system}"
    );
    assert!(
        system.contains("Stored results from this turn"),
        "a later system message must not be dropped: {system}"
    );
    assert!(
        system.find("you are a helpful agent") < system.find("Stored results"),
        "order must be preserved so the persona still leads: {system}"
    );
}

#[test]
fn a_conversation_with_no_system_message_yields_none() {
    let messages = vec![Message::user("hello")];
    assert!(coalesce_system_prompt(&messages).is_none());
    // Blank system messages are not a prompt either.
    let blank = vec![Message::system("   ")];
    assert!(coalesce_system_prompt(&blank).is_none());
}
