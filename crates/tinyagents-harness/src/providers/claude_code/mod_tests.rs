use super::*;

#[test]
fn chat_model_profile_advertises_streaming_without_native_tools() {
    let workspace = tempfile::tempdir().expect("workspace");
    let project = tempfile::tempdir().expect("project");
    let provider = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project.path().to_path_buf(),
        None,
    );

    let profile = provider.profile().expect("profile");
    assert_eq!(profile.provider.as_deref(), Some("claude-code"));
    assert_eq!(profile.model.as_deref(), Some("claude-sonnet-4-6"));
    assert!(!profile.tool_calling);
    assert!(!profile.parallel_tool_calls);
    assert!(profile.streaming);
    assert!(!profile.streaming_tool_chunks);
}

#[test]
fn thread_key_uses_caller_supplied_metadata() {
    let mut first = ModelRequest::new(vec![Message::user("hello")]);
    first.metadata = serde_json::json!({"thread_id": "thread-a"});
    let mut second = ModelRequest::new(vec![Message::user("hello")]);
    second.metadata = serde_json::json!({"thread_id": "thread-b"});
    assert_ne!(
        thread_key_from_request(&first),
        thread_key_from_request(&second)
    );
    assert_eq!(thread_key_from_request(&first), "thread-a");
}

#[test]
fn thread_key_without_id_is_ephemeral() {
    let request = ModelRequest::new(vec![Message::user("same text")]);
    let first = thread_key_from_request(&request);
    let second = thread_key_from_request(&request);
    assert_ne!(first, second);
    assert!(first.starts_with("ephemeral_"));
}

#[test]
fn every_system_message_is_coalesced_in_order() {
    let messages = vec![
        ChatMessage::system("base"),
        ChatMessage::user("question"),
        ChatMessage::system("middleware addition"),
    ];
    assert_eq!(
        coalesce_system_prompt(&messages).as_deref(),
        Some("base\n\nmiddleware addition")
    );
}

#[test]
fn cache_identity_includes_project_scope() {
    let workspace = tempfile::tempdir().expect("workspace");
    let project_a = tempfile::tempdir().expect("project a");
    let project_b = tempfile::tempdir().expect("project b");
    let first = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project_a.path().to_path_buf(),
        None,
    );
    let second = ClaudeCodeProvider::new(
        "claude-sonnet-4-6",
        PathBuf::from("claude"),
        workspace.path().to_path_buf(),
        project_b.path().to_path_buf(),
        None,
    );
    assert_ne!(first.cache_identity(), second.cache_identity());
}
