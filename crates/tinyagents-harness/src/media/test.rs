//! Tests for the media generation tools.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tinyinference_image::{MediaReference, MockImageGenerator};
use tinyinference_video::{JobState, MockVideoGenerator, MockVideoScript, WaitPolicy};
use tinytools::{Tool, ToolCallOptions, ToolRunContext, ToolTimeout, WorkspaceDescriptor};

use super::{GenerateImageTool, GenerateVideoTool, MediaOutput};

struct Workspace(WorkspaceDescriptor);

impl ToolRunContext for Workspace {
    fn workspace(&self) -> Option<&WorkspaceDescriptor> {
        Some(&self.0)
    }
}

fn workspace(root: &std::path::Path) -> Workspace {
    Workspace(WorkspaceDescriptor::new(root.to_path_buf()))
}

fn text(result: &tinytools::ToolResult) -> String {
    serde_json::to_string(result).unwrap()
}

#[tokio::test]
async fn image_tool_saves_into_the_workspace_and_reports_paths() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockImageGenerator::new());
    let tool = GenerateImageTool::new(generator.clone(), MediaOutput::new("/nonexistent"))
        .with_name("media_generate_image");
    let result = tool
        .execute_with_context(
            json!({ "prompt": "a comic", "n": "2", "aspectRatio": "landscape", "seed": 5 }),
            ToolCallOptions::default(),
            Some(&workspace(dir.path())),
        )
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    let saved: Vec<_> = std::fs::read_dir(dir.path().join("generated-media"))
        .unwrap()
        .collect();
    assert_eq!(saved.len(), 2);

    let request = generator.requests().pop().unwrap();
    assert_eq!(request.n, Some(2), "numeric strings are accepted");
    assert_eq!(
        request.aspect_ratio.as_deref(),
        Some("landscape"),
        "camelCase alias accepted"
    );
    assert_eq!(request.seed, Some(5));
    assert_eq!(tool.name(), "media_generate_image");
}

/// Regression (R2): a billed call that returned nothing must surface as an
/// error telling the model not to call again — the incident retried three
/// times and paid three times.
#[tokio::test]
async fn billed_non_delivery_tells_the_model_not_to_retry() {
    let dir = tempfile::tempdir().unwrap();
    let tool = GenerateImageTool::new(
        Arc::new(MockImageGenerator::returning_no_media()),
        MediaOutput::new(dir.path()),
    );
    let result = tool.execute(json!({ "prompt": "x" })).await.unwrap();
    assert!(result.is_error);
    let message = text(&result);
    assert!(
        message.contains("billed") && message.contains("do not retry"),
        "{message}"
    );
    assert!(
        message.contains("mock-request"),
        "request id is reported: {message}"
    );
}

#[tokio::test]
async fn image_tool_requires_a_prompt() {
    let tool = GenerateImageTool::new(
        Arc::new(MockImageGenerator::new()),
        MediaOutput::new("/tmp"),
    );
    let result = tool.execute(json!({})).await.unwrap();
    assert!(result.is_error);
}

#[tokio::test]
async fn references_resolve_inside_the_workspace_and_are_confined_to_it() {
    let dir = tempfile::tempdir().unwrap();
    // Create the referenced file so canonicalize succeeds
    let art_dir = dir.path().join("art");
    std::fs::create_dir(&art_dir).unwrap();
    std::fs::write(art_dir.join("ref.png"), b"fake image").unwrap();

    let generator = Arc::new(MockImageGenerator::new());
    let tool = GenerateImageTool::new(generator.clone(), MediaOutput::new("/nonexistent"));
    let context = workspace(dir.path());

    let result = tool
        .execute_with_context(
            json!({ "prompt": "x", "references": ["art/ref.png", "https://x.test/r.png"] }),
            ToolCallOptions::default(),
            Some(&context),
        )
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    let request = generator.requests().pop().unwrap();
    // Canonicalize the expected path to match what the reference function returns
    let canonical_ref = art_dir.join("ref.png").canonicalize().unwrap();
    assert_eq!(
        request.references,
        vec![
            MediaReference::Path(canonical_ref),
            MediaReference::Url("https://x.test/r.png".into()),
        ]
    );

    // Test that tilde-prefixed references stay within the workspace.
    // Tilde is not expanded by the reference parser, so it resolves as a literal
    // subdirectory inside the workspace.
    for escape in ["../secret.png", "/etc/passwd", "~/.ssh/id_rsa"] {
        let result = tool
            .execute_with_context(
                json!({ "prompt": "x", "references": [escape] }),
                ToolCallOptions::default(),
                Some(&context),
            )
            .await
            .unwrap();
        // All three should be refused: `..` is outside, `/etc/passwd` is outside,
        // and `~/.ssh/id_rsa` must be inside the workspace (tilde is literal, so it
        // would be <workspace>/~/.ssh/id_rsa, which also doesn't exist).
        assert!(
            result.is_error,
            "{escape} must be refused: {}",
            text(&result)
        );
    }
}

#[tokio::test]
async fn a_host_reference_policy_replaces_the_default_confinement() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockImageGenerator::new());

    // Test 1: a policy that rejects all references. Since the custom policy is
    // called after canonicalize, it rejects the canonical path.
    let output = MediaOutput::new("/nonexistent").with_reference_policy(Arc::new(|path| {
        Err(format!("host refused {}", path.display()))
    }));
    let tool = GenerateImageTool::new(generator.clone(), output);
    // Create a real file so canonicalize succeeds, then the policy rejects it
    std::fs::write(dir.path().join("a.png"), b"fake").unwrap();
    let context = workspace(dir.path());
    let result = tool
        .execute_with_context(
            json!({ "prompt": "x", "references": ["a.png"] }),
            ToolCallOptions::default(),
            Some(&context),
        )
        .await
        .unwrap();
    assert!(text(&result).contains("host refused"), "{}", text(&result));

    // Test 2: a policy that admits out-of-workspace paths. Create a file in the
    // test directory and a policy that allows it to be referenced.
    std::fs::write(dir.path().join("ref2.png"), b"fake").unwrap();
    let output =
        MediaOutput::new(dir.path()).with_reference_policy(Arc::new(|path| Ok(path.to_path_buf())));
    let tool = GenerateImageTool::new(generator, output);
    let result = tool
        .execute(json!({ "prompt": "x", "references": ["ref2.png"] }))
        .await
        .unwrap();
    // The custom policy allows the reference
    assert!(!result.is_error, "{}", text(&result));
}

fn fast() -> WaitPolicy {
    WaitPolicy::new(Duration::from_millis(1), Duration::from_secs(5))
}

#[tokio::test]
async fn video_tool_waits_for_delivery_and_saves_the_clip() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockVideoGenerator::new(MockVideoScript {
        polls: vec![
            (JobState::InProgress, 0),
            (JobState::Completed, 0),
            (JobState::Completed, 1),
        ],
        error: None,
    }));
    let tool = GenerateVideoTool::new(generator.clone(), MediaOutput::new(dir.path()))
        .with_wait_policy(fast());
    let result = tool
        .execute(json!({
            "prompt": "a lighthouse", "durationSeconds": 5, "aspect_ratio": "16:9",
            "first_frame": "https://x.test/f.png", "generate_audio": "true"
        }))
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    assert!(text(&result).contains("mock-job"));
    let saved: Vec<_> = std::fs::read_dir(dir.path().join("generated-media"))
        .unwrap()
        .collect();
    assert_eq!(saved.len(), 1);

    let request = generator.requests().pop().unwrap();
    assert_eq!(request.duration_s, Some(5), "legacy durationSeconds alias");
    assert_eq!(request.generate_audio, Some(true));
    assert!(request.first_frame.is_some());
    assert_eq!(tool.timeout_policy(&json!({})), ToolTimeout::Unbounded);
}

/// Regression (R2): a timed-out job names its id and points at resume instead
/// of a new (billed) submit; resuming collects it without a second submit.
#[tokio::test]
async fn video_timeout_names_the_job_and_resume_collects_it() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockVideoGenerator::new(MockVideoScript {
        polls: vec![(JobState::InProgress, 0)],
        error: None,
    }));
    let tool =
        GenerateVideoTool::new(generator.clone(), MediaOutput::new(dir.path())).with_wait_policy(
            WaitPolicy::new(Duration::from_millis(1), Duration::from_millis(10)),
        );
    let result = tool.execute(json!({ "prompt": "x" })).await.unwrap();
    assert!(result.is_error);
    let message = text(&result);
    assert!(
        message.contains("mock-job") && message.contains("do not resubmit"),
        "{message}"
    );

    let delivered = Arc::new(MockVideoGenerator::new(MockVideoScript::delivers()));
    let resume = GenerateVideoTool::new(delivered.clone(), MediaOutput::new(dir.path()))
        .with_wait_policy(fast());
    let result = resume
        .execute(json!({ "resume_job_id": "mock-job" }))
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    assert!(
        delivered.requests().is_empty(),
        "resume must not submit a new job"
    );
}

#[tokio::test]
async fn video_failure_surfaces_the_provider_reason() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockVideoGenerator::new(MockVideoScript {
        polls: vec![(JobState::Failed, 0)],
        error: Some("safety filter".into()),
    }));
    let tool =
        GenerateVideoTool::new(generator, MediaOutput::new(dir.path())).with_wait_policy(fast());
    let result = tool.execute(json!({ "prompt": "x" })).await.unwrap();
    assert!(result.is_error);
    assert!(text(&result).contains("safety filter"));
}

#[test]
fn media_tools_declare_billing_side_effects() {
    let tool = GenerateImageTool::new(
        Arc::new(MockImageGenerator::new()),
        MediaOutput::new("/tmp"),
    );
    let policy = tool.policy();
    assert!(
        policy.side_effects.payment
            && policy.side_effects.network
            && policy.side_effects.writes_files
    );
    assert!(!policy.runtime.idempotent, "a replay would bill again");
    assert!(tool.external_effect());
}

#[test]
fn media_errors_map_onto_harness_errors() {
    use crate::TinyAgentsError;
    let unsupported: TinyAgentsError = tinyinference_image::Error::Unsupported {
        model: "m".into(),
        field: "aspect_ratio".into(),
        value: "16:9".into(),
        allowed: vec!["1:1".into()],
    }
    .into();
    assert!(matches!(unsupported, TinyAgentsError::Validation(_)));
    let no_media: TinyAgentsError = tinyinference_image::Error::NoMedia { request_id: None }.into();
    assert!(matches!(no_media, TinyAgentsError::Model(ref m) if m.contains("do not retry")));
    let timeout: TinyAgentsError = tinyinference_video::Error::Timeout {
        job_id: "j".into(),
        waited_secs: 1,
        last_state: "pending".into(),
    }
    .into();
    assert!(matches!(timeout, TinyAgentsError::Timeout(ref m) if m.contains("j")));
}

#[tokio::test]
async fn malformed_string_options_are_rejected_not_ignored() {
    let generator = Arc::new(MockImageGenerator::new());
    let tool = GenerateImageTool::new(generator.clone(), MediaOutput::new("/tmp"));
    let result = tool
        .execute(json!({ "prompt": "x", "n": "two" }))
        .await
        .unwrap();
    assert!(result.is_error && text(&result).contains("`n` must be an integer"));
    assert!(
        generator.requests().is_empty(),
        "nothing billed on a malformed option"
    );

    let video = Arc::new(MockVideoGenerator::new(MockVideoScript::delivers()));
    let tool =
        GenerateVideoTool::new(video.clone(), MediaOutput::new("/tmp")).with_wait_policy(fast());
    let result = tool
        .execute(json!({ "prompt": "x", "generate_audio": "maybe" }))
        .await
        .unwrap();
    assert!(result.is_error && text(&result).contains("`generate_audio` must be true or false"));
    assert!(video.requests().is_empty());
}

#[test]
fn video_schema_exposes_size() {
    let tool = GenerateVideoTool::new(
        Arc::new(MockVideoGenerator::new(MockVideoScript::delivers())),
        MediaOutput::new("/tmp"),
    );
    assert!(tool.parameters_schema()["properties"].get("size").is_some());
}

#[tokio::test]
async fn negative_counts_are_rejected_but_negative_seeds_are_not() {
    let generator = Arc::new(MockImageGenerator::new());
    let tool = GenerateImageTool::new(generator.clone(), MediaOutput::new("/tmp"));
    let result = tool
        .execute(json!({ "prompt": "x", "n": -1 }))
        .await
        .unwrap();
    assert!(result.is_error && text(&result).contains("non-negative"));
    let result = tool
        .execute(json!({ "prompt": "x", "count": "-1" }))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(generator.requests().is_empty());

    let dir = tempfile::tempdir().unwrap();
    let tool = GenerateImageTool::new(generator.clone(), MediaOutput::new(dir.path()));
    let result = tool
        .execute(json!({ "prompt": "x", "seed": -7 }))
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    assert_eq!(generator.requests().pop().unwrap().seed, Some(-7));
}

#[test]
fn schemas_advertise_a_single_reference_string() {
    let image = GenerateImageTool::new(
        Arc::new(MockImageGenerator::new()),
        MediaOutput::new("/tmp"),
    );
    assert_eq!(
        image.parameters_schema()["properties"]["references"]["type"],
        json!(["array", "string"])
    );
    let video = GenerateVideoTool::new(
        Arc::new(MockVideoGenerator::new(MockVideoScript::delivers())),
        MediaOutput::new("/tmp"),
    );
    assert_eq!(
        video.parameters_schema()["properties"]["references"]["type"],
        json!(["array", "string"])
    );
}
