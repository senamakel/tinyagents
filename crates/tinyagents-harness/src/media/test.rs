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
    assert_eq!(request.aspect_ratio.as_deref(), Some("landscape"), "camelCase alias accepted");
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
    assert!(message.contains("billed") && message.contains("do not retry"), "{message}");
    assert!(message.contains("mock-request"), "request id is reported: {message}");
}

#[tokio::test]
async fn image_tool_requires_a_prompt() {
    let tool = GenerateImageTool::new(Arc::new(MockImageGenerator::new()), MediaOutput::new("/tmp"));
    let result = tool.execute(json!({})).await.unwrap();
    assert!(result.is_error);
}

#[tokio::test]
async fn references_resolve_inside_the_workspace_and_are_confined_to_it() {
    let dir = tempfile::tempdir().unwrap();
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
    assert_eq!(
        request.references,
        vec![
            MediaReference::Path(dir.path().join("art/ref.png")),
            MediaReference::Url("https://x.test/r.png".into()),
        ]
    );

    for escape in ["../secret.png", "/etc/passwd", "~/.ssh/id_rsa"] {
        let result = tool
            .execute_with_context(
                json!({ "prompt": "x", "references": [escape] }),
                ToolCallOptions::default(),
                Some(&context),
            )
            .await
            .unwrap();
        // `~` is not expanded, so it stays inside the root; the other two are refused.
        if escape.starts_with('~') {
            continue;
        }
        assert!(result.is_error, "{escape} must be refused: {}", text(&result));
    }
}

#[tokio::test]
async fn a_host_reference_policy_replaces_the_default_confinement() {
    let generator = Arc::new(MockImageGenerator::new());
    let output = MediaOutput::new("/nonexistent")
        .with_reference_policy(Arc::new(|path| Err(format!("host refused {}", path.display()))));
    let tool = GenerateImageTool::new(generator, output);
    let result = tool
        .execute(json!({ "prompt": "x", "references": ["a.png"] }))
        .await
        .unwrap();
    assert!(text(&result).contains("host refused"));
}

fn fast() -> WaitPolicy {
    WaitPolicy::new(Duration::from_millis(1), Duration::from_secs(5))
}

#[tokio::test]
async fn video_tool_waits_for_delivery_and_saves_the_clip() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockVideoGenerator::new(MockVideoScript {
        polls: vec![(JobState::InProgress, 0), (JobState::Completed, 0), (JobState::Completed, 1)],
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
    let saved: Vec<_> = std::fs::read_dir(dir.path().join("generated-media")).unwrap().collect();
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
    let tool = GenerateVideoTool::new(generator.clone(), MediaOutput::new(dir.path()))
        .with_wait_policy(WaitPolicy::new(Duration::from_millis(1), Duration::from_millis(10)));
    let result = tool.execute(json!({ "prompt": "x" })).await.unwrap();
    assert!(result.is_error);
    let message = text(&result);
    assert!(message.contains("mock-job") && message.contains("do not resubmit"), "{message}");

    let delivered = Arc::new(MockVideoGenerator::new(MockVideoScript::delivers()));
    let resume = GenerateVideoTool::new(delivered.clone(), MediaOutput::new(dir.path()))
        .with_wait_policy(fast());
    let result = resume
        .execute(json!({ "resume_job_id": "mock-job" }))
        .await
        .unwrap();
    assert!(!result.is_error, "{}", text(&result));
    assert!(delivered.requests().is_empty(), "resume must not submit a new job");
}

#[tokio::test]
async fn video_failure_surfaces_the_provider_reason() {
    let dir = tempfile::tempdir().unwrap();
    let generator = Arc::new(MockVideoGenerator::new(MockVideoScript {
        polls: vec![(JobState::Failed, 0)],
        error: Some("safety filter".into()),
    }));
    let tool = GenerateVideoTool::new(generator, MediaOutput::new(dir.path())).with_wait_policy(fast());
    let result = tool.execute(json!({ "prompt": "x" })).await.unwrap();
    assert!(result.is_error);
    assert!(text(&result).contains("safety filter"));
}

#[test]
fn media_tools_declare_billing_side_effects() {
    let tool = GenerateImageTool::new(Arc::new(MockImageGenerator::new()), MediaOutput::new("/tmp"));
    let policy = tool.policy();
    assert!(policy.side_effects.payment && policy.side_effects.network && policy.side_effects.writes_files);
    assert!(!policy.runtime.idempotent, "a replay would bill again");
    assert!(tool.external_effect());
}
