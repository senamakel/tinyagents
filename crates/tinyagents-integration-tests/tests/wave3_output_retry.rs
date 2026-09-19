//! Coverage for A3 — the output-validation retry loop.
//!
//! `docs/runtime-comparison/plan.md` Phase 2 item A3 adds
//! `RunPolicy::output_retry`, an `OutputValidator<State, Ctx>` trait
//! registered via `AgentHarness::with_output_validator`, and
//! `AgentRun::structured_as::<T>()`. This file exercises: a validator that
//! rejects once then accepts (two model calls, `AgentEvent::OutputRetry`
//! emitted), retry exhaustion failing the run, and the typed
//! `structured_as` accessor.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::runtime::{AgentHarness, OutputRetryPolicy, RunPolicy};
use tinyagents_harness::structured::OutputValidator;
use tinyagents_harness::testkit::EventRecorder;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ResponseFormat;
use tinyinference_llm::providers::MockModel;

#[derive(Debug, Deserialize, PartialEq)]
struct Answer {
    score: i64,
}

fn object_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": { "score": { "type": "integer" } },
        "required": ["score"]
    })
}

/// Rejects any value whose `score` is below `threshold` with `ModelRetry`.
struct MinScore {
    threshold: i64,
}

#[async_trait]
impl OutputValidator<()> for MinScore {
    async fn validate(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        output: &serde_json::Value,
    ) -> tinyagents_harness::Result<()> {
        let score = output.get("score").and_then(|v| v.as_i64()).unwrap_or(0);
        if score < self.threshold {
            return Err(TinyAgentsError::ModelRetry(format!(
                "score {score} is below the required {}",
                self.threshold
            )));
        }
        Ok(())
    }
}

#[tokio::test]
async fn validator_rejects_once_then_accepts() {
    // First answer fails validation (score too low); the model "fixes" it on
    // the re-ask.
    let scripted = MockModel::with_responses(vec![
        tinyinference_llm::model::ModelResponse::assistant(r#"{"score":1}"#),
        tinyinference_llm::model::ModelResponse::assistant(r#"{"score":99}"#),
    ]);
    let recorder = Arc::new(EventRecorder::new());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(scripted))
        .set_default_model("mock")
        .with_output_validator(Arc::new(MinScore { threshold: 50 }))
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema("answer", object_schema())),
            output_retry: OutputRetryPolicy {
                max_attempts: 2,
                ..OutputRetryPolicy::default()
            },
            ..RunPolicy::default()
        });

    let config = tinyagents_harness::context::RunConfig::new("validator-retry");
    let mut ctx: RunContext<()> = RunContext::new(config, ());
    ctx.events.subscribe(recorder.clone());
    let run = harness
        .invoke_in_context(&(), ctx, vec![Message::user("answer me")])
        .await
        .expect("the second attempt satisfies the validator");

    assert_eq!(run.model_calls, 2, "one rejected attempt plus one accepted retry");
    let structured = run.structured.expect("structured output is surfaced");
    assert_eq!(structured["score"], 99);

    let retries = recorder
        .events()
        .into_iter()
        .filter(|record| matches!(record.event, AgentEvent::OutputRetry { .. }))
        .count();
    assert_eq!(retries, 1, "exactly one OutputRetry event for the rejected attempt");
}

/// A validator that never accepts, to prove exhaustion fails the run instead
/// of retrying forever.
struct NeverAccepts {
    calls: AtomicUsize,
}

#[async_trait]
impl OutputValidator<()> for NeverAccepts {
    async fn validate(
        &self,
        _ctx: &mut RunContext<()>,
        _state: &(),
        _output: &serde_json::Value,
    ) -> tinyagents_harness::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(TinyAgentsError::ModelRetry("never good enough".to_string()))
    }
}

#[tokio::test]
async fn exhausting_output_retries_fails_the_run() {
    let scripted = MockModel::with_responses(vec![
        tinyinference_llm::model::ModelResponse::assistant(r#"{"score":1}"#),
        tinyinference_llm::model::ModelResponse::assistant(r#"{"score":2}"#),
    ]);
    let validator = Arc::new(NeverAccepts {
        calls: AtomicUsize::new(0),
    });

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", Arc::new(scripted))
        .set_default_model("mock")
        .with_output_validator(validator.clone())
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema("answer", object_schema())),
            output_retry: OutputRetryPolicy {
                max_attempts: 1,
                ..OutputRetryPolicy::default()
            },
            ..RunPolicy::default()
        });

    let err = harness
        .invoke_default(&(), vec![Message::user("answer me")])
        .await
        .expect_err("the validator never accepts, so retries exhaust and the run fails");

    assert!(
        matches!(err, TinyAgentsError::StructuredOutput(_)),
        "got {err:?}"
    );
    assert_eq!(
        validator.calls.load(Ordering::SeqCst),
        2,
        "the original attempt plus exactly max_attempts=1 retry"
    );
}

#[tokio::test]
async fn structured_as_deserializes_the_typed_output() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model(
            "mock",
            Arc::new(MockModel::constant(r#"{"score":7}"#)),
        )
        .set_default_model("mock")
        .with_policy(RunPolicy {
            default_response_format: Some(ResponseFormat::json_schema("answer", object_schema())),
            ..RunPolicy::default()
        });

    let run = harness
        .invoke_default(&(), vec![Message::user("answer me")])
        .await
        .expect("run succeeds");

    let answer: Answer = run.structured_as().expect("typed deserialize succeeds");
    assert_eq!(answer, Answer { score: 7 });
}

#[tokio::test]
async fn structured_as_errors_when_the_run_produced_no_structured_output() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("plain text")));

    let run = harness
        .invoke_default(&(), vec![Message::user("hi")])
        .await
        .expect("run succeeds");

    let err = run
        .structured_as::<Answer>()
        .expect_err("no structured output was produced");
    assert!(matches!(err, TinyAgentsError::StructuredOutput(_)));
}
