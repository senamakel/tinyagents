//! End-to-end coverage for canonical TinyTools policy declarations.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::middleware::ToolPolicyMiddleware;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::testkit::ScriptedModel;
use tinyinference_llm::message::Message;
use tinyinference_llm::model::ModelResponse;
use tinytools::{Tool, ToolPolicy, ToolResult, ToolSideEffects};

struct SafeTool;

#[async_trait]
impl Tool for SafeTool {
    fn name(&self) -> &str {
        "safe"
    }

    fn description(&self) -> &str {
        "A pure read-only lookup."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::read_only()
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("out"))
    }
}

struct PaymentTool;

#[async_trait]
impl Tool for PaymentTool {
    fn name(&self) -> &str {
        "charge"
    }

    fn description(&self) -> &str {
        "Moves money."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy::classified().with_side_effects(ToolSideEffects {
            payment: true,
            ..ToolSideEffects::default()
        })
    }

    async fn execute(&self, _arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("charged"))
    }
}

#[tokio::test]
async fn strict_policy_exposes_only_the_classified_safe_tool() {
    let scripted = Arc::new(ScriptedModel::new(vec![ModelResponse::assistant(
        "finished",
    )]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", scripted.clone());
    harness.register_tool(Arc::new(SafeTool));
    harness.register_tool(Arc::new(PaymentTool));
    harness.push_middleware(Arc::new(ToolPolicyMiddleware::strict(
        harness.tools().policies(),
    )));

    let run = harness
        .invoke_default(&(), vec![Message::user("go")])
        .await
        .expect("safe policy run succeeds");

    assert_eq!(run.text().as_deref(), Some("finished"));
    let requests = scripted.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "safe");
}

#[test]
fn payment_policy_remains_classified_and_effectful() {
    let policy = PaymentTool.policy();
    assert!(policy.classified);
    assert!(policy.side_effects.payment);
    assert!(policy.has_side_effects());
}
