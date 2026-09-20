//! Tests for [`ApprovalRequiredToolSet`].

use std::sync::Arc;

use serde_json::json;

use super::ApprovalRequiredToolSet;
use crate::tool::ToolRegistry;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::{EchoTool, ctx};

fn registry() -> Arc<dyn ToolSet<(), ()>> {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("delete_everything")));
    registry.register(Arc::new(EchoTool::new("read_only")));
    Arc::new(registry)
}

#[tokio::test]
async fn matching_tools_declare_approval_required() {
    let wrapped = ApprovalRequiredToolSet::for_names(registry(), ["delete_everything"]);
    let ctx = ctx();
    let tools = wrapped.tools(&ctx).await.expect("tools");

    let flagged = tools
        .iter()
        .find(|tool| tool.name() == "delete_everything")
        .expect("delete_everything is exposed");
    assert!(flagged.policy().access.approval_required);
    assert!(flagged.policy().classified);

    let unflagged = tools
        .iter()
        .find(|tool| tool.name() == "read_only")
        .expect("read_only is exposed");
    assert!(!unflagged.policy().access.approval_required);
}

#[tokio::test]
async fn calling_a_flagged_tool_still_delegates() {
    let wrapped = ApprovalRequiredToolSet::for_names(registry(), ["delete_everything"]);
    let ctx = ctx();
    let result = wrapped
        .call("delete_everything", json!({"text": "gone"}), &ctx)
        .await
        .expect("this adaptor does not itself block execution");
    assert!(!result.is_error);
}
