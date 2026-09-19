//! Tests for [`RenamedToolSet`].

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use super::RenamedToolSet;
use crate::tool::ToolRegistry;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::{EchoTool, ctx};

fn renamed_set() -> RenamedToolSet<(), ()> {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("search")));
    registry.register(Arc::new(EchoTool::new("untouched")));
    let mut renames = HashMap::new();
    renames.insert("search".to_string(), "web_search".to_string());
    RenamedToolSet::new(Arc::new(registry), renames)
}

#[tokio::test]
async fn renames_mapped_tools_and_leaves_others_untouched() {
    let renamed = renamed_set();
    let ctx = ctx();
    let mut names: Vec<_> = renamed
        .tools(&ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["untouched".to_string(), "web_search".to_string()]);
}

#[tokio::test]
async fn calls_the_renamed_tool_by_its_new_name() {
    let renamed = renamed_set();
    let ctx = ctx();
    let result = renamed
        .call("web_search", json!({"text": "hi"}), &ctx)
        .await
        .expect("renamed tool resolves");
    assert!(!result.is_error);
}

#[tokio::test]
async fn original_name_of_a_renamed_tool_is_no_longer_reachable() {
    let renamed = renamed_set();
    let ctx = ctx();
    let err = renamed
        .call("search", json!({"text": "hi"}), &ctx)
        .await
        .expect_err("the tool was renamed away from `search`");
    assert!(matches!(err, crate::error::TinyAgentsError::ToolNotFound(_)));
}
