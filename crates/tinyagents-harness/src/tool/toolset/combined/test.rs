//! Tests for [`CombinedToolSet`], including the name-collision case
//! [`PrefixedToolSet`] exists to resolve.

use std::sync::Arc;

use serde_json::json;

use super::CombinedToolSet;
use crate::tool::ToolRegistry;
use crate::tool::toolset::test::{EchoTool, ctx};
use crate::tool::toolset::{PrefixedToolSet, ToolSet};

fn registry_with(name: &str) -> Arc<dyn ToolSet<(), ()>> {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new(name)));
    Arc::new(registry)
}

#[tokio::test]
async fn concatenates_every_member_tools_list() {
    let combined = CombinedToolSet::new(vec![registry_with("alpha"), registry_with("beta")]);
    let ctx = ctx();
    let mut names: Vec<_> = combined
        .tools(&ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    names.sort();
    assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
}

#[tokio::test]
async fn dispatches_to_the_owning_member() {
    let combined = CombinedToolSet::new(vec![registry_with("alpha"), registry_with("beta")]);
    let ctx = ctx();
    let result = combined
        .call("beta", json!({"text": "hi"}), &ctx)
        .await
        .expect("beta is owned by the second member");
    assert!(!result.is_error);
}

#[tokio::test]
async fn unowned_name_reports_tool_not_found() {
    let combined = CombinedToolSet::new(vec![registry_with("alpha")]);
    let ctx = ctx();
    let err = combined
        .call("missing", json!({}), &ctx)
        .await
        .expect_err("no member owns `missing`");
    assert!(matches!(
        err,
        crate::error::TinyAgentsError::ToolNotFound(_)
    ));
}

/// Two members that would otherwise both expose a `search` tool: prefixing
/// each before combining avoids the collision and keeps both reachable
/// under distinct, unambiguous names.
#[tokio::test]
async fn prefixed_members_avoid_a_name_collision_when_combined() {
    let first: Arc<dyn ToolSet<(), ()>> =
        Arc::new(PrefixedToolSet::new(registry_with("search"), "weather_"));
    let second: Arc<dyn ToolSet<(), ()>> =
        Arc::new(PrefixedToolSet::new(registry_with("search"), "news_"));
    let combined = CombinedToolSet::new(vec![first, second]);

    let ctx = ctx();
    let mut names: Vec<_> = combined
        .tools(&ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["news_search".to_string(), "weather_search".to_string()]
    );

    let weather = combined
        .call("weather_search", json!({"text": "sunny"}), &ctx)
        .await
        .expect("weather_search resolves to the first member's `search`");
    assert!(!weather.is_error);
    let news = combined
        .call("news_search", json!({"text": "breaking"}), &ctx)
        .await
        .expect("news_search resolves to the second member's `search`");
    assert!(!news.is_error);
}
