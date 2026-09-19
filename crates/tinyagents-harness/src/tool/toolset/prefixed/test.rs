//! Tests for [`PrefixedToolSet`].

use std::sync::Arc;

use serde_json::json;

use super::PrefixedToolSet;
use crate::tool::ToolRegistry;
use crate::tool::toolset::ToolSet;
use crate::tool::toolset::test::{EchoTool, ctx};

#[tokio::test]
async fn prefixes_every_advertised_name() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("forecast")));
    let prefixed = PrefixedToolSet::new(Arc::new(registry), "weather_");

    let ctx = ctx();
    let names: Vec<_> = prefixed
        .tools(&ctx)
        .await
        .expect("tools")
        .into_iter()
        .map(|tool| tool.name().to_string())
        .collect();
    assert_eq!(names, vec!["weather_forecast".to_string()]);
}

#[tokio::test]
async fn strips_prefix_before_delegating_a_call() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("forecast")));
    let prefixed = PrefixedToolSet::new(Arc::new(registry), "weather_");

    let ctx = ctx();
    let result = prefixed
        .call("weather_forecast", json!({"text": "sunny"}), &ctx)
        .await
        .expect("prefixed name resolves to the inner tool");
    assert!(!result.is_error);
}

#[tokio::test]
async fn unprefixed_name_is_not_found() {
    let mut registry: ToolRegistry<(), ()> = ToolRegistry::new();
    registry.register(Arc::new(EchoTool::new("forecast")));
    let prefixed = PrefixedToolSet::new(Arc::new(registry), "weather_");

    let ctx = ctx();
    let err = prefixed
        .call("forecast", json!({"text": "sunny"}), &ctx)
        .await
        .expect_err("the unprefixed name was never advertised");
    assert!(matches!(err, crate::error::TinyAgentsError::ToolNotFound(_)));
}
