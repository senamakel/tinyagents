//! End-to-end provider-KV-cache coverage for harness-generated requests.
//!
//! The mock provider models a server-side KV cache: it accepts the harness's
//! `prompt_cache_key`, retains the corresponding rendered stable prefix, and
//! reports a hit only when a later model turn presents identical prefix bytes.
//! Two tool turns followed by a final answer therefore exercise three actual
//! model requests and prove that the tool schemas remain cacheable.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use tinyagents_harness::cache::PROMPT_CACHE_KEY_OPTION;
use tinyagents_harness::runtime::{AgentHarness, RunPolicy};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;
use tinytools::{Tool, ToolResult};

#[derive(Default)]
struct KvCacheMockServer {
    prefixes: Mutex<HashMap<String, Vec<u8>>>,
    requests: Mutex<Vec<ModelRequest>>,
    hits: Mutex<u32>,
}

impl KvCacheMockServer {
    fn hit_ratio(&self) -> f64 {
        let requests = self.requests.lock().expect("requests lock poisoned").len();
        let hits = *self.hits.lock().expect("hits lock poisoned") as usize;
        hits as f64 / requests as f64
    }

    fn stable_prefix(request: &ModelRequest) -> Vec<u8> {
        let system_messages = request
            .messages
            .iter()
            .take_while(|message| matches!(message, Message::System(_)))
            .collect::<Vec<_>>();
        serde_json::to_vec(&(system_messages, &request.tools))
            .expect("mock provider can serialize the stable prompt prefix")
    }
}

#[async_trait]
impl ChatModel<()> for KvCacheMockServer {
    fn cache_identity(&self) -> Option<String> {
        Some("mock-kv-provider".to_string())
    }

    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let key = request.provider_options[PROMPT_CACHE_KEY_OPTION]
            .as_str()
            .expect("harness must send a prompt-cache key for a protected prefix")
            .to_string();
        let prefix = Self::stable_prefix(&request);
        let hit = self
            .prefixes
            .lock()
            .expect("prefixes lock poisoned")
            .get(&key)
            .is_some_and(|cached| cached == &prefix);
        if hit {
            *self.hits.lock().expect("hits lock poisoned") += 1;
        } else {
            self.prefixes
                .lock()
                .expect("prefixes lock poisoned")
                .insert(key, prefix);
        }
        let call = self.requests.lock().expect("requests lock poisoned").len();
        self.requests
            .lock()
            .expect("requests lock poisoned")
            .push(request);

        if call < 2 {
            Ok(ModelResponse {
                message: AssistantMessage {
                    id: Some(format!("mock-{call}")),
                    content: Vec::new(),
                    tool_calls: vec![ToolCall::new(
                        format!("lookup-{call}"),
                        "lookup",
                        json!({ "query": format!("part-{call}") }),
                    )],
                    usage: Some(Usage::new(100, 10)),
                },
                usage: Some(Usage::new(100, 10)),
                finish_reason: Some("tool_calls".to_string()),
                raw: None,
                resolved_model: None,
                continue_turn: None,
                served_from_cache: false,
                correlation: None,
                resolved_route: None,
            })
        } else {
            Ok(ModelResponse::assistant("weekend planned"))
        }
    }
}

struct LookupTool;

#[async_trait]
impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "Looks up an itinerary item."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["query"],
            "properties": { "query": { "type": "string" } }
        })
    }

    async fn execute(&self, arguments: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success(format!(
            "result for {}",
            arguments["query"]
        )))
    }
}

#[tokio::test]
async fn harness_tool_turns_reuse_the_provider_kv_prompt_prefix() {
    let server = Arc::new(KvCacheMockServer::default());
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock-kv", server.clone());
    harness.register_tool(Arc::new(LookupTool));
    let mut policy = RunPolicy::default();
    policy.cache.protect_prompt_prefix = true;
    harness.with_policy(policy);

    let run = harness
        .invoke_default(
            &(),
            vec![
                Message::system("You are a careful travel planner."),
                Message::user("Plan my weekend."),
            ],
        )
        .await
        .expect("mock KV run succeeds");

    assert_eq!(run.text().as_deref(), Some("weekend planned"));
    assert_eq!(
        server
            .requests
            .lock()
            .expect("requests lock poisoned")
            .len(),
        3
    );
    assert_eq!(*server.hits.lock().expect("hits lock poisoned"), 2);
    assert!(
        (server.hit_ratio() - (2.0 / 3.0)).abs() < f64::EPSILON,
        "two of the three provider turns should reuse the stable prefix"
    );

    let requests = server.requests.lock().expect("requests lock poisoned");
    assert!(requests.iter().all(|request| {
        request
            .cache_segments
            .iter()
            .any(|segment| segment.id == "system" && segment.cacheable)
    }));
    assert!(requests.iter().all(|request| {
        request
            .cache_segments
            .iter()
            .any(|segment| segment.id == "tools" && segment.cacheable)
    }));
    assert!(
        requests
            .iter()
            .all(|request| request.prompt_fingerprint.is_some())
    );
    assert!(
        requests
            .windows(2)
            .all(|pair| pair[0].tools == pair[1].tools)
    );
}
