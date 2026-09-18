//! Claude Code CLI model adapter.

pub mod auth;
pub mod auth_status;
mod bridge;
pub mod driver;
mod event_mapper;
mod input_builder;
mod session_store;
pub mod settings;
mod stream_parser;
pub mod types;
pub mod version_check;

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bridge::{ChatMessage, ChatResponse, ProviderDelta};
use tinyinference_llm::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use tinyinference_llm::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
};
use tinyinference_llm::usage::Usage;
use tokio::sync::Semaphore;

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Provider string prefix used by host routing grammars.
pub const PROVIDER_PREFIX: &str = "claude-code:";
/// Maximum concurrent Claude subprocess turns per provider.
pub const MAX_CONCURRENT_TURNS: usize = 4;

/// Renders the JSONL stdin payload Claude Code receives for a model request.
///
/// This is exposed for hosts that need to verify their typed multimodal
/// conversion without depending on the adapter's internal bridge types.
pub fn render_request_stdin(request: &ModelRequest, is_new_session: bool) -> Vec<u8> {
    input_builder::build_stdin(&request_messages(request), is_new_session)
}

#[cfg(test)]
pub(crate) static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn test_set_env(key: impl AsRef<std::ffi::OsStr>, value: impl AsRef<std::ffi::OsStr>) {
    // SAFETY: every moved environment-mutating test serializes access through
    // `ENV_TEST_LOCK`; no provider work runs concurrently in those tests.
    unsafe { std::env::set_var(key, value) }
}

#[cfg(test)]
pub(crate) fn test_remove_env(key: impl AsRef<std::ffi::OsStr>) {
    // SAFETY: see `test_set_env`.
    unsafe { std::env::remove_var(key) }
}

/// Claude Code CLI-backed inference model.
#[derive(Clone)]
pub struct ClaudeCodeProvider {
    /// Model name passed to the CLI.
    pub model: String,
    bin_path: PathBuf,
    workspace_dir: PathBuf,
    project_dir: PathBuf,
    anthropic_api_key: Option<String>,
    semaphore: Arc<Semaphore>,
    session_store: Arc<session_store::SessionStore>,
    profile: ModelProfile,
    mcp_provider: Option<Arc<dyn driver::McpEndpointProvider>>,
}

impl std::fmt::Debug for ClaudeCodeProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClaudeCodeProvider")
            .field("model", &self.model)
            .field("bin_path", &self.bin_path)
            .field("workspace_dir", &self.workspace_dir)
            .field("project_dir", &self.project_dir)
            .finish_non_exhaustive()
    }
}

impl ClaudeCodeProvider {
    /// Construct a provider with an already-resolved CLI binary.
    pub fn new(
        model: impl Into<String>,
        bin_path: PathBuf,
        workspace_dir: PathBuf,
        project_dir: PathBuf,
        anthropic_api_key: Option<String>,
    ) -> Self {
        let model = model.into();
        Self {
            profile: ModelProfile {
                provider: Some("claude-code".into()),
                model: Some(model.clone()),
                tool_calling: true,
                parallel_tool_calls: true,
                streaming: true,
                streaming_tool_chunks: true,
                ..Default::default()
            },
            model,
            bin_path,
            project_dir,
            session_store: Arc::new(session_store::SessionStore::open(&workspace_dir)),
            workspace_dir,
            anthropic_api_key,
            semaphore: Arc::new(Semaphore::new(MAX_CONCURRENT_TURNS)),
            mcp_provider: None,
        }
    }

    /// Resolve the installed CLI and environment authentication.
    pub fn from_env(
        model: impl Into<String>,
        workspace_dir: PathBuf,
        project_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        match version_check::probe() {
            types::CliStatus::Ok { path, .. } => {
                let (_, key) = auth::resolve();
                Ok(Self::new(
                    model,
                    path.into(),
                    workspace_dir,
                    project_dir,
                    key,
                ))
            }
            types::CliStatus::NotInstalled => anyhow::bail!(
                "[claude-code] `claude` CLI not installed; require >= {}",
                types::MIN_CLI_VERSION
            ),
            types::CliStatus::Outdated {
                version,
                min_required,
                path,
            } => anyhow::bail!(
                "[claude-code] `claude` CLI at {path} is version {version}; require >= {min_required}"
            ),
            types::CliStatus::Unusable { path, reason } => {
                anyhow::bail!("[claude-code] `claude` CLI at {path} unusable: {reason}")
            }
        }
    }

    /// Attach a host-provided authenticated MCP endpoint resolver.
    pub fn with_mcp_provider(mut self, provider: Arc<dyn driver::McpEndpointProvider>) -> Self {
        self.mcp_provider = Some(provider);
        self
    }

    async fn run_chat(
        &self,
        messages: &[ChatMessage],
        stream: Option<&tokio::sync::mpsc::Sender<ProviderDelta>>,
        model_override: Option<&str>,
    ) -> anyhow::Result<ChatResponse> {
        let _permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| anyhow::anyhow!("claude-code semaphore closed: {error}"))?;
        let append_system_prompt = messages
            .iter()
            .find(|message| message.role == "system")
            .map(|message| message.content.clone());
        driver::run_turn(driver::TurnContext {
            bin_path: self.bin_path.clone(),
            workspace_dir: self.workspace_dir.clone(),
            project_dir: self.project_dir.clone(),
            thread_id: thread_key_from_messages(messages),
            model: model_override.unwrap_or(&self.model).to_string(),
            append_system_prompt,
            messages,
            session_store: self.session_store.clone(),
            stream,
            anthropic_api_key: self.anthropic_api_key.clone(),
            mcp_provider: self.mcp_provider.clone(),
        })
        .await
    }
}

fn request_messages(request: &ModelRequest) -> Vec<ChatMessage> {
    request
        .messages
        .iter()
        .map(|message| {
            let role = match message {
                Message::System(_) => "system",
                Message::User(_) => "user",
                Message::Assistant(_) => "assistant",
                Message::Tool(_) => "tool",
            };
            let content = match message {
                Message::System(value) => render_content(&value.content),
                Message::User(value) => render_content(&value.content),
                Message::Assistant(value) => render_content(&value.content),
                Message::Tool(value) => render_content(&value.content),
            };
            ChatMessage::new(role, content)
        })
        .collect()
}

fn render_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.clone()),
            ContentBlock::Image(image) => Some(format!("[OH_IMAGE:{}]", image.url)),
            ContentBlock::Json(value) | ContentBlock::ProviderExtension(value) => {
                Some(value.to_string())
            }
            ContentBlock::Thinking { text, .. } => Some(text.clone()),
            ContentBlock::RedactedThinking { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn model_response(response: ChatResponse) -> ModelResponse {
    let usage = response.usage.map(|value| Usage {
        input_tokens: value.input_tokens,
        output_tokens: value.output_tokens,
        total_tokens: value.input_tokens.saturating_add(value.output_tokens),
        cache_read_tokens: value.cached_input_tokens,
        cache_creation_tokens: value.cache_creation_tokens,
        reasoning_tokens: value.reasoning_tokens,
        charged_amount: None,
        context_window_tokens: None,
    });
    ModelResponse {
        message: AssistantMessage {
            id: None,
            content: response.text.into_iter().map(ContentBlock::Text).collect(),
            tool_calls: Vec::new(),
            usage,
        },
        usage,
        finish_reason: Some("stop".into()),
        raw: response
            .usage
            .filter(|value| value.charged_amount_usd > 0.0)
            .map(|value| serde_json::json!({"total_cost_usd": value.charged_amount_usd})),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn map_error(error: anyhow::Error) -> tinyinference_llm::Error {
    let message = format!("claude-code model call failed: {error}");
    if !matches!(
        tinyinference_llm::classify_provider_failure(None, None, &message),
        tinyinference_llm::ProviderFailureClass::NonRetryable
            | tinyinference_llm::ProviderFailureClass::NonRetryableRateLimit
    ) {
        tinyinference_llm::Error::Model(message)
    } else {
        tinyinference_llm::Error::Validation(message)
    }
}

#[async_trait]
impl ChatModel<()> for ClaudeCodeProvider {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    fn cache_identity(&self) -> Option<String> {
        Some(format!(
            "claude_code:{}:{}",
            self.bin_path.display(),
            self.model
        ))
    }
    async fn invoke(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let messages = request_messages(&request);
        self.run_chat(&messages, None, None)
            .await
            .map(model_response)
            .map_err(map_error)
    }
    async fn stream(
        &self,
        _state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelStream> {
        let provider = self.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = AbortOnDrop(tokio::spawn(async move {
            let _ = tx.send(ModelStreamItem::Started);
            let messages = request_messages(&request);
            let (delta_tx, mut delta_rx) = tokio::sync::mpsc::channel(64);
            let call = provider.run_chat(&messages, Some(&delta_tx), None);
            tokio::pin!(call);
            let response = loop {
                tokio::select! { delta = delta_rx.recv() => if let Some(delta) = delta { forward_delta(&tx, delta); }, response = &mut call => break response }
            };
            while let Ok(delta) = delta_rx.try_recv() {
                forward_delta(&tx, delta);
            }
            let terminal = response
                .map(model_response)
                .map(ModelStreamItem::Completed)
                .unwrap_or_else(|error| ModelStreamItem::Failed(map_error(error).to_string()));
            let _ = tx.send(terminal);
        }));
        let stream =
            futures::stream::unfold((rx, Some(handle)), |(mut receiver, handle)| async move {
                receiver.recv().await.map(|item| (item, (receiver, handle)))
            });
        Ok(ModelStream::new(Box::pin(stream)))
    }
}

fn forward_delta(
    sender: &tokio::sync::mpsc::UnboundedSender<ModelStreamItem>,
    delta: ProviderDelta,
) {
    let item = match delta {
        ProviderDelta::TextDelta { delta } => MessageDelta::text(delta),
        ProviderDelta::ThinkingDelta { delta } => MessageDelta::reasoning(delta),
    };
    let _ = sender.send(ModelStreamItem::MessageDelta(item));
}

fn thread_key_from_messages(messages: &[ChatMessage]) -> String {
    use sha2::{Digest, Sha256};
    let first = messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .unwrap_or("");
    let digest = Sha256::digest(first.as_bytes());
    format!(
        "hash_{:032x}",
        u128::from_be_bytes(digest[..16].try_into().expect("SHA-256 prefix"))
    )
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "pipeline_test.rs"]
mod pipeline_test;
