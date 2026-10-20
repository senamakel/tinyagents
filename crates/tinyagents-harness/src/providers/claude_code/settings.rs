//! Host-bound Claude Code settings stored through the TinyStorage KV port.

use serde::{Deserialize, Serialize};
use tinyagents_storage_kit::primitives::Kv;
use tinystoragedrivers_types::StorageError;

const SETTINGS_KEY: &str = "settings";
const OWNER: &str = "tinyagents-harness";
const NAMESPACE: &str = "claude_code_settings";

/// Persisted Claude Code provider settings. Defaults are the safe posture.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeCodeSettings {
    /// When true, Claude Code runs with `--permission-mode bypassPermissions`.
    #[serde(default)]
    pub full_access: bool,
}

pub fn store(
    docs: &std::sync::Arc<dyn tinystoragedrivers_api::DocumentStore>,
) -> Result<Kv, StorageError> {
    Kv::ns(docs, OWNER, NAMESPACE)
}

/// Load settings. Missing or malformed settings fail closed to the safe default.
pub async fn load(kv: &Kv) -> Result<ClaudeCodeSettings, StorageError> {
    match kv.get(SETTINGS_KEY).await {
        Ok(Some(settings)) => Ok(settings),
        Ok(None) => Ok(ClaudeCodeSettings::default()),
        Err(error) => {
            tracing::warn!(
                "[claude-code][settings] failed to read settings; using safe defaults: {error}"
            );
            Ok(ClaudeCodeSettings::default())
        }
    }
}

/// Persist settings to the host-bound KV namespace.
pub async fn save(kv: &Kv, settings: &ClaudeCodeSettings) -> Result<(), StorageError> {
    kv.put(SETTINGS_KEY, settings).await?;
    Ok(())
}

/// Import old `claude_code_settings.json` bytes only when no v2 value exists.
/// The host owns locating and reading the old file; repeated imports are safe.
pub async fn import_legacy(kv: &Kv, bytes: &[u8]) -> Result<bool, StorageError> {
    let settings: ClaudeCodeSettings = serde_json::from_slice(bytes)
        .map_err(|error| StorageError::serialization(error.to_string()))?;
    let imported = kv
        .update(SETTINGS_KEY, |current: Option<ClaudeCodeSettings>| {
            if current.is_some() {
                Ok(None)
            } else {
                Ok(Some(settings.clone()))
            }
        })
        .await?;
    Ok(imported.is_some())
}

#[cfg(test)]
#[path = "settings_tests.rs"]
mod tests;
