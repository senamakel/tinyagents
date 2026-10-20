//! Shared configuration for the media generation tools.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tinyagents_storage_kit::BlobStore;
use tinyinference_image::MediaReference;

/// Sub-directory of the output root that generated media is written into.
pub const DEFAULT_MEDIA_SUBDIR: &str = "harness/generated_media";

/// Decides whether a local reference path may be read and sent to a provider,
/// returning the path to read. Relative paths arrive already joined to the
/// workspace root.
pub type ReferencePathPolicy = Arc<dyn Fn(&Path) -> Result<PathBuf, String> + Send + Sync>;

/// Where a tool writes artifacts and which local references it may read.
#[derive(Clone)]
pub struct MediaOutput {
    /// Root used when the run context carries no workspace.
    pub fallback_root: PathBuf,
    /// Directory under the root that artifacts are written into.
    pub subdir: String,
    /// Local-reference admission policy; `None` confines references to the
    /// workspace (or fallback) root.
    pub reference_policy: Option<ReferencePathPolicy>,
    /// Where generated artifacts are stored. Hosts pass their scoped blob
    /// handle when opening this output target.
    pub blobs: Arc<dyn BlobStore>,
}

impl MediaOutput {
    /// Resolves local references against `fallback_root` when no workspace is
    /// known and stores generated artifacts through the supplied host port.
    #[must_use]
    pub fn new(fallback_root: impl Into<PathBuf>, blobs: Arc<dyn BlobStore>) -> Self {
        Self {
            fallback_root: fallback_root.into(),
            subdir: DEFAULT_MEDIA_SUBDIR.to_owned(),
            reference_policy: None,
            blobs,
        }
    }

    /// Stores generated artifacts on `blobs` instead of the in-memory default.
    #[must_use]
    pub fn with_blobs(mut self, blobs: Arc<dyn BlobStore>) -> Self {
        self.blobs = blobs;
        self
    }

    /// Changes the artifact sub-directory.
    #[must_use]
    pub fn with_subdir(mut self, subdir: impl Into<String>) -> Self {
        self.subdir = subdir.into();
        self
    }

    /// Installs a host policy for local reference paths.
    #[must_use]
    pub fn with_reference_policy(mut self, policy: ReferencePathPolicy) -> Self {
        self.reference_policy = Some(policy);
        self
    }

    /// The root for this call: the run's workspace, else the fallback.
    pub(crate) fn root<'a>(&'a self, workspace: Option<&'a Path>) -> &'a Path {
        workspace.unwrap_or(&self.fallback_root)
    }

    /// Validates the artifact sub-directory and returns the blob key prefix
    /// (`<subdir>/`) generated artifacts are stored under.
    ///
    /// Checked before a generation request is made, so a bad host
    /// configuration is refused before the call can be billed.
    pub(crate) fn prefix(&self) -> Result<String, String> {
        let subdir = Path::new(&self.subdir);
        if self.subdir.trim().is_empty()
            || subdir
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || self.subdir.contains('\\')
        {
            return Err(format!(
                "artifact subdirectory `{}` must be a relative path below the output root",
                self.subdir
            ));
        }
        let joined = subdir
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        Ok(format!("{joined}/"))
    }

    /// Stores one artifact without replacing an existing blob, and returns its
    /// key (the path the tool reports).
    pub(crate) async fn persist(
        &self,
        prefix: &str,
        stem: &str,
        extension: &str,
        bytes: &[u8],
        content_type: &str,
    ) -> Result<String, String> {
        if !is_filename_component(stem) || !is_filename_component(extension) {
            return Err(
                "artifact name components must be non-empty filename components".to_owned(),
            );
        }
        let key = format!("{prefix}{stem}.{extension}");
        match self.blobs.head(&key).await {
            Ok(None) => {}
            Ok(Some(_)) => return Err(format!("artifact {key} already exists")),
            Err(error) => return Err(format!("artifact {key} could not be checked: {error}")),
        }
        self.blobs
            .put(&key, bytes.to_vec(), Some(content_type))
            .await
            .map_err(|error| format!("artifact {key} could not be stored: {error}"))?;
        Ok(key)
    }

    /// Converts a model-supplied reference string into a [`MediaReference`],
    /// resolving and admitting local paths.
    pub(crate) fn reference(
        &self,
        raw: &str,
        workspace: Option<&Path>,
    ) -> Result<MediaReference, String> {
        match MediaReference::parse(raw) {
            MediaReference::Path(path) => {
                let root = self.root(workspace);
                let joined = if path.is_absolute() {
                    path
                } else {
                    root.join(path)
                };
                // Canonicalize the path to resolve symlinks before checking confinement.
                // This prevents symlink attacks where a symlink inside the workspace
                // points to a location outside it.
                let canonical = joined.canonicalize().map_err(|e| {
                    format!(
                        "reference path {} could not be resolved: {}",
                        joined.display(),
                        e
                    )
                })?;
                // Canonicalize the root if it exists, for accurate symlink-safe
                // comparison. If the root is a fallback path that doesn't exist,
                // fall back to a lexical check after canonicalizing the path.
                let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
                let admitted = match &self.reference_policy {
                    Some(policy) => policy(&canonical)?,
                    None => confine(&canonical, &canonical_root)?,
                };
                Ok(MediaReference::Path(admitted))
            }
            other => Ok(other),
        }
    }
}

fn is_filename_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && Path::new(value)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

impl std::fmt::Debug for MediaOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaOutput")
            .field("fallback_root", &self.fallback_root)
            .field("subdir", &self.subdir)
            .field("reference_policy", &self.reference_policy.is_some())
            .field("blobs", &self.blobs)
            .finish()
    }
}

/// Default reference policy: the path must stay inside `root` when canonicalized,
/// so a model cannot read and upload arbitrary files via symlinks.
fn confine(path: &Path, root: &Path) -> Result<PathBuf, String> {
    // Both paths are already canonicalized by the caller, so we check the
    // resolved path against the resolved root.
    if !path.starts_with(root) {
        return Err(format!(
            "reference path {} is outside the workspace; use a URL or a file inside the workspace",
            path.display()
        ));
    }
    Ok(path.to_path_buf())
}

/// Reads the first present string among `keys`.
pub(crate) fn arg_str<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// Rejects a present-but-malformed option instead of silently dropping it.
///
/// The schemas accept numeric and boolean options as strings too (models emit
/// both), so a value such as `"n": "two"` passes schema validation; without
/// this check it would be ignored and the call would run — and bill — with
/// the default instead of what was asked.
pub(crate) fn check_option_types(
    args: &Value,
    unsigned: &[&str],
    signed: &[&str],
    booleans: &[&str],
) -> Result<(), String> {
    for key in unsigned {
        match args.get(*key) {
            None | Some(Value::Null) => {}
            Some(Value::Number(number)) if number.is_u64() => {}
            Some(Value::String(text)) if text.trim().parse::<u64>().is_ok() => {}
            Some(other) => {
                return Err(format!(
                    "`{key}` must be a non-negative integer, got {other}"
                ));
            }
        }
    }
    for key in signed {
        match args.get(*key) {
            None | Some(Value::Null) => {}
            Some(Value::Number(number)) if number.is_i64() => {}
            Some(Value::String(text)) if text.trim().parse::<i64>().is_ok() => {}
            Some(other) => return Err(format!("`{key}` must be an integer, got {other}")),
        }
    }
    for key in booleans {
        match args.get(*key) {
            None | Some(Value::Null) | Some(Value::Bool(_)) => {}
            Some(Value::String(text)) if text.trim().parse::<bool>().is_ok() => {}
            Some(other) => return Err(format!("`{key}` must be true or false, got {other}")),
        }
    }
    Ok(())
}

/// Rejects present string options that would otherwise be treated as absent.
pub(crate) fn check_string_options(args: &Value, keys: &[&str]) -> Result<(), String> {
    for key in keys {
        match args.get(*key) {
            None | Some(Value::Null) | Some(Value::String(_)) => {}
            Some(other) => return Err(format!("`{key}` must be a string, got {other}")),
        }
    }
    Ok(())
}

/// Rejects a present list unless every entry is a non-empty string.
pub(crate) fn check_string_lists(args: &Value, keys: &[&str]) -> Result<(), String> {
    for key in keys {
        match args.get(*key) {
            None | Some(Value::Null) | Some(Value::String(_)) => {}
            Some(Value::Array(items))
                if items
                    .iter()
                    .all(|item| item.as_str().is_some_and(|s| !s.trim().is_empty())) => {}
            Some(other) => {
                return Err(format!(
                    "`{key}` must be a string or an array of non-empty strings, got {other}"
                ));
            }
        }
    }
    Ok(())
}

/// Reads the first present unsigned integer among `keys` (numbers or numeric
/// strings, since models emit both).
pub(crate) fn arg_u64(args: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads the first present integer among `keys`.
pub(crate) fn arg_i64(args: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads the first present boolean among `keys`.
pub(crate) fn arg_bool(args: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| match args.get(*key)? {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

/// Reads string lists among `keys`, accepting a single string too.
pub(crate) fn arg_list(args: &Value, keys: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for key in keys {
        match args.get(*key) {
            Some(Value::Array(items)) => out.extend(
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
            ),
            Some(Value::String(item)) if !item.trim().is_empty() => {
                out.push(item.trim().to_owned())
            }
            _ => {}
        }
    }
    out
}
