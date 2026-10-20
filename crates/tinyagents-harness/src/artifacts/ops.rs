//! Write, pointer-render, and handoff plumbing for the artifact-offload
//! convention.
//!
//! [`ArtifactOffload`] is the per-run writer: it resolves a target key via
//! [`super::paths::resolve_artifact_key`], applies the host redactor from
//! [`super::policy`], and stores the redacted body on a
//! [`BlobStore`] port. [`should_offload`]
//! and [`effective_offload_threshold`] decide *whether* a result should be
//! written; [`offload_oversized_result`] ties threshold decision, write, and
//! pointer rendering into the single entry point most callers use.
//! [`extract_artifact_paths`] and [`note_artifact_handoff`] are the reader
//! side: parsing pointers back out of a handoff payload and logging both ends
//! of the exchange. See [`super`] for the convention this module implements.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tinyagents_storage_kit::BlobStore;

use super::paths::{resolve_artifact_key, sanitize_component};
use super::policy::{ArtifactPathPolicy, ArtifactRedactor};
use super::types::{
    ABSTRACT_BUDGET_CHARS, ARTIFACT_POINTER_PREFIX, ArtifactKind, OUTPUTS_DIR, OffloadError,
    OffloadedArtifact, SCRATCH_DIR,
};

/// Whether a payload of `bytes` should be offloaded at `threshold_bytes`.
///
/// A zero threshold disables offload entirely, so a run can opt out without
/// threading an `Option` through every call site.
pub fn should_offload(bytes: usize, threshold_bytes: usize) -> bool {
    threshold_bytes > 0 && bytes > threshold_bytes
}

/// Condense `content` to at most `budget_chars` characters for a pointer's
/// abstract.
///
/// Prefers cutting at a line break, then at a word break, so the reader gets a
/// whole thought rather than a word sliced in half. Falls back to a hard
/// character cut only when neither boundary sits in the back half of the budget.
pub fn build_abstract(content: &str, budget_chars: usize) -> String {
    let trimmed = content.trim();
    if budget_chars == 0 {
        return String::new();
    }
    if trimmed.chars().count() <= budget_chars {
        return trimmed.to_string();
    }

    let mut head: String = trimmed.chars().take(budget_chars).collect();
    let floor = head.len() / 2;
    if let Some(idx) = head.rfind('\n').filter(|idx| *idx >= floor) {
        head.truncate(idx);
    } else if let Some(idx) = head.rfind(' ').filter(|idx| *idx >= floor) {
        head.truncate(idx);
    }
    format!("{}...", head.trim_end())
}

/// Render the text a worker hands back in place of an offloaded payload.
///
/// The first line is machine-readable ([`extract_artifact_paths`] parses it);
/// the rest is for the model reading the handoff.
///
/// `read_tool` names the tool the reader should call. It is a parameter rather
/// than a constant because tool names are host vocabulary — a crate that
/// hard-coded one would put a tool the host may not have into its prompts.
pub fn render_artifact_pointer(
    artifact: &OffloadedArtifact,
    abstract_text: &str,
    read_tool: &str,
) -> String {
    let redaction_note = if artifact.redacted {
        " Credential/PII redaction was applied before storage."
    } else {
        ""
    };
    format!(
        "{ARTIFACT_POINTER_PREFIX} kind={kind} path={path} bytes={bytes}\n\
         read_with: {read_tool} {{\"path\":\"{path}\"}}\n\
         note: The full result was stored as an artifact instead of being inlined. \
         Read the artifact for complete fidelity; the abstract below is not exhaustive.{redaction_note}\n\n\
         [abstract]\n{abstract_text}",
        kind = artifact.kind.as_str(),
        path = artifact.relative_path,
        bytes = artifact.stored_bytes,
    )
}

/// Pull every artifact path out of a handoff payload.
///
/// Scans for [`ARTIFACT_POINTER_PREFIX`] lines and returns their `path=` values
/// in encounter order, de-duplicated. A worker that inlined a pointer by hand —
/// following the prompt contract rather than being offloaded by the harness — is
/// picked up by exactly the same parse.
pub fn extract_artifact_paths(text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        if !line.starts_with(ARTIFACT_POINTER_PREFIX) {
            continue;
        }
        let Some(rest) = line.split(" path=").nth(1) else {
            continue;
        };
        // Split on the FIRST whitespace rather than `split_whitespace`, which
        // skips leading separators: an empty `path=` followed by another field
        // would otherwise yield that next field (`bytes=1`) as the "path".
        let path = rest.split(char::is_whitespace).next().unwrap_or_default();
        if path.is_empty() || found.iter().any(|existing| existing == path) {
            continue;
        }
        found.push(path.to_string());
    }
    found
}

/// Emit an `[artifact]` reference log for every path crossing a handoff, and
/// return how many were surfaced.
///
/// `stage` distinguishes the two ends of the same pointer so a run journal can
/// tell them apart instead of showing the identical line twice: the child
/// *recording* paths onto its outcome, and the parent *consuming* them. Use
/// [`HANDOFF_STAGE_RECORDED`] / [`HANDOFF_STAGE_CONSUMED`].
pub fn note_artifact_handoff(
    stage: &str,
    agent_id: &str,
    task_id: &str,
    paths: &[String],
) -> usize {
    for path in paths {
        tracing::info!(
            stage = %stage,
            agent_id = %agent_id,
            task_id = %task_id,
            path = %path,
            "[artifact] handoff carried an artifact path"
        );
    }
    paths.len()
}

/// Producing side: the child recorded these paths onto its outcome.
pub const HANDOFF_STAGE_RECORDED: &str = "recorded_by_child";

/// Consuming side: the parent took delivery of these paths.
pub const HANDOFF_STAGE_CONSUMED: &str = "consumed_by_parent";

/// Effective offload threshold for an agent whose definition caps its result at
/// `max_result_chars`.
///
/// A cap below the default would otherwise truncate the result before offload
/// ever fired, so the artifact would never reach disk. Offloading at the tighter
/// of the two means anything the cap would have cut is on disk first. Chars are
/// treated as a byte budget, which is conservative for multibyte text: it
/// offloads slightly earlier, never later.
pub fn effective_offload_threshold(
    default_threshold_bytes: usize,
    max_result_chars: Option<usize>,
) -> usize {
    match max_result_chars {
        Some(cap) if cap > 0 => default_threshold_bytes.min(cap),
        _ => default_threshold_bytes,
    }
}

/// Per-run writer for the offload convention.
///
/// Holds the blob store the artifacts land on, the host policies used for the
/// fail-closed key checks and redaction, and the identifiers that name
/// generated artifacts and tag the `[artifact]` logs.
#[derive(Clone)]
pub struct ArtifactOffload {
    blobs: Arc<dyn BlobStore>,
    policy: Option<Arc<dyn ArtifactPathPolicy>>,
    redactor: Option<Arc<dyn ArtifactRedactor>>,
    agent_id: String,
    task_id: String,
}

impl std::fmt::Debug for ArtifactOffload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactOffload")
            .field("blobs", &self.blobs)
            .field("policy", &self.policy.as_ref().map(|_| "set"))
            .field("redactor", &self.redactor.as_ref().map(|_| "set"))
            .field("agent_id", &self.agent_id)
            .field("task_id", &self.task_id)
            .finish()
    }
}

impl ArtifactOffload {
    /// A writer over `blobs` (a scoped tinystoragedrivers blob handle) for the
    /// given agent and task.
    ///
    /// Starts with no path policy and no redactor: add them with
    /// [`with_path_policy`](Self::with_path_policy) and
    /// [`with_redactor`](Self::with_redactor). A writer with no redactor stores
    /// bytes verbatim — see [`ArtifactRedactor`].
    pub fn new(
        blobs: Arc<dyn BlobStore>,
        agent_id: impl Into<String>,
        task_id: impl Into<String>,
    ) -> Self {
        Self {
            blobs,
            policy: None,
            redactor: None,
            agent_id: agent_id.into(),
            task_id: task_id.into(),
        }
    }

    /// Consults `policy` for host-internal key refusals.
    pub fn with_path_policy(mut self, policy: Arc<dyn ArtifactPathPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Scrubs bodies through `redactor` before they are stored.
    pub fn with_redactor(mut self, redactor: Arc<dyn ArtifactRedactor>) -> Self {
        self.redactor = Some(redactor);
        self
    }

    /// Convention-stable name for a worker's offloaded final result:
    /// `<agent_id>/<task_id>-result.md`, both components sanitized.
    pub fn default_result_name(&self) -> String {
        format!(
            "{}/{}-result.md",
            sanitize_component(&self.agent_id),
            sanitize_component(&self.task_id)
        )
    }

    /// Resolve `relative` to its blob key without writing.
    ///
    /// Exposed so callers can validate a model-supplied name before acting on it.
    pub fn resolve(&self, kind: ArtifactKind, relative: &str) -> Result<String, OffloadError> {
        resolve_artifact_key(self.policy.as_deref(), kind, relative)
    }

    /// Store `content` under `relative` in the convention namespace for `kind`.
    ///
    /// The body passes through the host redactor before it is stored.
    pub async fn write(
        &self,
        kind: ArtifactKind,
        relative: &str,
        content: &str,
    ) -> Result<OffloadedArtifact, OffloadError> {
        self.write_returning_stored(kind, relative, content)
            .await
            .map(|(artifact, _stored)| artifact)
    }

    /// Same as [`write`](Self::write), but also hands back the **stored**
    /// (redacted) body.
    ///
    /// Callers surfacing any part of the artifact back into a model's context
    /// must render it from this value, never from their own input: building a
    /// preview from the raw text would re-expose exactly the credentials the
    /// redactor just scrubbed out of the artifact.
    pub async fn write_returning_stored(
        &self,
        kind: ArtifactKind,
        relative: &str,
        content: &str,
    ) -> Result<(OffloadedArtifact, String), OffloadError> {
        let key = self.resolve(kind, relative)?;

        let stored = match self.redactor.as_deref() {
            Some(redactor) => redactor.redact(content),
            None => super::policy::Redacted::unchanged(content),
        };
        self.blobs
            .put(
                &key,
                stored.text.clone().into_bytes(),
                Some("text/markdown; charset=utf-8"),
            )
            .await?;

        let artifact = OffloadedArtifact {
            kind,
            relative_path: key,
            stored_bytes: stored.text.len(),
            original_bytes: content.len(),
            redacted: stored.changed,
        };

        tracing::info!(
            agent_id = %self.agent_id,
            task_id = %self.task_id,
            kind = artifact.kind.as_str(),
            path = %artifact.relative_path,
            original_bytes = artifact.original_bytes,
            stored_bytes = artifact.stored_bytes,
            redacted = artifact.redacted,
            "[artifact] stored worker artifact on the blob store"
        );

        Ok((artifact, stored.text))
    }

    /// Check that the filesystem copy of a relative artifact key is allowed.
    ///
    /// This must run before [`write`](Self::write) when the caller requires
    /// the blob store and filesystem copy to have the same path boundary.
    pub async fn validate_materialization_target(
        &self,
        root: &Path,
        key: &str,
    ) -> Result<PathBuf, OffloadError> {
        let (kind, relative) = split_key(key).ok_or_else(|| OffloadError::PathEscape {
            root: format!("{OUTPUTS_DIR}|{SCRATCH_DIR}"),
            path: key.to_string(),
        })?;
        let key = self.resolve(kind, relative)?;
        let canonical_root = tokio::fs::canonicalize(root)
            .await
            .map_err(|error| OffloadError::Materialization(error.to_string()))?;
        let target = canonical_root.join(&key);
        let parent = target
            .parent()
            .ok_or_else(|| OffloadError::Materialization("artifact target has no parent".into()))?;

        // Resolve the deepest existing ancestor so an existing symlink cannot
        // redirect the eventual write outside the chosen artifact root.
        let (canonical_parent, unresolved_parent) = canonicalize_existing_ancestor(parent).await?;
        if !canonical_parent.starts_with(&canonical_root) {
            return Err(OffloadError::PathEscape {
                root: canonical_root.display().to_string(),
                path: canonical_parent.display().to_string(),
            });
        }
        let resolved_parent = canonical_parent.join(unresolved_parent);
        let resolved_target = resolved_parent.join(target.file_name().unwrap_or_default());
        self.check_materialization_policy(&resolved_target)?;
        Ok(target)
    }

    fn check_materialization_policy(&self, target: &Path) -> Result<(), OffloadError> {
        if let Some(policy) = self.policy.as_deref() {
            if policy.is_internal_state(target) {
                return Err(OffloadError::InternalState {
                    path: target.display().to_string(),
                });
            }
            if policy
                .internal_root()
                .is_some_and(|root| target.starts_with(root))
            {
                return Err(OffloadError::InternalRoot {
                    path: target.display().to_string(),
                });
            }
        }
        Ok(())
    }

    /// Materialize a previously stored artifact under `root` using an atomic
    /// same-directory temporary file. Existing destination symlinks are
    /// replaced by rename rather than opened or followed.
    pub async fn materialize_to_fs(&self, root: &Path, key: &str) -> Result<PathBuf, OffloadError> {
        let target = self.validate_materialization_target(root, key).await?;
        let body = self.read(key).await?.ok_or_else(|| {
            OffloadError::Materialization("artifact is missing from blob store".into())
        })?;
        let parent = target
            .parent()
            .ok_or_else(|| OffloadError::Materialization("artifact target has no parent".into()))?;
        let checked = self.validate_materialization_target(root, key).await?;
        if checked != target {
            return Err(OffloadError::Materialization(
                "artifact target changed during validation".into(),
            ));
        }

        #[cfg(unix)]
        {
            let root = tokio::fs::canonicalize(root)
                .await
                .map_err(|error| OffloadError::Materialization(error.to_string()))?;
            let relative_parent = parent
                .strip_prefix(&root)
                .map_err(|error| OffloadError::Materialization(error.to_string()))?
                .to_path_buf();
            let file_name = target
                .file_name()
                .ok_or_else(|| {
                    OffloadError::Materialization("artifact target has no file name".into())
                })?
                .to_os_string();
            let bytes = body.into_bytes();
            tokio::task::spawn_blocking(move || {
                atomic_write_under_dir(&root, &relative_parent, &file_name, &bytes)
            })
            .await
            .map_err(|error| OffloadError::Materialization(error.to_string()))?
            .map_err(|error| OffloadError::Materialization(error.to_string()))?;
        }
        #[cfg(not(unix))]
        {
            let canonical_root = tokio::fs::canonicalize(root)
                .await
                .map_err(|error| OffloadError::Materialization(error.to_string()))?;
            let relative_parent = parent
                .strip_prefix(&canonical_root)
                .map_err(|error| OffloadError::Materialization(error.to_string()))?
                .to_path_buf();
            let file_name = target
                .file_name()
                .ok_or_else(|| {
                    OffloadError::Materialization("artifact target has no file name".into())
                })?
                .to_os_string();
            let bytes = body.into_bytes();
            tokio::task::spawn_blocking(move || {
                atomic_write_with_capability(&canonical_root, &relative_parent, &file_name, &bytes)
            })
            .await
            .map_err(|error| OffloadError::Materialization(error.to_string()))?
            .map_err(|error| OffloadError::Materialization(error.to_string()))?;
        }
        Ok(target)
    }

    /// Read an artifact back by the key a pointer carried
    /// ([`OffloadedArtifact::relative_path`]).
    ///
    /// The key passes the same fail-closed checks as a write, so a pointer a
    /// model rewrote by hand cannot address anything outside the two
    /// convention namespaces. `Ok(None)` means no such artifact.
    pub async fn read(&self, key: &str) -> Result<Option<String>, OffloadError> {
        let (kind, relative) = split_key(key).ok_or_else(|| OffloadError::PathEscape {
            root: format!("{OUTPUTS_DIR}|{SCRATCH_DIR}"),
            path: key.to_string(),
        })?;
        let key = self.resolve(kind, relative)?;
        let blob = self.blobs.get(&key).await?;
        Ok(blob.map(|blob| String::from_utf8_lossy(&blob.bytes).into_owned()))
    }

    /// The blob store artifacts are written to, for hosts that materialize or
    /// list them.
    pub fn blobs(&self) -> &Arc<dyn BlobStore> {
        &self.blobs
    }
}

#[cfg(any(not(unix), test))]
pub(super) fn atomic_write_with_capability(
    root: &Path,
    relative_parent: &Path,
    target_name: &std::ffi::OsStr,
    body: &[u8],
) -> std::io::Result<()> {
    use cap_std::ambient_authority;
    use cap_std::fs::{Dir, OpenOptions};
    use std::{
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let root = Dir::open_ambient_dir(root, ambient_authority())?;
    root.create_dir_all(relative_parent)?;
    let parent = root.open_dir(relative_parent)?;
    let mut temp = None;
    for _ in 0..32 {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let name = format!(".artifact-{}-{id}.tmp", std::process::id());
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        match parent.open_with(&name, &options) {
            Ok(mut file) => {
                if let Err(error) = file.write_all(body) {
                    let _ = parent.remove_file(&name);
                    return Err(error);
                }
                file.sync_all()?;
                temp = Some(name);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let temp =
        temp.ok_or_else(|| std::io::Error::other("could not allocate temporary artifact file"))?;
    if let Err(error) = parent.rename(&temp, &parent, target_name) {
        let _ = parent.remove_file(&temp);
        return Err(error);
    }
    Ok(())
}

async fn canonicalize_existing_ancestor(path: &Path) -> Result<(PathBuf, PathBuf), OffloadError> {
    let mut cursor = path.to_path_buf();
    let mut suffix = Vec::new();
    loop {
        match tokio::fs::canonicalize(&cursor).await {
            Ok(canonical) => {
                let mut unresolved = PathBuf::new();
                for component in suffix.iter().rev() {
                    unresolved.push(component);
                }
                return Ok((canonical, unresolved));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = cursor.file_name().ok_or_else(|| {
                    OffloadError::Materialization(format!(
                        "no existing parent for artifact target {}",
                        path.display()
                    ))
                })?;
                suffix.push(name.to_os_string());
                if !cursor.pop() {
                    return Err(OffloadError::Materialization(error.to_string()));
                }
            }
            Err(error) => return Err(OffloadError::Materialization(error.to_string())),
        }
    }
}

#[cfg(unix)]
pub(super) fn atomic_write_under_dir(
    root: &Path,
    relative_parent: &Path,
    target_name: &std::ffi::OsStr,
    body: &[u8],
) -> std::io::Result<()> {
    use rustix::fs::{AtFlags, Mode, OFlags, mkdirat, open, openat, renameat, unlinkat};
    use std::{
        io::Write,
        os::fd::OwnedFd,
        sync::atomic::{AtomicU64, Ordering},
    };

    let mut directory: OwnedFd = open(
        root,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )?;
    for component in relative_parent.components() {
        let std::path::Component::Normal(name) = component else {
            continue;
        };
        // Each segment is opened relative to the already pinned parent and
        // with NOFOLLOW. A replaced symlink therefore fails closed.
        let opened = openat(
            &directory,
            name,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        );
        directory = match opened {
            Ok(directory) => directory,
            Err(error) if error == rustix::io::Errno::NOENT => {
                match mkdirat(&directory, name, Mode::RUSR | Mode::WUSR | Mode::XUSR) {
                    Ok(()) | Err(rustix::io::Errno::EXIST) => {}
                    Err(error) => return Err(error.into()),
                }
                openat(
                    &directory,
                    name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
                    Mode::empty(),
                )?
            }
            Err(error) => return Err(error.into()),
        };
    }

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let mut temp_name = None;
    let mut file = None;
    for _ in 0..32 {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let name = format!(".artifact-{}-{id}.tmp", std::process::id());
        match openat(
            &directory,
            name.as_str(),
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::RUSR | Mode::WUSR,
        ) {
            Ok(fd) => {
                temp_name = Some(name);
                file = Some(std::fs::File::from(fd));
                break;
            }
            Err(error) if error == rustix::io::Errno::EXIST => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let temp_name = temp_name.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate temporary artifact file",
        )
    })?;
    let mut file = file.expect("temporary name and file are set together");
    if let Err(error) = file.write_all(body).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = unlinkat(&directory, temp_name.as_str(), AtFlags::empty());
        return Err(error);
    }
    drop(file);
    if let Err(error) = renameat(&directory, temp_name.as_str(), &directory, target_name) {
        let _ = unlinkat(&directory, temp_name.as_str(), AtFlags::empty());
        return Err(error.into());
    }
    Ok(())
}

/// The pre-v2 directory names, so a path a transcript quoted before the keys
/// moved under `harness/` still resolves to its (migrated) artifact.
const LEGACY_DIRS: [(ArtifactKind, &str); 2] = [
    (ArtifactKind::Output, "outputs"),
    (ArtifactKind::Scratch, "workspace"),
];

/// Split `harness/outputs/x` / `harness/workspace/x` (or the legacy `outputs/x` /
/// `workspace/x`) into its kind and the name below it.
fn split_key(key: &str) -> Option<(ArtifactKind, &str)> {
    let key = key.trim();
    [ArtifactKind::Output, ArtifactKind::Scratch]
        .into_iter()
        .map(|kind| (kind, kind.subdir()))
        .chain(LEGACY_DIRS)
        .find_map(|(kind, dir)| {
            key.strip_prefix(dir)
                .and_then(|rest| rest.strip_prefix('/'))
                .map(|rest| (kind, rest))
        })
}

/// Offload `output` when it exceeds `threshold_bytes`, returning the text the
/// parent should receive plus the artifact when one was written.
///
/// This is the deterministic half of the convention: it fires whether or not the
/// worker followed the prompt contract. **Every failure mode is soft** — the
/// caller gets its original payload back, and whatever summarisation or
/// truncation backstop it already had stays in charge.
pub async fn offload_oversized_result(
    output: String,
    offload: &ArtifactOffload,
    threshold_bytes: usize,
    read_tool: &str,
) -> (String, Option<OffloadedArtifact>) {
    if !should_offload(output.len(), threshold_bytes) {
        return (output, None);
    }

    let name = offload.default_result_name();
    match offload
        .write_returning_stored(ArtifactKind::Output, &name, &output)
        .await
    {
        Ok((artifact, stored)) => {
            // Build the abstract from the STORED (redacted) body, never from
            // `output`. The pointer goes straight into the parent's context, so
            // rendering it from the raw text would re-expose the very
            // credentials `write` just scrubbed out of the file.
            let abstract_text = build_abstract(&stored, ABSTRACT_BUDGET_CHARS);
            let pointer = render_artifact_pointer(&artifact, &abstract_text, read_tool);
            tracing::info!(
                path = %artifact.relative_path,
                inline_bytes = output.len(),
                pointer_bytes = pointer.len(),
                threshold_bytes,
                "[artifact] replaced oversized worker result with a path + abstract"
            );
            (pointer, Some(artifact))
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                inline_bytes = output.len(),
                threshold_bytes,
                "[artifact] offload refused; keeping the inline result (host backstop applies)"
            );
            (output, None)
        }
    }
}

/// Offload an oversized result only when its eventual filesystem target passes
/// the host's path policy, then materialize the stored (redacted) artifact.
///
/// The materialization target is checked before the first blob-store write.
/// Any refusal or I/O failure keeps the original inline result.
pub async fn offload_oversized_result_to_fs(
    output: String,
    offload: &ArtifactOffload,
    threshold_bytes: usize,
    read_tool: &str,
    materialize_root: &Path,
) -> (String, Option<OffloadedArtifact>) {
    if !should_offload(output.len(), threshold_bytes) {
        return (output, None);
    }
    let name = offload.default_result_name();
    let key = match offload.resolve(ArtifactKind::Output, &name) {
        Ok(key) => key,
        Err(error) => {
            tracing::warn!(error = %error, "[artifact] refusing unsafe filesystem target");
            return (output, None);
        }
    };
    if let Err(error) = offload
        .validate_materialization_target(materialize_root, &key)
        .await
    {
        tracing::warn!(error = %error, path = %key, "[artifact] refusing unsafe filesystem target before storage");
        return (output, None);
    }
    let (artifact, stored) = match offload
        .write_returning_stored(ArtifactKind::Output, &name, &output)
        .await
    {
        Ok(value) => value,
        Err(error) => {
            tracing::warn!(error = %error, "[artifact] offload refused; keeping inline result");
            return (output, None);
        }
    };
    if let Err(error) = offload
        .materialize_to_fs(materialize_root, &artifact.relative_path)
        .await
    {
        tracing::warn!(error = %error, path = %artifact.relative_path, "[artifact] materialization failed; keeping inline result");
        return (output, None);
    }
    let pointer = render_artifact_pointer(
        &artifact,
        &build_abstract(&stored, ABSTRACT_BUDGET_CHARS),
        read_tool,
    );
    (pointer, Some(artifact))
}
