//! The fail-closed path gate for a [`WorkspaceDescriptor`].
//!
//! The descriptor and its lexical `allows` check live in `tinytools`, which
//! owns the tool vocabulary. What stays here is the half that needs this
//! crate: emitting a [`WorkspaceViolation`][crate::events::AgentEvent::WorkspaceViolation]
//! and returning this crate's error type. It is a free function rather than an
//! inherent method because the descriptor is now a foreign type.

use std::path::{Path, PathBuf};

use tinytools::WorkspaceDescriptor;

use crate::Result;
use crate::events::{AgentEvent, EventSink};

/// Fail-closed path gate to call *before* a tool touches `path`.
///
/// When the path is outside every allowed root, emits a
/// [`AgentEvent::WorkspaceViolation`] on `events` and returns a validation
/// error so the caller blocks the operation. Returns `Ok(())` when the path is
/// allowed.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Validation`][crate::error::TinyAgentsError::Validation]
/// when `path` lies outside the descriptor's root and trusted roots.
pub fn enforce_workspace_path(
    workspace: &WorkspaceDescriptor,
    path: &Path,
    events: &EventSink,
) -> Result<()> {
    if workspace.allows(path) && resolved_path_is_allowed(workspace, path) {
        return Ok(());
    }
    let rendered = path.display().to_string();
    events.emit(AgentEvent::WorkspaceViolation {
        path: rendered.clone(),
    });
    Err(crate::error::TinyAgentsError::Validation(format!(
        "path `{rendered}` is outside the allowed workspace roots"
    )))
}

/// Resolves existing filesystem components before checking containment so an
/// in-workspace symlink cannot redirect a tool to an untrusted root. For a new
/// path, its nearest existing ancestor is resolved and the remaining components
/// are appended; the eventual open must still use no-follow semantics where a
/// host supports them to close the final TOCTOU window.
fn resolved_path_is_allowed(workspace: &WorkspaceDescriptor, path: &Path) -> bool {
    let candidate = canonicalize_with_missing_tail(path);
    let Some(candidate) = candidate else {
        return false;
    };
    std::iter::once(&workspace.root)
        .chain(workspace.trusted_roots.iter())
        .filter_map(|root| std::fs::canonicalize(root).ok())
        .any(|root| candidate.starts_with(root))
}

fn canonicalize_with_missing_tail(path: &Path) -> Option<PathBuf> {
    let mut candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut missing = Vec::new();
    while !candidate.exists() {
        missing.push(candidate.file_name()?.to_os_string());
        candidate = candidate.parent()?.to_path_buf();
    }
    let mut resolved = std::fs::canonicalize(candidate).ok()?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Some(resolved)
}
