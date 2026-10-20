//! Transcript stem sanitization and test-only legacy `.md` fixture paths.

#[cfg(test)]
use std::path::{Path, PathBuf};

/// Sanitize a user-supplied transcript stem so it never escapes the
/// `session_raw/` directory. Allows ASCII alphanumerics plus a small
/// punctuation set (`_`, `-`, `.`); every other byte is replaced with
/// `_`. Empty inputs fall back to `"session"`.
pub(super) fn sanitize_stem(stem: &str) -> String {
    let cleaned: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "session".to_string()
    } else {
        cleaned
    }
}

/// Given a `session_raw/{stem}.jsonl` path, derive the companion
/// `sessions/YYYY_MM_DD/{stem}.md` path. The date is taken from the
/// local clock at write time — fine for browsing because the source
/// of truth lives in the flat raw dir; the `.md` is purely a view.
///
/// Legacy `session_raw/DDMMYYYY/{stem}.jsonl` paths (still on disk
/// from older releases until they roll forward) keep their date
/// component when generating the companion so we don't accidentally
/// stamp old transcripts with today's date.
///
/// If no `session_raw` component is present (tests using a flat
/// tempdir), the companion sits alongside as a sibling `.md`.
#[cfg(test)]
pub(super) fn md_companion_path(jsonl_path: &Path) -> PathBuf {
    let components: Vec<_> = jsonl_path.components().collect();

    let raw_idx = components
        .iter()
        .position(|comp| matches!(comp, std::path::Component::Normal(s) if *s == "session_raw"));

    let Some(raw_idx) = raw_idx else {
        return jsonl_path.with_extension("md");
    };

    let mut out = PathBuf::new();
    for comp in &components[..raw_idx] {
        out.push(comp.as_os_str());
    }
    out.push("sessions");

    // Tail after `session_raw`:
    //   * Flat: ["{stem}.jsonl"] — prepend today's YYYY_MM_DD.
    //   * Legacy: ["DDMMYYYY", "{stem}.jsonl"] — keep the existing
    //     date dir so we don't relabel old transcripts.
    let tail = &components[raw_idx + 1..];
    if tail.len() <= 1 {
        out.push(chrono::Local::now().format("%Y_%m_%d").to_string());
    }
    for comp in tail {
        out.push(comp.as_os_str());
    }

    out.with_extension("md")
}

/// Test fixture: `{workspace}/session_raw/{stem}.jsonl`, creating the
/// directory.
#[cfg(test)]
pub(crate) fn resolve_keyed_transcript_path(
    workspace_dir: &Path,
    stem: &str,
) -> anyhow::Result<PathBuf> {
    let raw_dir = workspace_dir.join("session_raw");
    std::fs::create_dir_all(&raw_dir)?;
    Ok(raw_dir.join(format!("{}.jsonl", sanitize_stem(stem))))
}
