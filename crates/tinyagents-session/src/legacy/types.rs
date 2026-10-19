//! Result types of the legacy imports.

use super::transcripts::TranscriptImportReport;

/// How a one-time import ended.
#[derive(Debug, Clone, PartialEq)]
pub enum OnceOutcome {
    /// No workspace path was supplied, so no file import was attempted.
    NotAttempted,
    /// A marker from an earlier launch says the import already ran.
    AlreadyImported,
    /// This pass ran. `marked` is false when some file did not parse, so the
    /// next launch tries again instead of treating the import as complete.
    Imported {
        /// What the pass did.
        report: TranscriptImportReport,
        /// Whether the completion marker was written.
        marked: bool,
    },
}

/// What [`rewrite_legacy_transcripts`](super::rewrite_legacy_transcripts) did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegacyRewriteReport {
    /// JSONL transcripts found.
    pub scanned: usize,
    /// Transcripts the rewrite changed and that were written back.
    pub rewritten: usize,
    /// Transcripts the rewrite left as they were.
    pub unchanged: usize,
    /// Transcripts that could not be read or written back.
    pub errors: usize,
}
