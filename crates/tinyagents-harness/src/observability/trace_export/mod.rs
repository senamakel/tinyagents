//! Pure span/trace export helpers: the span data model, NDJSON serialization,
//! OTLP/HTTP JSON conversion in Langfuse conventions, Langfuse ingestion-batch
//! chunking and the pure parts of journal export.
//!
//! Nothing here performs I/O. Hosts own credentials, URL resolution, transport
//! and the progress-event collector that produces the spans; they pass an
//! [`ExportBrand`] so product identity strings stay out of this crate.

pub mod ingestion_batch;
pub mod ingestion_otlp;
#[cfg(feature = "langfuse")]
pub mod journal_export;
pub mod otlp;
pub mod serialize;
mod types;

pub use types::{RunType, SpanKind, SpanStatus, TraceContext, TraceSpan, trace_session_id};

/// Product identity stamped onto exported telemetry.
///
/// `product` becomes the OTLP `service.name`, the `<product>.agent` scope, the
/// `<product>_trace_id` trace metadata key and the `run.total` generation id
/// and source labels. `version` becomes the Langfuse release/version and the
/// `app.version` metadata.
#[derive(Debug, Clone, Copy)]
pub struct ExportBrand<'a> {
    /// Product slug, e.g. `"openhuman"`.
    pub product: &'a str,
    /// Product version string.
    pub version: &'a str,
}
