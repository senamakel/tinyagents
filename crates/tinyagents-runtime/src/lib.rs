//! Stateful, host-neutral sessions for TinyAgents.
//!
//! A [`Session<C>`] owns conversation history and its durable transcript view.
//! Hosts supply the model/tool execution through [`SessionDriver`] and own all
//! policy, prompt composition, model selection, and authorization.  This
//! crate deliberately has no host configuration or product dependencies.

mod builder;
mod driver;
mod error;
mod hooks;
mod prefix;
mod session;
mod tools;
mod types;

pub use builder::SessionBuilder;
pub use driver::{DriverFailure, DriverOutcome, DriverRequest, HarnessDriver, SessionDriver};
pub use error::RuntimeError;
pub use hooks::{NoopSessionHooks, SessionHooks};
pub use prefix::PrefixSnapshot;
pub use session::Session;
pub use tinyagents_session::transcript::TranscriptPartial;
pub use tools::ToolSnapshot;
pub use types::{
    ResumeMode, SessionResume, SessionTerminal, SessionTurnOutcome, SessionTurnRequest,
    TranscriptTurnOptions, TurnOptions,
};

/// Converts between a host's lossless durable transcript dialect and the
/// inference messages driven by a [`SessionDriver<C>`].
///
/// It is object-safe so a host can keep its conversion and metadata ownership
/// outside this reusable runtime.  The runtime never narrows transcript data
/// itself; a codec must explicitly decide how every field is represented.
pub trait TranscriptCodec<C: Clone + Send + Sync + 'static = ()>: Send + Sync {
    /// Decodes a durable transcript for model execution.
    fn decode_history(
        &self,
        transcript: &tinyagents_session::transcript::SessionTranscript,
    ) -> Result<Vec<tinyinference_llm::message::Message>, RuntimeError>;

    /// Reconciles a model-history transition with the prior lossless durable
    /// rows.  `prior` must be treated as authoritative for fields absent from
    /// inference messages (provider metadata, raw arguments, reasoning, ids,
    /// and host extensions).  The returned rows are the complete next logical
    /// durable set; the history layer writes its delta atomically.
    fn reconcile(
        &self,
        prior: &[tinyagents_session::transcript::TranscriptMessage],
        previous: &[tinyinference_llm::message::Message],
        next: &[tinyinference_llm::message::Message],
        options: &TranscriptTurnOptions<C>,
    ) -> Result<Vec<tinyagents_session::transcript::TranscriptMessage>, RuntimeError>;
}

#[cfg(test)]
mod test;
