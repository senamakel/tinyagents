//! Built-in middleware library.
//!
//! This module collects the ready-to-use middleware that ship with the harness.
//! They use the model-wrap and lifecycle extension surfaces from
//! [`crate::middleware`]:
//!
//! - **Resilience (wrap)** — [`RetryMiddleware`], [`TimeoutMiddleware`],
//!   [`ModelFallbackMiddleware`], and [`RateLimitMiddleware`] implement the
//!   around-call [`ModelMiddleware`] trait and surround the real model call.
//! - **Policy / guard / observation (lifecycle)** —
//!   [`ToolAllowlistMiddleware`], [`DynamicToolSelectionMiddleware`],
//!   [`HumanApprovalMiddleware`], [`StructuredOutputValidatorMiddleware`],
//!   [`DynamicPromptMiddleware`], [`RedactionMiddleware`], and
//!   [`TracingMiddleware`] implement the lifecycle [`Middleware`] trait.
//!
//! - **Context ladder and loop guards** — [`ImageAwareMessageTrimMiddleware`],
//!   [`ArtifactIndexTocMiddleware`], [`FinalCallWrapUpMiddleware`],
//!   [`RepeatProgressMiddleware`] / [`RepeatEvictionObserver`],
//!   [`ArgRecoveryMiddleware`] and [`CredentialScrubMiddleware`]. All are
//!   generic over the run-context payload; host policy (tool names, the
//!   cleared-result placeholder, exemptions, per-tool scrubbing) is injected
//!   through constructor arguments.
//!
//! Type definitions live in `types`; this file holds the constructors and
//! trait impls. Tests live in `test.rs`.
//!
//! # Testability
//!
//! None of these middleware sleep on the wall clock in a way that tests cannot
//! control: [`RetryMiddleware`] sleeps on backoff only when its policy opts in
//! via [`RetryPolicy::with_backoff_sleep`] (off by default),
//! [`TimeoutMiddleware`] is exercised under `tokio::time` paused-time tests, and
//! [`RateLimitMiddleware`] takes an injectable clock and a configurable poll
//! interval so its wait loop can be driven deterministically.

mod types;

pub use types::*;

use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::context::{MiddlewareControl, RunConfig, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::events::AgentEvent;
use crate::ids::CallId;
use crate::middleware::{
    Middleware, MiddlewareModelOutcome, ModelHandler, ModelMiddleware, ToolInvocationIdentity,
};
use crate::retry::{RateLimiter, RetryPolicy, is_retryable};
use crate::structured::{StructuredExtractor, StructuredStrategy};
use tinyinference_llm::message::{ContentBlock, Message};
use tinyinference_llm::model::{ModelDelta, ModelRequest, ModelResponse, ResponseFormat};
use tinyinference_llm::tool::{ToolCall, ToolDelta, ToolSchema};
use tinytools::ToolResult;

mod arg_recovery;
mod artifact_toc;
mod budget;
mod compaction_pressure;
mod context;
mod credential_scrub;
mod image_trim;
mod memory_protocol;
mod memory_protocol_middleware;
mod observe;
mod policy_gate;
mod repeat_progress;
mod resilience;
mod tool_policy;
mod wrap_up;

pub use arg_recovery::ArgRecoveryMiddleware;
pub use artifact_toc::{
    ARTIFACT_INDEX_NAMESPACE, ArtifactIndexTocMiddleware, FOOTER_ALLOWANCE, NO_WINDOW_ALLOWANCE,
    split_input_allowance,
};
pub use credential_scrub::{
    CredentialScrubMiddleware, REDACTION_PLACEHOLDER, ToolScrubber, redaction_notice,
    scrub_with_notice,
};
pub use image_trim::{
    IMAGE_MARKER_TOKEN_COST, ImageAwareMessageTrimMiddleware, estimate_message_tokens,
    estimate_text_tokens, legacy_max_input_tokens,
};
pub use memory_protocol::{
    MEMORY_PROTOCOL_MARKER, MemoryOp, MemoryProtocolObservation, MemoryProtocolSpec,
    MemoryProtocolTracker, ModeTool,
};
pub use memory_protocol_middleware::MemoryProtocolMiddleware;
pub use policy_gate::{
    ApprovalGateMiddleware, ApprovalResolution, ApprovalResolver, DenialRenderer, GateVerdict,
    PolicyDecision, ToolCallPolicy, ToolPolicyGate, ToolPolicyGateMiddleware, record_approved,
};
pub use repeat_progress::{
    HaltSummarySlot, RepeatEvictionObserver, RepeatExemption, RepeatProgressMiddleware,
};
pub use wrap_up::{
    CapturedOutcomes, DEFAULT_CLEARED_PLACEHOLDER, FinalCallWrapUpMiddleware, OutcomesUnavailable,
};

#[cfg(test)]
#[path = "arg_recovery_tests.rs"]
mod arg_recovery_test;
#[cfg(test)]
#[path = "artifact_toc_tests.rs"]
mod artifact_toc_test;
#[cfg(test)]
#[path = "context_fold_tests.rs"]
mod context_fold_test;
#[cfg(test)]
#[path = "context_loop_tests.rs"]
mod context_loop_test;
#[cfg(test)]
#[path = "context_task_state_tests.rs"]
mod context_task_state_test;
#[cfg(test)]
#[path = "credential_scrub_tests.rs"]
mod credential_scrub_test;
#[cfg(test)]
#[path = "image_trim_tests.rs"]
mod image_trim_test;
#[cfg(test)]
#[path = "policy_gate_tests.rs"]
mod policy_gate_test;
#[cfg(test)]
#[path = "repeat_progress_tests.rs"]
mod repeat_progress_test;
#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;
#[cfg(test)]
#[path = "wrap_up_tests.rs"]
mod wrap_up_test;
