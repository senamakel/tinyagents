//! Host-driven tool-call policy gate with an approval outcome.
//!
//! [`ToolPolicyMiddleware`] enforces *declarative* per-tool metadata
//! ([`tinytools::ToolPolicy`]). This module is its runtime counterpart: a host
//! supplies a [`ToolCallPolicy`] that inspects each concrete call (name,
//! arguments, the run's host context) and answers with a [`PolicyDecision`] —
//! allow, deny, or **require approval**. An [`ApprovalResolver`] is the seam a
//! host implements to actually ask a human (or a policy store) and to record
//! the outcome afterwards.
//!
//! Two middleware are provided:
//!
//! - [`ApprovalGateMiddleware`] — routes calls the resolver says need approval
//!   through [`ApprovalResolver::resolve`], short-circuits a denial as a
//!   model-consumable error result, and records the terminal outcome of an
//!   approved call.
//! - [`ToolPolicyGateMiddleware`] — consults a [`ToolCallPolicy`]. `Deny` always
//!   short-circuits. `RequireApproval` fails closed unless an
//!   [`ApprovalResolver`] is attached with
//!   [`ToolPolicyGateMiddleware::with_approval_resolver`].
//!
//! Hosts that interleave the gate with their own checks can call
//! [`ToolPolicyGate::check`] directly instead of installing the middleware.

use std::sync::Arc;

use async_trait::async_trait;

use crate::context::{MiddlewareControl, RunContext};
use crate::error::{Result, TinyAgentsError};
use crate::middleware::{MiddlewareToolOutcome, ToolHandler, ToolMiddleware};
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

/// What a [`ToolCallPolicy`] decided about one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The call may run.
    Allow,
    /// The call may run only after an approval handoff.
    RequireApproval {
        /// Why approval is needed (shown to the model when it fails closed).
        reason: String,
    },
    /// The call must not run.
    Deny {
        /// Why the call is refused.
        reason: String,
    },
}

impl PolicyDecision {
    /// Builds a [`PolicyDecision::RequireApproval`].
    pub fn require_approval(reason: impl Into<String>) -> Self {
        Self::RequireApproval {
            reason: reason.into(),
        }
    }

    /// Builds a [`PolicyDecision::Deny`].
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// The reason a fail-closed executor must refuse the call, or `None` for
    /// [`PolicyDecision::Allow`].
    pub fn blocking_reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::RequireApproval { reason } | Self::Deny { reason } => Some(reason.as_str()),
        }
    }
}

/// Host policy consulted before a tool call executes.
///
/// `Ctx` is the host's run-context payload ([`RunContext::data`]); the policy
/// receives the whole [`RunContext`] so it can read session, channel or
/// workspace identity without the harness knowing their shape.
#[async_trait]
pub trait ToolCallPolicy<Ctx: Send + Sync = ()>: Send + Sync {
    /// Stable policy name for logs and denial messages.
    fn name(&self) -> &str;

    /// Inspects one tool call and decides whether it can execute.
    async fn check(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> PolicyDecision;
}

/// How an [`ApprovalResolver`] settled one approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalResolution {
    /// Run the call. `ticket` is handed back to [`ApprovalResolver::record`]
    /// with the call's terminal result (an audit id, say); `None` records
    /// nothing.
    Allow {
        /// Opaque host identifier for the approved request.
        ticket: Option<String>,
    },
    /// Do not run the call; `reason` becomes the tool-error the model sees.
    Deny {
        /// Why the request was refused.
        reason: String,
    },
}

/// Host seam that asks for approval and audits the outcome.
#[async_trait]
pub trait ApprovalResolver<Ctx: Send + Sync = ()>: Send + Sync {
    /// Whether this call must be routed through [`Self::resolve`] at all.
    async fn requires_approval(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> bool;

    /// Settles the approval request (prompting a human, consulting a store,
    /// waiting for a decision or an expiry).
    async fn resolve(&self, ctx: &RunContext<Ctx>, call: &ToolCall) -> ApprovalResolution;

    /// Optionally admits an already-allowed call without prompting or
    /// requesting human approval, and attaches an audit or admission ticket.
    /// Calls to this method may run concurrently; implementations must be
    /// thread-safe. The harness does not serialize this callback with the
    /// approval prompt lock, which serializes calls to [`Self::resolve`] only.
    /// This callback must never widen the policy decision; returning `Deny`
    /// can only fail the call closed. The default preserves existing resolver
    /// behavior and leaves allowed calls ticket-free.
    async fn resolve_allowed(
        &self,
        _ctx: &RunContext<Ctx>,
        _call: &ToolCall,
    ) -> ApprovalResolution {
        ApprovalResolution::Allow { ticket: None }
    }

    /// Records a returned tool result for older resolvers. New resolvers
    /// should implement [`Self::record_terminal`] to observe every terminal
    /// outcome for an acquired ticket.
    fn record(&self, ticket: &str, result: &ToolResult);

    /// Records the terminal outcome of an approved call. This callback must
    /// not panic: it runs during unwinding when an acquired ticket is abandoned.
    /// `Abandoned` means the harness dropped the in-flight future; the tool may
    /// already have produced partial side effects.
    fn record_terminal(&self, ticket: &str, outcome: ApprovalExecutionOutcome<'_>) {
        if let ApprovalExecutionOutcome::Result(result) = outcome {
            self.record(ticket, result);
        }
    }
}

/// A terminal outcome observed after an approval ticket was acquired.
#[derive(Debug)]
#[non_exhaustive]
pub enum ApprovalExecutionOutcome<'a> {
    /// The wrapped middleware returned a tool result.
    Result(&'a ToolResult),
    /// The wrapped middleware returned a harness error other than cancellation.
    Failed(&'a TinyAgentsError),
    /// The wrapped middleware returned a typed cancellation error.
    Cancelled(&'a TinyAgentsError),
    /// The wrapped middleware returned control instead of a tool result.
    Control(&'a MiddlewareControl),
    /// The in-flight future was dropped before the harness observed a result.
    Abandoned,
}

struct TicketGuard<'a, Ctx: Send + Sync> {
    resolver: &'a dyn ApprovalResolver<Ctx>,
    ticket: Option<String>,
}

impl<'a, Ctx: Send + Sync> TicketGuard<'a, Ctx> {
    fn new(resolver: &'a dyn ApprovalResolver<Ctx>, ticket: Option<String>) -> Self {
        Self { resolver, ticket }
    }

    fn complete(&mut self, outcome: ApprovalExecutionOutcome<'_>) {
        if let Some(ticket) = self.ticket.take() {
            self.resolver.record_terminal(&ticket, outcome);
        }
    }
}

impl<Ctx: Send + Sync> Drop for TicketGuard<'_, Ctx> {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            self.resolver
                .record_terminal(&ticket, ApprovalExecutionOutcome::Abandoned);
        }
    }
}

/// What a host must do with a call after [`ToolPolicyGate::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateVerdict {
    /// Run the call. `ticket` is what the resolver handed back for the
    /// approved request; pass it to [`record_approved`] with the call's result.
    Proceed {
        /// Opaque host identifier for the approved request, if any.
        ticket: Option<String>,
    },
    /// The policy refused the call, or required approval and no resolver was
    /// supplied (fail closed). Carries the policy's own decision so the host
    /// can render its wording.
    Blocked(PolicyDecision),
    /// The policy required approval and the resolver refused it; `reason` is
    /// the resolver's text, shown to the model as-is.
    Refused {
        /// Why the approval request was refused.
        reason: String,
    },
}

/// A [`ToolCallPolicy`] plus the context-free glue shared by its consumers.
pub struct ToolPolicyGate<Ctx: Send + Sync = ()> {
    policy: Arc<dyn ToolCallPolicy<Ctx>>,
}

impl<Ctx: Send + Sync> Clone for ToolPolicyGate<Ctx> {
    fn clone(&self) -> Self {
        Self {
            policy: self.policy.clone(),
        }
    }
}

impl<Ctx: Send + Sync> ToolPolicyGate<Ctx> {
    /// Wraps a host policy.
    pub fn new(policy: Arc<dyn ToolCallPolicy<Ctx>>) -> Self {
        Self { policy }
    }

    /// The wrapped policy's name.
    pub fn policy_name(&self) -> &str {
        self.policy.name()
    }

    /// The effective decision for one call. When `waive_approval` is true a
    /// [`PolicyDecision::RequireApproval`] is downgraded to
    /// [`PolicyDecision::Allow`] (the host has already decided approvals do not
    /// apply to this call); `Deny` is never waived.
    pub async fn check(
        &self,
        ctx: &RunContext<Ctx>,
        call: &ToolCall,
        waive_approval: bool,
    ) -> PolicyDecision {
        let decision = self.policy.check(ctx, call).await;
        if waive_approval && matches!(decision, PolicyDecision::RequireApproval { .. }) {
            return PolicyDecision::Allow;
        }
        decision
    }
}

impl<Ctx: Send + Sync> ToolPolicyGate<Ctx> {
    /// Decides one call end to end for hosts that interleave the gate with
    /// their own checks: the policy decision (with the `waive_approval`
    /// downgrade of [`Self::check`]), then, for a
    /// [`PolicyDecision::Allow`] admission or [`PolicyDecision::RequireApproval`]
    /// settlement through `resolver`.
    /// Without a resolver an approval requirement fails closed as
    /// [`GateVerdict::Blocked`].
    pub async fn evaluate(
        &self,
        ctx: &RunContext<Ctx>,
        call: &ToolCall,
        waive_approval: bool,
        resolver: Option<&dyn ApprovalResolver<Ctx>>,
    ) -> GateVerdict {
        let decision = self.check(ctx, call, waive_approval).await;
        match (decision, resolver) {
            (PolicyDecision::Allow, Some(resolver)) => {
                match resolver.resolve_allowed(ctx, call).await {
                    ApprovalResolution::Allow { ticket } => GateVerdict::Proceed { ticket },
                    ApprovalResolution::Deny { reason } => GateVerdict::Refused { reason },
                }
            }
            (PolicyDecision::Allow, None) => GateVerdict::Proceed { ticket: None },
            (PolicyDecision::RequireApproval { .. }, Some(resolver)) => {
                match resolver.resolve(ctx, call).await {
                    ApprovalResolution::Allow { ticket } => GateVerdict::Proceed { ticket },
                    ApprovalResolution::Deny { reason } => GateVerdict::Refused { reason },
                }
            }
            (decision, _) => GateVerdict::Blocked(decision),
        }
    }
}

/// Records a returned tool result or control outcome. A no-op when there is no
/// ticket. Middleware should use the terminal guard to include errors and
/// cancellation.
pub fn record_approved<Ctx: Send + Sync>(
    resolver: &dyn ApprovalResolver<Ctx>,
    ticket: Option<&str>,
    outcome: &MiddlewareToolOutcome,
) {
    if let Some(ticket) = ticket {
        match outcome {
            MiddlewareToolOutcome::Result(result) => {
                resolver.record_terminal(ticket, ApprovalExecutionOutcome::Result(result))
            }
            MiddlewareToolOutcome::Command { control } => {
                resolver.record_terminal(ticket, ApprovalExecutionOutcome::Control(control))
            }
        }
    }
}

async fn run_approved<State: Send + Sync, Ctx: Send + Sync>(
    resolver: &dyn ApprovalResolver<Ctx>,
    prompt_lock: &tokio::sync::Mutex<()>,
    ctx: &RunContext<Ctx>,
    state: &State,
    call: ToolCall,
    next: ToolHandler<'_, State, Ctx>,
) -> Result<MiddlewareToolOutcome> {
    let resolution = {
        // Calls of one batch overlap, but a host routes one prompt at a time
        // (e.g. per chat thread), so only the interactive part is serialised.
        let _prompting = prompt_lock.lock().await;
        resolver.resolve(ctx, &call).await
    };
    match resolution {
        ApprovalResolution::Deny { reason } => {
            Ok(MiddlewareToolOutcome::Result(ToolResult::error(reason)))
        }
        ApprovalResolution::Allow { ticket } => {
            let mut guard = TicketGuard::new(resolver, ticket);
            match next.run(ctx, state, call).await {
                Ok(outcome) => {
                    match &outcome {
                        MiddlewareToolOutcome::Result(result) => {
                            guard.complete(ApprovalExecutionOutcome::Result(result))
                        }
                        MiddlewareToolOutcome::Command { control } => {
                            guard.complete(ApprovalExecutionOutcome::Control(control))
                        }
                    }
                    Ok(outcome)
                }
                Err(error) => {
                    if matches!(error, TinyAgentsError::Cancelled) {
                        guard.complete(ApprovalExecutionOutcome::Cancelled(&error));
                    } else {
                        guard.complete(ApprovalExecutionOutcome::Failed(&error));
                    }
                    Err(error)
                }
            }
        }
    }
}

/// Tool-wrap middleware that routes calls through an [`ApprovalResolver`].
pub struct ApprovalGateMiddleware<Ctx: Send + Sync = ()> {
    label: &'static str,
    resolver: Arc<dyn ApprovalResolver<Ctx>>,
    /// Serialises [`ApprovalResolver::resolve`] across the overlapping calls
    /// of a concurrent batch. Calls that need no approval never take it.
    prompt_lock: tokio::sync::Mutex<()>,
}

impl<Ctx: Send + Sync> ApprovalGateMiddleware<Ctx> {
    /// Creates the middleware under a stable event `label`.
    pub fn new(label: &'static str, resolver: Arc<dyn ApprovalResolver<Ctx>>) -> Self {
        Self {
            label,
            resolver,
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolMiddleware<State, Ctx>
    for ApprovalGateMiddleware<Ctx>
{
    fn name(&self) -> &str {
        self.label
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        if !self.resolver.requires_approval(ctx, &call).await {
            return next.run(ctx, state, call).await;
        }
        run_approved(
            self.resolver.as_ref(),
            &self.prompt_lock,
            ctx,
            state,
            call,
            next,
        )
        .await
    }
}

/// Renders the model-facing text for a blocked call: tool name, policy name,
/// the blocking decision.
pub type DenialRenderer = Arc<dyn Fn(&ToolCall, &str, &PolicyDecision) -> String + Send + Sync>;

/// Host wording for each part of a structured policy denial.
///
/// The harness supplies stable section labels; hosts supply the actual
/// product-specific copy, including any typed marker they need to preserve.
/// [`StructuredDenialRenderOptions`] lets a host select its reason label and
/// punctuation without replacing the shared renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredDenialParts {
    /// What was blocked, including a marker when downstream policy needs it.
    pub blocked: String,
    /// Why the action was blocked.
    pub why: String,
    /// A permitted next step.
    pub workaround: String,
    /// Instruction to relay the result instead of silently stopping.
    pub relay: String,
}

/// Host-selected labels and punctuation for structured policy denials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredDenialRenderOptions<'a> {
    /// Label for the denial's explanation section.
    pub reason_label: &'a str,
    /// Text between the workaround and the instruction to relay the denial.
    pub relay_separator: &'a str,
}

impl<'a> Default for StructuredDenialRenderOptions<'a> {
    fn default() -> Self {
        Self {
            reason_label: "Why",
            relay_separator: ". ",
        }
    }
}

impl StructuredDenialParts {
    /// Creates the four host-authored sections of a denial.
    pub fn new(
        blocked: impl Into<String>,
        why: impl Into<String>,
        workaround: impl Into<String>,
        relay: impl Into<String>,
    ) -> Self {
        Self {
            blocked: blocked.into(),
            why: why.into(),
            workaround: workaround.into(),
            relay: relay.into(),
        }
    }

    /// Renders the shared blocked/why/workaround/relay structure.
    pub fn render(&self) -> String {
        self.render_with_options(&StructuredDenialRenderOptions::default())
    }

    /// Renders the shared denial structure with host-selected labels and
    /// punctuation. The default options preserve [`Self::render`] byte for
    /// byte.
    pub fn render_with_options(&self, options: &StructuredDenialRenderOptions<'_>) -> String {
        format!(
            "Blocked: {}. {}: {}. Workaround: {}{}{}",
            self.blocked,
            options.reason_label,
            self.why,
            self.workaround,
            options.relay_separator,
            self.relay
        )
    }
}

/// Creates a [`DenialRenderer`] with shared structure and host-owned wording.
pub fn structured_denial_renderer(
    wording: impl Fn(&ToolCall, &str, &PolicyDecision) -> StructuredDenialParts + Send + Sync + 'static,
) -> DenialRenderer {
    Arc::new(move |call, policy, decision| wording(call, policy, decision).render())
}

/// Optional per-session ceiling consulted before the builder-level tool policy.
/// The host supplies it from its own session snapshot; an `Allow` only means
/// this ceiling permits the call, after which the primary policy still runs.
pub type SessionToolCeiling<Ctx> = Arc<dyn ToolCallPolicy<Ctx>>;

/// Host callback that filters candidate tool names for a session-scoped
/// listing such as `use_skill`. It receives only names already available to the
/// current run and must return a subset of those names.
pub type SessionToolListing<Ctx> =
    Arc<dyn Fn(&RunContext<Ctx>, &[String]) -> Vec<String> + Send + Sync>;

/// Optional host response applied after both policy gates allow a call and
/// before the wrapped tool executes. Hosts use this for session-scoped
/// disclosures such as filtering a `use_skill` listing.
pub type ToolPolicyPostGate<Ctx> =
    Arc<dyn Fn(&RunContext<Ctx>, &ToolCall) -> Option<ToolResult> + Send + Sync>;

fn default_denial(call: &ToolCall, policy: &str, decision: &PolicyDecision) -> String {
    let (action, reason) = match decision {
        PolicyDecision::RequireApproval { reason } => ("requires approval", reason.as_str()),
        PolicyDecision::Deny { reason } => ("denied", reason.as_str()),
        PolicyDecision::Allow => ("allowed", ""),
    };
    format!(
        "Tool '{}' {action} by policy '{policy}': {reason}",
        call.name
    )
}

/// Tool-wrap middleware that enforces a [`ToolCallPolicy`].
pub struct ToolPolicyGateMiddleware<Ctx: Send + Sync = ()> {
    gate: ToolPolicyGate<Ctx>,
    session_ceiling: Option<SessionToolCeiling<Ctx>>,
    session_listing: Option<SessionToolListing<Ctx>>,
    resolver: Option<Arc<dyn ApprovalResolver<Ctx>>>,
    render: DenialRenderer,
    post_gate: Option<ToolPolicyPostGate<Ctx>>,
    /// Serialises calls to `resolve` across an overlapping concurrent batch;
    /// already-allowed `resolve_allowed` admissions do not take this lock.
    prompt_lock: tokio::sync::Mutex<()>,
}

impl<Ctx: Send + Sync> ToolPolicyGateMiddleware<Ctx> {
    /// Creates a fail-closed gate over `policy`.
    pub fn new(policy: Arc<dyn ToolCallPolicy<Ctx>>) -> Self {
        Self {
            gate: ToolPolicyGate::new(policy),
            session_ceiling: None,
            session_listing: None,
            resolver: None,
            render: Arc::new(default_denial),
            post_gate: None,
            prompt_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Settles [`PolicyDecision::RequireApproval`] through `resolver` instead
    /// of failing closed.
    pub fn with_approval_resolver(mut self, resolver: Arc<dyn ApprovalResolver<Ctx>>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Overrides the text returned to the model for a blocked call.
    pub fn with_denial_renderer(mut self, render: DenialRenderer) -> Self {
        self.render = render;
        self
    }

    /// Adds a session permission ceiling that runs before the builder policy.
    /// `Allow` delegates to the builder policy; a block or approval requirement
    /// is handled by this middleware's normal denial/resolver path.
    pub fn with_session_ceiling(mut self, policy: SessionToolCeiling<Ctx>) -> Self {
        self.session_ceiling = Some(policy);
        self
    }

    /// Adds the session-scoped name filter used by host-owned skill/listing
    /// adapters. Names not present in `candidates` are discarded even if the
    /// callback returns them.
    pub fn with_session_listing(mut self, listing: SessionToolListing<Ctx>) -> Self {
        self.session_listing = Some(listing);
        self
    }

    /// Applies an optional host response after both gates allow a call and
    /// before invoking its wrapped tool.
    pub fn with_post_gate(mut self, post_gate: ToolPolicyPostGate<Ctx>) -> Self {
        self.post_gate = Some(post_gate);
        self
    }

    /// Filters candidate names through the configured session listing policy.
    /// Without a listing callback, all candidates are returned unchanged.
    pub fn visible_tool_names(&self, ctx: &RunContext<Ctx>, candidates: &[String]) -> Vec<String> {
        let Some(listing) = &self.session_listing else {
            return candidates.to_vec();
        };
        let selected = listing(ctx, candidates);
        selected
            .into_iter()
            .filter(|name| candidates.contains(name))
            .collect()
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolMiddleware<State, Ctx>
    for ToolPolicyGateMiddleware<Ctx>
{
    fn name(&self) -> &str {
        "tool_policy_gate"
    }

    async fn wrap_tool(
        &self,
        ctx: &RunContext<Ctx>,
        state: &State,
        call: ToolCall,
        next: ToolHandler<'_, State, Ctx>,
    ) -> Result<MiddlewareToolOutcome> {
        // Evaluate without the resolver first so only a call that actually
        // needs approval waits on the prompt lock; the rest stay concurrent.
        let mut policy_name = self.gate.policy_name().to_owned();
        let session_decision = if let Some(session) = &self.session_ceiling {
            session.check(ctx, &call).await
        } else {
            PolicyDecision::Allow
        };
        // A session denial is decisive and avoids consulting a wider builder
        // policy. Otherwise both decisions are required before resolving any
        // approval: neither policy may waive the other's denial.
        let mut verdict = match session_decision {
            decision @ PolicyDecision::Deny { .. } => {
                policy_name = self
                    .session_ceiling
                    .as_ref()
                    .map(|session| session.name().to_owned())
                    .unwrap_or_else(|| self.gate.policy_name().to_owned());
                GateVerdict::Blocked(decision)
            }
            session_decision => {
                let builder_decision = self.gate.check(ctx, &call, false).await;
                match (builder_decision, session_decision) {
                    (decision @ PolicyDecision::Deny { .. }, _) => GateVerdict::Blocked(decision),
                    (PolicyDecision::RequireApproval { reason }, PolicyDecision::Allow) => {
                        GateVerdict::Blocked(PolicyDecision::RequireApproval { reason })
                    }
                    (PolicyDecision::Allow, PolicyDecision::RequireApproval { reason }) => {
                        policy_name = self
                            .session_ceiling
                            .as_ref()
                            .map(|session| session.name().to_owned())
                            .unwrap_or_else(|| self.gate.policy_name().to_owned());
                        GateVerdict::Blocked(PolicyDecision::RequireApproval { reason })
                    }
                    (
                        PolicyDecision::RequireApproval { reason: builder },
                        PolicyDecision::RequireApproval { reason: session },
                    ) => GateVerdict::Blocked(PolicyDecision::RequireApproval {
                        reason: format!("{builder}; {session}"),
                    }),
                    (PolicyDecision::Allow, PolicyDecision::Allow) => {
                        if let Some(resolver) = self.resolver.as_deref() {
                            match resolver.resolve_allowed(ctx, &call).await {
                                ApprovalResolution::Allow { ticket } => {
                                    GateVerdict::Proceed { ticket }
                                }
                                ApprovalResolution::Deny { reason } => {
                                    GateVerdict::Refused { reason }
                                }
                            }
                        } else {
                            GateVerdict::Proceed { ticket: None }
                        }
                    }
                    // A session denial was returned above.
                    (_, PolicyDecision::Deny { .. }) => {
                        unreachable!("session denial handled first")
                    }
                }
            }
        };
        if let (GateVerdict::Blocked(PolicyDecision::RequireApproval { .. }), Some(resolver)) =
            (&verdict, self.resolver.as_deref())
        {
            let _prompting = self.prompt_lock.lock().await;
            verdict = match resolver.resolve(ctx, &call).await {
                ApprovalResolution::Allow { ticket } => GateVerdict::Proceed { ticket },
                ApprovalResolution::Deny { reason } => GateVerdict::Refused { reason },
            };
        }
        match verdict {
            GateVerdict::Proceed { ticket } => {
                let Some(resolver) = &self.resolver else {
                    if let Some(result) = self
                        .post_gate
                        .as_ref()
                        .and_then(|post_gate| post_gate(ctx, &call))
                    {
                        return Ok(MiddlewareToolOutcome::Result(result));
                    }
                    return next.run(ctx, state, call).await;
                };
                let mut guard = TicketGuard::new(resolver.as_ref(), ticket);
                if let Some(result) = self
                    .post_gate
                    .as_ref()
                    .and_then(|post_gate| post_gate(ctx, &call))
                {
                    guard.complete(ApprovalExecutionOutcome::Result(&result));
                    return Ok(MiddlewareToolOutcome::Result(result));
                }
                match next.run(ctx, state, call).await {
                    Ok(outcome) => {
                        match &outcome {
                            MiddlewareToolOutcome::Result(result) => {
                                guard.complete(ApprovalExecutionOutcome::Result(result))
                            }
                            MiddlewareToolOutcome::Command { control } => {
                                guard.complete(ApprovalExecutionOutcome::Control(control))
                            }
                        }
                        Ok(outcome)
                    }
                    Err(error) => {
                        if matches!(error, TinyAgentsError::Cancelled) {
                            guard.complete(ApprovalExecutionOutcome::Cancelled(&error));
                        } else {
                            guard.complete(ApprovalExecutionOutcome::Failed(&error));
                        }
                        Err(error)
                    }
                }
            }
            GateVerdict::Refused { reason } => {
                Ok(MiddlewareToolOutcome::Result(ToolResult::error(reason)))
            }
            GateVerdict::Blocked(decision) => {
                let text = (self.render)(&call, &policy_name, &decision);
                Ok(MiddlewareToolOutcome::Result(ToolResult::error(text)))
            }
        }
    }
}
