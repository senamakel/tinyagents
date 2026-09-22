use std::sync::Arc;

use tinyagents_session::transcript::{SessionRef, TranscriptLocator, TranscriptMeta, session_stem};

use crate::{
    NoopSessionHooks, PrefixSnapshot, RuntimeError, Session, SessionDriver, SessionHooks,
    ToolSnapshot, TranscriptCodec,
};

/// Configures a directly-owned, reusable [`Session`].
pub struct SessionBuilder<C: Clone + Send + Sync + 'static = ()> {
    driver: Arc<dyn SessionDriver<C>>,
    codec: Option<Arc<dyn TranscriptCodec<C>>>,
    hooks: Arc<dyn SessionHooks<C>>,
    prefix: PrefixSnapshot,
    tools: ToolSnapshot,
    transcript: Option<TranscriptConfig>,
}

struct TranscriptConfig {
    locator: Arc<dyn TranscriptLocator>,
    stem: String,
    session: Option<SessionRef>,
    resume_agent: Option<String>,
    meta: TranscriptMeta,
}

impl<C: Clone + Send + Sync + 'static> SessionBuilder<C> {
    /// Starts a builder over an object-safe execution driver.
    pub fn new(driver: Arc<dyn SessionDriver<C>>) -> Self {
        Self {
            driver,
            codec: None,
            hooks: Arc::new(NoopSessionHooks),
            prefix: PrefixSnapshot::default(),
            tools: ToolSnapshot::default(),
            transcript: None,
        }
    }

    /// Installs the host-owned lossless transcript conversion.
    pub fn codec(mut self, codec: Arc<dyn TranscriptCodec<C>>) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Installs optional host preparation/observation hooks.
    pub fn hooks(mut self, hooks: Arc<dyn SessionHooks<C>>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Freezes the prefix used to initialize this session's history.
    pub fn prefix(mut self, prefix: PrefixSnapshot) -> Self {
        self.prefix = prefix;
        self
    }

    /// Freezes the tool declarations exposed to each driver invocation.
    pub fn tool_snapshot(mut self, tools: ToolSnapshot) -> Self {
        self.tools = tools;
        self
    }

    /// Enables append-only transcript persistence through a session-owned
    /// locator, stem, and neutral metadata seed.
    pub fn transcript(
        mut self,
        locator: Arc<dyn TranscriptLocator>,
        stem: impl Into<String>,
        meta: TranscriptMeta,
    ) -> Self {
        self.transcript = Some(TranscriptConfig {
            locator,
            stem: stem.into(),
            session: None,
            resume_agent: None,
            meta,
        });
        self
    }

    /// Enables persistence addressed by durable session identity.
    ///
    /// Prefer this over [`Self::transcript`]: the stem it derives is stable
    /// across processes and launches, so one conversation stays in one
    /// transcript instead of accumulating a file per cold boot. It is also what
    /// [`ResumeMode::Session`](crate::ResumeMode::Session) resolves against.
    pub fn session(
        mut self,
        locator: Arc<dyn TranscriptLocator>,
        session: SessionRef,
        mut meta: TranscriptMeta,
    ) -> Self {
        meta.session_id = Some(session.session_id());
        meta.parent_session_id = session.parent_session_id();
        self.transcript = Some(TranscriptConfig {
            locator,
            stem: session_stem(&session),
            session: Some(session),
            resume_agent: None,
            meta,
        });
        self
    }

    /// Uses a distinct agent key for `ResumeMode::LatestForAgent` lookup.
    ///
    /// Previously reachable only through a hook-supplied `ResumePreparation`,
    /// which meant a host that simply wanted a different resume key had to
    /// implement a hook to say so.
    pub fn resume_agent(mut self, resume_agent: impl Into<String>) -> Self {
        if let Some(config) = self.transcript.as_mut() {
            config.resume_agent = Some(resume_agent.into());
        }
        self
    }

    /// Builds a session. A codec is required only when transcript persistence
    /// or transcript resume is configured.
    pub fn build(self) -> Result<Session<C>, RuntimeError> {
        let target = self.transcript.map(|config| crate::TranscriptTarget {
            locator: config.locator,
            stem: config.stem,
            resume_agent: config.resume_agent,
            session: config.session,
            meta: config.meta,
        });
        if target.is_some() && self.codec.is_none() {
            return Err(RuntimeError::MissingDependency("TranscriptCodec"));
        }
        Ok(Session::<C>::new(
            self.driver,
            self.codec,
            self.hooks,
            self.prefix,
            self.tools,
            target,
        ))
    }
}
