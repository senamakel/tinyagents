//! Test fixtures over a store-port transcript locator.
//!
//! Runtime tests bind sessions to an in-memory locator and read back what the
//! session persisted by stem, the way a host reads a store, rather than
//! naming any file.

use std::sync::Arc;

use tinyagents_session::port::{
    AgentStorageFactory, ConditionedAgentStorageFactory, DriverSessionStores, SessionStoreProvider,
};
use tinyagents_session::transcript::{
    DisplaySessionTranscript, SessionAdoption, SessionRef, SessionTranscript, ThreadTranscripts,
    TranscriptHistory, TranscriptLocator, TranscriptMessage, TranscriptMeta, TranscriptRead,
    TranscriptTurn,
};
use tinyagents_storage_kit::{Scope, ScopedStorage, WriteCondition};
use tinystoragedrivers_api::{DocumentStore, Storage, StorageBackend, StreamStore};
use tinystoragedrivers_types::{Capabilities, Result};

/// A driver-backed session provider bound to `scope`, for identity-admission tests.
pub(crate) fn driver_session_stores(scope: Scope) -> Arc<DriverSessionStores> {
    driver_session_stores_over(sqlite_backend(), scope)
}

/// A SQLite backend whose temporary directory stays alive as long as the
/// returned backend. Driver session tests exercise lease-conditioned generation
/// writes, which the testkit's simpler in-memory backend does not implement.
fn sqlite_backend() -> Arc<dyn StorageBackend> {
    let directory = tempfile::tempdir().expect("SQLite session test directory");
    let config = tinystoragedrivers::StorageConfig::parse(&format!(
        "sqlite:{}",
        directory.path().join("storage").display()
    ))
    .expect("SQLite session test URL");
    let backend = std::thread::spawn(move || {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("SQLite session test runtime")
            .block_on(tinystoragedrivers::open(&config))
            .expect("SQLite session test backend")
    })
    .join()
    .expect("SQLite session test thread");
    Arc::new(TemporarySqliteBackend {
        _directory: directory,
        backend,
    })
}

#[derive(Debug)]
struct TemporarySqliteBackend {
    _directory: tempfile::TempDir,
    backend: Arc<dyn StorageBackend>,
}

impl StorageBackend for TemporarySqliteBackend {
    fn capabilities(&self) -> Capabilities {
        self.backend.capabilities()
    }

    fn for_scope(&self, scope: &Scope) -> Result<Storage> {
        self.backend.for_scope(scope)
    }
}

/// A driver-backed session provider bound to `scope` over a shared backend.
pub(crate) fn driver_session_stores_over(
    backend: Arc<dyn StorageBackend>,
    scope: Scope,
) -> Arc<DriverSessionStores> {
    let scope_for_factory = scope.clone();
    let factory: AgentStorageFactory = Arc::new(move |_| {
        let storage = backend.for_scope(&scope_for_factory)?;
        Ok(ScopedStorage::new(storage, scope_for_factory.clone()))
    });
    Arc::new(DriverSessionStores::new(factory).expect("driver session stores over memory"))
}

/// An empty transcript store: the production driver-backed locator, over memory.
pub(crate) fn memory() -> Arc<dyn TranscriptLocator> {
    driver()
}

/// Writes `messages` as the first turn of the transcript `stem`.
pub(crate) fn seed(
    locator: &dyn TranscriptLocator,
    stem: &str,
    messages: &[TranscriptMessage],
    meta: &TranscriptMeta,
) {
    locator
        .open_stem(stem, meta.clone())
        .expect("open stem")
        .append_turn(TranscriptTurn {
            prev: &[],
            next: messages,
            meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .expect("seed transcript");
}

/// The model-context replay of the transcript `stem`; panics when it is absent.
pub(crate) fn read(locator: &dyn TranscriptLocator, stem: &str) -> SessionTranscript {
    try_read(locator, stem).unwrap_or_else(|| panic!("transcript {stem} is absent"))
}

/// The model-context replay of the transcript `stem`, when it exists.
pub(crate) fn try_read(locator: &dyn TranscriptLocator, stem: &str) -> Option<SessionTranscript> {
    locator
        .open_stem(stem, placeholder_meta())
        .expect("open stem")
        .read_session()
        .expect("read transcript")
}

/// The display projection of the transcript `stem`; panics when it is absent.
pub(crate) fn read_display(
    locator: &dyn TranscriptLocator,
    stem: &str,
) -> DisplaySessionTranscript {
    locator
        .open_stem(stem, placeholder_meta())
        .expect("open stem")
        .read_display()
        .expect("read display")
        .unwrap_or_else(|| panic!("transcript {stem} is absent"))
}

/// A `_meta` for opening a stem only to read it (never written: the stem exists
/// or the read is `None`).
fn placeholder_meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: String::new(),
        agent_id: None,
        agent_type: None,
        dispatcher: String::new(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// Everything `handle` would show a reader (model context and display), as one
/// comparable value, so a test can assert a transcript is byte-for-byte what
/// it was.
pub(crate) fn fingerprint(handle: &dyn tinyagents_session::transcript::TranscriptRead) -> String {
    format!(
        "{:?}\n{:?}",
        handle.read_session().expect("read transcript"),
        handle.read_display().expect("read display")
    )
}

/// An in-memory store addressed under a fixed destination key, so tests can
/// hand out separate `Arc`s that stand for one durable store (or, with
/// another key, for a different one).
pub(crate) struct Destination {
    key: String,
    store: Arc<dyn TranscriptLocator>,
}

impl Destination {
    pub(crate) fn new(key: &str, store: &Arc<dyn TranscriptLocator>) -> Self {
        Self {
            key: key.to_string(),
            store: Arc::clone(store),
        }
    }
}

impl TranscriptLocator for Destination {
    fn destination_key(&self) -> Option<String> {
        Some(self.key.clone())
    }
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.store.latest_for_agent(agent_name)
    }
    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.store.root_for_thread(thread_id)
    }
    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        self.store.root_for_thread_scoped(thread_id, agent_id)
    }
    fn thread_transcripts(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<ThreadTranscripts> {
        self.store.thread_transcripts(thread_id, agent_id)
    }
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.store.open_stem(stem, seed)
    }
    fn session_exists(&self, session: &SessionRef) -> bool {
        self.store.session_exists(session)
    }
    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        self.store.read_session_transcript(session)
    }
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.store.begin_generation(session, seed)
    }
    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.store
            .begin_generation_from_baseline(session, seed, baseline)
    }
}

/// An empty driver-backed transcript store (the production store, over SQLite).
pub(crate) fn driver() -> Arc<dyn TranscriptLocator> {
    let scope = Scope::new("runtime-tests").expect("runtime test scope");
    let stores = conditioned_driver_session_stores(scope.clone());
    Arc::new(OwnedTestLocator {
        locator: stores.for_agent("runtime-tests").transcripts,
        _stores: stores,
    })
}

fn conditioned_driver_session_stores(scope: Scope) -> Arc<DriverSessionStores> {
    use tinystoragedrivers_connect::sqlite::SqliteConnection;
    use tinystoragedrivers_document::sqlite::{SqliteDocuments, SqliteStreams};

    fn storage(
        path: &std::path::Path,
        scope: &Scope,
        memory: &tinystoragedrivers_testkit::MemoryBackend,
        condition: Option<&WriteCondition>,
    ) -> Result<ScopedStorage> {
        let connection = SqliteConnection::open(path)
            .map_err(|error| tinystoragedrivers_types::StorageError::backend(error.to_string()))?;
        let documents = match condition {
            Some(condition) => Arc::new(SqliteDocuments::conditioned(
                connection.clone(),
                scope.clone(),
                condition.clone(),
            )) as Arc<dyn DocumentStore>,
            None => Arc::new(SqliteDocuments::new(connection.clone(), scope.clone()))
                as Arc<dyn DocumentStore>,
        };
        let streams =
            Arc::new(SqliteStreams::new(connection, scope.clone())) as Arc<dyn StreamStore>;
        let ports = memory
            .for_scope(scope)
            .map_err(|error| tinystoragedrivers_types::StorageError::backend(error.to_string()))?;
        Ok(ScopedStorage::new(
            Storage::new(
                documents,
                streams,
                ports.kv().clone(),
                ports.blobs().clone(),
                ports.queues().clone(),
                ports.secrets().clone(),
            ),
            scope.clone(),
        ))
    }

    let directory = Arc::new(tempfile::tempdir().expect("SQLite session test directory"));
    let path = directory.path().join("session.sqlite");
    let plain_path = path.clone();
    let memory = Arc::new(tinystoragedrivers_testkit::MemoryBackend::new());
    let plain_scope = scope.clone();
    let plain_dir = Arc::clone(&directory);
    let plain_memory = Arc::clone(&memory);
    let factory: AgentStorageFactory = Arc::new(move |_| {
        let _keep_directory = &plain_dir;
        storage(&plain_path, &plain_scope, &plain_memory, None)
    });
    let condition_scope = scope;
    let condition_dir = Arc::clone(&directory);
    let condition_memory = memory;
    let conditioned: ConditionedAgentStorageFactory = Arc::new(move |_, condition| {
        let _keep_directory = &condition_dir;
        storage(&path, &condition_scope, &condition_memory, Some(condition))
    });
    Arc::new(
        DriverSessionStores::new(factory)
            .expect("conditioned SQLite test stores")
            .with_conditioned_storage_factory(conditioned),
    )
}

/// An empty in-memory session locator for behavior tests that do not certify
/// the driver's storage adapter.
pub(crate) fn in_memory() -> Arc<dyn TranscriptLocator> {
    tinyagents_session::port::InMemorySessionStores::new()
        .for_agent("runtime-tests")
        .transcripts
}

/// A transcript handle plus its backend, which retains the fixture directory.
struct OwnedTestLocator {
    locator: Arc<dyn TranscriptLocator>,
    _stores: Arc<DriverSessionStores>,
}

impl TranscriptLocator for OwnedTestLocator {
    fn destination_key(&self) -> Option<String> {
        self.locator.destination_key()
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.locator.latest_for_agent(agent_name)
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.locator.root_for_thread(thread_id)
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.locator.open_stem(stem, seed)
    }

    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<SessionAdoption>> {
        self.locator.adopt_legacy(session, thread_id, seed)
    }
}
