//! Per-thread Claude Code session state on the host-bound storage ports.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tinyagents_storage_kit::primitives::Kv;
use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_types::StorageError;

/// Maximum delivered-turn fingerprints remembered per thread.
const MAX_DELIVERED_PER_THREAD: usize = 256;
const OWNER: &str = "tinyagents-harness";
const NAMESPACE: &str = "claude_code_sessions";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct SessionRecord {
    session_id: Option<String>,
    #[serde(default)]
    delivered: Vec<String>,
}

/// Serialize async read/reserve and persist/release sequences in this process.
static STORE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A process-local reservation. It is keyed by the identity of the bound
/// DocumentStore and thread, and is never persisted before CLI acknowledgement.
static IN_FLIGHT: Mutex<Vec<(usize, String, String)>> = Mutex::new(Vec::new());

/// Persistent per-thread session state. The host supplies an already scoped
/// document handle; this type never discovers or reads workspace files.
#[derive(Debug, Clone)]
pub struct SessionStore {
    kv: Kv,
    store_identity: usize,
}

impl SessionStore {
    pub fn new(docs: &Arc<dyn DocumentStore>) -> Result<Self, StorageError> {
        Ok(Self {
            kv: Kv::ns(docs, OWNER, NAMESPACE)?,
            store_identity: Arc::as_ptr(docs) as *const () as usize,
        })
    }

    pub async fn get(&self, thread_id: &str) -> Result<Option<String>, StorageError> {
        let _guard = STORE_LOCK.lock().await;
        Ok(self
            .record(thread_id)
            .await?
            .and_then(|record| record.session_id))
    }

    #[cfg(test)]
    pub async fn delivered(&self, thread_id: &str) -> Result<HashSet<String>, StorageError> {
        let _guard = STORE_LOCK.lock().await;
        Ok(self
            .record(thread_id)
            .await?
            .map_or_else(HashSet::new, |record| {
                record.delivered.into_iter().collect()
            }))
    }

    pub async fn claim_delivered(
        self: &Arc<Self>,
        thread_id: &str,
        fingerprints: &[String],
    ) -> Result<(HashSet<String>, DeliveryClaim), StorageError> {
        let _guard = STORE_LOCK.lock().await;
        let record = self.record(thread_id).await?.unwrap_or_default();
        let mut already: HashSet<String> = record.delivered.into_iter().collect();
        let mut in_flight = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        already.extend(
            in_flight
                .iter()
                .filter(|(store, thread, _)| *store == self.store_identity && thread == thread_id)
                .map(|(_, _, fingerprint)| fingerprint.clone()),
        );
        let mut claimed = Vec::new();
        for fingerprint in fingerprints {
            if !already.contains(fingerprint) && !claimed.contains(fingerprint) {
                claimed.push(fingerprint.clone());
            }
        }
        in_flight.extend(claimed.iter().map(|fingerprint| {
            (
                self.store_identity,
                thread_id.to_owned(),
                fingerprint.clone(),
            )
        }));
        drop(in_flight);
        Ok((
            already,
            DeliveryClaim {
                store: Arc::clone(self),
                thread_id: thread_id.to_owned(),
                claimed,
            },
        ))
    }

    pub async fn record_delivered(
        &self,
        thread_id: &str,
        fingerprints: &[String],
    ) -> Result<(), StorageError> {
        if fingerprints.is_empty() {
            return Ok(());
        }
        let _guard = STORE_LOCK.lock().await;
        self.kv
            .update(thread_id, |current: Option<SessionRecord>| {
                let mut record = current.unwrap_or_default();
                for fingerprint in fingerprints {
                    if !record.delivered.contains(fingerprint) {
                        record.delivered.push(fingerprint.clone());
                    }
                }
                trim(&mut record.delivered, fingerprints.len());
                Ok(Some(record))
            })
            .await?;
        Ok(())
    }

    pub async fn set(&self, thread_id: &str, session_id: &str) -> Result<(), StorageError> {
        let _guard = STORE_LOCK.lock().await;
        self.kv
            .update(thread_id, |current: Option<SessionRecord>| {
                let mut record = current.unwrap_or_default();
                if record.session_id.as_deref() != Some(session_id) {
                    record.delivered.clear();
                }
                record.session_id = Some(session_id.to_owned());
                Ok(Some(record))
            })
            .await?;
        Ok(())
    }

    pub async fn remove_if(
        &self,
        thread_id: &str,
        expected_session_id: &str,
    ) -> Result<bool, StorageError> {
        let _guard = STORE_LOCK.lock().await;
        let removed = self
            .kv
            .update(thread_id, |current: Option<SessionRecord>| {
                let Some(record) = current else {
                    return Ok(None);
                };
                if record.session_id.as_deref() != Some(expected_session_id) {
                    return Ok(None);
                }
                Ok(Some(SessionRecord::default()))
            })
            .await?;
        Ok(removed.is_some())
    }

    async fn record(&self, thread_id: &str) -> Result<Option<SessionRecord>, StorageError> {
        self.kv.get(thread_id).await
    }

    async fn import_record(
        &self,
        thread_id: &str,
        incoming: SessionRecord,
    ) -> Result<(), StorageError> {
        let _guard = STORE_LOCK.lock().await;
        self.kv
            .update(thread_id, |current: Option<SessionRecord>| {
                let Some(mut current) = current else {
                    return Ok(Some(incoming.clone()));
                };
                if current.session_id == incoming.session_id {
                    for fingerprint in &incoming.delivered {
                        if !current.delivered.contains(fingerprint) {
                            current.delivered.push(fingerprint.clone());
                        }
                    }
                    trim(&mut current.delivered, 0);
                    Ok(Some(current))
                } else {
                    Ok(None)
                }
            })
            .await?;
        Ok(())
    }

    fn release_in_flight(&self, thread_id: &str, fingerprints: &[String]) {
        let mut in_flight = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        in_flight.retain(|(store, thread, fingerprint)| {
            !(*store == self.store_identity
                && thread == thread_id
                && fingerprints.contains(fingerprint))
        });
    }
}

fn trim(delivered: &mut Vec<String>, keep_at_least: usize) {
    let limit = MAX_DELIVERED_PER_THREAD.max(keep_at_least);
    if delivered.len() > limit {
        delivered.drain(..delivered.len() - limit);
    }
}

/// Importer input for the old `claude-code-sessions.json` file. The host owns
/// filesystem discovery and supplies the bytes; malformed input is rejected.
#[derive(Debug, Default, Deserialize)]
struct LegacySessionFile {
    #[serde(default)]
    sessions: std::collections::HashMap<String, String>,
    #[serde(default)]
    delivered: std::collections::HashMap<String, Vec<String>>,
}

/// Import old session JSON bytes without accessing the filesystem. Existing
/// session IDs win; matching IDs merge delivered fingerprints, while a
/// different current ID is left untouched. Safe to call repeatedly.
pub async fn import_legacy_sessions(
    store: &SessionStore,
    bytes: &[u8],
) -> Result<usize, StorageError> {
    let legacy: LegacySessionFile = serde_json::from_slice(bytes)
        .map_err(|error| StorageError::serialization(error.to_string()))?;
    let mut imported = 0;
    let thread_ids: std::collections::HashSet<_> = legacy
        .sessions
        .keys()
        .chain(legacy.delivered.keys())
        .cloned()
        .collect();
    for thread_id in thread_ids {
        let record = SessionRecord {
            session_id: legacy.sessions.get(&thread_id).cloned(),
            delivered: legacy
                .delivered
                .get(&thread_id)
                .cloned()
                .unwrap_or_default(),
        };
        let has_legacy_state = record.session_id.is_some() || !record.delivered.is_empty();
        let before = store.record(&thread_id).await?;
        store.import_record(&thread_id, record).await?;
        if before.is_none() && has_legacy_state {
            imported += 1;
        }
    }
    Ok(imported)
}

/// Reservation of fingerprints; dropping releases it, commit ends it after
/// the CLI accepted the user turn and durable state was recorded.
#[derive(Debug)]
pub struct DeliveryClaim {
    store: Arc<SessionStore>,
    thread_id: String,
    claimed: Vec<String>,
}

impl DeliveryClaim {
    pub fn commit(self) {}
}

impl Drop for DeliveryClaim {
    fn drop(&mut self) {
        self.store.release_in_flight(&self.thread_id, &self.claimed);
    }
}

/// Random RFC-4122 v4 UUID, formatted lower-case with hyphens.
pub fn generate_uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// CC accepts only RFC-4122 v4. Older stores might carry pre-v4 strings;
/// they are treated as missing and regenerated.
pub fn is_uuid_v4(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (index, byte) in bytes.iter().enumerate() {
        if [8, 13, 18, 23].contains(&index) {
            if *byte != b'-' {
                return false;
            }
        } else if !byte.is_ascii_hexdigit() {
            return false;
        }
    }
    bytes[14] == b'4' && matches!(bytes[19], b'8' | b'9' | b'a' | b'b' | b'A' | b'B')
}

#[cfg(test)]
#[path = "session_store_tests.rs"]
mod tests;
