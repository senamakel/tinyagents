//! Explicit declaration policy for driver-backed stores.
use tinyagents_storage_kit::{CollectionSpec, DocumentStore, Result};

/// How an adapter admits its required collection declarations.
///
/// This selects document schema behavior, not tenant, lease or native driver
/// authority. Hosts must separately supply suitably restricted native handles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SchemaAdmission {
    /// Declare compatible collections and indexes through the native port.
    #[default]
    Declare,
    /// Validate persisted declarations without creating or repairing them.
    ValidateOnly,
}

impl SchemaAdmission {
    /// Admit `spec` using the selected native operation.
    ///
    /// # Errors
    ///
    /// Native missing, incompatible, unsupported or fenced schema errors,
    /// preserving their kind and source. Validation never falls back to DDL.
    pub async fn admit(self, documents: &dyn DocumentStore, spec: &CollectionSpec) -> Result<()> {
        match self {
            Self::Declare => documents.ensure_collection(spec).await,
            Self::ValidateOnly => documents.validate_collection_spec(spec).await,
        }
    }
}
