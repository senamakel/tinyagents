//! Truthful introspection for a registry containing local and external steps.
use super::{Item, Owner, TransformDescriptor, TransformKind};
use serde::{Deserialize, Serialize};

/// Owned declaration metadata for one owner in a composed registry.
/// Executable closures are represented only by their record family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaMetadata {
    /// Owner of these items and steps.
    pub owner: Owner,
    /// Stored items in declaration order.
    pub items: Vec<Item>,
    /// Migration steps in execution order.
    pub steps: Vec<StepMetadata>,
}

/// Owned migration identity and execution metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepMetadata {
    /// Stable progress identifier within its item.
    pub id: String,
    /// Destination item.
    pub item: String,
    /// Input version.
    pub from: u32,
    /// Output version.
    pub to: u32,
    /// Optional legacy source.
    pub source: Option<String>,
    /// Whether this step invokes a closure or an external descriptor.
    pub execution: StepExecution,
}

/// Describes execution without pretending a local closure has a wire selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepExecution {
    /// An in-process closure, whose body is deliberately absent.
    Local {
        /// Input and output record family.
        kind: TransformKind,
    },
    /// An owner operation whose exact wire descriptor is preserved.
    External(TransformDescriptor),
}
