use std::collections::BTreeMap;

use tinytools::ToolSpec;

use crate::RuntimeError;

/// An immutable, model-visible tool declaration set for one session turn.
///
/// It records declarations only; choosing which tools are permitted and wiring
/// their executors remains a host/driver responsibility.
#[derive(Clone, Debug, Default)]
pub struct ToolSnapshot {
    specs: Vec<ToolSpec>,
    exact: bool,
}

impl ToolSnapshot {
    /// Validates and freezes a tool declaration set.
    ///
    /// Identical repeated declarations are deduplicated.  A shared name with
    /// different contents is rejected rather than silently choosing one.
    pub fn new(specs: Vec<ToolSpec>) -> Result<Self, RuntimeError> {
        let mut names = BTreeMap::<String, ToolSpec>::new();
        for spec in specs {
            if let Some(existing) = names.get(&spec.name) {
                let same = existing.description == spec.description
                    && existing.parameters == spec.parameters;
                if !same {
                    return Err(RuntimeError::ToolNameCollision(spec.name));
                }
                continue;
            }
            names.insert(spec.name.clone(), spec);
        }
        Ok(Self {
            specs: names.into_values().collect(),
            exact: false,
        })
    }

    /// Marks this snapshot as a one-off declaration set. It is sent exactly
    /// as supplied and is not retained for a later turn.
    pub fn exact(mut self) -> Self {
        self.exact = true;
        self
    }

    pub(crate) fn is_exact(&self) -> bool {
        self.exact
    }

    /// Returns the frozen declarations in stable name order.
    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// The declarations as the JSON a transcript `tools` record stores.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.specs).unwrap_or(serde_json::Value::Array(Vec::new()))
    }

    /// Restores a snapshot from a transcript `tools` record.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, RuntimeError> {
        let specs: Vec<ToolSpec> = serde_json::from_value(value.clone()).map_err(|error| {
            RuntimeError::Persistence(format!("decode recorded tool declarations: {error}"))
        })?;
        Self::new(specs)
    }

    /// This snapshot plus every declaration of `recorded` whose name it does
    /// not already carry. A name present in both keeps this snapshot's
    /// declaration: the live host is authoritative for a tool it still
    /// supplies. Returns the merged snapshot and how many were retained.
    pub fn with_retained(&self, recorded: &ToolSnapshot) -> Result<(Self, usize), RuntimeError> {
        let retained: Vec<ToolSpec> = recorded
            .specs
            .iter()
            .filter(|spec| !self.specs.iter().any(|live| live.name == spec.name))
            .cloned()
            .collect();
        if retained.is_empty() {
            return Ok((self.clone(), 0));
        }
        let count = retained.len();
        let merged = Self::new(self.specs.iter().cloned().chain(retained).collect())?;
        Ok((merged, count))
    }
}
