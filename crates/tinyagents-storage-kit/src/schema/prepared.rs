//! Owned, private normalization shared by both registry APIs.
use super::{
    BlobStepFn, Item, Owner, Schema, SchemaDeclaration, StepBody, StepFn, StreamStepFn,
    TransformDescriptor, TransformKind,
};
use crate::{Result, StorageError};

#[derive(Clone)]
pub(super) enum Transform<T> {
    Local(T),
    External(TransformDescriptor),
}

#[derive(Clone)]
pub(super) enum PreparedBody {
    Documents(Transform<StepFn>),
    Streams(Transform<StreamStepFn>),
    Blobs(Transform<BlobStepFn>),
}
impl std::fmt::Debug for PreparedBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.execution().fmt(f)
    }
}
impl PreparedBody {
    fn execution(&self) -> super::StepExecution {
        let (kind, external) = match self {
            Self::Documents(body) => (
                TransformKind::Documents,
                match body {
                    Transform::External(d) => Some(d),
                    Transform::Local(_) => None,
                },
            ),
            Self::Streams(body) => (
                TransformKind::Stream,
                match body {
                    Transform::External(d) => Some(d),
                    Transform::Local(_) => None,
                },
            ),
            Self::Blobs(body) => (
                TransformKind::Blob,
                match body {
                    Transform::External(d) => Some(d),
                    Transform::Local(_) => None,
                },
            ),
        };
        external.cloned().map_or(
            super::StepExecution::Local { kind },
            super::StepExecution::External,
        )
    }
}

#[derive(Debug, Clone)]
pub(super) struct PreparedStep {
    pub id: String,
    pub item: String,
    pub from: u32,
    pub to: u32,
    pub source: Option<String>,
    pub run: PreparedBody,
}
impl PreparedStep {
    pub(super) fn source_collection(&self) -> &str {
        self.source.as_deref().unwrap_or(&self.item)
    }
}
#[derive(Debug)]
pub(super) struct PreparedSchema {
    pub owner: Owner,
    pub items: Vec<Item>,
    pub steps: Vec<PreparedStep>,
}
impl PreparedSchema {
    pub(super) fn local(schema: &Schema) -> Self {
        Self {
            owner: schema.owner.clone(),
            items: schema.items.clone(),
            steps: schema
                .steps
                .iter()
                .map(|step| PreparedStep {
                    id: step.id.to_owned(),
                    item: step.item.to_owned(),
                    from: step.from,
                    to: step.to,
                    source: step.source.map(str::to_owned),
                    run: match &step.run {
                        StepBody::Documents(run) => {
                            PreparedBody::Documents(Transform::Local(run.clone()))
                        }
                        StepBody::Streams(run) => {
                            PreparedBody::Streams(Transform::Local(run.clone()))
                        }
                        StepBody::Blobs(run) => PreparedBody::Blobs(Transform::Local(run.clone())),
                    },
                })
                .collect(),
        }
    }
    pub(super) fn external(schema: SchemaDeclaration) -> Result<Self> {
        let steps = schema.steps.into_iter().map(|step| {
            let descriptor = step.transform;
            if descriptor.version != 1 || descriptor.operation.is_empty()
                || descriptor.operation.len() > 256
                || !descriptor.operation.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
            {
                return Err(StorageError::invalid_input(format!("invalid transform descriptor for step '{}' (operation must be a plain selector and version must be 1)", step.id)));
            }
            let run = match descriptor.kind {
                TransformKind::Documents => PreparedBody::Documents(Transform::External(descriptor)),
                TransformKind::Stream => PreparedBody::Streams(Transform::External(descriptor)),
                TransformKind::Blob => PreparedBody::Blobs(Transform::External(descriptor)),
            };
            Ok(PreparedStep { id: step.id, item: step.item, from: step.from, to: step.to, source: step.source, run })
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owner: schema.owner,
            items: schema.items,
            steps,
        })
    }
    pub(super) fn metadata(&self) -> super::SchemaMetadata {
        super::SchemaMetadata {
            owner: self.owner.clone(),
            items: self.items.clone(),
            steps: self
                .steps
                .iter()
                .map(|step| super::StepMetadata {
                    id: step.id.clone(),
                    item: step.item.clone(),
                    from: step.from,
                    to: step.to,
                    source: step.source.clone(),
                    execution: step.run.execution(),
                })
                .collect(),
        }
    }
}
