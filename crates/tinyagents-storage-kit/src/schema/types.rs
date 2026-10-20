//! The declaration types: [`Owner`], [`Schema`], [`Item`], [`ScopeClass`],
//! [`Step`] and the records a step reads and writes.

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

use super::seal::SealPolicy;
use crate::document::CollectionSpec;
use crate::error::Result;

pub use super::declaration::{Expiry, Item, Owner, ScopeClass, SourceRecord, TargetRecord};

/// The transformation of a document [`Step`]: a pure function of its batch, so
/// a re-run produces the same output.
pub type StepFn = Arc<dyn Fn(Vec<SourceRecord>) -> Result<Vec<TargetRecord>> + Send + Sync>;

/// The transformation of a stream [`Step`]: `(source stream name, its entries
/// in order)` to `Some((target stream name, new entries))`, or `None` to drop
/// the stream. Pure, like [`StepFn`]: a resumed run calls it again and must
/// get the same entries.
pub type StreamStepFn =
    Arc<dyn Fn(String, Vec<Value>) -> Result<Option<(String, Vec<Value>)>> + Send + Sync>;

/// What a blob [`Step`] writes: `(target key, content type, bytes)`.
pub type BlobOutput = (String, Option<String>, Vec<u8>);

/// The transformation of a blob [`Step`]: `(source key, content type, bytes)`
/// to `Some((target key, content type, bytes))`, or `None` to drop the blob.
/// Pure, like [`StepFn`].
pub type BlobStepFn =
    Arc<dyn Fn(String, Option<String>, Vec<u8>) -> Result<Option<BlobOutput>> + Send + Sync>;

/// What a [`Step`] does, and so which kind of [`Item`] it can target.
#[derive(Clone)]
pub enum StepBody {
    /// Rewrite the documents of a [`Item::Collection`].
    Documents(StepFn),
    /// Move the legacy streams under a prefix into an [`Item::Stream`].
    Streams(StreamStepFn),
    /// Move the legacy blobs under a prefix into an [`Item::Blobs`].
    Blobs(BlobStepFn),
}

impl StepBody {
    fn kind(&self) -> &'static str {
        match self {
            Self::Documents(_) => "documents",
            Self::Streams(_) => "streams",
            Self::Blobs(_) => "blobs",
        }
    }
}

impl fmt::Debug for StepBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StepBody::{}", self.kind())
    }
}

/// Brings one item forward from version `from` to `to`.
///
/// - [`Step::new`] rewrites the records of a **collection**. By default the
///   source is the collection itself (an in-place rewrite of the records
///   still below version `to`). [`Step::source`] names another collection
///   instead, normally a legacy v1 name; its rows are deleted once the step
///   completes.
/// - [`Step::stream`] moves **streams**: every stream whose name starts with
///   the [`Step::source_prefix`] is read whole, mapped, written under its new
///   name inside the item, and deleted.
/// - [`Step::blobs`] moves **blobs** the same way, one blob at a time.
///
/// Stream and blob versions are labels for ordering the steps of an item
/// (they start at 1 and must not skip); unlike a collection, the item has no
/// declared version to end at.
#[derive(Clone)]
pub struct Step {
    /// Identifies the step in `tsd.migrations`; unique within its item.
    pub id: &'static str,
    /// The item the step brings to version `to`.
    pub item: &'static str,
    /// The version it reads.
    pub from: u32,
    /// The version it writes.
    pub to: u32,
    /// A legacy collection to read instead of `item` (document steps), or the
    /// legacy stream name / blob key prefix to migrate (stream and blob
    /// steps, where it is required).
    pub source: Option<&'static str>,
    /// The transformation.
    pub run: StepBody,
}

impl Step {
    /// A document step over collection `item` from version `from` to `to`.
    #[must_use]
    pub fn new(
        id: &'static str,
        item: &'static str,
        from: u32,
        to: u32,
        run: impl Fn(Vec<SourceRecord>) -> Result<Vec<TargetRecord>> + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(id, item, from, to, StepBody::Documents(Arc::new(run)))
    }

    /// A stream step into the stream family `item`. Set the legacy streams
    /// with [`Step::source_prefix`].
    ///
    /// The target name `run` returns must be `item` itself or start with
    /// `item/`. Migrated streams start at offset 0, whatever the offsets of
    /// their source were: a source that had been truncated keeps only the
    /// entries `read_window` can still see, renumbered from 0, because a
    /// stream's base cannot be set from outside the driver. Owners that keep
    /// offsets as external references must translate them with the source's
    /// base. A sealed source produces a sealed target.
    #[must_use]
    pub fn stream(
        id: &'static str,
        item: &'static str,
        from: u32,
        to: u32,
        run: impl Fn(String, Vec<Value>) -> Result<Option<(String, Vec<Value>)>> + Send + Sync + 'static,
    ) -> Self {
        Self::with_body(id, item, from, to, StepBody::Streams(Arc::new(run)))
    }

    /// A blob step into the prefix `item`. Set the legacy blobs with
    /// [`Step::source_prefix`].
    ///
    /// The target key `run` returns must start with `item`. An existing
    /// target is kept (nothing is overwritten).
    #[must_use]
    pub fn blobs(
        id: &'static str,
        item: &'static str,
        from: u32,
        to: u32,
        run: impl Fn(String, Option<String>, Vec<u8>) -> Result<Option<BlobOutput>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::with_body(id, item, from, to, StepBody::Blobs(Arc::new(run)))
    }

    fn with_body(id: &'static str, item: &'static str, from: u32, to: u32, run: StepBody) -> Self {
        Self {
            id,
            item,
            from,
            to,
            source: None,
            run,
        }
    }

    /// Read `name` (a legacy collection) instead of the item itself.
    #[must_use]
    pub fn source(mut self, name: &'static str) -> Self {
        self.source = Some(name);
        self
    }

    /// The legacy stream name prefix or blob key prefix a stream or blob step
    /// migrates. It must not overlap the item's own name, so the runner never
    /// reads its own output.
    #[must_use]
    pub fn source_prefix(self, prefix: &'static str) -> Self {
        self.source(prefix)
    }
}

impl fmt::Debug for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Step")
            .field("id", &self.id)
            .field("kind", &self.run.kind())
            .field("item", &self.item)
            .field("from", &self.from)
            .field("to", &self.to)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// What one owner stores and how older versions are brought forward. Each
/// owning crate exports `pub fn storage_schema() -> Schema`.
///
/// Built with chained calls; nothing is validated until
/// [`Registry::compose`](super::Registry::compose).
#[derive(Debug, Clone)]
pub struct Schema {
    /// The owner every name must start with.
    pub owner: Owner,
    /// Everything the owner stores.
    pub items: Vec<Item>,
    /// The migration steps, in the order they run.
    pub steps: Vec<Step>,
}

impl Schema {
    /// An empty schema for `owner`.
    #[must_use]
    pub fn new(owner: Owner) -> Self {
        Self {
            owner,
            items: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Declare collection `name` at `version`, with no indexes.
    #[must_use]
    pub fn collection(self, name: &str, version: u32, scope: ScopeClass) -> Self {
        self.collection_with(version, CollectionSpec::new(name), scope)
    }

    /// Declare a collection from a full spec (indexes, search, ttl); the item
    /// takes the spec's name.
    #[must_use]
    pub fn collection_with(
        mut self,
        version: u32,
        spec: CollectionSpec,
        scope: ScopeClass,
    ) -> Self {
        self.items.push(Item::Collection {
            name: spec.name.clone(),
            version,
            spec,
            scope,
        });
        self
    }

    /// Declare stream family `name`.
    #[must_use]
    pub fn stream(self, name: &str, scope: ScopeClass) -> Self {
        self.stream_with(name, scope, SealPolicy::default())
    }

    /// Declare a stream family with its seal policy.
    #[must_use]
    pub fn stream_with(mut self, name: &str, scope: ScopeClass, seal: SealPolicy) -> Self {
        self.items.push(Item::Stream {
            name: name.to_owned(),
            scope,
            seal,
        });
        self
    }

    /// Declare a blob prefix.
    #[must_use]
    pub fn blobs(mut self, prefix: &str, scope: ScopeClass) -> Self {
        self.items.push(Item::Blobs {
            prefix: prefix.to_owned(),
            scope,
        });
        self
    }

    /// Declare a secret prefix.
    #[must_use]
    pub fn secrets(mut self, prefix: &str, scope: ScopeClass) -> Self {
        self.items.push(Item::Secrets {
            prefix: prefix.to_owned(),
            scope,
        });
        self
    }

    /// Add a migration step.
    #[must_use]
    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    /// Collection declarations owned by this schema.
    #[must_use]
    pub fn collections(&self) -> Vec<CollectionSpec> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Item::Collection { spec, .. } => Some(spec.clone()),
                Item::Stream { .. } | Item::Blobs { .. } | Item::Secrets { .. } => None,
            })
            .collect()
    }
}
