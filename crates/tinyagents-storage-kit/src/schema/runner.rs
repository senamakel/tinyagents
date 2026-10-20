//! The resumable schema step runner.

use std::collections::{HashMap, HashSet};

use crate::backend::ScopedStorage;
use crate::document::{
    CollectionSpec, Cursor, DocumentStore, Precondition, Query, Version, WriteOptions,
};
use crate::error::{ErrorKind, Result, StorageError};
use crate::filter::{Filter, ID_FIELD, SCHEMA_FIELD, Sort};

use super::TransformExecutor;
use super::compose::PreparedRegistry;
use super::migrations::{Migrations, Progress, migrations_spec, step_key};
use super::prepared::{PreparedBody as StepBody, PreparedStep, Transform};
use super::types::{Expiry, Owner, SourceRecord, StepFn, TargetRecord};
use super::{runner_blobs, runner_streams};

/// Options for running the schema migration steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrateOptions {
    /// Records per batch: the page size read from the source, and the unit of
    /// progress that survives a crash.
    pub batch_size: usize,
}

impl MigrateOptions {
    /// 100 records per batch.
    #[must_use]
    pub const fn new() -> Self {
        Self { batch_size: 100 }
    }

    /// Set the batch size (must be at least 1).
    #[must_use]
    pub const fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl Default for MigrateOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A step that ran in this call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepReport {
    /// The owner of the step.
    pub owner: String,
    /// The collection the step brought forward.
    pub item: String,
    /// The step id.
    pub step: String,
    /// Records written in total, including batches finished by an earlier
    /// interrupted call.
    pub written: u64,
}

/// Report of schema migration steps run in one call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MigrationReport {
    /// Steps that ran to completion in this call, in order.
    pub ran: Vec<StepReport>,
    /// Steps skipped because `tsd.migrations` already had them done.
    pub already_done: usize,
}

impl PreparedRegistry {
    /// Run every pending step over `storage`'s scope, in schema order and step
    /// order, then open the migrated collections for writes.
    ///
    /// For each step the runner pages the source (the item itself, filtered to
    /// records below version `to`, or the legacy collection named by
    /// [`Step::source`]) by id with a keyset cursor, `batch_size` records at a
    /// time. It writes every output with `Precondition::Absent` and the
    /// schema version `to`. The batch's writes and its progress record are
    /// two steps, so a crash between them leaves outputs the cursor does not
    /// know; on resume the same outputs are produced again, and an output
    /// that already exists counts as written **only when it holds exactly the
    /// body the step produced**. A different document under the id, or a
    /// unique-index collision with another document, stops the step with the
    /// driver's `AlreadyExists` or `Conflict`. An in-place rewrite of a record
    /// under the same id replaces it instead, guarded by the version it read;
    /// a source record of an in-place step that is not re-emitted under its
    /// id (the step renamed or dropped it) is deleted once the batch's
    /// outputs are written. After each batch it stores `{cursor, written}` in
    /// `tsd.migrations`. Only when the source is exhausted does it delete a
    /// legacy source's rows and mark the step done. A crash or error anywhere leaves the progress of the
    /// last finished batch; calling `migrate` again resumes there and yields
    /// the same final state as an uninterrupted run, because steps are pure.
    ///
    /// **Stream steps** ([`Step::stream`]) list the streams under the step's
    /// source prefix and move them one at a time: read the source whole (paged
    /// by `batch_size`), map it, append the result to its new name in batches,
    /// seal the target if the source was sealed (as the stream backend
    /// reports it), then delete the source. Appends are conditional on the
    /// length the run observed. The
    /// per-source progress (target name and entries appended) is saved before
    /// the first append and after every batch. On resume the target's length
    /// says where to continue, so no entry is appended twice; a target that
    /// was not started by this step is refused with `Conflict`. Migrated
    /// streams start at offset 0 (see [`Step::stream`]).
    ///
    /// **Blob steps** ([`Step::blobs`]) move one blob at a time: put the
    /// target, then delete the source. An existing target is accepted only when
    /// its bytes and content type equal the step's output (an interrupted
    /// earlier write); otherwise the step fails with `Conflict` and keeps the
    /// source. The step's
    /// count is saved after every blob and a resume lists what is left.
    ///
    /// Run one `migrate` per scope at a time (hold a lease). Two concurrent
    /// runners do not corrupt the data (document writes are idempotent and
    /// stream appends are conditional, so the loser gets a `Conflict`) but
    /// waste work or fail.
    ///
    /// Readiness is recorded per scope (`storage.scope()`): see
    /// [`Registry::check_ready_in`]. Run `migrate` over every scope the host
    /// serves before relying on the unscoped [`Registry::check_ready`].
    /// The target collections must have been declared with
    /// [`Registry::prepare`]; the progress collection and a legacy source are
    /// declared here.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] for a zero batch size or a step that emits
    /// a record for a collection its owner does not declare; whatever the step
    /// function or the document store returns otherwise.
    pub(super) async fn migrate(
        &self,
        storage: &ScopedStorage,
        options: &MigrateOptions,
        executor: Option<&dyn TransformExecutor>,
    ) -> Result<MigrationReport> {
        if options.batch_size == 0 {
            return Err(StorageError::invalid_input(
                "migration batch size must be at least 1",
            ));
        }
        let docs = storage.documents();
        docs.ensure_collection(&migrations_spec()).await?;
        let migrations = Migrations::new(storage);

        let mut plan = Vec::new();
        let mut remaining: HashMap<&str, usize> = HashMap::new();
        for (owner, step) in self.steps() {
            let key = step_key(owner.as_str(), &step.item, &step.id)?;
            let progress = migrations.load(&key).await?;
            if !progress.as_ref().is_some_and(|found| found.done) {
                *remaining.entry(&step.item).or_default() += 1;
            }
            plan.push((owner, step, key, progress));
        }
        self.set_pending(
            storage.scope(),
            remaining
                .keys()
                .map(|item| (*item).to_owned())
                .collect::<HashSet<_>>(),
        );

        let mut report = MigrationReport::default();
        for (owner, step, key, progress) in plan {
            if progress.as_ref().is_some_and(|found| found.done) {
                report.already_done += 1;
                continue;
            }
            let written = match &step.run {
                StepBody::Documents(run) => {
                    StepRun {
                        registry: self,
                        owner,
                        step,
                        run,
                        executor,
                        docs,
                        batch_size: options.batch_size,
                    }
                    .run(&migrations, &key, progress)
                    .await?
                }
                StepBody::Streams(run) => {
                    let context = Context {
                        storage,
                        executor,
                        owner,
                        step,
                        migrations: &migrations,
                        key: &key,
                        batch_size: options.batch_size,
                    };
                    runner_streams::run(&context, run, progress).await?
                }
                StepBody::Blobs(run) => {
                    let context = Context {
                        storage,
                        executor,
                        owner,
                        step,
                        migrations: &migrations,
                        key: &key,
                        batch_size: options.batch_size,
                    };
                    runner_blobs::run(&context, run, progress).await?
                }
            };
            report.ran.push(StepReport {
                owner: owner.to_string(),
                item: step.item.clone(),
                step: step.id.clone(),
                written,
            });
            if let Some(left) = remaining.get_mut(step.item.as_str()) {
                *left -= 1;
                if *left == 0 {
                    self.mark_ready(storage.scope(), &step.item);
                }
            }
        }
        Ok(report)
    }
}

/// What a stream or blob step needs from the runner.
pub(super) struct Context<'a> {
    pub(super) storage: &'a ScopedStorage,
    pub(super) owner: &'a Owner,
    pub(super) step: &'a PreparedStep,
    pub(super) executor: Option<&'a dyn TransformExecutor>,
    pub(super) migrations: &'a Migrations,
    pub(super) key: &'a crate::Key,
    pub(super) batch_size: usize,
}

struct StepRun<'a> {
    registry: &'a PreparedRegistry,
    owner: &'a Owner,
    step: &'a PreparedStep,
    run: &'a Transform<StepFn>,
    executor: Option<&'a dyn TransformExecutor>,
    docs: &'a std::sync::Arc<dyn DocumentStore>,
    batch_size: usize,
}

impl StepRun<'_> {
    fn in_place(&self) -> bool {
        self.step.source_collection() == self.step.item
    }

    async fn run(
        &self,
        migrations: &Migrations,
        key: &crate::Key,
        progress: Option<Progress>,
    ) -> Result<u64> {
        let source = self.step.source_collection();
        if !self.in_place() {
            self.docs
                .ensure_collection(&CollectionSpec::new(source))
                .await?;
        }
        let mut progress = progress.unwrap_or_else(|| Progress {
            owner: self.owner.to_string(),
            step: self.step.id.clone(),
            from: self.step.from,
            to: self.step.to,
            cursor: None,
            written: 0,
            done: false,
        });
        // An in-place rewrite must not pick its own output up again, so it
        // reads only records still below the target version.
        let filter = if self.in_place() {
            Filter::lt(SCHEMA_FIELD, self.step.to)
        } else {
            Filter::default()
        };
        loop {
            let mut query = Query::filter(filter.clone())
                .sort(Sort::asc(ID_FIELD))
                .limit(self.batch_size);
            if let Some(cursor) = &progress.cursor {
                query = query.after(Cursor(cursor.clone()));
            }
            let page = self.docs.query(source, &query).await?;
            if !page.items.is_empty() {
                progress.written += self.write_batch(page.items).await?;
                progress.cursor = page.next.as_ref().map(|cursor| cursor.0.clone());
                migrations.save(key, &progress).await?;
            }
            if page.next.is_none() {
                break;
            }
        }
        // The source goes only now, with every batch written; a crash before
        // `done` is recorded re-runs against an emptier source and finishes.
        if !self.in_place() {
            self.docs.delete_where(source, &Filter::default()).await?;
        }
        progress.cursor = None;
        progress.done = true;
        migrations.save(key, &progress).await?;
        Ok(progress.written)
    }

    async fn map_records(&self, records: Vec<SourceRecord>) -> Result<Vec<TargetRecord>> {
        match self.run {
            Transform::Local(run) => run(records),
            Transform::External(descriptor) => {
                super::executor::required(self.executor)?
                    .documents(
                        self.owner,
                        super::DocumentsTransformRequest {
                            operation: descriptor.operation.clone(),
                            version: descriptor.version,
                            records,
                        },
                    )
                    .await
            }
        }
    }

    async fn write_batch(&self, items: Vec<crate::Versioned<serde_json::Value>>) -> Result<u64> {
        let versions: HashMap<String, Version> = items
            .iter()
            .map(|item| (item.id.clone(), item.version))
            .collect();
        // The expiry of each source record, for an output that rewrites it.
        let expiries: HashMap<String, Option<i64>> = items
            .iter()
            .map(|item| (item.id.clone(), item.meta.expires_ms))
            .collect();
        let records = items
            .into_iter()
            .map(|item| SourceRecord {
                id: item.id,
                body: item.doc,
                meta: item.meta,
            })
            .collect();
        let outputs = self.map_records(records).await?;
        // Validate every output before writing any, so a bad step leaves the
        // batch untouched.
        let schemas = outputs
            .iter()
            .map(|output| self.target_schema(output))
            .collect::<Result<Vec<_>>>()?;
        let count = outputs.len() as u64;
        // Ids the step re-emitted in place; every other source record was
        // replaced (or dropped) by the batch and goes once the outputs landed.
        let kept: HashSet<String> = if self.in_place() {
            outputs
                .iter()
                .filter(|output| output.collection == self.step.item)
                .map(|output| output.id.clone())
                .collect()
        } else {
            HashSet::new()
        };
        for (output, schema) in outputs.into_iter().zip(schemas) {
            let expires_ms = match output.expires {
                Expiry::At(ms) => Some(ms),
                // Only the record rewritten under its own id keeps its
                // deadline; a renamed or emitted-elsewhere record is new.
                Expiry::Carry if output.collection == self.step.item => {
                    expiries.get(&output.id).copied().flatten()
                }
                Expiry::Carry | Expiry::Never => None,
            };
            let mut write = WriteOptions::new().schema(schema);
            // Creation time is host-owned and the storage bus rejects every
            // module attempt to write it. Existing-row writes retain their
            // timestamp in the backend; new rows receive the host clock.
            let existing_same_row = self.in_place()
                && output.collection == self.step.item
                && versions.contains_key(&output.id);
            if let Some(expires) = expires_ms {
                write = write.expires_ms(expires);
            }
            let precondition = if self.in_place() && output.collection == self.step.item {
                versions
                    .get(&output.id)
                    .map_or(Precondition::Absent, |version| {
                        Precondition::Version(*version)
                    })
            } else {
                Precondition::Absent
            };
            let (collection, id, body) = (output.collection, output.id, output.body);
            let expected_created_ms = output.created_ms.filter(|_| existing_same_row);
            match self
                .docs
                .put_with(&collection, &id, body.clone(), precondition, write)
                .await
            {
                Ok(_) => {}
                // Only an interrupted run of this step may have left the
                // output behind, and then it holds exactly this body, schema,
                // and expiry. Any
                // other conflict (a stranger under the id, a unique-index
                // collision with another document, a concurrent edit) stops
                // the step.
                Err(error)
                    if matches!(error.kind(), ErrorKind::AlreadyExists | ErrorKind::Conflict) =>
                {
                    let stored = self.docs.get(&collection, &id).await?;
                    let identical = stored.is_some_and(|found| {
                        found.doc == body
                            && found.meta.schema == schema
                            && expected_created_ms
                                .is_none_or(|created| found.meta.created_ms == created)
                            && found.meta.expires_ms == expires_ms
                    });
                    if !identical {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        if self.in_place() {
            for (id, version) in &versions {
                if !kept.contains(id) {
                    self.remove_replaced(id, *version).await?;
                }
            }
        }
        Ok(count)
    }

    /// Delete an in-place source record the step replaced under another id.
    /// One already gone was removed by an earlier, interrupted run.
    async fn remove_replaced(&self, id: &str, version: Version) -> Result<()> {
        match self
            .docs
            .delete(&self.step.item, id, Precondition::Version(version))
            .await
        {
            Err(error) if error.kind() == ErrorKind::Conflict => {
                if self.docs.get(&self.step.item, id).await?.is_some() {
                    return Err(error);
                }
                Ok(())
            }
            other => other.map(|_| ()),
        }
    }

    /// The schema version stamped on an output, after checking it targets a
    /// collection of the step's owner.
    fn target_schema(&self, output: &TargetRecord) -> Result<u32> {
        let owned = output
            .collection
            .strip_prefix(self.owner.as_str())
            .is_some_and(|rest| rest.starts_with('.'));
        let info = self
            .registry
            .collections
            .get(&output.collection)
            .filter(|_| owned)
            .ok_or_else(|| {
                StorageError::invalid_input(format!(
                    "step `{}` of `{}` emitted a record for `{}`, which is not a collection declared by owner `{}`",
                    self.step.id, self.step.item, output.collection, self.owner
                ))
            })?;
        if self.in_place() && output.collection == self.step.source_collection() {
            return Ok(self.step.to);
        }
        Ok(if output.collection == self.step.item {
            self.step.to
        } else {
            info.version
        })
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
