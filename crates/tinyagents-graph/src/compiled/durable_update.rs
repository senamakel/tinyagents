//! Best-effort serialization of a completed task's `Update` for durable
//! persistence, without adding a `Serialize`/`DeserializeOwned` bound to
//! [`super::CompiledGraph`]'s `Update` type parameter.
//!
//! See the C1/C2 findings in `docs/runtime-comparison/code-review-graph.md`:
//! a durable backend recorded only a *completion marker* (`payload: null`)
//! for each finished task, because the executor is generic over `Update`
//! with no `Serialize` bound, so it had no way to persist what a task
//! actually wrote. The applied value is already durable in the checkpoint's
//! `state`, but the write ledger's `payload` — meant to let a resume
//! inspect/replay a specific task's write independent of the merged state —
//! carried nothing.
//!
//! [`DurableUpdate`] closes that gap for the (common) case where the
//! concrete `Update` a graph is compiled with happens to implement
//! [`serde::Serialize`], while leaving graphs whose `Update` does not
//! implement it exactly as before (a `null` marker payload). This is done
//! with the "autoref specialization" pattern: [`durable_payload`] resolves,
//! at compile time and without any bound on the call site, to the
//! `Serialize`-based impl when the argument's concrete type supports it, and
//! to the fallback otherwise. No trait object, `Any`, or public API change
//! is involved — this is purely an internal helper used by the checkpoint
//! persistence path in `boundary.rs`.
//!
//! # Why not a real trait bound
//!
//! Bounding `Update: Serialize + serde::de::DeserializeOwned` on
//! `CompiledGraph`/`StepRunner`/etc. would be a breaking API change for
//! every existing caller whose `Update` type is not (de)serializable — and
//! the in-memory execution path has never needed that bound, since applied
//! updates only ever need to be *moved*, not persisted. Autoref
//! specialization keeps the bound-free API while still extracting a real
//! payload wherever the concrete type allows it.

use serde::Serialize;

/// Wraps a `&T` so inherent method resolution can pick between the
/// `Serialize`-bounded impl and the unconditional fallback below, based on
/// whether `T: Serialize` holds for the concrete type at the call site.
struct Wrap<'a, T>(&'a T);

/// Fallback: implemented for `&Wrap<'_, T>` (one level of autoref) for any
/// `T`, unconditionally. Reached only when the specialized impl below does
/// not apply to `T`.
trait FallbackPayload {
    fn durable_payload(&self) -> Option<serde_json::Value>;
}

impl<T> FallbackPayload for &Wrap<'_, T> {
    fn durable_payload(&self) -> Option<serde_json::Value> {
        None
    }
}

/// Specialized: implemented directly for `Wrap<'_, T>` (zero levels of
/// autoref) whenever `T: Serialize`. Method resolution tries the
/// zero-autoref candidate first, so this wins over the fallback whenever it
/// is available.
trait SerializedPayload {
    fn durable_payload(&self) -> Option<serde_json::Value>;
}

impl<T: Serialize> SerializedPayload for Wrap<'_, T> {
    fn durable_payload(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.0).ok()
    }
}

/// Best-effort serialization of `value` for the durable checkpoint write
/// ledger: `Some(payload)` when the concrete `Update` type implements
/// [`serde::Serialize`] and serialized successfully, `None` otherwise (the
/// caller falls back to a `null` completion marker, preserving the
/// pre-existing behavior for non-serializable `Update` types).
pub(super) fn durable_payload<T>(value: &T) -> Option<serde_json::Value> {
    // `Wrap(value).durable_payload()`: method lookup tries the receiver's
    // by-value type `Wrap<T>` first (matching `SerializedPayload for
    // Wrap<'_, T>` when `T: Serialize`), and only autorefs to `&Wrap<T>` —
    // matching the unconditional `FallbackPayload for &Wrap<'_, T>` impl —
    // when the specialized impl does not exist for `T`.
    Wrap(value).durable_payload()
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(serde::Serialize)]
    struct Serializable {
        value: u32,
    }

    struct NotSerializable(#[allow(dead_code)] u32);

    #[test]
    fn serializable_update_yields_payload() {
        let value = Serializable { value: 7 };
        assert_eq!(
            durable_payload(&value),
            Some(serde_json::json!({ "value": 7 }))
        );
    }

    #[test]
    fn non_serializable_update_yields_none() {
        let value = NotSerializable(7);
        assert_eq!(durable_payload(&value), None);
    }
}
