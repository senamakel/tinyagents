use super::*;

/// Legacy SQL errors remain structured sources instead of becoming another display string.
#[test]
fn legacy_storage_error_keeps_original_type() {
    let error: LedgerError =
        tinyagents_harness::TinyAgentsError::Storage("unavailable".into()).into();
    assert!(
        matches!(error, LedgerError::Legacy(ref original) if matches!(**original, tinyagents_harness::TinyAgentsError::Storage(_)))
    );
}

/// Native unsupported failures retain their exact driver capability and error kind.
#[test]
fn native_driver_error_preserves_capability_and_kind() {
    use tinyagents_storage_kit::{Capability, ErrorKind, StorageError};
    let error: LedgerError = StorageError::unsupported(Capability::Transactions, "required").into();
    assert!(
        matches!(error, LedgerError::Driver(ref native) if native.kind() == ErrorKind::Unsupported(Capability::Transactions))
    );
}
