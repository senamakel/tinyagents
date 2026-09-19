//! Public integration coverage for workspace descriptors and the canonical
//! TinyTools workspace vocabulary.

use std::path::Path;

use tinytools::{SandboxMode, WorkspaceDescriptor};

#[test]
fn descriptor_allows_declared_roots_and_rejects_lexical_escapes() {
    let workspace = WorkspaceDescriptor::new("/work/a")
        .with_trusted_root("/shared")
        .with_sandbox(SandboxMode::Required);

    assert!(workspace.allows(Path::new("/work/a/output.txt")));
    assert!(workspace.allows(Path::new("/shared/input.txt")));
    assert!(!workspace.allows(Path::new("/etc/passwd")));
    assert!(!workspace.allows(Path::new("/work/a/../private/secret")));
}

#[test]
fn descriptor_round_trips_without_a_harness_reexport() {
    let workspace = WorkspaceDescriptor::new("/work/a")
        .with_trusted_root("/shared")
        .with_sandbox(SandboxMode::Required);
    let encoded = serde_json::to_string(&workspace).expect("workspace serializes");
    let decoded: WorkspaceDescriptor = serde_json::from_str(&encoded).expect("workspace parses");

    assert_eq!(decoded.root, workspace.root);
    assert_eq!(decoded.trusted_roots, workspace.trusted_roots);
    assert_eq!(decoded.sandbox, SandboxMode::Required);
}
