use super::*;

#[test]
fn composite_keys_keep_the_legacy_encoding_for_unicode_and_reserved_delimiters() {
    assert_eq!(
        composite_key(["~owner", "a/b", "100%", "☃"]).unwrap(),
        "%7Eowner/a%2Fb/100%25/☃"
    );
    let long = "x".repeat(513);
    assert_eq!(composite_key([long.as_str()]).unwrap().len(), 53);
}

#[test]
fn scoped_storage_exposes_the_queue_and_secret_ports_from_its_bundle() {
    let storage = Storage::unsupported();
    let scoped = ScopedStorage::new(storage.clone(), Scope::new("local").unwrap());
    assert!(std::sync::Arc::ptr_eq(storage.queues(), scoped.queues()));
    assert!(std::sync::Arc::ptr_eq(storage.secrets(), scoped.secrets()));
}
