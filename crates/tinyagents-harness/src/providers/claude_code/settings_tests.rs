use super::*;

#[test]
fn load_missing_file_returns_safe_defaults() {
    let dir = std::env::temp_dir().join("oh_cc_settings_missing_test");
    let _ = std::fs::remove_dir_all(&dir);
    let s = load(&dir);
    assert!(!s.full_access, "missing settings must default to OFF");
}

#[test]
fn save_then_load_roundtrips() {
    let dir = std::env::temp_dir().join("oh_cc_settings_roundtrip_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    save(&dir, &ClaudeCodeSettings { full_access: true }).unwrap();
    assert!(
        load(&dir).full_access,
        "saved full_access=true must persist"
    );
    save(&dir, &ClaudeCodeSettings { full_access: false }).unwrap();
    assert!(
        !load(&dir).full_access,
        "toggling back to false must persist"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_corrupt_file_returns_safe_defaults() {
    let dir = std::env::temp_dir().join("oh_cc_settings_corrupt_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(settings_path(&dir), b"{not json").unwrap();
    assert!(
        !load(&dir).full_access,
        "corrupt settings must fail safe to OFF"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
