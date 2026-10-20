#[path = "../old-release/session_store.rs"]
mod session_store;
#[path = "../old-release/settings.rs"]
mod settings;

use std::path::PathBuf;

fn main() {
    let output = PathBuf::from(std::env::args().nth(1).expect("fixture output directory"));
    let session_dir = output.join("sessions");
    std::fs::create_dir_all(&session_dir).expect("create session fixture directory");

    let sessions = session_store::SessionStore::open(&session_dir);
    sessions.set("same", "uuid-a").expect("write matching session");
    sessions
        .set("different", "uuid-old")
        .expect("write replaced session");
    sessions.set("new", "uuid-new").expect("write new session");
    sessions
        .record_delivered("same", &["old-turn".to_owned()])
        .expect("write delivered session turn");
    sessions
        .record_delivered("orphan", &["orphan-turn".to_owned()])
        .expect("write delivered-only thread record");

    let settings_dir = output.join("settings");
    settings::save(
        &settings_dir,
        &settings::ClaudeCodeSettings { full_access: true },
    )
    .expect("write settings fixture");
}
