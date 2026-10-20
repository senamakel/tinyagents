# Task storage compatibility fixture

`legacy-31cc-tasks.json` was generated through the unmodified public
`DriverTaskStore` API available before this epic at TinyAgents commit
`31cc57180f67b2f539a3c98d03388251084f0da0`, using its checked-in
`Cargo.lock`. That lock resolves `tinystoragedrivers-core` to the canonical
`v1.0.0` tag at `141fd9ed0ad5d14a8fb5a1b3e837a719e81ac08d`.

To recreate the artifact from a clean checkout of that commit, run these
commands from the superproject root. The archival checkout is an ignored build
fixture under `target/`; the checked-in generator is copied into that checkout
without changing its source revision:

```sh
git clone https://github.com/tinyhumansai/tinyagents vendor/tinyagents/target/legacy-31cc-source
git -C vendor/tinyagents/target/legacy-31cc-source checkout --detach 31cc57180f67b2f539a3c98d03388251084f0da0
git -C vendor/tinyagents/target/legacy-31cc-source submodule update --init --recursive
mkdir -p vendor/tinyagents/target/legacy-31cc-source/crates/tinyagents-tasks/examples
cp vendor/tinyagents/crates/tinyagents-tasks/src/storage/fixtures/generate-31cc.rs \
  vendor/tinyagents/target/legacy-31cc-source/crates/tinyagents-tasks/examples/generate_storage_fixture.rs
cd vendor/tinyagents/target/legacy-31cc-source
cargo run --locked -p tinyagents-tasks --example generate_storage_fixture -- legacy-31cc-tasks.json
```

It inserts a task and records its pending, running, and completed snapshots
through the pre-epic public task-store API, then queries the public document
port and serializes the resulting document rows. The current task store test
seeds these rows into a real testkit memory backend and verifies that owner
open and the current public store read them back.

`legacy-v2.9.1-tasks.json` was generated the same way from the released
TinyAgents `v2.9.1` tag at `88690ac669fae677f2ebab506f03f6185f8ccc3a`. Its
checked-in lock resolves `tinystoragedrivers-core` to `v0.11.0` at
`9eeef54aea403ac07461a612189bdf4982782823`. Recreate it from this superproject
root with:

```sh
git clone https://github.com/tinyhumansai/tinyagents vendor/tinyagents/target/legacy-v2.9.1-source
git -C vendor/tinyagents/target/legacy-v2.9.1-source checkout --detach v2.9.1
git -C vendor/tinyagents/target/legacy-v2.9.1-source submodule update --init --recursive
mkdir -p vendor/tinyagents/target/legacy-v2.9.1-source/crates/tinyagents-tasks/examples
cp vendor/tinyagents/crates/tinyagents-tasks/src/storage/fixtures/generate-v2.9.1.rs \
  vendor/tinyagents/target/legacy-v2.9.1-source/crates/tinyagents-tasks/examples/generate_storage_fixture.rs
cd vendor/tinyagents/target/legacy-v2.9.1-source
cargo run --locked -p tinyagents-tasks --example generate_storage_fixture -- legacy-v2.9.1-tasks.json
```
