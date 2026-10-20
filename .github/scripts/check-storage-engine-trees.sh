#!/usr/bin/env bash
set -euo pipefail

blocked='rusqlite|libsqlite3-sys|mongodb|bson|aws-sdk-s3|lapin|amq-protocol|sqlx|redb|sled|rocksdb|^tinystoragedrivers$|tinystoragedrivers-(core|sqlite|mongodb|document|kv|blob|queue|secrets|connect)'
root=$(pwd -P)

has_blocked_dependency() {
  awk '{print $1}' | grep -xE "$blocked"
}

self_test() {
  local benign
  local engine
  benign=$(printf '%s\n' \
    'tinyagents-storage-kit v2.2.0' \
    'tinystoragedrivers-api v2.0.0' \
    'tinystoragedrivers-types v2.0.0' \
    'tinystoragedrivers-testkit v2.0.0')
  if has_blocked_dependency <<<"$benign" >/dev/null; then
    echo 'storage-engine dependency check rejected a port or testkit crate' >&2
    return 1
  fi
  for engine in \
    'rusqlite v0.31.0' \
    'tinystoragedrivers v2.0.0' \
    'tinystoragedrivers-mongodb v1.0.0' \
    'tinystoragedrivers-document v2.0.0'; do
    if ! has_blocked_dependency <<<"$engine" >/dev/null; then
      printf 'storage-engine dependency check did not reject %s\n' "$engine" >&2
      return 1
    fi
  done
}

self_test

mapfile -t packages < <(
  cargo metadata --no-deps --format-version 1 |
    jq -r --arg root "$root" '
      .packages[]
      | select(.manifest_path | startswith($root + "/crates/"))
      | select(any(.targets[]; any(.kind[]; IN("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"))))
      | .name
    ' | sort -u
)

if ((${#packages[@]} == 0)); then
  echo 'no TinyAgents library targets found in workspace metadata' >&2
  exit 1
fi

for package in "${packages[@]}"; do
  tree=$(cargo tree -p "$package" --all-features -e normal,build --prefix none)
  if matches=$(has_blocked_dependency <<<"$tree"); then
    printf 'normal/build dependency tree for %s contains a storage engine:\n%s\n' "$package" "$matches" >&2
    exit 1
  fi
  printf 'checked %s\n' "$package"
done
