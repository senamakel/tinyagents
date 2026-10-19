#!/usr/bin/env bash
# Start the same single-node replica-set fixture used by tinystoragedrivers.
set -euo pipefail

port="${1:-27017}"
image="${2:-mongo:7}"
name="tinyagents-session-mongo"

cleanup() {
  if [[ $? -ne 0 ]]; then
    docker rm -f "$name" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

docker run -d --rm --name "$name" -p "127.0.0.1:${port}:27017" "$image" \
  --replSet rs0 --bind_ip_all >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$name" mongosh --quiet --eval 'db.adminCommand({ping: 1}).ok' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

docker exec "$name" mongosh --quiet --eval \
  'rs.initiate({_id: "rs0", members: [{_id: 0, host: "localhost:27017"}]})' >/dev/null

for _ in $(seq 1 60); do
  if [[ "$(docker exec "$name" mongosh --quiet --eval 'db.hello().isWritablePrimary')" == "true" ]]; then
    echo "MongoDB replica set ready on port ${port}"
    exit 0
  fi
  sleep 1
done

echo "MongoDB replica set did not elect a primary" >&2
exit 1
