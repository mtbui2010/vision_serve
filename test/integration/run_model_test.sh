#!/usr/bin/env bash
# Functional integration test for one model using the Docker container.
#
# Usage:
#   ./run_model_test.sh MODEL TASK PORT [IMAGE_TAG] [PROMPT] [BOX] [EXTRA_PULLS]
#
# EXTRA_PULLS: space-separated list of models to pull BEFORE the main model.
#              Used by grounded-sam (needs grounding-dino + mobile-sam pulled first).
#
# Exit codes: 0=PASS, 1=FAIL, 2=SKIP (pull not available)
set -euo pipefail

MODEL=${1:?Usage: run_model_test.sh MODEL TASK PORT [IMAGE_TAG] [PROMPT] [BOX] [EXTRA_PULLS]}
TASK=${2:?task required}
PORT=${3:?port required}
IMAGE_TAG=${4:-visionserve:v0.1.0}
PROMPT=${5:-}
BOX=${6:-}
EXTRA_PULLS=${7:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAMPLE_IMAGE="$REPO_ROOT/test/testdata/sample.jpg"
MODELS_DIR="/tmp/vs-test-${MODEL}-$$"
CONTAINER_NAME="vs-test-${MODEL}-$$"
# Run Docker as the current user so downloaded files are host-writable (not root-owned).
DOCKER_USER="$(id -u):$(id -g)"

log() { echo "[$(date +%H:%M:%S)] [$MODEL] $*"; }

cleanup() {
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm   "$CONTAINER_NAME" 2>/dev/null || true
  rm -rf "$MODELS_DIR"
}
trap cleanup EXIT

mkdir -p "$MODELS_DIR"

# ── Wait for image to be available (build may be in progress) ──────────────
log "waiting for Docker image $IMAGE_TAG..."
for i in $(seq 1 40); do
  if docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    log "image ready"
    break
  fi
  if [[ $i -eq 40 ]]; then
    log "TIMEOUT: image not ready after 20 min"
    exit 1
  fi
  sleep 30
done

# ── Pull dependency weights (e.g. for grounded-sam) ───────────────────────
for dep in $EXTRA_PULLS; do
  log "pulling dependency: $dep"
  if ! docker run --rm --user "$DOCKER_USER" -v "$MODELS_DIR:/models" "$IMAGE_TAG" pull "$dep" --models /models; then
    log "SKIP: dependency pull failed for $dep"
    exit 2
  fi
done

# ── Pull model weights ─────────────────────────────────────────────────────
log "pulling model weights..."
PULL_OUT=$(docker run --rm --user "$DOCKER_USER" -v "$MODELS_DIR:/models" \
  "$IMAGE_TAG" pull "$MODEL" --models /models 2>&1) && PULL_OK=1 || PULL_OK=0
echo "$PULL_OUT"
if [[ $PULL_OK -eq 0 ]]; then
  if echo "$PULL_OUT" | grep -q "unknown model"; then
    # Model not in catalog: fall back to local manifest (e.g. grounded-sam reuses
    # sibling weights pulled by EXTRA_PULLS).
    LOCAL_MANIFEST="$REPO_ROOT/models/$MODEL/manifest.yaml"
    if [[ -f "$LOCAL_MANIFEST" ]]; then
      log "not in catalog — using local manifest (pipeline model)"
      mkdir -p "$MODELS_DIR/$MODEL"
      cp "$LOCAL_MANIFEST" "$MODELS_DIR/$MODEL/manifest.yaml"
    else
      log "SKIP: model not in catalog and no local manifest"
      exit 2
    fi
  else
    log "SKIP: pull failed (network error / auth required)"
    exit 2
  fi
fi

# ── CLI test (visionserve run — in-process, no server needed) ─────────────
log "CLI test (visionserve run)..."
CLI_ARGS=(run "$MODEL" /test/sample.jpg --models /models)
[[ -n "$PROMPT" ]] && CLI_ARGS+=(--prompt "$PROMPT")
[[ -n "$BOX"    ]] && CLI_ARGS+=(--box    "$BOX")

CLI_OUT=$(docker run --rm --user "$DOCKER_USER" \
  -v "$MODELS_DIR:/models" \
  -v "$SAMPLE_IMAGE:/test/sample.jpg:ro" \
  "$IMAGE_TAG" \
  "${CLI_ARGS[@]}")

# Use bash substring to avoid SIGPIPE when JSON is large (e.g. depth maps)
echo "${CLI_OUT:0:300}"
# Verify JSON output has the task field
echo "$CLI_OUT" | python3 -c "
import sys, json
j = json.load(sys.stdin)
assert j.get('task'), 'missing task field'
print('  CLI: JSON ok, task=' + j['task'])
"

# ── Start server container ─────────────────────────────────────────────────
log "starting server container on port $PORT..."
docker run -d --name "$CONTAINER_NAME" --user "$DOCKER_USER" \
  -p "$PORT:11435" \
  -v "$MODELS_DIR:/models" \
  "$IMAGE_TAG" \
  serve --addr :11435 --models /models

# Wait for health (up to 60s)
log "waiting for server health..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/api/health" 2>/dev/null | grep -q '"ok"'; then
    log "server ready (${i}×2s)"
    break
  fi
  if [[ $i -eq 30 ]]; then
    log "FAIL: server did not become healthy"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -20
    exit 1
  fi
  sleep 2
done

# ── Python client test ─────────────────────────────────────────────────────
log "Python client test..."
PYTHON_ARGS=(
  --host "http://localhost:$PORT"
  --model "$MODEL"
  --task  "$TASK"
  --image "$SAMPLE_IMAGE"
)
[[ -n "$PROMPT" ]] && PYTHON_ARGS+=(--prompt "$PROMPT")
[[ -n "$BOX"    ]] && PYTHON_ARGS+=(--box    "$BOX")

python3 "$SCRIPT_DIR/test_model.py" "${PYTHON_ARGS[@]}"

# ── JS client test ─────────────────────────────────────────────────────────
log "JS client test..."
node "$SCRIPT_DIR/test_model.mjs" \
  "http://localhost:$PORT" "$MODEL" "$TASK" "$SAMPLE_IMAGE" "$PROMPT" "$BOX"

log "=== PASS ==="
