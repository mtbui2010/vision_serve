#!/bin/bash
# E1 — device×EP matrix on a Jetson edge host.
# Runs every (model × EP) cell against VisionServe, capturing latency/throughput/energy
# via edge_bench.py. The EP is forced per run with VISIONSERVE_EP (cpu|cuda|tensorrt) and
# read back from Result.device. Pin clocks first for canonical numbers:
#     sudo nvpmodel -m 0 && sudo jetson_clocks      # MAXN
# then re-run; the nvpmodel mode is recorded in the CSV sidecar either way.
#
# Env:
#   ORT_DYLIB    path to a GPU-capable libonnxruntime.so (CUDA+TensorRT providers)
#   MODELS_DIR   model registry (default ./models)
#   OUT          results CSV (default eval/results/e1_matrix.csv)
#   EPS          space list of EPs (default "cpu cuda tensorrt")
#   CONC         concurrency list (default "1,8")
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)

ORT_DYLIB=${ORT_DYLIB:-$HOME/.local/ort-jetson/libonnxruntime.so}
MODELS_DIR=${MODELS_DIR:-./models}
OUT=${OUT:-eval/results/e1_matrix.csv}
EPS=${EPS:-"cpu cuda tensorrt"}
CONC=${CONC:-"1,8"}
BIN=${BIN:-$HOME/go/bin/visionserve}
PORT=11435
TARGET="http://localhost:$PORT"
IMAGES="test/testdata/sample.jpg,demo/images/000000039769.jpg,demo/images/000000001268.jpg,demo/images/000000000139.jpg"

# model -> extra args (segmentation PipelineModels need a box prompt)
declare -A EXTRA
EXTRA[mobilenet-v3]=""
EXTRA[efficientnet-b0]=""
EXTRA[rf-detr-nano]=""
EXTRA[mobile-sam]="--box 200,200,400,400"
MODELS=${MODELS:-"mobilenet-v3 efficientnet-b0 rf-detr-nano mobile-sam"}

rm -f "$OUT"
echo "== E1 matrix → $OUT (power mode: $(nvpmodel -q 2>/dev/null | grep -oE 'MODE_\w+|MAXN'))"

for ep in $EPS; do
  echo "=== EP=$ep : starting server ==="
  ORT_DYLIB_PATH="$ORT_DYLIB" VISIONSERVE_EP="$ep" LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}" \
    nohup "$BIN" serve --addr ":$PORT" --models "$MODELS_DIR" > /tmp/e1_$ep.log 2>&1 &
  PID=$!
  sleep 6
  if ! curl -s --max-time 5 "$TARGET/api/health" >/dev/null; then
    echo "server failed to start (EP=$ep); log:"; tail -5 /tmp/e1_$ep.log; kill $PID 2>/dev/null; continue
  fi
  for m in $MODELS; do
    echo "--- $m @ $ep ---"
    ORT_DYLIB_PATH="$ORT_DYLIB" VISIONSERVE_EP="$ep" \
      python3 eval/edge/edge_bench.py --target "$TARGET" --model "$m" \
        --images "$IMAGES" --concurrency "$CONC" --duration 10 \
        --server-pid "$PID" --out "$OUT" ${EXTRA[$m]:-}
  done
  kill $PID 2>/dev/null; sleep 2
done

echo "== done. matrix:"; column -t -s, "$OUT"
