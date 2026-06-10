#!/usr/bin/env bash
# D1 (anti-strawman: multi-worker baseline) + D2 (heavier-model concurrency).
#
# D1: FastAPI+ORT baseline swept at uvicorn --workers {1,2,4,8} vs VisionServe, mobilenet-v3.
#     Answers reviewer: how much does a multi-worker baseline narrow VisionServe's ~8x peak?
# D2: rf-detr-nano (heavier detection model) concurrency sweep, VisionServe vs baseline
#     (1 worker + 4 workers). Answers reviewer: does the crossover hold beyond a tiny classifier?
#
# Closed-loop sweep, errors=0 validity gate enforced by sweep_openloop. CSVs -> eval/results/.
set -uo pipefail

REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
PY=$VSEVAL/bin/python
SP=$VSEVAL/lib/python3.12/site-packages
ORT_LIB=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
BIN=/tmp/visionserve
VS_GPU=2
BL_GPU=3
IMAGE=$REPO/test/testdata/sample.jpg
RESULTS=$REPO/eval/results
LABELS=$REPO/internal/catalog/labels/imagenet1k.txt
mkdir -p "$RESULTS"

export ORT_DYLIB_PATH=$ORT_LIB
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO

log(){ echo "[$(date +%H:%M:%S)] $*"; }
PIDS=()
kill_all(){ for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; PIDS=(); sleep 2; }
trap kill_all EXIT
wait_health(){ for i in $(seq 1 180); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

log "build visionserve"
( cd "$REPO" && go build -o "$BIN" ./cmd/visionserve ) || { log "BUILD FAIL"; exit 1; }

start_vs(){ # model
  CUDA_VISIBLE_DEVICES=$VS_GPU "$BIN" serve --addr :11436 --models "$REPO/models" --preload "$1" \
    >/tmp/vs_$1.log 2>&1 & PIDS+=($!); }
start_bl(){ # model task w h letterbox workers
  local m=$1 task=$2 w=$3 h=$4 lb=$5 wk=$6
  CUDA_VISIBLE_DEVICES=$BL_GPU ONNX_PATH=$REPO/models/$m/$7 MODEL_NAME=$m TASK=$task \
    INPUT_W=$w INPUT_H=$h LETTERBOX=$lb LABELS_PATH=$LABELS \
    EP=CUDAExecutionProvider,CPUExecutionProvider \
    $VSEVAL/bin/uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port 8001 --workers $wk \
    >/tmp/bl_${m}_w${wk}.log 2>&1 & PIDS+=($!); }

############################ D1: mobilenet-v3 multi-worker ############################
log "=== D1: mobilenet-v3 ==="
start_vs mobilenet-v3
wait_health http://localhost:11436/api/health || { log "VS FAIL"; tail -20 /tmp/vs_mobilenet-v3.log; }
$PY -m eval.loadgen.sweep_openloop --target http://localhost:11436 --label visionserve \
  --model mobilenet-v3 --image "$IMAGE" --concurrency 1,8,32,64 --duration 15 \
  --out "$RESULTS/d1_visionserve_mobilenet-v3.csv"
kill_all

for WK in 1 2 4 8; do
  log "D1 baseline workers=$WK"
  start_bl mobilenet-v3 classification 224 224 false $WK model.onnx
  wait_health http://localhost:8001/api/health || { log "BL w$WK FAIL"; tail -20 /tmp/bl_mobilenet-v3_w${WK}.log; kill_all; continue; }
  $PY -m eval.loadgen.sweep_openloop --target http://localhost:8001 --label "fastapi-ort-w${WK}" \
    --model mobilenet-v3 --image "$IMAGE" --concurrency 1,8,32,64 --duration 15 \
    --out "$RESULTS/d1_fastapi-ort-w${WK}_mobilenet-v3.csv"
  kill_all
done

############################ D2: rf-detr-nano concurrency ############################
log "=== D2: rf-detr-nano ==="
start_vs rf-detr-nano
wait_health http://localhost:11436/api/health || { log "VS FAIL"; tail -20 /tmp/vs_rf-detr-nano.log; }
$PY -m eval.loadgen.sweep_openloop --target http://localhost:11436 --label visionserve \
  --model rf-detr-nano --image "$IMAGE" --concurrency 1,2,4,8,16,32,64 --duration 15 \
  --out "$RESULTS/d2_visionserve_rf-detr-nano.csv"
kill_all

for WK in 1 4; do
  log "D2 baseline workers=$WK"
  start_bl rf-detr-nano detection 384 384 true $WK rf-detr-base.onnx
  wait_health http://localhost:8001/api/health || { log "BL w$WK FAIL"; tail -20 /tmp/bl_rf-detr-nano_w${WK}.log; kill_all; continue; }
  $PY -m eval.loadgen.sweep_openloop --target http://localhost:8001 --label "fastapi-ort-w${WK}" \
    --model rf-detr-nano --image "$IMAGE" --concurrency 1,2,4,8,16,32,64 --duration 15 \
    --out "$RESULTS/d2_fastapi-ort-w${WK}_rf-detr-nano.csv"
  kill_all
done

log "ALL DONE — CSVs in $RESULTS/ (d1_*, d2_*)"
