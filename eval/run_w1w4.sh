#!/usr/bin/env bash
# W1 (engine-controlled baseline) + W4 (concurrency sweep) runner.
#
# Brings up, for ONE classification model, the two systems-under-test on separate GPUs:
#   - VisionServe (Go)         GPU $VS_GPU   :11436
#   - FastAPI + onnxruntime    GPU $BL_GPU   :8001   (same .onnx, CUDA EP -> W1 control)
# then runs the pure-Python concurrency sweep (eval/loadgen/sweep_openloop.py) against each
# and writes CSVs to eval/results/. No numbers are committed; CSVs land in the git-ignored dir.
#
# Usage:  eval/run_w1w4.sh <model> [concurrency] [duration_s]
# Example: eval/run_w1w4.sh mobilenet-v3 1,2,4,8,16,32,64 20
set -uo pipefail

REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
PY=$VSEVAL/bin/python
SP=$VSEVAL/lib/python3.12/site-packages
ORT_LIB=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
BIN=/tmp/visionserve

MODEL=${1:?usage: run_w1w4.sh <model> [concurrency] [duration]}
CONC=${2:-1,2,4,8,16,32,64}
DUR=${3:-20}
VS_GPU=${VS_GPU:-2}
BL_GPU=${BL_GPU:-3}
IMAGE=$REPO/test/testdata/sample.jpg
RESULTS=$REPO/eval/results
mkdir -p "$RESULTS"

export ORT_DYLIB_PATH=$ORT_LIB
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO

ONNX=$REPO/models/$MODEL/model.onnx
LABELS=$REPO/internal/catalog/labels/imagenet1k.txt

log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ [[ -n "${VS_PID:-}" ]] && kill "$VS_PID" 2>/dev/null; [[ -n "${BL_PID:-}" ]] && kill "$BL_PID" 2>/dev/null; }
trap cleanup EXIT

# ---- 1. VisionServe (Go) on VS_GPU --------------------------------------------------------
log "starting VisionServe (GPU $VS_GPU) :11436 model=$MODEL"
CUDA_VISIBLE_DEVICES=$VS_GPU VISIONSERVE_MODELS=$REPO/models \
  "$BIN" serve --addr :11436 --models "$REPO/models" --preload "$MODEL" \
  >/tmp/vs_$MODEL.log 2>&1 &
VS_PID=$!

# ---- 2. FastAPI + onnxruntime baseline on BL_GPU ------------------------------------------
log "starting FastAPI+ORT baseline (GPU $BL_GPU) :8001 model=$MODEL"
CUDA_VISIBLE_DEVICES=$BL_GPU ONNX_PATH=$ONNX MODEL_NAME=$MODEL TASK=classification \
  LABELS_PATH=$LABELS EP=CUDAExecutionProvider,CPUExecutionProvider \
  $VSEVAL/bin/uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port 8001 \
  >/tmp/bl_$MODEL.log 2>&1 &
BL_PID=$!

# ---- 3. wait for both to be healthy -------------------------------------------------------
wait_health(){ for i in $(seq 1 120); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
log "waiting for VisionServe health..."
wait_health http://localhost:11436/api/health || { log "VisionServe FAILED"; tail -30 /tmp/vs_$MODEL.log; exit 1; }
log "waiting for baseline health..."
wait_health http://localhost:8001/api/health || { log "baseline FAILED"; tail -30 /tmp/bl_$MODEL.log; exit 1; }
log "both healthy."

# ---- 4. sweeps (sequential: keep client CPU uncontended for fair latency) ------------------
log "sweep: VisionServe ..."
$PY -m eval.loadgen.sweep_openloop --target http://localhost:11436 --label visionserve \
  --model "$MODEL" --image "$IMAGE" --concurrency "$CONC" --duration "$DUR" \
  --out "$RESULTS/w1w4_${MODEL}_visionserve.csv"

log "sweep: FastAPI+ORT baseline ..."
$PY -m eval.loadgen.sweep_openloop --target http://localhost:8001 --label fastapi-ort \
  --model "$MODEL" --image "$IMAGE" --concurrency "$CONC" --duration "$DUR" \
  --out "$RESULTS/w1w4_${MODEL}_fastapi-ort.csv"

log "DONE model=$MODEL — CSVs in $RESULTS/"
