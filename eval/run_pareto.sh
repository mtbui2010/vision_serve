#!/usr/bin/env bash
# Pareto / open-loop tail (W4b): sweep offered rate for VisionServe (GPU2) vs FastAPI+ORT (GPU3),
# both on the same ONNX, recording measured goodput + p50/p95/p99 at each rate.
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval; SP=$VSEVAL/lib/python3.12/site-packages
PY=$VSEVAL/bin/python; BIN=/tmp/visionserve
MODEL=${1:-mobilenet-v3}
RATES=${2:-25,50,100,200,400,600,800,1000,1400}
DUR=${3:-15}
IMAGE=$REPO/test/testdata/sample.jpg; RESULTS=$REPO/eval/results
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ pkill -f 'visionserve serve --addr :11436' 2>/dev/null; pkill -f 'fastapi_ort:app' 2>/dev/null; }
trap cleanup EXIT
wait_health(){ for i in $(seq 1 120); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

log "start VS GPU2:11436 + FastAPI+ORT GPU3:8001 ($MODEL)"
CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11436 --models "$REPO/models" --preload "$MODEL" >/tmp/pareto_vs.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 ONNX_PATH=$REPO/models/$MODEL/model.onnx MODEL_NAME=$MODEL TASK=classification \
  EP=CUDAExecutionProvider,CPUExecutionProvider \
  $VSEVAL/bin/uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port 8001 >/tmp/pareto_bl.log 2>&1 &
wait_health http://localhost:11436/api/health || { log "VS FAIL"; tail /tmp/pareto_vs.log; exit 1; }
wait_health http://localhost:8001/api/health || { log "BL FAIL"; tail /tmp/pareto_bl.log; exit 1; }
log "both healthy; rate sweep ($RATES)"

$PY -m eval.loadgen.rate_sweep --target http://localhost:11436 --label visionserve \
  --model "$MODEL" --image "$IMAGE" --rates "$RATES" --duration "$DUR" \
  --out "$RESULTS/pareto_${MODEL}_visionserve.csv"
$PY -m eval.loadgen.rate_sweep --target http://localhost:8001 --label fastapi-ort \
  --model "$MODEL" --image "$IMAGE" --rates "$RATES" --duration "$DUR" \
  --out "$RESULTS/pareto_${MODEL}_fastapi-ort.csv"
log "PARETO_DONE"
