#!/usr/bin/env bash
# Benchmark the `grasp` task (7th capability): class-agnostic MobileSAM automask -> pure-Go
# analytic mask2grasp. Single-stream latency + low concurrency + cold-start + VRAM.
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
PY=$VSEVAL/bin/python
SP=$VSEVAL/lib/python3.12/site-packages
BIN=/tmp/vs-grasp
IMAGE=$REPO/test/testdata/sample.jpg
RESULTS=$REPO/eval/results
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
log(){ echo "[$(date +%H:%M:%S)] $*"; }

cd "$REPO" && go build -o "$BIN" ./cmd/visionserve || { log "BUILD FAIL"; exit 1; }

log "start server (GPU 2) :11438 preload grasp"
CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11438 --models "$REPO/models" --preload grasp \
  >/tmp/vs_grasp.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf http://localhost:11438/api/health >/dev/null 2>&1 && break; sleep 1; done

log "smoke test: POST grasp"
curl -s -m 120 -X POST http://localhost:11438/api/predict \
  -F "model=grasp" -F "image=@$IMAGE" -o /tmp/grasp_resp.json -w "HTTP %{http_code} %{time_total}s\n"
echo "  response head:"; head -c 400 /tmp/grasp_resp.json; echo
echo "  #grasps:"; grep -o '"x":' /tmp/grasp_resp.json | wc -l

log "VRAM during inference:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed -n '3p'

log "cold-start from log:"; grep -iE "preloaded|cold" /tmp/vs_grasp.log | head

log "concurrency sweep (C=1,2,4; dur 15s; timeout 120s/req)"
$PY -m eval.loadgen.sweep_openloop --target http://localhost:11438 --label visionserve \
  --model grasp --image "$IMAGE" --concurrency 1,2,4 --duration 15 --timeout 120 \
  --out "$RESULTS/grasp_visionserve.csv" 2>&1 | tail -12

log "DONE — $RESULTS/grasp_visionserve.csv"
