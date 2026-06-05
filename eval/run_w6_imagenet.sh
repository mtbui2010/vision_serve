#!/usr/bin/env bash
# W6 Part B: ImageNet-val Top-1 measured THROUGH the running VisionServe server.
# Runs two classification models in parallel, one per free GPU, against the 5k val subset.
set -uo pipefail

REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
PY=$VSEVAL/bin/python
SP=$VSEVAL/lib/python3.12/site-packages
BIN=/tmp/visionserve
VAL=/mnt/nas/huggingface/trung_w6/imagenet_val
LABELS=$REPO/internal/catalog/labels/imagenet1k.txt
RESULTS=$REPO/eval/results
LIMIT=${1:-5000}
mkdir -p "$RESULTS"

export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO

log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ pkill -f 'visionserve serve --addr :11436' 2>/dev/null; pkill -f 'visionserve serve --addr :11437' 2>/dev/null; }
trap cleanup EXIT

log "start VisionServe GPU2 :11436 (mobilenet-v3) + GPU3 :11437 (efficientnet-b0)"
CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11436 --models "$REPO/models" --preload mobilenet-v3   >/tmp/w6_vs_mnet.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 "$BIN" serve --addr :11437 --models "$REPO/models" --preload efficientnet-b0 >/tmp/w6_vs_enet.log 2>&1 &

wait_health(){ for i in $(seq 1 120); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
wait_health http://localhost:11436/api/health || { log "mnet server FAILED"; tail -20 /tmp/w6_vs_mnet.log; exit 1; }
wait_health http://localhost:11437/api/health || { log "enet server FAILED"; tail -20 /tmp/w6_vs_enet.log; exit 1; }
log "both healthy; evaluating Top-1 on $LIMIT images (parallel)"

$PY -m eval.accuracy.task_eval imagenet --target http://localhost:11436 --model mobilenet-v3 \
  --images "$VAL/images" --gt "$VAL/gt.txt" --labels "$LABELS" --limit "$LIMIT" \
  --out "$RESULTS/w6_top1_mobilenet-v3.json" >/tmp/w6_eval_mnet.log 2>&1 &
P1=$!
$PY -m eval.accuracy.task_eval imagenet --target http://localhost:11437 --model efficientnet-b0 \
  --images "$VAL/images" --gt "$VAL/gt.txt" --labels "$LABELS" --limit "$LIMIT" \
  --out "$RESULTS/w6_top1_efficientnet-b0.json" >/tmp/w6_eval_enet.log 2>&1 &
P2=$!
wait $P1; wait $P2
log "=== RESULTS ==="
echo "--- mobilenet-v3 ---"; cat "$RESULTS/w6_top1_mobilenet-v3.json"
echo "--- efficientnet-b0 ---"; cat "$RESULTS/w6_top1_efficientnet-b0.json"
log "W6_DONE"
