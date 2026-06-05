#!/usr/bin/env bash
# W6 Part B (detection): COCO val2017 mAP@[.5:.95] for RF-DETR, measured THROUGH the server.
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval; SP=$VSEVAL/lib/python3.12/site-packages
PY=$VSEVAL/bin/python; BIN=/tmp/visionserve
COCO=/mnt/nas/huggingface/trung_w6/coco
LIMIT=${1:-1000}
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ pkill -f 'visionserve serve --addr :11436' 2>/dev/null; }
trap cleanup EXIT

pkill -f 'visionserve serve --addr :11436' 2>/dev/null; sleep 1
log "start RF-DETR (GPU2 :11436)"
CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11436 --models "$REPO/models" --preload rf-detr >/tmp/rfdetr.log 2>&1 &
for i in $(seq 1 90); do curl -sf localhost:11436/api/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf localhost:11436/api/health >/dev/null 2>&1 || { log "server FAILED"; tail -20 /tmp/rfdetr.log; exit 1; }
# wait for preload to finish
for i in $(seq 1 60); do grep -q 'preloaded: rf-detr' /tmp/rfdetr.log && break; sleep 1; done
log "healthy; running COCO mAP on $LIMIT images (single-threaded through server)"

$PY -m eval.accuracy.task_eval coco --target http://localhost:11436 --model rf-detr \
  --images "$COCO/val2017" --gt "$COCO/annotations/instances_val2017.json" \
  --limit "$LIMIT" --out "$REPO/eval/results/w6_coco_rf-detr.json"
log "W6_COCO_DONE"