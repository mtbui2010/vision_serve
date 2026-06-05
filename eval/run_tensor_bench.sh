#!/usr/bin/env bash
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval; SP=$VSEVAL/lib/python3.12/site-packages
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cleanup(){ pkill -f 'visionserve serve --addr :11436' 2>/dev/null; }
trap cleanup EXIT
pkill -f 'visionserve serve --addr :11436' 2>/dev/null; sleep 1
log "start VS (GPU2 :11436) mobilenet-v3"
CUDA_VISIBLE_DEVICES=2 /tmp/visionserve serve --addr :11436 --models "$REPO/models" --preload mobilenet-v3 >/tmp/vs_tensor.log 2>&1 &
for i in $(seq 1 90); do curl -sf localhost:11436/api/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf localhost:11436/api/health >/dev/null 2>&1 || { log "FAIL"; tail /tmp/vs_tensor.log; exit 1; }
log "tensor-in benchmark ..."
$VSEVAL/bin/python -m eval.baselines.vs_tensor_bench --url http://localhost:11436 \
  --model mobilenet-v3 --shape 1,3,224,224 --concurrency 1,2,4,8,16,32,64 --duration 20 \
  --out "$REPO/eval/results/vs_tensor_mobilenet-v3.csv"
log "TENSOR_BENCH_DONE"
