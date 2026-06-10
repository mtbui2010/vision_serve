#!/usr/bin/env bash
# Benchmark the remaining composite pipelines: grounded-sam (text->box->mask),
# grasp-rfdetr (rf-detr boxes -> SAM -> analytic grasp), grasp-gd (text -> SAM -> grasp).
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
PY=$VSEVAL/bin/python
SP=$VSEVAL/lib/python3.12/site-packages
BIN=/tmp/vs-remain
IMAGE=$REPO/test/testdata/sample.jpg
RESULTS=$REPO/eval/results
PROMPT="person. car. dog. bag. chair. cup. bottle. book."
export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO
log(){ echo "[$(date +%H:%M:%S)] $*"; }
cd "$REPO" && go build -o "$BIN" ./cmd/visionserve || { log "BUILD FAIL"; exit 1; }

bench(){ # model  prompt(optional)
  local m=$1 prompt=${2:-}
  log "=== $m ==="
  CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11439 --models "$REPO/models" --preload "$m" \
    >/tmp/vs_$m.log 2>&1 & local srv=$!
  for i in $(seq 1 90); do curl -sf http://localhost:11439/api/health >/dev/null 2>&1 && break; sleep 1; done
  local pf=(); [[ -n "$prompt" ]] && pf=(-F "prompt=$prompt")
  log "smoke:"; curl -s -m 180 -X POST http://localhost:11439/api/predict \
    -F "model=$m" "${pf[@]}" -F "image=@$IMAGE" -o /tmp/resp_$m.json \
    -w "  HTTP %{http_code} %{time_total}s\n"
  echo "  #detections=$(grep -o '"bbox"' /tmp/resp_$m.json | wc -l)  #grasps=$(grep -o '"theta"' /tmp/resp_$m.json | wc -l)  #masks=$(grep -o '"rle"' /tmp/resp_$m.json | wc -l)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed -n '3p' | sed 's/^/  VRAM: /'
  grep -iE "preloaded|error|refus" /tmp/vs_$m.log | head -3
  local promptarg=(); [[ -n "$prompt" ]] && promptarg=(--prompt "$prompt")
  $PY -m eval.loadgen.sweep_openloop --target http://localhost:11439 --label visionserve \
    --model "$m" --image "$IMAGE" "${promptarg[@]}" --concurrency 1,2,4 --duration 15 --timeout 180 \
    --out "$RESULTS/${m}_visionserve.csv" 2>&1 | grep -E '^\[sweep\]|wrote' | tail -8
  kill $srv 2>/dev/null; sleep 3
}

bench grasp-rfdetr ""
bench grounded-sam "$PROMPT"
bench grasp-gd "$PROMPT"
log "ALL DONE — CSVs in $RESULTS/ ({grasp-rfdetr,grounded-sam,grasp-gd}_visionserve.csv)"
