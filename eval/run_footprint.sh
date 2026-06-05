#!/usr/bin/env bash
# Multi-model footprint (Exp 4): ONE Go binary serving N models vs N Python (FastAPI+ORT)
# processes serving one ONNX each. Measures idle (loaded, no traffic) RSS + VRAM.
# Substantiates the "single binary / shared runtime eliminates per-process overhead" claim.
# A6000-only, no core change. GPUs: VS on 2, Python procs on 3.
set -uo pipefail

REPO=/home/trung/trung_workdir/vision_serve
VSEVAL=/home/trung/miniconda3/envs/vseval
SP=$VSEVAL/lib/python3.12/site-packages
BIN=/tmp/visionserve
RESULTS=$REPO/eval/results
OUT=$RESULTS/footprint.json
mkdir -p "$RESULTS"

export ORT_DYLIB_PATH=$SP/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/onnxruntime/capi:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$REPO

MODELS=(mobilenet-v3 efficientnet-b0 clip scrfd)
log(){ echo "[$(date +%H:%M:%S)] $*"; }

PIDS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; pkill -f 'visionserve serve --addr :114' 2>/dev/null; pkill -f 'fastapi_ort:app' 2>/dev/null; }
trap cleanup EXIT

rss_kb(){ awk '/VmRSS/{print $2}' /proc/$1/status 2>/dev/null; }
wait_health(){ for i in $(seq 1 120); do curl -sf "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
# VRAM (MiB) summed for a set of pids on a given GPU index
vram_for_pids(){ local gpu=$1; shift; local pids="$* ";
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk -v want=" $pids" '{p=$1; gsub(/,/,"",p); if (index(want, " " p " ")>0) s+=$2} END{print s+0}'; }

# ---- A) ONE Go binary serving ALL models (GPU 2) ------------------------------------------
log "VisionServe: one process, preload ${MODELS[*]} (GPU 2 :11440)"
CUDA_VISIBLE_DEVICES=2 "$BIN" serve --addr :11440 --models "$REPO/models" \
  --preload "$(IFS=,; echo "${MODELS[*]}")" >/tmp/fp_vs.log 2>&1 &
VS_PID=$!; PIDS+=("$VS_PID")
wait_health http://localhost:11440/api/health || { log "VS FAILED"; tail -20 /tmp/fp_vs.log; exit 1; }
# give preloads time to finish (background goroutines)
for i in $(seq 1 60); do n=$(grep -c 'preloaded:' /tmp/fp_vs.log); [ "$n" -ge "${#MODELS[@]}" ] && break; sleep 1; done
sleep 3
VS_RSS=$(rss_kb "$VS_PID"); VS_VRAM=$(vram_for_pids 2 "$VS_PID")
log "VisionServe RSS=${VS_RSS}KB VRAM=${VS_VRAM}MiB (preloaded $(grep -c 'preloaded:' /tmp/fp_vs.log)/${#MODELS[@]})"

# ---- B) N Python processes, one ONNX each (GPU 3) -----------------------------------------
PY_PIDS=""; port=8011
for M in "${MODELS[@]}"; do
  log "FastAPI+ORT: $M (GPU 3 :$port)"
  CUDA_VISIBLE_DEVICES=3 ONNX_PATH=$REPO/models/$M/model.onnx MODEL_NAME=$M TASK=classification \
    EP=CUDAExecutionProvider,CPUExecutionProvider \
    $VSEVAL/bin/uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port $port \
    >/tmp/fp_py_$M.log 2>&1 &
  pid=$!; PIDS+=("$pid"); PY_PIDS="$PY_PIDS $pid"
  wait_health http://localhost:$port/api/health || { log "FastAPI $M FAILED"; tail -15 /tmp/fp_py_$M.log; }
  port=$((port+1))
done
sleep 3
PY_RSS=0; for p in $PY_PIDS; do r=$(rss_kb "$p"); PY_RSS=$((PY_RSS + ${r:-0})); done
PY_VRAM=$(vram_for_pids 3 $PY_PIDS)
log "Python total RSS=${PY_RSS}KB VRAM=${PY_VRAM}MiB across ${#MODELS[@]} procs"

# ---- report ------------------------------------------------------------------------------
python3 - "$VS_RSS" "$VS_VRAM" "$PY_RSS" "$PY_VRAM" "${#MODELS[@]}" >"$OUT" <<'PYEOF'
import json,sys
vs_rss,vs_vram,py_rss,py_vram,n=sys.argv[1:6]
vs_rss=int(vs_rss or 0); py_rss=int(py_rss or 0); n=int(n)
d={"n_models":n,"models":"mobilenet-v3,efficientnet-b0,clip,scrfd",
   "visionserve_one_process":{"rss_mb":round(vs_rss/1024,1),"vram_mb":int(vs_vram or 0)},
   "python_n_processes":{"rss_mb":round(py_rss/1024,1),"vram_mb":int(py_vram or 0)},
   "rss_saving_mb":round((py_rss-vs_rss)/1024,1),
   "vram_saving_mb":int((py_vram or 0))-int(vs_vram or 0),
   "note":"idle (models loaded, no traffic). RSS via /proc/pid/VmRSS; VRAM via nvidia-smi compute-apps."}
print(json.dumps(d,indent=2))
PYEOF
log "FOOTPRINT_DONE"; cat "$OUT"
