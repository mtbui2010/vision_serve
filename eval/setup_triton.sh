#!/usr/bin/env bash
# Triton baseline setup (Exp6): build an ONNX model repository + pull the Triton server image.
# Does NOT start the GPU container (run that after the Pareto sweep frees GPU3, to avoid contention).
set -uo pipefail
REPO=/home/trung/trung_workdir/vision_serve
TR=/tmp/triton_repo
MODEL=${1:-mobilenet-v3}
log(){ echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$TR/$MODEL/1"
cp "$REPO/models/$MODEL/model.onnx" "$TR/$MODEL/1/model.onnx"
# ONNX I/O verified earlier: input 'x' [1,3,224,224], output '400' [1,1000]; max_batch_size 0.
cat > "$TR/$MODEL/config.pbtxt" <<EOF
name: "$MODEL"
backend: "onnxruntime"
max_batch_size: 0
input [ { name: "x", data_type: TYPE_FP32, dims: [1, 3, 224, 224] } ]
output [ { name: "400", data_type: TYPE_FP32, dims: [1, 1000] } ]
instance_group [ { kind: KIND_GPU, count: 1, gpus: [0] } ]
EOF
log "model repo ready at $TR/$MODEL"

IMG=""
for tag in nvcr.io/nvidia/tritonserver:24.08-py3 nvcr.io/nvidia/tritonserver:24.06-py3 nvcr.io/nvidia/tritonserver:23.10-py3; do
  log "pulling $tag ..."
  if docker pull "$tag" >/tmp/triton_pull.log 2>&1; then IMG="$tag"; break; fi
  log "  pull failed for $tag (see /tmp/triton_pull.log tail): $(tail -1 /tmp/triton_pull.log)"
done
if [ -z "$IMG" ]; then log "ERROR: could not pull any tritonserver image"; exit 1; fi
echo "$IMG" > /tmp/triton_image.txt
log "TRITON_IMAGE=$IMG ready. SETUP_DONE"
