# VisionServe evaluation harness

Runnable, reviewer-proof experiment harness for the VisionServe paper revision.
It implements the experiment *designs* from `paper/REVISION-PLAN.md` (Sections 2–3):

| Dir                  | Covers          | What it does                                                                       |
|----------------------|-----------------|------------------------------------------------------------------------------------|
| `baselines/`         | W1              | Engine-controlled serving baselines that load the **same ONNX** as VisionServe.    |
| `loadgen/`           | W1, W4, W4b     | Open-loop, coordinated-omission-correct concurrency + constant-rate load harness.  |
| `accuracy/`          | W6              | Export numerical parity + task metrics measured through the running server.        |
| `latency_breakdown/` | W8              | Decompose end-to-end latency (decode/preprocess/ORT Run/postprocess/encode).       |

> **NO experimental numbers are committed in this tree.** This harness only *produces*
> numbers when run on the target hardware (RTX A6000, Jetson, etc.). Any CSV/JSON it emits
> lands under `eval/results/` (git-ignored by convention — do not commit measured numbers
> into the repo until they have actually been measured). The paper is being revised
> precisely because it previously shipped unmeasured/copied figures; **do not** hand-fill
> any output file. If a value is unknown, leave it blank / `TODO`, never fabricate.

---

## 0. Prerequisites

### Hardware
- Primary host: **RTX A6000** (48 GB), pinned GPU clocks (`nvidia-smi -lgc <min>,<max>`).
- Edge (Phase 3, optional): Jetson AGX Orin / Orin Nano (TensorRT EP), Raspberry Pi 5 (CPU EP),
  Apple Silicon (CoreML EP), Intel (OpenVINO EP). The edge runs reuse the *same* scripts here;
  only the EP and device label change.

### Software
Pin versions so a reviewer can reproduce. Suggested:

```
# ONNX Runtime (must match the libonnxruntime.so VisionServe links against)
onnxruntime-gpu==1.20.1        # or onnxruntime==1.20.1 for CPU-only hosts
# Baselines
fastapi>=0.110  uvicorn[standard]>=0.29  gunicorn>=21  python-multipart>=0.0.9
torch>=2.2  torchvision>=0.17            # only for fastapi_torch.py / parity.py
tritonclient[http]>=2.40                 # only if driving Triton via perf_analyzer/client
torchserve  torch-model-archiver         # only for the TorchServe baseline
# Load generation (open-loop, CO-correct) — install at least one
#   wrk2:   https://github.com/giltene/wrk2        (build from source)
#   vegeta: https://github.com/tsenart/vegeta      (go install / release binary)
# Latency parsing
hdrhistogram>=0.10                       # HdrHistogram for percentiles
numpy>=1.24
# Accuracy
pycocotools>=2.0.7                       # COCO mAP
pillow>=10                               # image IO for clients
requests>=2.31
```

`requirements.txt` next to this README lists the same set; `pip install -r requirements.txt`.

### Datasets (supply paths via env / CLI flags — **none are bundled**)
- ImageNet-val (~5 k subset is enough for Top-1): images + the val ground-truth label index.
  Class names come from the repo file `internal/catalog/labels/imagenet1k.txt`.
- COCO val2017 + `instances_val2017.json` (for mAP via pycocotools).
- WiderFace val (easy split) + GT (for SCRFD face AP) — `task_eval.py` marks this `TODO`.
- A fixed image replay set for load tests (e.g. 50–200 COCO-val JPEGs) so every system sees
  identical input bytes.

### ONNX exports / checkpoints
- The **same** `.onnx` files VisionServe serves (under `~/.visionserve/models/<name>/` or the
  repo `models/<name>/`). The baselines deliberately load these exact files so any speedup is
  attributable to the serving layer, not the runtime (W1 control).
- For parity (`accuracy/parity.py`) you also need the **original PyTorch checkpoint** per model.
  These are not bundled; pass `--torch-ckpt`. Where a checkpoint/loader is model-specific the
  code is `TODO`-marked rather than guessed.

---

## 1. Bring up the systems under test

All systems must serve the **identical** ONNX file with the **CUDA** EP for the W1 control.

```bash
# (a) VisionServe itself — built from the repo root
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
  go run ./cmd/visionserve serve --addr :11435          # default port 11435

# (b) FastAPI + onnxruntime-python (cleanest engine control)
ONNX_PATH=$HOME/.visionserve/models/mobilenet-v3/model.onnx \
MODEL_NAME=mobilenet-v3 TASK=classification \
  uvicorn eval.baselines.fastapi_ort:app --host 0.0.0.0 --port 8001

# (c) FastAPI + PyTorch  (model-loading is TODO-marked, see file)
#   uvicorn eval.baselines.fastapi_torch:app --host 0.0.0.0 --port 8002

# (d) Triton  — copy ONNX into the model_repository, then:
#   tritonserver --model-repository=eval/baselines/triton/model_repository

# (e) TorchServe — archive with eval/baselines/torchserve/handler.py, then `torchserve --start`
```

Every baseline exposes `POST /api/predict` matching VisionServe's JSON shape
(`{"model": ..., "image_base64": ...}` or multipart `model=`,`image=`) so the **same**
loadgen/accuracy clients hit all systems unchanged. See `eval/common/api.py`.

---

## 2. Run the experiments

```bash
# W1 + W4/W4b — concurrency sweep + open-loop constant rate, percentiles -> CSV
python -m eval.loadgen.sweep \
    --target http://localhost:11435 --model mobilenet-v3 \
    --image test/testdata/sample.jpg \
    --concurrency 1,2,4,8,16,32,64 \
    --open-loop-rate 200 --duration 30 \
    --pool-sizes 1,2,4,8 \
    --out eval/results/w1_mobilenet.csv

# W6 part A — export parity (PyTorch checkpoint vs ORT export)
python -m eval.accuracy.parity \
    --onnx $HOME/.visionserve/models/mobilenet-v3/model.onnx \
    --torch-ckpt /path/to/mobilenet_v3.pth \
    --model mobilenet-v3

# W6 part B — task metric THROUGH the running server
python -m eval.accuracy.task_eval imagenet \
    --target http://localhost:11435 --model mobilenet-v3 \
    --images /data/imagenet/val --gt /data/imagenet/val_gt.txt

# W8 — latency decomposition
python -m eval.latency_breakdown.breakdown \
    --target http://localhost:11435 --model mobilenet-v3 \
    --image test/testdata/sample.jpg --n 500
```

---

## 3. The SessionPool-size sweep knob (W4)

VisionServe's per-role `SessionPool` size is currently **hardcoded per model in Go**
(`internal/models/*/*.go` `PoolSizes()`); there is **no** runtime env var for it today.
So `sweep.py --pool-sizes 1,2,4,8` cannot reconfigure a running server by itself. Two options,
both documented in `loadgen/sweep.py`:

1. **Restart-per-size (default):** the sweep restarts the server once per pool size using a
   user-supplied `--server-cmd` template containing `{pool}` (you wire `{pool}` to whatever
   mechanism your build exposes — e.g. a patched manifest field or a new env var if/when one
   is added to core). The harness only *orchestrates*; it does not edit core code.
2. **Single-size:** omit `--pool-sizes` to sweep concurrency at the server's built-in pool size.

This keeps the harness strictly outside core (Stream 4 owns only `eval/`).

---

## 4. Output / provenance

Each run writes a CSV plus a sidecar `*.meta.json` capturing: git commit, hostname, EP selected
(read back from the server `Result.device` field), GPU clock, n requests, loadgen tool + version,
dataset path, and a UTC timestamp — so every number is traceable. Fill nothing by hand.
