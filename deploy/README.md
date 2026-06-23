# VisionServe — Docker deployment

Self-contained images published on Docker Hub at
[`mtbui2010/visionserve`](https://hub.docker.com/r/mtbui2010/visionserve).
The Go binary and ONNX Runtime are bundled — no host setup required.
Model weights are **not** baked in; they are downloaded on first use via `visionserve pull`.

* Default port: **11435**
* Health endpoint: `GET /api/health`
* ONNX Runtime: **v1.20.1** (loaded at runtime via `ORT_DYLIB_PATH`)

---

## Images

| Tag | Platform | Contents | Size |
|-----|----------|----------|------|
| `latest`, `latest-gpu` | x86-64 NVIDIA | CUDA 12.4 + cuDNN 9 (no TensorRT) | ~4 GB |
| `latest-cpu` | x86-64 | CPU only — no GPU required | ~141 MB |
| `latest-arm` | Jetson arm64 | CUDA + TensorRT EP (JetPack 6) | ~4 GB |

> **`latest` = GPU image.** Use `latest-cpu` explicitly on machines without an NVIDIA GPU.
> Immutable versioned tags (`vX.Y.Z`, `vX.Y.Z-cpu`, `vX.Y.Z-arm`) are also published for
> pinning — see the [tags page](https://hub.docker.com/r/mtbui2010/visionserve/tags).

---

## Quick start (Ollama-style)

### Step 1 — Start the server

```bash
# GPU (default / recommended)
docker run -d \
  --gpus all \
  -p 11435:11435 \
  -v ~/.visionserve_models:/root/.models \
  --name visionserve \
  mtbui2010/visionserve:latest

# CPU only
docker run -d \
  -p 11435:11435 \
  -v ~/.visionserve_models:/root/.models \
  --name visionserve \
  mtbui2010/visionserve:latest-cpu
```

The registry lives at `/root/.models` inside the container (`VISIONSERVE_MODELS`).
Bind-mounting the host folder `~/.visionserve_models` onto it means pulled models
**persist on the host** (visible in plain `~/.visionserve_models/`), and any **local
model** you drop there (`manifest.yaml` + `.onnx`) shows up in `list` immediately —
no `pull` / `docker cp` needed. Prefer no mount? Omit `-v`: the image declares
`/root/.models` as a `VOLUME`, so catalog pulls still persist via an anonymous volume
(but local host folders won't be visible — use `pull <path>` to copy them in).

### Step 2 — Pull a model (no restart needed)

```bash
docker exec -it visionserve visionserve pull rf-detr
```

The server detects newly pulled models automatically — no restart required.

All available models (Apache-2.0 / MIT):

```bash
docker exec -it visionserve visionserve pull rf-detr          # detection
docker exec -it visionserve visionserve pull rf-detr-nano     # detection, faster
docker exec -it visionserve visionserve pull rt-detr          # detection, COCO-80
docker exec -it visionserve visionserve pull mobile-sam       # segmentation
docker exec -it visionserve visionserve pull efficient-sam    # segmentation
docker exec -it visionserve visionserve pull sam2             # segmentation (SAM2)
docker exec -it visionserve visionserve pull grounding-dino   # open-vocab detection
docker exec -it visionserve visionserve pull midas            # depth estimation
docker exec -it visionserve visionserve pull depth-anything-v2
docker exec -it visionserve visionserve pull efficientnet-b0  # classification
docker exec -it visionserve visionserve pull mobilenet-v3
docker exec -it visionserve visionserve pull clip             # image embeddings
docker exec -it visionserve visionserve pull scrfd            # face detection
docker exec -it visionserve visionserve pull paddle-ocr       # OCR

# Grounded-SAM (text → boxes → masks): pull dependencies first, then grounded-sam
docker exec -it visionserve visionserve pull grounding-dino
docker exec -it visionserve visionserve pull mobile-sam
docker exec -it visionserve visionserve pull grounded-sam
```

### Step 3 — Call the API

```bash
curl http://localhost:11435/api/health
# {"status":"ok"}

curl -s -F model=rf-detr -F image=@photo.jpg \
  http://localhost:11435/api/predict | python3 -m json.tool

# Open-vocab detection (requires a text prompt)
curl -s -F model=grounding-dino -F image=@photo.jpg -F prompt="cat. remote." \
  http://localhost:11435/api/predict | python3 -m json.tool

# Segmentation with a box prompt
curl -s -F model=mobile-sam -F image=@photo.jpg -F box="100,80,440,300" \
  http://localhost:11435/api/predict | python3 -m json.tool

# Segmentation — no prompt → Automatic Mask Generator (segment everything, ~256 masks)
curl -s -F model=mobile-sam -F image=@photo.jpg \
  http://localhost:11435/api/predict | python3 -m json.tool
```

---

## Keeping models resident (disabling idle unload)

**Symptom:** after the server has sat idle for a while, the **first** inference is
noticeably slow (a multi-second pause), then subsequent requests are fast again.

**Cause:** each model has an idle auto-unload reaper (manifests default to
`idle_unload_seconds: 300`, i.e. 5 min). Once a model has been idle past that window
it is unloaded from VRAM, so the next request pays for a full reload — ONNX session
re-create + CUDA init + first-inference autotune.

**Fix:** override the reaper with the `serve` flag `--idle-unload-seconds`:

* `0` — **never unload.** Models stay resident in VRAM, so the first request after an
  idle pause is *not* slowed by a reload. Tradeoff: VRAM is held continuously.
* `-1` — use each manifest's value (the default; 300 s / 5 min).
* `N` — override every model to `N` seconds.

Because the image's `CMD` is `serve --addr :11435`, args after the image name **replace**
the whole CMD (the `visionserve` entrypoint stays), so you must repeat `--addr :11435`:

```bash
# GPU — keep models resident (never unload)
docker run -d \
  --gpus all \
  -p 11435:11435 \
  -v ~/.visionserve_models:/root/.models \
  --name visionserve \
  mtbui2010/visionserve:latest \
  serve --addr :11435 --idle-unload-seconds 0

# CPU — keep models resident (never unload)
docker run -d \
  -p 11435:11435 \
  -v ~/.visionserve_models:/root/.models \
  --name visionserve \
  mtbui2010/visionserve:latest-cpu \
  serve --addr :11435 --idle-unload-seconds 0
```

> Running locally instead of in Docker? The `make serve` equivalent is `make serve IDLE=0`.

---

## One-shot inference (no server)

`visionserve run` loads the model in-process, infers, and exits:

```bash
docker run --rm --gpus all \
  -v visionserve:/root/.models \
  -v "$PWD/photo.jpg:/img.jpg:ro" \
  mtbui2010/visionserve:latest \
  run rf-detr /img.jpg

# With prompts
docker run --rm --gpus all \
  -v visionserve:/root/.models \
  -v "$PWD/photo.jpg:/img.jpg:ro" \
  mtbui2010/visionserve:latest \
  run grounded-sam /img.jpg --prompt "person. car."
```

---

## GPU image details (x86-64)

The GPU image (`latest` / `latest-gpu`) bundles **CUDA 12.4 + cuDNN 9** and includes
`libonnxruntime_providers_tensorrt.so`. TensorRT EP is used **automatically** when
`libnvinfer.so.10` is found on the host; otherwise VisionServe silently falls back to
CUDA EP — no crash.

**Why TRT matters:** GroundingDINO and MobileSAM use custom attention ops that ORT's
CUDA EP falls back to CPU for. Without TRT they run at CPU speed (~6 s GDINO, ~1.7 s SAM).
With TRT: ~70 ms GDINO, ~160 ms SAM.

**Check TRT status:**
```bash
docker exec visionserve visionserve version
# TensorRT: available (/usr/lib/x86_64-linux-gnu/libnvinfer.so.10)
# — or —
# TensorRT: not found — install for 10-50× faster GPU inference
```

**To enable TRT:** install TensorRT 10.x on the host and mount the lib:
```bash
# Option A — install TRT on the host (Ubuntu/Debian)
sudo apt-get install tensorrt   # or download from https://developer.nvidia.com/tensorrt

# Option B — mount an existing TRT lib into the container
docker run ... -v /usr/lib/x86_64-linux-gnu/libnvinfer.so.10:/usr/lib/x86_64-linux-gnu/libnvinfer.so.10:ro ...
```

**Prerequisites on the host (CUDA EP, no TRT):**
- NVIDIA driver ≥ 550
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

**Prerequisites on the host:**
- NVIDIA driver ≥ 550
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

## Jetson / arm64

The ARM image (`latest-arm`) is built with `deploy/Dockerfile.jetson` using
`nvcr.io/nvidia/l4t-ml:r36.3.0` (JetPack 6.x, CUDA 12.2 + TensorRT 8.6) as the ORT source.

**Build on a machine with `docker buildx` + QEMU or directly on the Jetson:**

```bash
# Requires: docker login nvcr.io (free NVIDIA NGC account)
make docker-arm ORT_SOURCE=jetson
make push-docker-arm
```

**Run on Jetson:**

```bash
docker run -d \
  -v visionserve:/root/.models \
  --runtime nvidia \
  -p 11435:11435 \
  --name visionserve \
  mtbui2010/visionserve:latest-arm
```

---

## Docker Compose

```bash
# GPU server (default)
docker compose -f deploy/docker-compose.yml --profile gpu up

# CPU server
docker compose -f deploy/docker-compose.yml up
```

---

## Build from source

```bash
make docker                   # CPU image  → visionserve:<version>-cpu
make docker ORT_VARIANT=gpu   # GPU image  → visionserve:<version>-gpu
make docker-arm ORT_SOURCE=jetson  # Jetson → visionserve:<version>-arm
make push-docker              # push CPU + GPU to Docker Hub
make push-docker-arm          # push ARM to Docker Hub
```

**Build notes:**
- CGO is required (`yalue/onnxruntime_go` uses the ORT C API via cgo).
- `libonnxruntime.so` is only needed at **runtime** (dlopen), not at build time.
- Run `cp deploy/.dockerignore .dockerignore` before building locally (CI does this
  automatically); it excludes `.git`, `*.onnx`, `bin/`, `demo/`, `clients/` from the
  build context.
