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
| `latest`, `v0.1.2-gpu` | x86-64 NVIDIA | CUDA 12.4 + cuDNN 9 (no TensorRT) | ~4 GB |
| `v0.1.2`, `v0.1.2-cpu` | x86-64 | CPU only — no GPU required | ~141 MB |
| `v0.1.2-arm` | Jetson arm64 | CUDA + TensorRT EP (JetPack 6) | ~4 GB |

> **`latest` = GPU image.** Use `v0.1.2-cpu` explicitly on machines without an NVIDIA GPU.

---

## Quick start (Ollama-style)

### Step 1 — Start the server

```bash
# GPU (default / recommended)
docker run -d \
  -v visionserve:/root/.visionserve \
  --gpus all \
  -p 11435:11435 \
  --name visionserve \
  mtbui2010/visionserve:latest

# CPU only
docker run -d \
  -v visionserve:/root/.visionserve \
  -p 11435:11435 \
  --name visionserve \
  mtbui2010/visionserve:v0.1.2-cpu
```

The named volume `visionserve:/root/.visionserve` persists downloaded models across
restarts. Omit `-v` if persistence is not needed.

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

# Grounded-SAM (text → boxes → masks): pull both dependencies first
docker exec -it visionserve visionserve pull grounding-dino
docker exec -it visionserve visionserve pull mobile-sam
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
```

---

## One-shot inference (no server)

`visionserve run` loads the model in-process, infers, and exits:

```bash
docker run --rm --gpus all \
  -v visionserve:/root/.visionserve \
  -v "$PWD/photo.jpg:/img.jpg:ro" \
  mtbui2010/visionserve:latest \
  run rf-detr /img.jpg

# With prompts
docker run --rm --gpus all \
  -v visionserve:/root/.visionserve \
  -v "$PWD/photo.jpg:/img.jpg:ro" \
  mtbui2010/visionserve:latest \
  run grounded-sam /img.jpg --prompt "person. car."
```

---

## GPU image details (x86-64)

The GPU image (`latest` / `v0.1.2-gpu`) bundles **CUDA 12.4 + cuDNN 9**.
TensorRT is **intentionally excluded** — it caused hard crashes on systems where
`libnvinfer.so` was absent or a different version. ORT uses the **CUDA EP** directly,
which is still 3–5× faster than CPU.

For TensorRT acceleration on Jetson, use the `v0.1.2-arm` image instead.

**Prerequisites on the host:**
- NVIDIA driver ≥ 550
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

## Jetson / arm64

The ARM image (`v0.1.2-arm`) is built with `deploy/Dockerfile.jetson` using
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
  -v visionserve:/root/.visionserve \
  --runtime nvidia \
  -p 11435:11435 \
  --name visionserve \
  mtbui2010/visionserve:v0.1.2-arm
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
make docker                   # CPU image  → visionserve:v0.1.2-cpu
make docker ORT_VARIANT=gpu   # GPU image  → visionserve:v0.1.2-gpu
make docker-arm ORT_SOURCE=jetson  # Jetson → visionserve:v0.1.2-arm
make push-docker              # push CPU + GPU to Docker Hub
make push-docker-arm          # push ARM to Docker Hub
```

**Build notes:**
- CGO is required (`yalue/onnxruntime_go` uses the ORT C API via cgo).
- `libonnxruntime.so` is only needed at **runtime** (dlopen), not at build time.
- Run `cp deploy/.dockerignore .dockerignore` before building locally (CI does this
  automatically); it excludes `.git`, `*.onnx`, `bin/`, `demo/`, `clients/` from the
  build context.
