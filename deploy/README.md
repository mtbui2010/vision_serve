# VisionServe — Docker deployment

Self-contained images for the VisionServe **server**: the Go binary and ONNX Runtime are
bundled, so no host setup is required. Model weights are **not** baked into the image
(they are large and not committed to git) — mount them into the `/models` volume.

* Default port: **11435**
* ONNX Runtime: **pinned to v1.20.1** (official GitHub releases), loaded at runtime via
  `ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so`
* Health endpoint: `GET /api/health`

> Why a download instead of `pip`? VisionServe is a small Go edge binary; no Python is
> pulled into the runtime. ONNX Runtime is fetched as the official prebuilt C/C++ shared
> library during the image build.

---

## Images

| File                   | Target                    | ONNX Runtime build it bundles                          |
|------------------------|---------------------------|--------------------------------------------------------|
| `deploy/Dockerfile`    | x86_64 / linux-amd64      | CPU (default) or GPU via `--build-arg ORT_VARIANT=gpu` |
| `deploy/Dockerfile.edge` | arm64 (incl. Jetson)    | linux-aarch64 CPU (default), or Jetson TensorRT/CUDA   |

---

## Quick start (x86_64, CPU)

```bash
# Build (run from the repo ROOT; the .dockerignore lives in deploy/, copy it to the root)
cp deploy/.dockerignore .dockerignore
docker build -f deploy/Dockerfile -t visionserve .

# Run — mount your local models directory into /models
docker run --rm -p 11435:11435 -v "$PWD/models:/models" visionserve

# Check it
curl http://localhost:11435/api/health
```

Sanity-check the binary without starting the server:

```bash
docker run --rm visionserve version
docker run --rm visionserve --help
```

### docker compose

```bash
docker compose -f deploy/docker-compose.yml up --build
```

---

## Getting model weights into `/models`

Weights (`.onnx`) are not in git. Each model under `models/<name>/README.md` documents how
to export/download them. Two ways to populate the mounted volume:

1. **On the host** — follow `models/rf-detr/README.md`, dropping `rf-detr-base.onnx` next to
   its `manifest.yaml` inside `./models/rf-detr/`. The container mounts that directory.

2. **With the `pull` command** (one-shot container) — fetches weights into the same volume:

   ```bash
   docker compose -f deploy/docker-compose.yml --profile pull run --rm pull-rfdetr
   # or directly:
   docker run --rm -v "$PWD/models:/models" visionserve pull rf-detr --models /models
   ```

After the volume has the weights, start the server normally.

---

## GPU (x86_64, CUDA + TensorRT EP)

The GPU image includes everything for maximum performance:

| Layer | What it provides |
|---|---|
| `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | CUDA 12.4 runtime + cuDNN 9 |
| NVIDIA apt repo → `libnvinfer10` | **TensorRT 10** (libnvinfer.so.10) |
| ORT GPU tgz | `libonnxruntime_providers_tensorrt.so` + CUDA EP |

With TensorRT EP the manifest's `runtime.prefer: [tensorrt, cuda, cpu]` chain actually
reaches TRT, giving **2–4× speedup** over the plain CUDA EP:

| Model | CUDA EP | TensorRT EP (expected) |
|---|---|---|
| RF-DETR-nano | ~23 ms | ~5–10 ms |
| RF-DETR-base | ~41 ms | ~10–20 ms |
| GroundingDINO | ~325 ms | ~80–150 ms |

> TensorRT JIT-compiles an optimized engine on **first load** — this can take 1–5 minutes
> depending on the model. Subsequent loads reuse the cached engine (in the `/models`
> volume). Set `VISIONSERVE_TRACE=1` to confirm the TRT EP loaded.

```bash
# Build (~3–5 GB, downloads CUDA base + TRT apt packages + ORT GPU tgz)
make docker ORT_VARIANT=gpu
# or directly:
docker build -f deploy/Dockerfile --build-arg ORT_VARIANT=gpu -t visionserve:gpu .

# Run — requires nvidia-container-toolkit on the host
docker run --rm --gpus all -p 11435:11435 \
  -v "$PWD/models:/models" \
  -e VISIONSERVE_TRACE=1 \
  visionserve:gpu

# Confirm TensorRT EP loaded (look for "[trace]" lines mentioning "tensorrt")
docker logs <container> 2>&1 | grep -i 'trace\|tensorrt'

# Compose (profile "gpu"):
docker compose -f deploy/docker-compose.yml --profile gpu up --build visionserve-gpu
```

> **Prerequisites on the host:**
> - NVIDIA driver ≥ 550 (matches CUDA 12.4)
> - [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
> - The GPU image does NOT need TensorRT on the host — TRT libraries are inside the image.

---

## Edge / arm64 / Jetson

The edge image (`deploy/Dockerfile.edge`) has two runtime sources, picked with
`--build-arg ORT_SOURCE=...`:

### Portable arm64 CPU (default)

Downloads the official `linux-aarch64` ONNX Runtime CPU build. Runs on any arm64 box and on
Jetson, but **CPU only** (no GPU acceleration).

```bash
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.edge -t visionserve:edge .
docker run --rm -p 11435:11435 -v "$PWD/models:/models" visionserve:edge
```

### Jetson with TensorRT/CUDA EP (hardware-specific)

There is **no stable public download URL** for an ONNX Runtime *C/C++ shared library* built
with the TensorRT/CUDA EP for Jetson — that build must match your device's JetPack
(CUDA + cuDNN + TensorRT) and is produced from source or a JetPack-matched package.

Assumptions for this path:

* Build **on the Jetson** (so the NVIDIA container runtime provides CUDA/cuDNN/TensorRT),
  or use an L4T *ML* base that ships them — set `--build-arg L4T_BASE=...` to match your
  JetPack (e.g. `nvcr.io/nvidia/l4t-base:r36.3.0` for JetPack 6.x).
* Place your JetPack-matched `libonnxruntime.so*` under `deploy/ort-jetson/` in the build
  context (the Dockerfile copies it in; the build fails loudly if it is missing, so you
  never silently ship a CPU-only image expecting GPU acceleration).

```bash
mkdir -p deploy/ort-jetson
cp /path/to/jetson/libonnxruntime.so* deploy/ort-jetson/
docker build -f deploy/Dockerfile.edge \
  --build-arg ORT_SOURCE=jetson \
  --build-arg L4T_BASE=nvcr.io/nvidia/l4t-base:r36.3.0 \
  -t visionserve:jetson .
docker run --rm --runtime nvidia -p 11435:11435 -v "$PWD/models:/models" visionserve:jetson
```

---

## Published images (GHCR)

On a `v*` release tag, CI builds and pushes a **multi-arch** (linux/amd64 + linux/arm64,
CPU) image to GitHub Container Registry:

```bash
docker pull ghcr.io/<owner>/visionserve:latest
docker run --rm -p 11435:11435 -v "$PWD/models:/models" ghcr.io/<owner>/visionserve:latest
```

Replace `<owner>` with the GitHub org/user that owns the repository. The GPU and Jetson
images are intentionally not published (they need a CUDA/JetPack-matched runtime — build
them on the target host as shown above).

---

## Build notes

* **CGO is required.** The `yalue/onnxruntime_go` binding links the ONNX Runtime C API via
  cgo, so the build stage uses `CGO_ENABLED=1` (the `golang` image already has gcc).
  `CGO_ENABLED=0` fails with *"build constraints exclude all Go files"*. The
  `libonnxruntime.so` itself is only needed at **runtime** (dlopen), not at build time.
* The `.dockerignore` keeps `.git`, `bin/`, `*.onnx`, `demo/`, `clients/`, and sample
  images out of the build context. Docker reads it from the **context root**, so copy it
  there before building (`cp deploy/.dockerignore .dockerignore`); CI does this
  automatically.
