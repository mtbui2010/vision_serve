# VisionServe — Docker deployment

Self-contained images published on Docker Hub at
[`mtbui2010/visionserve`](https://hub.docker.com/r/mtbui2010/visionserve).
The Go binary and ONNX Runtime are bundled — no host setup required.
Model weights are **not** baked in; they are downloaded on first use via `visionserve pull`.

* Default port: **11435**
* Listen address: `visionserve serve` binds **`127.0.0.1:11435`** by default (loopback only, like
  Ollama: the API has no authentication, so it is not exposed to the network unless you ask).
  The images' `CMD` passes **`--addr :11435`** explicitly so the published port works; keep that
  flag whenever you override the command (see [Keeping models resident](#keeping-models-resident-disabling-idle-unload)).
* Health endpoint: `GET /api/health`
* ONNX Runtime: **v1.20.1** (loaded at runtime via `ORT_DYLIB_PATH`)

---

## Images

| Tag | Platform | Contents | Size |
|-----|----------|----------|------|
| `latest`, `latest-gpu` | x86-64 NVIDIA | CUDA 12.4 + cuDNN 9 (no TensorRT) | ~4 GB |
| `latest-cpu` | x86-64 | CPU only — no GPU required | ~141 MB |
| `latest-arm` | Jetson arm64 | CUDA + TensorRT EP (JetPack 6.1 / 6.2, built with `ORT_SOURCE=jetson`) | ~10 GB |

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
# (runs CUDA → CPU; TensorRT is opt-in with -e VISIONSERVE_TENSORRT=1, see "GPU image details")

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
docker exec -it visionserve visionserve pull mobile-sam       # segmentation
docker exec -it visionserve visionserve pull efficient-sam    # segmentation
docker exec -it visionserve visionserve pull sam2             # segmentation (SAM2)
docker exec -it visionserve visionserve pull grounding-dino   # open-vocab detection (alias: groundingdino)
docker exec -it visionserve visionserve pull rfdetr-small-etri  # detection, 22 tabletop classes
docker exec -it visionserve visionserve pull siglip-image     # SigLIP image tower (crop embeddings)
docker exec -it visionserve visionserve pull siglip-text      # SigLIP text tower + Go tokenizer
docker exec -it visionserve visionserve pull midas            # depth estimation
docker exec -it visionserve visionserve pull depth-anything-v2
docker exec -it visionserve visionserve pull efficientnet-b0  # classification
docker exec -it visionserve visionserve pull mobilenet-v3
docker exec -it visionserve visionserve pull clip             # image embeddings
docker exec -it visionserve visionserve pull scrfd            # face detection
docker exec -it visionserve visionserve pull paddle-ocr       # OCR

# Composed models — ONE command; missing dependencies are pulled automatically
docker exec -it visionserve visionserve pull grounded-sam               # GroundingDINO → MobileSAM masks
docker exec -it visionserve visionserve pull rfdetr-gdino               # hybrid router, COCO
docker exec -it visionserve visionserve pull rfdetr-gdino-siglip        # + SigLIP-crop rescoring
docker exec -it visionserve visionserve pull rfdetr-gdino-etri          # hybrid router, tabletop
docker exec -it visionserve visionserve pull rfdetr-gdino-sam-etri      # + MobileSAM masks
docker exec -it visionserve visionserve pull rfdetr-gdino-siglip-etri   # + SigLIP-crop rescoring
docker exec -it visionserve visionserve pull rfdetr-gdino-siglip-sam-etri  # + rescoring + MobileSAM masks
docker exec -it visionserve visionserve pull gdino-siglip               # GroundingDINO + SigLIP rescoring, no RF-DETR
docker exec -it visionserve visionserve pull gdino-siglip-sam           # + MobileSAM masks
```

`rt-detr` is still listed by `visionserve pull` but cannot be pulled: its upstream repository
(`onnx-community/RT-DETR-l-hf`) returns 401 and the catalog marks it unverified.

`visionserve pull grounding-dino` (or `groundingdino`) always installs the **corrected**
fixed-mask export (`mtbui2010/grounding-dino-tiny-fixedmask-ONNX`), never the onnx-community
graph that only masks the first phrase. A dependency that is already installed is reused, not
re-downloaded. Re-pulling a model refreshes a `manifest.yaml` that `pull` generated when the
catalog changed (e.g. an old GroundingDINO install pointing at `model.onnx`); a manifest you
edited by hand is kept unless you pass `--force` — also when you keep its `# Generated by` header
line (the header records a hash of the generated body, and an edit no longer matches it). If a composed model reports a missing file
inside a dependency, re-pull that dependency with `--force`, as the message says.

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
the whole CMD (the `visionserve` entrypoint stays), so you must repeat `--addr :11435`.
Without it the server binds its default `127.0.0.1:11435`, which inside a container is the
container's own loopback: `-p 11435:11435` then reaches nothing and every request fails with
"connection reset". The same applies to a compose `command:`.

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

## Bringing your own checkpoint (`visionserve convert`)

The server only runs ONNX. A checkpoint in another format — an RF-DETR `.pth`, a HuggingFace
model, TorchScript, a PyTorch `state_dict`, TensorFlow SavedModel / Keras / TFLite — is converted
by a **separate converter image** (`mtbui2010/visionserve-convert`, Python + PyTorch + TensorFlow).
It writes the converted model straight into the models folder the server uses, and the running
server picks it up without a restart. The server image itself stays small and Python-free.

```bash
# RF-DETR fine-tune: variant, resolution and class names are read from the checkpoint
visionserve convert rfdetr ./checkpoint_best_total.pth --name my-detector

# HuggingFace (local dir or hub id): grounding-dino, siglip, clip, rt_detr,
# image classification, depth estimation
visionserve convert hf google/vit-base-patch16-224 --name vit

# formats that do not say what the model is need --task and --input
visionserve convert keras ./model.keras --name my-cls --task classification --input 224x224 \
    --license Apache-2.0 --labels ./classes.txt
visionserve convert torchscript ./model.pt --name my-depth --task depth --input 256x256 --license MIT
visionserve convert pytorch ./weights.pth --script ./build.py --name my-net --task classification \
    --input 224x224 --license BSD-3-Clause --labels ./classes.txt

visionserve convert --help      # every format and flag
```

`visionserve convert` runs on the host and needs Docker. It mounts the checkpoint read-only and
the models folder read-write, then runs the converter as your user so the files are not
root-owned. It installs into `~/.visionserve_models` when that is the folder your container
mounts; pass `--models DIR` to choose another one.

Without the host binary, run the image directly:

```bash
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp \
  -v ~/.visionserve_models:/root/.models -v "$PWD":/in:ro \
  mtbui2010/visionserve-convert rfdetr /in/checkpoint_best_total.pth --name my-detector
```

Nothing is installed unless every check passes:

- **License.** `--license` must be Apache-2.0, MIT, BSD-3-Clause or BSD-2-Clause. It is required
  unless the checkpoint declares one (a HuggingFace model card). A model card with a copyleft
  license is refused whatever `--license` says. Ultralytics checkpoints (YOLO, FastSAM,
  YOLO-World) are AGPL and always refused.
- **Parity.** The ONNX graph is run next to the original framework on the same input. If the
  outputs differ beyond `--tolerance`, the model is not installed.
- **I/O contract.** The real tensor shapes must match what the VisionServe architecture decodes.
  For example, a detection model must emit logits `[1,Q,C]` plus boxes `[1,Q,4]`, and the number
  of labels must equal `C`.
- **Registry validation.** The result is installed through `visionserve pull <folder>`, the same
  check as a hand-made model folder.

`--dry-run` converts and checks without installing, and prints the generated `manifest.yaml`.

### Checking that the served model behaves like the original

Converting correctly is not enough. The server prepares images and text in Go; if that differs
from how the model was trained, the model runs, reports no error, and is quietly worse. This
project lost 7.35 mAP to a letterbox/squash mismatch and 4.3 mAP to a padding token that way.
Give the converter some of your images and it checks the whole path:

| Tier | Compares | Catches |
|---|---|---|
| A (always) | ONNX vs the framework on the same tensor | a broken export |
| B1 (`--images DIR`) | the tensor the server builds vs the framework's own preprocessing | letterbox vs squash, wrong mean/std, missing /255, centre crop, RGB/BGR, tokenizer padding |
| B2 (`--images DIR`) | server results vs the original model's (boxes, classes, embeddings) | anything above, plus decode/coordinate errors |
| C (`--eval PATH`) | mAP (COCO json) or top-1 (ImageFolder) of the original vs the served model | the real accuracy cost |
| speed (`--bench`) | p50/p95 of the framework, the ONNX model alone, and a whole server request | where the time goes |

```bash
visionserve convert rfdetr ./checkpoint_best_total.pth --name my-detector \
    --images ./val_images --eval ./val/_annotations.coco.json --bench
```

Each tier reports PASS / WARN / FAIL. A FAIL uninstalls the model (unless `--keep-on-fail`) and
exits non-zero. Tier C warns above 0.5 points of mAP (or top-1) lost and fails above 1.0
(`--max-map-drop`). The report is printed, saved as `<model>/convert-report.json`, and
summarised in the header of the installed `manifest.yaml`. The checks need a server. The
converter uses yours if it can reach it, and otherwise starts a temporary one by itself.

For a RF-DETR or HuggingFace checkpoint, the reference is the framework's own preprocessing and
post-processing. For TorchScript, PyTorch or TensorFlow there is no official preprocessing, so
pass yours with `--reference-script FILE.py` (`preprocess(pil) -> CHW array`, optionally
`predict(pil)`); without it, B1 checks the Go code against the manifest's declared preprocessing
only.

### From Python, without Docker

```bash
pip install "visionserve[convert]"          # PyTorch / RF-DETR / HuggingFace
pip install "visionserve[convert-tf]"       # TensorFlow / Keras / TFLite, in its OWN environment
```

```python
from visionserve.convert import export

report = export(model,                       # an nn.Module in memory, or a checkpoint path / hub id
                name="my-cls", task="classification", input="224x224",
                labels=classes, license="Apache-2.0",
                preprocess=my_val_transform,  # YOUR training transform: the strongest B1 check
                images="val_images/", eval="val_imagefolder/", bench=True)
print(report)          # report.ok is False on FAIL
```

`visionserve-convert` is the same tool as a command (`visionserve-convert rfdetr ckpt.pth --name ...`).

To compare the server's preprocessing with your own pipeline at any time, use the client:

```python
import visionserve as vs
c = vs.Client()
x = c.preprocess("my-detector", "img.jpg")   # exactly what the model is fed (no inference)
x.inputs["input"], x.meta                     # numpy array + scale/pad back to the original image
c.tokenize("siglip-text", "a photo of a cup") # token ids exactly as served
```

These call `POST /api/preprocess`, which returns the tensors a model would feed its first ONNX
session, bit for bit.

---

## Updating to a new release

Run the server from a small compose file, so you can update with one command whenever a new
image is published. Create `~/visionserve/compose.yaml`:

```yaml
services:
  visionserve:
    image: mtbui2010/visionserve:latest
    container_name: visionserve
    command: ["serve", "--addr", ":11435", "--idle-unload-seconds", "0"]
    ports: ["11435:11435"]
    volumes: ["${HOME}/.visionserve_models:/root/.models"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
```

- `${HOME}/.visionserve_models` is where your pulled models live on the host. They are kept
  across updates, so you do not need to pull them again.
- CPU-only machine: use `mtbui2010/visionserve:latest-cpu` and delete the `deploy:` block.

Start it the first time with `cd ~/visionserve && docker compose up -d`. After that, each time a
new version is released, one command updates it:

```bash
cd ~/visionserve && docker compose pull && docker compose up -d && docker image prune -f
```

This downloads the new image, recreates the container from it, and deletes the old image.

Or add an alias to `~/.bashrc` and run `update_visionserve`:

```bash
alias update_visionserve='docker compose -f ~/visionserve/compose.yaml pull && { docker rm -f visionserve >/dev/null 2>&1; docker compose -f ~/visionserve/compose.yaml up -d; } && docker image prune -f'
```

The `docker rm -f visionserve` step removes a container named `visionserve` that was started
earlier with a plain `docker run`. Without it, `docker compose up` fails with a
container-name conflict.

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
`libonnxruntime_providers_tensorrt.so`, but it runs **CUDA → CPU** by default (every shipped
model prefers `[cuda, cpu]`). TensorRT is opt-in, with `VISIONSERVE_TENSORRT=1` (below).

**Why not TensorRT by default:** measured on GroundingDINO (same weights, 247 tabletop images),
TensorRT is ~1.5x faster but drops boxes and scores unseen object names **6.8 mAP lower**
(37.78 vs 44.61), and it rebuilds its engine for every new prompt length (10-40 s stalls). The
CUDA EP runs GroundingDINO in ~150 ms and the SigLIP-rescored routers in ~200 ms per request.

**Check the EP chain:**
```bash
docker exec visionserve visionserve version
# EP chain: CUDA → CPU (default). TensorRT: off; enable with --tensorrt or VISIONSERVE_TENSORRT=1 (libnvinfer.so.10 not found)
# — with -e VISIONSERVE_TENSORRT=1 and the lib mounted —
# EP chain: TensorRT → CUDA → CPU (TensorRT ON via --tensorrt / VISIONSERVE_TENSORRT; libnvinfer: /usr/lib/x86_64-linux-gnu/libnvinfer.so.10)
```
The server prints the same lines at startup.

**To enable TensorRT** you need both: the TensorRT 10.x libraries inside the container (the image
does not ship them), and the opt-in. Without the libraries the opt-in falls back to CUDA (no
crash), and each GPU response carries a `hint` saying TensorRT could not load.
```bash
docker run -d --gpus all -p 11435:11435 \
  -e VISIONSERVE_TENSORRT=1 \
  -v /usr/lib/x86_64-linux-gnu/libnvinfer.so.10:/usr/lib/x86_64-linux-gnu/libnvinfer.so.10:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10:/usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10:ro \
  -v ~/.visionserve_models:/root/.models \
  --name visionserve \
  mtbui2010/visionserve:latest
# host side: sudo apt-get install tensorrt, or https://developer.nvidia.com/tensorrt
```
Every chain that contains `cuda` becomes `tensorrt → cuda → cpu`. Passing `--tensorrt` after
`serve` in the container command does the same. `VISIONSERVE_EP` is different: it **replaces**
every model's chain (`VISIONSERVE_EP=tensorrt` gives `tensorrt → cpu`, without CUDA), and it
wins over the opt-in. With Compose, see the commented `VISIONSERVE_TENSORRT` line in
`deploy/docker-compose.yml`.

**Prerequisites on the host (CUDA EP, no TRT):**
- NVIDIA driver ≥ 550
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

---

## Jetson / arm64

The ARM image (`latest-arm`) is built with `deploy/Dockerfile.edge`. With `ORT_SOURCE=jetson` it
targets JetPack 6.1 / 6.2 (L4T r36.4): the runtime base is NVIDIA's
`nvcr.io/nvidia/l4t-jetpack:r36.4.0`, which carries CUDA 12.6, cuDNN 9.3 and TensorRT 10.3 inside
the image (since JetPack 5, `l4t-base` no longer mounts them from the host), and ONNX Runtime
1.24.0 comes from the Jetson AI Lab `jp6/cu126` `onnxruntime-gpu` wheel (CUDA + TensorRT
providers, pinned by SHA-256). Without it (`ORT_SOURCE=cpu`, the default) the image is a portable
arm64 CPU image (Debian + the official ONNX Runtime aarch64 CPU build).

**Build on any machine with `docker buildx` (no QEMU needed) or directly on the Jetson:**

```bash
make docker-arm ORT_SOURCE=jetson     # pulls nvcr.io/nvidia/l4t-jetpack:r36.4.0 (~5.6 GB download, ~10 GB image)
make push-docker-arm
```

Every stage that runs a command runs on the build platform; the arm64 stages only copy files.

**Run on Jetson:**

```bash
docker run -d \
  -v visionserve:/root/.models \
  --runtime nvidia \
  -p 11435:11435 \
  --name visionserve \
  mtbui2010/visionserve:latest-arm
```

> **Checked 2026-10-05:** the previous default, `nvcr.io/nvidia/l4t-ml:r36.3.0`, does not exist on
> NGC (`docker manifest inspect`: "no such manifest"; l4t-ml stops at `r36.2.0-py3`), and
> `l4t-ml:r36.2.0-py3` ships ONNX Runtime 1.16.3, older than the 1.20 C API this binary's
> `onnxruntime_go` binding needs. `l4t-jetpack:r36.2.0`, `r36.3.0` and `r36.4.0` exist. Both
> variants of `Dockerfile.edge` build with `docker buildx --platform linux/arm64` on an x86-64
> host without QEMU, and the Jetson image contains every library ONNX Runtime's CUDA and
> TensorRT providers link against (cuDNN 9, cuBLAS 12, cuFFT 11, cudart 12, TensorRT 10). It has
> **not run on an Orin yet**: on the device, `docker run ... visionserve:<version>-arm bench MODEL
> --ep cuda` must report `gpu:0`. For JetPack 6.0 (r36.2/r36.3, CUDA 12.2), pass
> `--build-arg L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.3.0` and a matching wheel index
> (`--build-arg JETSON_ORT_WHEEL_INDEX=... JETSON_ORT_VERSION=... JETSON_ORT_WHEEL_SHA256=...`).

### Jetson Thor (JetPack 7): `deploy/Dockerfile.thor` (untested on hardware)

JetPack 7 is SBSA-aligned (standard arm64 CUDA 13 packages; no l4t-* images) and Thor's GPU is
Blackwell (sm_110), so the JetPack 6 image does not apply. `Dockerfile.thor` uses
`nvcr.io/nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04` and the ONNX Runtime 1.24 libraries of the
Jetson AI Lab `sbsa/cu130` wheel (CUDA + TensorRT providers, sm_110 kernels, pinned by SHA-256).
It builds without QEMU (every `RUN` runs on the build platform):

```bash
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.thor -t visionserve:thor --load .
docker run -d --runtime nvidia --gpus all -p 11435:11435 -v visionserve:/root/.models visionserve:thor
```

It has not run on a Thor yet: see the checklist in `website/docs/guides/edge.md`.

---

## Docker Compose

```bash
# GPU server — name the service: `--profile gpu up` alone starts BOTH services, and the CPU one
# and the GPU one would then fight over port 11435.
docker compose -f deploy/docker-compose.yml --profile gpu up visionserve-gpu

# CPU server
docker compose -f deploy/docker-compose.yml up visionserve
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
- Run `cp deploy/.dockerignore .dockerignore` before building locally (the `make docker*`
  targets and CI do this automatically). It keeps the build context to the Go sources plus
  `clients/python` and `convert/` (for the converter image): `.git`, `models/`, every
  `**/*.onnx` / `**/*.data` / `**/*.pt` and image file, `bin/`, `demo/`, `docs/`, `paper/` are
  excluded. Its patterns are anchored at the context root, so anything that can appear in a
  sub-directory must be written `**/<pattern>` — a plain `*.onnx` only matches files at the root.
