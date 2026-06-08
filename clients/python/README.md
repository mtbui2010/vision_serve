# VisionServe Python Client

> Prefer JavaScript/TypeScript? There's a sibling client with the same API in
> [`clients/js/`](../js/).

A lightweight Python **client** SDK for the [VisionServe](../../) HTTP server. It talks
to the Go runtime over HTTP (default `http://localhost:11435`) — it is **not** the
inference runtime and pulls no inference engine into Python. Think of it like Ollama's
Python client.

The transport uses only the Python **standard library** (`urllib`), so the client has
no required third-party dependencies. `numpy` and `pillow` are **optional** and only
needed for:
- passing `numpy.ndarray` / `PIL.Image` images to `predict()`, and
- decoding masks with `Mask.to_ndarray()`.

## Contents

- [Install](#install)
- [Run the server](#run-the-server-first)
- [Quickstart](#quickstart)
- [CLI](#cli)
  - [Output](#output)
  - [Examples](#examples-1)
- [Public API](#public-api)
  - [Image inputs](#image-inputs)
  - [Result types](#result-types)
  - [Detection](#detection)
  - [Segmentation](#segmentation)
  - [Open-vocab / Grounded-SAM](#open-vocab--grounded-sam)
  - [Depth estimation](#depth-estimation)
  - [Classification](#classification)
  - [CLIP embeddings](#clip-embeddings)
  - [SCRFD / OCR](#scrfd--ocr)
  - [Grasp detection](#grasp-detection)
- [Post-processing](#post-processing)
  - [Grasp post-processing](#grasp-post-processing)
  - [Size filtering](#size-filtering----resultfilter_by_size)
  - [Visualization](#visualization----draw--resultvisualize)
- [Script examples](#examples)
- [Tests](#tests)

## Install

```bash
pip install visionserve               # from PyPI (recommended)
# or from source:
pip install -e clients/python
# optional extras for ndarray/PIL image inputs and mask decoding:
pip install -e 'clients/python[images]'
```

## Run the server first

The client needs a running VisionServe server (which in turn needs the ONNX Runtime
shared library at runtime). From the repo root:

```bash
make serve            # starts the Go server on :11435
```

## Quickstart

```python
from visionserve import Client

c = Client()                       # http://localhost:11435, timeout=120s
print(c.health())                  # {"status": "ok"}

for m in c.list_models():
    print(m.name, m.task, m.license, m.state)

c.load("rf-detr")
res = c.predict("rf-detr", "cat.jpg")
print(res.task, res.duration_ms)
for d in res.detections:
    print(d.cls, d.conf, d.bbox)   # bbox = [x, y, w, h] in ORIGINAL image pixels

print([m.name for m in c.ps()])    # currently loaded models
```

## CLI

Installing the package registers a `visionserve` console command. Like the SDK it is a
thin **HTTP client** — it runs no inference itself, it just talks to a running
VisionServe server (the Go binary `visionserve serve`, default `http://localhost:11435`).
The bare command needs only the standard library; `--save` (annotated images)
additionally needs Pillow:

```bash
pip install visionserve            # CLI works with stdlib only
pip install 'visionserve[images]'  # add Pillow for --save
```

Commands:

| Command | Description |
| --- | --- |
| `visionserve predict <model> <image> [flags]` | Run a prediction; JSON to stdout |
| `visionserve list` (aliases `models`, `ls`) | List available models |
| `visionserve ps` | List currently loaded models |
| `visionserve load <model>` | Load a model into memory |
| `visionserve unload <model>` (alias `rm`) | Unload a model |
| `visionserve health` | Server health check |

Global flags (accepted **before or after** the subcommand):

| Flag | Description |
| --- | --- |
| `--host <url>` | Server base URL (default `http://localhost:11435`) |
| `--timeout <sec>` | Per-request timeout in seconds (default `120`) |
| `--version`, `-h`/`--help` | Print version / help |

`predict` flags:

| Flag | Description |
| --- | --- |
| `--prompt "<text>"` | Open-vocab text prompt, e.g. `"cat. remote."` (GroundingDINO / Grounded-SAM / grasp-gd) |
| `--box x,y,w,h` | SAM box prompt(s) in ORIGINAL image pixels; multiple boxes separated by `;` |
| `--point x,y[,l]` | SAM point prompt(s); label `1`=fg, `0`=bg; multiple separated by `;` |
| `--min-size PCT` / `--max-size PCT` | Drop objects whose bbox area is below/above PCT% of the image (e.g. `0.1`, `90`) |
| `--gripper-min PX` / `--gripper-max PX` | Grasp models only: jaw-opening bounds in original-image pixels |
| `--save` | Save an annotated image, auto-named `<stem>.python.<model>.<task>.png` |
| `--save-as PATH` | Save the annotated image to this exact path (extension picks the format) |
| `--alpha FLOAT` | Mask overlay opacity for `--save` (0..1, default `0.45`) |
| `--max-grasps-per-object N` | For grasp results, draw at most N grasps per object (`<=0` = all; default `3`) |
| `--compact` | Print result JSON on a single line (default: pretty-printed) |
| `--quiet` | Suppress the stderr summary line |

`list` / `ps` accept `--json` to print JSON instead of a table.

### Output

`predict` prints the unified result as JSON to **stdout** (pipe-friendly; field names
match the server wire schema, e.g. `class`, with empty arrays omitted). A one-line
summary goes to **stderr**:

```
predict: model=rf-detr task=detection device=gpu:0  client=42.1ms server=12.3ms  (12 detections)
```

Here `client` is the wall-clock time around the `predict()` HTTP round-trip and `server`
is the server's `duration_ms` (inference only). Both are captured **before** the image is
drawn or saved, so visualization cost never inflates the reported latency. The `--save`
filename `<stem>.python.<model>.<task>.png` embeds the client type (`python`), so outputs
from the Python/JS/Go CLIs for the same image never collide.

### Examples

```bash
# Start the server first (Go binary):
visionserve serve &

# Detection -> JSON on stdout, summary on stderr
visionserve predict rf-detr cat.jpg

# Open-vocab + save annotated image (auto-named cat.python.grounding-dino.open_vocab.png)
visionserve predict grounding-dino cat.jpg --prompt "cat. remote." --save

# Box-prompted segmentation, explicit output path
visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save-as dog_masks.png

# Open-vocab grasps, remote server, top-1 grasp per object
visionserve --host http://10.0.0.5:11435 predict grasp-gd bin.jpg --prompt "mug." --max-grasps-per-object 1 --save

# Registry / memory management
visionserve list
visionserve ps
visionserve load rf-detr
```

## Public API

### `Client(host="http://localhost:11435", timeout=120)`

| Method | HTTP | Returns |
| --- | --- | --- |
| `health()` | `GET /api/health` | `{"status": "ok"}` |
| `list_models()` | `GET /api/models` | `list[ModelInfo]` |
| `load(model)` | `POST /api/load` | `{"model", "state"}` |
| `unload(model)` | `POST /api/unload` | `{"model", "state"}` |
| `ps()` | `GET /api/models` (filtered) | loaded `list[ModelInfo]` |
| `predict(model, image, *, prompt=None, box=None, point=None, min_size=0, max_size=0, gripper_min=None, gripper_max=None)` | `POST /api/predict` | `Result` |

`min_size` / `max_size` filter by bounding-box area as a **percentage of the image area** (0–100; `0` = no limit). Example: `min_size=0.5` keeps only objects covering at least 0.5% of the image. The conversion to absolute pixels is done server-side using the uploaded image dimensions.

### Image inputs

`predict(model, image, ...)` accepts four image types — choose whichever fits your pipeline:

```python
from visionserve import Client
from PIL import Image
import numpy as np

c = Client()

# 1. File path (str or os.PathLike) — simplest
res = c.predict("rf-detr", "photo.jpg")

# 2. PIL.Image — e.g. after augmentation or crop
pil_img = Image.open("photo.jpg")
res = c.predict("rf-detr", pil_img)

# 3. numpy ndarray — HWC uint8 (H×W×3), or float [0,1] scaled to uint8 automatically
arr = np.array(pil_img)           # shape (H, W, 3), dtype uint8
res = c.predict("rf-detr", arr)

# 4. raw bytes — already-encoded PNG/JPEG, sent verbatim
with open("photo.jpg", "rb") as f:
    raw = f.read()
res = c.predict("rf-detr", raw)
```

Grayscale `(H, W)` ndarrays are automatically promoted to RGB. Float arrays in `[0, 1]`
are scaled to `uint8`. Encoded to PNG client-side before upload.

Prompts (serialized to the server's string format):
- `box`: `[x, y, w, h]` or a list of boxes → `"x,y,w,h"` joined by `;`.
- `point`: `[x, y]` / `[x, y, label]` or a list (label 1=fg, 0=bg) → `"x,y[,label]"` joined by `;`.
- `prompt`: free text, e.g. `"cat. remote."`.
- `gripper_min` / `gripper_max`: grasp models only — jaw-opening bounds in **original-image pixels** (e.g. `gripper_min=20, gripper_max=150`). Server filters out grasps outside the range.

### Result types

```python
Detection(bbox: list[float], cls: str, conf: float)       # bbox = [x, y, w, h], original px
Mask(rle: str, bbox: list[float], conf: float)
Classification(cls: str, conf: float)                      # top-K prediction
Grasp(
    x: float, y: float,       # grasp center in ORIGINAL image pixels
    theta: float,             # in-plane gripper-closing angle in radians
    width: float,             # jaw opening in ORIGINAL image pixels
    quality: float,           # analytic grasp score in [0, 1]
    cls: str = "",            # object class label ("" = class-agnostic)
    conf: float = 0.0,        # detector confidence (0.0 = class-agnostic)
)
Result(
    task, model, duration_ms,
    detections: list[Detection],
    masks: list[Mask],
    classifications: list[Classification],                 # task="classification"
    grasps: list[Grasp],                                   # task="grasp"
    depth_map: list[float] | None,                         # task="depth", row-major H×W
    depth_width: int | None,
    depth_height: int | None,
    embeddings: list[list[float]],                         # task="embedding" (CLIP)
)
```

`Mask.to_ndarray(width, height) -> np.ndarray` decodes the COCO-style **column-major**
uncompressed RLE into a boolean `(height, width)` array (requires numpy). It is the
exact inverse of the server's encoder.

### Detection

```python
# RF-DETR (COCO) or RT-DETR
res = c.predict("rf-detr", "photo.jpg")
for d in res.detections:
    print(d.cls, round(d.conf, 3), d.bbox)   # bbox = [x, y, w, h] in original px
```

### Segmentation

```python
# Box-prompted segmentation
from PIL import Image
res = c.predict("mobile-sam", "img.jpg", box=[50, 40, 120, 90])
w, h = Image.open("img.jpg").size
mask = res.masks[0].to_ndarray(width=w, height=h)   # bool (h, w) numpy array

# No-prompt → Automatic Mask Generator (segment everything, ~256 masks)
res = c.predict("mobile-sam", "img.jpg")
for m in res.masks:
    print(m.bbox, round(m.conf, 3))

# EfficientSAM and SAM2 — same prompt interface
res = c.predict("efficient-sam", "img.jpg", box=[50, 40, 120, 90])
res = c.predict("sam2", "img.jpg", box=[50, 40, 120, 90])
```

### Open-vocab / Grounded-SAM

```python
# GroundingDINO — text → boxes
res = c.predict("grounding-dino", "img.jpg", prompt="cat. remote.")
for d in res.detections:
    print(d.cls, round(d.conf, 3), d.bbox)

# Grounded-SAM — text → boxes → masks
res = c.predict("grounded-sam", "img.jpg", prompt="cat. remote.")
print([d.cls for d in res.detections], "→", len(res.masks), "masks")
```

### Depth estimation

```python
import numpy as np
res = c.predict("depth-anything-v2", "img.jpg")
depth = np.array(res.depth_map).reshape(res.depth_height, res.depth_width)
# or MiDaS (faster, 256×256)
res = c.predict("midas", "img.jpg")
```

### Classification

```python
res = c.predict("efficientnet-b0", "img.jpg")
for cls_pred in res.classifications:
    print(cls_pred.cls, round(cls_pred.conf, 3))
# or MobileNetV3 (lighter)
res = c.predict("mobilenet-v3", "img.jpg")
```

### CLIP embeddings

```python
import numpy as np
res = c.predict("clip", "img.jpg")
vec = np.array(res.embeddings[0])     # shape (512,)
vec /= np.linalg.norm(vec)            # L2-normalize before cosine similarity
```

### SCRFD / OCR

```python
# Face detection — detections with cls="face" and 5 keypoints in the bbox extensions
res = c.predict("scrfd", "photo.jpg")
for d in res.detections:
    print(d.cls, round(d.conf, 3), d.bbox)

# OCR — text regions as detections; cls = recognized text string
res = c.predict("paddle-ocr", "doc.jpg")
for d in res.detections:
    print(d.cls, round(d.conf, 3), d.bbox)
```

### Grasp detection

For planar grasp detection, results come back in `grasps`:

```python
# Class-agnostic — whole-image automask → grasps
res = c.predict("grasp", "bin.jpg", gripper_min=20, gripper_max=150)
for g in res.grasps:
    print(f"q={g.quality:.3f}  x={g.x:.1f} y={g.y:.1f}  θ={g.theta:.3f}  w={g.width:.1f}")

# Class-aware — text-prompted detector → per-object grasps
res = c.predict("grasp-gd", "table.jpg", prompt="mug. bottle.",
                gripper_min=20, gripper_max=150)
for g in res.grasps:
    print(g.cls, round(g.quality, 3), g.contacts())  # contacts() → [[x0,y0],[x1,y1]]

# Pick the best grasp for a target class / pixel
from visionserve import select_target_grasp
target = select_target_grasp(res.grasps, cls="mug",
                              gripper_min=20, gripper_max=150)
if target:
    print("best grasp:", target.x, target.y, target.theta)
```

See [Grasp post-processing](#grasp-post-processing) below for the full selection and filtering API.

## Post-processing

All methods return a **new** `Result`; the original is not modified. They work on
`detections`, `masks`, and `classifications` as appropriate.

```python
from visionserve import Client, get_depth_at_detection

client = Client()
result = client.predict("rf-detr", "photo.jpg")

# Keep only high-confidence detections
result = result.filter_by_conf(min_conf=0.5)

# NMS to remove overlapping boxes
result = result.nms(iou_threshold=0.45)

# Top-5 predictions
result = result.top_k(5)

# Sort and group by class
result = result.sort_by_conf()
by_class = result.group_by_class()
for cls, r in by_class.items():
    print(f"{cls}: {len(r.detections)} detections")

# Combine depth model with detection
depth = client.predict("midas", "photo.jpg")
depths = get_depth_at_detection(depth, result)
for det, d in zip(result.detections, depths):
    print(f"{det.cls}: depth={d:.1f}" if d else f"{det.cls}: no depth")
```

| Method | Signature | Description |
| --- | --- | --- |
| `filter_by_conf` | `(min_conf=0.0, max_conf=1.0)` | Keep predictions with conf in `[min_conf, max_conf]` |
| `sort_by_conf` | `(*, descending=True)` | Sort predictions by confidence |
| `top_k` | `(k)` | Retain top-k predictions by confidence |
| `nms` | `(iou_threshold=0.5)` | Greedy NMS on detections; no-op if no detections |
| `group_by_class` | `()` | Returns `Dict[str, Result]` keyed by class label |

`get_depth_at_detection(depth_result, det_result, *, mode="median")` (from
`visionserve.postprocess` or the top-level `visionserve` package) returns
`List[Optional[float]]` — one depth value per detection/mask, or `None` when
the box falls outside the depth map. `mode` is `"median"` (default) or `"mean"`.

### Grasp post-processing

All grasp post-processing helpers are importable from `visionserve` or
`visionserve.postprocess`.

#### `Grasp` fields and helpers

```python
g = res.grasps[0]

# Robot-ready pose as a flat list
print(g.pose)              # [x, y, width, theta]  — for robot control

# Jaw-contact points
print(g.contacts())        # [[x0, y0], [x1, y1]]  — nested
print(g.contacts_flat())   # [x0, y0, x1, y1]      — flat, ready for ROS/serial
```

#### `result.filter_grasps(max_per_object)`

Limit the number of grasps per detected object — keeps the `max_per_object`
highest-quality grasps inside each detection/mask bbox. Also available as a
parameter to `predict()`.

```python
# Via predict() — applied immediately after the server response:
res = c.predict("grasp-gd", "bin.jpg", prompt="mug.", max_grasps_per_object=3)

# Or on an existing result:
res = res.filter_grasps(max_per_object=3)   # None keeps all
```

#### `select_target_grasp()`

Pick the single best grasp for execution. Candidates are first filtered by class
and gripper-width feasibility, then scored on one or more criteria (each
normalised to `[0, 1]`):

| Criterion | Keyword | Description |
|-----------|---------|-------------|
| `quality` | — | Analytic grasp quality score (default when nothing else is set) |
| `near` | `target_point=(x, y)` | 2D pixel distance from grasp centre to a target pixel — nearest wins |
| `distance` | `target_distance=d` | True 3D camera→grasp Euclidean distance (Gaussian centred on `target_distance`); needs `depth_result` + `intrinsics` |
| `width` | — | Preference for a mid-range jaw opening within `[gripper_min, gripper_max]` |

By default the most-specific available criterion is used (`distance > near > quality`).
Pass `weights={"quality": 0.5, "near": 0.5}` for a weighted composite.

```python
from visionserve import select_target_grasp, CameraIntrinsics

res = c.predict("grasp-gd", "bin.jpg", prompt="mug.", max_grasps_per_object=3)

# 1. By quality alone (default)
target = select_target_grasp(res.grasps)

# 2. Prefer grasps near a pixel (e.g. robot workspace centre)
target = select_target_grasp(res.grasps, target_point=(320, 240))

# 3. Prefer grasps at a specific 3D distance (needs depth model)
depth = c.predict("depth-anything-v2", "bin.jpg")
K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
target = select_target_grasp(
    res.grasps,
    target_distance=0.55,       # metres
    depth_result=depth,
    intrinsics=K,
)

# 4. Filter by class + gripper bounds, weighted composite
target = select_target_grasp(
    res.grasps,
    cls="mug",
    gripper_min=20, gripper_max=150,
    target_point=(320, 240),
    weights={"quality": 0.4, "near": 0.6},
)

if target:
    print(target.pose)           # [x, y, width, theta]
    print(target.contacts_flat())# [x0, y0, x1, y1]
```

`return_index=True` returns `(grasp_or_None, index)` into the original list.

#### `select_target_object()`

For pipelines that need to pick ONE object first (e.g. "closest mug") before
computing grasps, use `select_target_object()`:

```python
from visionserve import select_target_object

det = c.predict("rf-detr", "scene.jpg")
obj, idx = select_target_object(
    det, cls="cup",
    near_point="center",          # or (x, y) pixel
    image_size=(1280, 720),
    return_index=True,
)
```

Same scoring criteria as `select_target_grasp` but operates on `Detection` /
`Mask` objects using `conf`, `area`, `near`, and `distance`.

#### `grasp_distances()` / `object_distances()`

True Euclidean camera→grasp (or camera→object) distance, computed by sampling the
depth map at each grasp centre and back-projecting through the camera intrinsics.

```python
from visionserve import grasp_distances, object_distances, CameraIntrinsics

K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
depth = c.predict("depth-anything-v2", "scene.jpg")

# Per-grasp distances (metres)
dists = grasp_distances(depth, res.grasps, K)
for g, d in zip(res.grasps, dists):
    print(f"grasp ({g.x:.0f},{g.y:.0f}) → {d:.2f}m" if d else "no depth")

# Per-object distances
odists = object_distances(depth, det, K)
```

#### Visualization with target grasp highlighted

```python
from visionserve import draw, select_target_grasp

res = c.predict("grasp-gd", "bin.jpg", prompt="mug.", max_grasps_per_object=3)
target = select_target_grasp(res.grasps)

# All grasps drawn in quality colour; target drawn in red on top
annotated = draw(res, "bin.jpg", target_grasp=target)
annotated.save("out.png")

# Or via result.visualize():
res.visualize("bin.jpg", target_grasp=target).save("out.png")
```

### Size filtering — `Result.filter_by_size()`

Remove detections/masks whose bounding-box area is outside a range (client-side, on
already-received results). The server's `min_size` / `max_size` fields do the same
filtering server-side (as % of image area) before the response is sent.

```python
# Absolute mode — area in pixels²
big = res.filter_by_size(min_size=5000)           # keep objects ≥ 5000 px²
small = res.filter_by_size(max_size=2000)          # keep objects ≤ 2000 px²
mid = res.filter_by_size(min_size=500, max_size=50000)

# Relative mode — fraction of image area (0.0–1.0), requires image dimensions
res_rel = res.filter_by_size(
    min_size=0.005,  # at least 0.5% of image area
    max_size=0.9,    # at most 90% of image area
    image_width=1280, image_height=720,
)
```

The method returns a **new** `Result`; the original is not modified.

### Visualization — `draw()` / `result.visualize()`

Requires **Pillow** (`pip install pillow` or `pip install 'visionserve[images]'`).

```python
from visionserve import draw   # or: from visionserve.visualize import draw

# Works with any task — detection, segmentation, classification, depth
res = c.predict("rf-detr", "photo.jpg")
annotated = draw(res, "photo.jpg")   # → PIL.Image
annotated.save("out.jpg")

# Convenience method on Result:
res.visualize("photo.jpg").save("out.jpg")

# Control mask overlay opacity:
annotated = draw(res, "photo.jpg", alpha=0.6)
```

What gets drawn per task:

| Task | Output |
|------|--------|
| `detection` / `open_vocab` | Colored bbox rectangles + `"class conf%"` labels |
| `segmentation` | Semi-transparent mask overlays + bbox outlines + confidence |
| `classification` | Top-K `"class conf%"` text lines in top-left corner |
| `depth` | Turbo colormap image (blue=near → red=far) — replaces original |
| `grasp` | Grasp lines (jaw contacts) + center dot + quality; pass `max_grasps_per_object=N` to limit crowding |

```python
annotated = draw(res, "bin.jpg", max_grasps_per_object=3)
```

## Examples

```bash
# RF-DETR / RT-DETR detection (optionally draw boxes):
python clients/python/examples/detect.py cat.jpg --model rf-detr --save out.png
python clients/python/examples/detect.py cat.jpg --model rt-detr --save out.png

# MobileSAM / EfficientSAM / SAM2 with a box prompt -> mask ndarray:
python clients/python/examples/segment.py img.jpg --model mobile-sam --box 50,40,120,90 --save mask.png
python clients/python/examples/segment.py img.jpg --model efficient-sam --box 50,40,120,90 --save mask.png
python clients/python/examples/segment.py img.jpg --model sam2 --box 50,40,120,90 --save mask.png

# Open-vocab (text prompt) — model must be available on the server:
python clients/python/examples/grounded.py img.jpg --prompt "cat. remote."

# Depth estimation:
python clients/python/examples/depth.py img.jpg --model depth-anything-v2 --save depth.png
python clients/python/examples/depth.py img.jpg --model midas --save depth.png

# Image classification:
python clients/python/examples/classify.py img.jpg --model efficientnet-b0
python clients/python/examples/classify.py img.jpg --model mobilenet-v3
```

## Tests

The test suite is fully offline — it spins up a mock HTTP server in a thread and also
round-trips the RLE codec against a reference port of the Go encoder. No running Go
server is required.

```bash
# with pytest:
python -m pytest clients/python/tests -v

# or as a dependency-free self-test:
python clients/python/tests/test_client.py
```
