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

## Public API

### `Client(host="http://localhost:11435", timeout=120)`

| Method | HTTP | Returns |
| --- | --- | --- |
| `health()` | `GET /api/health` | `{"status": "ok"}` |
| `list_models()` | `GET /api/models` | `list[ModelInfo]` |
| `load(model)` | `POST /api/load` | `{"model", "state"}` |
| `unload(model)` | `POST /api/unload` | `{"model", "state"}` |
| `ps()` | `GET /api/models` (filtered) | loaded `list[ModelInfo]` |
| `predict(model, image, *, prompt=None, box=None, point=None, min_size=0, max_size=0)` | `POST /api/predict` | `Result` |

`min_size` / `max_size` filter by bounding-box area as a **percentage of the image area** (0–100; `0` = no limit). Example: `min_size=0.5` keeps only objects covering at least 0.5% of the image. The conversion to absolute pixels is done server-side using the uploaded image dimensions.

`predict()` `image` accepts:
- `str` / `os.PathLike` — a path to an image file,
- `bytes` — already-encoded image (PNG/JPEG), sent verbatim,
- `PIL.Image.Image` — encoded to PNG client-side,
- `numpy.ndarray` — `HWC` uint8 (or float in `[0,1]` → scaled to uint8); grayscale
  `(H, W)` is promoted to RGB. Encoded to PNG client-side.

Prompts (serialized to the server's string format):
- `box`: `[x, y, w, h]` or a list of boxes → `"x,y,w,h"` joined by `;`.
- `point`: `[x, y]` / `[x, y, label]` or a list (label 1=fg, 0=bg) → `"x,y[,label]"` joined by `;`.
- `prompt`: free text, e.g. `"cat. remote."`.

### Result types

```python
Detection(bbox: list[float], cls: str, conf: float)       # bbox = [x, y, w, h], original px
Mask(rle: str, bbox: list[float], conf: float)
Classification(cls: str, conf: float)                      # top-K prediction
Result(
    task, model, duration_ms,
    detections: list[Detection],
    masks: list[Mask],
    classifications: list[Classification],                 # task="classification"
    depth_map: list[float] | None,                         # task="depth", row-major H×W
    depth_width: int | None,
    depth_height: int | None,
)
```

`Mask.to_ndarray(width, height) -> np.ndarray` decodes the COCO-style **column-major**
uncompressed RLE into a boolean `(height, width)` array (requires numpy). It is the
exact inverse of the server's encoder.

```python
# Box-prompted segmentation
res = c.predict("mobile-sam", "img.jpg", box=[50, 40, 120, 90])
from PIL import Image
w, h = Image.open("img.jpg").size
mask = res.masks[0].to_ndarray(width=w, height=h)   # bool (h, w)

# No-prompt → Automatic Mask Generator (segment everything, ~256 masks)
res = c.predict("mobile-sam", "img.jpg")
for m in res.masks:
    print(m.bbox, round(m.conf, 3))
```

For depth maps, reshape `depth_map` using `depth_width` / `depth_height`:

```python
import numpy as np
res = c.predict("depth-anything-v2", "img.jpg")
depth = np.array(res.depth_map).reshape(res.depth_height, res.depth_width)
```

For classification, iterate `classifications` (already sorted by confidence descending):

```python
res = c.predict("efficientnet-b0", "img.jpg")
for cls_pred in res.classifications:
    print(cls_pred.cls, round(cls_pred.conf, 3))
```

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
