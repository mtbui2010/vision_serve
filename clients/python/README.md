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

From the repository root:

```bash
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
| `predict(model, image, *, prompt=None, box=None, point=None)` | `POST /api/predict` | `Result` |

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
Detection(bbox: list[float], cls: str, conf: float)   # bbox = [x, y, w, h], original px
Mask(rle: str, bbox: list[float], conf: float)
Result(task, model, detections: list[Detection], masks: list[Mask], duration_ms)
```

`Mask.to_ndarray(width, height) -> np.ndarray` decodes the COCO-style **column-major**
uncompressed RLE into a boolean `(height, width)` array (requires numpy). It is the
exact inverse of the server's encoder.

```python
res = c.predict("mobile-sam", "img.jpg", box=[50, 40, 120, 90])
from PIL import Image
w, h = Image.open("img.jpg").size
mask = res.masks[0].to_ndarray(width=w, height=h)   # bool (h, w)
```

## Examples

```bash
# RF-DETR detection (optionally draw boxes):
python clients/python/examples/detect.py cat.jpg --model rf-detr --save out.png

# MobileSAM with a box prompt -> mask ndarray:
python clients/python/examples/segment.py img.jpg --box 50,40,120,90 --save mask.png

# Open-vocab (text prompt) — model must be available on the server:
python clients/python/examples/grounded.py img.jpg --prompt "cat. remote."
```

## Tests

The test suite is fully offline — it spins up a mock HTTP server in a thread and also
round-trips the RLE codec against a reference port of the Go encoder. No running Go
server is required.

```bash
# with pytest:
/home/trung/miniconda3/envs/label/bin/python3 -m pytest clients/python/tests -v

# or as a dependency-free self-test:
/home/trung/miniconda3/envs/label/bin/python3 clients/python/tests/test_client.py
```
