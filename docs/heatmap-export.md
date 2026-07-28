# Heatmap Visualization — ONNX Export Guide

VisionServe supports Grad-CAM-style heatmaps via two strategies:

| Strategy | Models | Inferences needed |
|----------|--------|-------------------|
| **Attention map** | Transformer-based (RF-DETR, GroundingDINO) | 1 |
| **Score-CAM** | CNN-based (ResNet, EfficientDet, …) | N (≤ `top_channels`, default 64) |

The server reads the extra tensors from the ONNX file. **You** export the ONNX once with
additional outputs declared; the server loads it lazily (only when `/api/explain` is called)
and runs regular detection through the original output set with zero overhead.

---

## 1 — Attention map (transformer models)

### What to expose

Add the **cross-attention weights** from the decoder as an extra output. The server averages
over layers and heads, then picks the spatial map for the queried detection.

### Required tensor format

| Property | Value |
|----------|-------|
| Node name | any — declare it in the manifest |
| Shape | `[num_layers, batch, num_heads, num_queries, num_spatial_tokens]` |
| dtype | `float32` |
| `num_spatial_tokens` | `(input_H / spatial_stride) × (input_W / spatial_stride)` |

Example — RF-DETR, 640 × 640 input, stride 32:
```
[6, 1, 8, 300, 400]   # 6 decoder layers, 8 heads, 300 queries, 20×20 spatial grid
```

Example — GroundingDINO (text-conditioned cross-attention), 800 × 800 input, stride 32:
```
[6, 1, 8, 900, 625]   # 900 queries, 25×25 grid
```

### PyTorch export (RF-DETR)

```python
import torch
from rfdetr import RFDETRBase   # or your model class

model = RFDETRBase()
model.load_state_dict(torch.load("rf_detr_base.pth"))
model.eval()

# ── Hook: collect cross-attention weights from all decoder layers ──────────
attn_store = []

def _hook(module, inp, out):
    # out is (attn_output, attn_weights) when need_weights=True
    if isinstance(out, tuple) and len(out) == 2:
        attn_store.append(out[1])   # [batch, heads, queries, keys]

for layer in model.transformer.decoder.layers:
    layer.cross_attn.register_forward_hook(_hook)

# ── Trace ──────────────────────────────────────────────────────────────────
dummy = torch.zeros(1, 3, 640, 640)

class ExportWrapper(torch.nn.Module):
    def forward(self, x):
        attn_store.clear()
        logits, boxes = model(x)          # your model's normal outputs
        # stack: [num_layers, batch, heads, queries, spatial_tokens]
        attn = torch.stack(attn_store, dim=0)
        return logits, boxes, attn

wrapper = ExportWrapper()

torch.onnx.export(
    wrapper,
    dummy,
    "rf_detr_explain.onnx",
    input_names=["images"],
    output_names=["logits", "boxes", "cross_attn_weights"],
    dynamic_axes={"images": {0: "batch"}},
    opset_version=17,
)
print("Exported rf_detr_explain.onnx")
```

### Manifest (attention)

```yaml
name: rf-detr-base
task: detection
license: Apache-2.0
model_file: rf_detr_explain.onnx    # the explain-enabled ONNX

input:
  width: 640
  height: 640
  layout: NCHW
  letterbox: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

postprocess:
  type: detr
  box_format: cxcywh
  conf_threshold: 0.5
  max_detections: 300

labels: coco.txt

explain:
  type: attention
  outputs:
    attention: cross_attn_weights   # must match the output_names in torch.onnx.export
  spatial_stride: 32                # backbone downsampling factor (32 for ResNet-50 based)
```

---

## 2 — Score-CAM (CNN-based models)

### What to expose

Add the **backbone feature maps** from the last convolutional layer as an extra output.
The server uses these as spatial masks, re-runs detection on each masked image, weights
the masks by the resulting detection score, and sums them into a heatmap.

### Required tensor format

| Property | Value |
|----------|-------|
| Node name | any — declare it in the manifest |
| Shape | `[batch, channels, H/stride, W/stride]` |
| dtype | `float32` |

Example — ResNet-50 backbone, 640 × 640 input, stride 32:
```
[1, 2048, 20, 20]
```

### PyTorch export (generic CNN detector)

```python
import torch

model = YourCNNDetector()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ── Hook: capture backbone feature maps ───────────────────────────────────
feat_store = []

def _feat_hook(module, inp, out):
    feat_store.append(out)

# Attach to the last conv block of your backbone (adjust the attribute path):
model.backbone.layer4.register_forward_hook(_feat_hook)

# ── Wrapper ────────────────────────────────────────────────────────────────
class ExportWrapper(torch.nn.Module):
    def forward(self, x):
        feat_store.clear()
        boxes, scores = model(x)
        return boxes, scores, feat_store[0]   # [1, C, H/s, W/s]

wrapper = ExportWrapper()
dummy = torch.zeros(1, 3, 640, 640)

torch.onnx.export(
    wrapper,
    dummy,
    "detector_explain.onnx",
    input_names=["images"],
    output_names=["boxes", "scores", "backbone_features"],
    dynamic_axes={"images": {0: "batch"}},
    opset_version=17,
)
print("Exported detector_explain.onnx")
```

### Manifest (Score-CAM)

```yaml
name: my-cnn-detector
task: detection
license: Apache-2.0
model_file: detector_explain.onnx

input:
  width: 640
  height: 640
  layout: NCHW
  letterbox: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

postprocess:
  type: detr
  conf_threshold: 0.5
  max_detections: 300

explain:
  type: score_cam
  outputs:
    features: backbone_features     # must match output_names in torch.onnx.export
  spatial_stride: 32
  top_channels: 64                  # default; override per-request via top_channels field
```

---

## 3 — Explain API

### Request

```
POST /api/explain
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | yes | model name (same as `/api/predict`) |
| `image` | file | yes | input image |
| `class` | string | no | class name to explain (e.g. `"cup"`). Uses highest-score detection matching the class. |
| `detection_idx` | int | no | 0-based index into detections; overrides `class` |
| `top_channels` | int | no | Score-CAM only — channels to sample (default: manifest value or 64) |
| `alpha` | float | no | PNG overlay opacity 0–1 (default 0.5) |
| `format` | string | no | `"png"` (default) \| `"numpy"` |

### Response — `format=png`

```
Content-Type: image/png
Body: heatmap overlaid on the original image (JET colormap)
```

### Response — `format=numpy`

```
Content-Type: application/octet-stream
X-Heatmap-Shape: 480,640        # H,W of the heatmap (matches original image)
X-Heatmap-Dtype: float32
Body: raw little-endian float32 bytes, values in [0, 1]
```

Python client reconstruction:

```python
import numpy as np, requests

resp = requests.post("http://localhost:11435/api/explain", files={
    "image": open("photo.jpg", "rb"),
}, data={"model": "rf-detr-base", "class": "cup", "format": "numpy"})

shape = tuple(int(x) for x in resp.headers["X-Heatmap-Shape"].split(","))
heatmap = np.frombuffer(resp.content, dtype=np.float32).reshape(shape)
# heatmap[y, x] ∈ [0, 1]: activation strength at pixel (x, y)
```

---

## 4 — How the server separates detect vs. explain sessions

The server creates **two ONNX sessions from the same file**:

```
manifest.explain.outputs.* values
        │
        ▼
detect_session  ←  all model outputs  minus  explain outputs   (always active)
explain_session ←  all model outputs  (lazy — created on first /api/explain call)
```

Regular `/api/predict` goes through `detect_session` only — the extra attention or feature
tensors are **never requested** and therefore **never computed** by ONNX Runtime (graph
pruning). Zero overhead for regular inference.

---

## 5 — Verifying your exported ONNX

```bash
# Inspect all output names and shapes (no server needed):
visionserve inspect ./rf_detr_explain.onnx

# Expected output:
# inputs:
#   images  [1, 3, 640, 640]  float32
# outputs:
#   logits              [1, 300, 91]    float32
#   boxes               [1, 300, 4]     float32
#   cross_attn_weights  [6, 1, 8, 300, 400]  float32   ← explain output
```

The explain output name in the `visionserve inspect` output must match the value you set
in `manifest.yaml` under `explain.outputs.attention` (or `explain.outputs.features`).
