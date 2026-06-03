# SCRFD — InsightFace Face Detector

**Architecture:** SCRFD-10GF  
**License:** MIT (InsightFace detection sub-project)  
**Task:** Face detection (no text prompt needed)  
**Input:** 640×640 NCHW, letterboxed  

## Model overview

SCRFD (Sample and Computation Redistribution for Face Detection) is a multi-scale
anchor-based face detector by InsightFace. The 10GF variant achieves strong WiderFace
accuracy with efficient computation, making it suitable for edge and server deployments.

Unlike RF-DETR, SCRFD is anchor-based and **requires NMS**.

## Downloading the model

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=scrfd    # downloads scrfd_10g_bnkps.onnx from HuggingFace
```

### Option B — download manually from HuggingFace

HuggingFace repo: [`cromsc/scrfd-10g`](https://huggingface.co/cromsc/scrfd-10g)

```bash
hf download cromsc/scrfd-10g scrfd_10g_bnkps.onnx \
    --local-dir models/scrfd/
mv models/scrfd/scrfd_10g_bnkps.onnx models/scrfd/det_10g.onnx
```

Place the file at `models/scrfd/det_10g.onnx` (next to `manifest.yaml`).

## Expected I/O (verified against cromsc/scrfd-10g)

Input `input.1` — dynamic shape `[1, 3, H, W]`, NCHW, normalized `(pixel − 127.5) / 128.0`.

Outputs are **9 numerically-named tensors** (no `score_8` etc.); the postprocessor
routes them by shape:

| Shape | Stride | Content |
|-------|--------|---------|
| `[1, 12800, 1]` | 8 | face score logits |
| `[1, 3200, 1]` | 16 | face score logits |
| `[1, 800, 1]` | 32 | face score logits |
| `[1, 12800, 4]` | 8 | bbox distances (l, t, r, b) |
| `[1, 3200, 4]` | 16 | bbox distances |
| `[1, 800, 4]` | 32 | bbox distances |
| `[1, 12800, 10]` | 8 | 5 keypoints × 2 coords (unused in v1) |
| `[1, 3200, 10]` | 16 | keypoints (unused) |
| `[1, 800, 10]` | 32 | keypoints (unused) |

Some exports omit the `kps` tensors; the postprocessor handles both (6-output and 9-output) variants.

## Example usage

**HTTP API** (no prompt needed):
```bash
curl -X POST http://localhost:11435/api/predict \
  -F model=scrfd \
  -F image=@photo.jpg
```

**JSON body:**
```bash
curl -X POST http://localhost:11435/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "scrfd",
    "image_base64": "'$(base64 -w0 photo.jpg)'"
  }'
```

**CLI:**
```bash
visionserve predict --model scrfd --image photo.jpg
```

**Example response:**
```json
{
  "task": "detection",
  "model": "scrfd",
  "detections": [
    {"bbox": [142.3, 80.1, 95.4, 120.7], "class": "face", "conf": 0.987},
    {"bbox": [310.0, 95.2, 88.1, 112.3], "class": "face", "conf": 0.931}
  ],
  "duration_ms": 12.4
}
```

`bbox` is `[x, y, w, h]` in **original image coordinates** (top-left corner + size).
All detections have `class: "face"`.

## Normalization note

SCRFD uses a non-standard normalization: `(pixel − 127.5) / 128.0`, NOT ImageNet stats.
The manifest declares `mean: [127.5, 127.5, 127.5]` and `std: [128.0, 128.0, 128.0]`,
which the preprocessor divides by 255 before passing to `ImageToCHWFloat` so the math
resolves correctly to the per-pixel formula above.

## Performance

Measured on NVIDIA RTX A6000 (48 GB VRAM), VisionServe Go HTTP server, 20 warm requests.

| Metric | Value |
|--------|-------|
| p50 latency (end-to-end HTTP) | 45 ms |
| p95 latency | 69 ms |
| Inference only (srv p50) | 23 ms |
| Throughput | 22.4 RPS |
| VRAM (GPU) | 420 MB |
| ONNX size | 16 MB |
| Cold-start | 3.9 s |

Most efficient model in the catalog by ONNX size (16 MB). Go preprocess+postprocess = 22 ms overhead.
