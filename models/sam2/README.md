# SAM2-Tiny (segmentation) — weights

> ONNX weight files are **NOT committed** to git. Download/export them with the
> instructions below (`*.onnx` is `.gitignore`d).

SAM2 (Segment Anything Model 2) is a promptable segmentation model from Meta AI.
The weights are **Apache-2.0** — fully free and open-source, including for commercial
use. SAM2-Tiny is the smallest (fastest) variant in the SAM2 family.

**License:** Apache-2.0 (Meta AI / facebookresearch).
**Source repo:** https://github.com/facebookresearch/segment-anything-2

## How it works

SAM2 is a **prompted, two-session model** (a `PipelineModel`). It requires a **PROMPT**
(a box or a point) and produces a segmentation mask.

Key difference from MobileSAM: SAM2's encoder outputs **three tensors** (multi-scale
features) instead of one. All three must be forwarded to the decoder.

```yaml
files:
  encoder: sam2_tiny_encoder.onnx
  decoder: sam2_tiny_decoder.onnx
```

### Encoder I/O

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `image` (input) | `[1, 3, 1024, 1024]` | float32 | NCHW, ImageNet-normalized |
| `image_embed` (output) | `[1, 256, 64, 64]` | float32 | primary feature map |
| `high_res_feats_0` (output) | `[1, 32, 256, 256]` | float32 | high-res level 0 |
| `high_res_feats_1` (output) | `[1, 64, 128, 128]` | float32 | high-res level 1 |

### Decoder I/O

| Name | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `image_embed` (input) | `[1, 256, 64, 64]` | float32 | from encoder |
| `high_res_feats_0` (input) | `[1, 32, 256, 256]` | float32 | from encoder |
| `high_res_feats_1` (input) | `[1, 64, 128, 128]` | float32 | from encoder |
| `point_coords` (input) | `[1, N, 2]` | float32 | in 1024-space |
| `point_labels` (input) | `[1, N]` | **float32** | 1=fg, 0=bg, 2=box-tl, 3=box-br |
| `mask_input` (input) | `[1, 1, 256, 256]` | float32 | zeros — required even when unused |
| `has_mask_input` (input) | `[1]` | float32 | 0.0 — required even when unused |
| `masks` (output) | `[1, M, H, W]` | float32 | logits — threshold at 0.0 |
| `iou_predictions` (output) | `[1, M]` | float32 | mask quality score |

> **Note:** `point_labels` is **float32** (not int64) — verified against
> `SharpAI/sam2-hiera-tiny-onnx`. `mask_input` and `has_mask_input` are required decoder
> inputs; pass zero tensors when no prior mask is available.

## Get the two ONNX files

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=sam2    # downloads encoder.onnx + decoder.onnx from HuggingFace
```

### Option B — download manually from HuggingFace

HuggingFace repo: [`SharpAI/sam2-hiera-tiny-onnx`](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx)
(Apache-2.0 — verified)

```bash
hf download SharpAI/sam2-hiera-tiny-onnx encoder.onnx decoder.onnx \
    --local-dir models/sam2/
mv models/sam2/encoder.onnx models/sam2/sam2_tiny_encoder.onnx
mv models/sam2/decoder.onnx models/sam2/sam2_tiny_decoder.onnx
```

### Option B — export from original checkpoint

```bash
# Install SAM2 (Apache-2.0)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Export encoder + decoder to ONNX
# (use a community export script or sam2's own tools)
# Place outputs as:
#   models/sam2/sam2_tiny_encoder.onnx
#   models/sam2/sam2_tiny_decoder.onnx
```

## Usage examples

```bash
# Box prompt (recommended — segment the region inside a box)
visionserve run sam2 image.jpg --box 100,50,200,150

# Point prompt (foreground click)
visionserve run sam2 image.jpg --point 200,175,1

# Multiple boxes (one mask per box)
visionserve run sam2 image.jpg --box 10,10,100,100 --box 200,200,50,50
```

## Preprocessing

The Go implementation:
1. Resizes the image so its long side equals 1024 (aspect-ratio preserved).
2. Zero-pads to exactly 1024×1024 (bottom/right padding).
3. Normalizes per-channel with ImageNet mean/std (unlike MobileSAM which bakes
   normalization into the graph and expects raw 0-255 input).

> If your ONNX export bakes normalization into the graph, edit the `sam2Mean`/`sam2Std`
> comment in `internal/models/sam2/preprocess.go` and pass raw 0..255 values instead.

## Result format

Output follows the unified VisionServe schema: `Mask.RLE` is column-major (COCO-style)
uncompressed RLE, `Mask.BBox` is `[x, y, w, h]` in mask-space coordinates,
`Mask.Conf` is the IoU prediction score.

## Performance

Measured on NVIDIA RTX A6000 (48 GB VRAM), VisionServe Go HTTP server, 20 warm requests.

| Metric | Value |
|--------|-------|
| p50 latency (end-to-end HTTP) | 242 ms |
| p95 latency | 544 ms |
| Inference only (srv p50) | 222 ms |
| Throughput | 3.9 RPS |
| VRAM (GPU) | 2508 MB |
| ONNX size | 148 MB |
| Cold-start | 5.3 s |

p95 of 544 ms reflects the multi-scale encoder loading variability. Best quality among SAM variants.
