# EfficientSAM (segmentation) — weights

> ONNX weight files are **NOT committed** to git. Download them with the instructions
> below (`*.onnx` is `.gitignore`d).

EfficientSAM is a lightweight promptable segmenter from the paper
["Efficient Segment Anything"](https://arxiv.org/abs/2312.00863) (Xiong et al., 2023).
The weights are **Apache-2.0** licensed (`yformer/EfficientSAM` on GitHub). This build
uses the ONNX export from [`yunyangx/EfficientSAM`](https://huggingface.co/yunyangx/EfficientSAM) on HuggingFace.

EfficientSAM replaces SAM's heavy ViT-H image encoder with a ViT-Tiny pre-trained with
the SAMI (Segment Anything with Mixed Images) strategy, making it significantly lighter
than original SAM while retaining strong zero-shot segmentation quality.

## How it works

EfficientSAM is a **prompted, two-session model** (a `PipelineModel`). It requires a
**PROMPT** (a box or point). The model runs as two ONNX graphs declared in
`manifest.yaml` under `files:`:

```yaml
files:
  encoder: efficient_sam_encoder.onnx
  decoder: efficient_sam_decoder.onnx
```

1. **Image encoder** (ViT-Tiny) — runs once per image, producing an image embedding
   that the decoder reuses for every prompt.
2. **Prompt encoder + mask decoder** — runs once per prompt, returning 4 candidate masks
   with IoU scores; the highest-scoring mask is selected.

The Go implementation lives in `internal/models/efficientsam/` and drives encoder →
decoder itself via a `Runner`. Inference (ONNX calls) is handled by the engine/lifecycle;
the model package only does pre/postprocessing and prompt encoding.

## Key differences from MobileSAM

| Feature | MobileSAM | EfficientSAM |
|---|---|---|
| Encoder | TinyViT | ViT-Tiny (SAMI) |
| Encoder input | `input_image` HWC float32, raw 0..255 (norm baked in graph) | `batched_images` NCHW float32, ImageNet-normalized |
| Decoder prompt format | `point_coords [1,N,2]` float32 | `batched_point_coords [1,1,N,2]` float32 |
| Decoder label type | float32 | **float32** (not int64) |
| Decoder output masks | `masks [1,N,H,W]` upsampled to orig size | `output_masks` 5-D `[1,1,M,H,W]` |
| IoU output shape | `[1,N]` | `iou_predictions [1,1,M]` |
| Orig size input | yes (`orig_im_size` int64 [2]) | **yes** — `orig_im_size [2]` int64 = [H, W] |

## Get the ONNX files

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=efficient-sam    # downloads both ONNX files from HuggingFace
```

### Option B — download manually from HuggingFace

HuggingFace repo: [`yunyangx/EfficientSAM`](https://huggingface.co/yunyangx/EfficientSAM)

```bash
MODEL_DIR=models/efficient-sam

hf download yunyangx/EfficientSAM efficientsam_ti_encoder.onnx \
    --local-dir $MODEL_DIR
mv $MODEL_DIR/efficientsam_ti_encoder.onnx $MODEL_DIR/efficient_sam_encoder.onnx

hf download yunyangx/EfficientSAM efficientsam_ti_decoder.onnx \
    --local-dir $MODEL_DIR
mv $MODEL_DIR/efficientsam_ti_decoder.onnx $MODEL_DIR/efficient_sam_decoder.onnx
```

## I/O contract (verified against yunyangx/EfficientSAM)

### Encoder (`efficientsam_ti_encoder.onnx`)

- Input `batched_images`: **NCHW float32** `[batch, 3, H, W]`, ImageNet-normalized
  (`(pixel/255 − mean) / std`). Go resizes the long side to 1024 (no padding needed).
- Output `image_embeddings`: **[batch, 256, 64, 64]**.
- `scale = 1024 / max(origW, origH)` maps original coordinates to the 1024-space used
  by the decoder's `batched_point_coords`.

### Decoder (`efficientsam_ti_decoder.onnx`)

| name | shape | dtype | notes |
|---|---|---|---|
| `image_embeddings` | [1, 256, 64, 64] | float32 | from encoder |
| `batched_point_coords` | [1, 1, N, 2] | float32 | scaled to 1024-space |
| `batched_point_labels` | [1, 1, N] | **float32** | 2=box TL, 3=box BR, 1=fg, 0=bg |
| `orig_im_size` | [2] | int64 | `[origH, origW]` of the original image |

A **box** `[x,y,w,h]` (original coords) becomes 2 points: top-left (label 2) + bottom-
right (label 3), scaled to 1024-space.

Outputs:

- `output_masks`: **[1, 1, M, H, W]** 5-D — M candidate masks.
- `iou_predictions`: **[1, 1, M]** — per-mask IoU scores; highest selects the best mask.

### Postprocess

- The mask with the **highest IoU score** is selected among the 4 candidates.
- Go upsamples from 256×256 to the original image size using **nearest-neighbor**.
- **Mask threshold = logit > 0** (SAM convention).
- Binary mask encoded as **column-major (Fortran-order) RLE**, COCO uncompressed style.

## Usage

EfficientSAM **requires a box or point prompt**:

```bash
visionserve run efficient-sam img.jpg --box x,y,w,h --out mask.png
```

- `--box "x,y,w,h"` — box prompt in original image pixels (multiple boxes: separate by `;`).
- `--point "x,y[,label]"` — point prompt (label 1=fg, 0=bg; multiple: separate by `;`).

Via the HTTP server:

```json
{ "model": "efficient-sam", "image_base64": "...", "box": "x,y,w,h" }
```

Running without a prompt is an error — EfficientSAM has nothing to segment around.

## Performance

Measured on NVIDIA RTX A6000 (48 GB VRAM), VisionServe Go HTTP server, 20 warm requests.

| Metric | Value |
|--------|-------|
| p50 latency (end-to-end HTTP) | 181 ms |
| p95 latency | 247 ms |
| Inference only (srv p50) | 158 ms |
| Throughput | 5.5 RPS |
| VRAM (GPU) | 1628 MB |
| ONNX size | 39 MB |
| Cold-start | 5.8 s |

VRAM is higher than MobileSAM despite smaller ONNX — ViT-Tiny encoder has larger activations.
