# NanoSAM (segmentation) — weights

> ONNX weight files are **NOT committed** to git. Download/export them with the
> instructions below (`*.onnx` is `.gitignore`d).

## What is NanoSAM

NanoSAM is **NVIDIA's edge-optimized Segment Anything Model** variant, designed for
real-time inference on **NVIDIA Jetson** devices. It is released under the **Apache-2.0 license**
(fully permissive, commercial use allowed, no AGPL restrictions).

Key architecture difference from MobileSAM:

| | MobileSAM | NanoSAM |
|---|---|---|
| Encoder | TinyViT | **ResNet-18** (NVIDIA-distilled) |
| Encoder input | HWC float32 0..255 (normalize baked in graph) | **NCHW float32 ImageNet-normalized** (Go normalizes) |
| Decoder | Standard SAM decoder | Standard SAM decoder (identical) |
| License | Apache-2.0 | **Apache-2.0** |
| Target device | CPU / generic GPU | **NVIDIA Jetson / TensorRT** |

NanoSAM reuses the standard SAM prompt-encoder and mask-decoder from the original SAM
(Meta AI). The encoder is a compact ResNet-18 distilled by NVIDIA specifically to run
fast on TensorRT.

## Prompt API

Same as `mobile-sam` — box or point prompts, no text:

```bash
# Box prompt
visionserve run nano-sam img.jpg --box x,y,w,h --out mask.png

# Point prompt (label 1=foreground, 0=background)
visionserve run nano-sam img.jpg --point x,y,1 --out mask.png
```

Via HTTP:
```json
{ "model": "nano-sam", "image_base64": "...", "box": "x,y,w,h" }
```

For text-driven segmentation (`"cat"` → mask), use the `grounded-sam` model instead.

## Expected speed

On NVIDIA Jetson (TensorRT EP), NanoSAM is designed to run significantly faster than
MobileSAM due to the smaller ResNet-18 encoder and TensorRT optimization. The
`prefer: [tensorrt, cuda, cpu]` fallback chain in `manifest.yaml` ensures TRT is used
when available and the model gracefully falls back to CUDA then CPU.

## How to get the ONNX files

The ONNX weights are available from the upstream GitHub repository:
**https://github.com/NVIDIA-AI-IOT/nanosam**

The two files needed are:
- `resnet18_image_encoder.onnx` — ResNet-18 image encoder
- `mobile_sam_mask_decoder.onnx` — standard SAM mask decoder

> **Note:** NanoSAM weights are hosted on GitHub / Google Drive (not HuggingFace).
> Check the upstream repo for the current download URLs:
> https://github.com/NVIDIA-AI-IOT/nanosam

### Option A — download pre-exported ONNX files

Follow the download links in the upstream README at
https://github.com/NVIDIA-AI-IOT/nanosam and place the files next to `manifest.yaml`:

```
models/nano-sam/
  manifest.yaml
  resnet18_image_encoder.onnx   ← encoder
  mobile_sam_mask_decoder.onnx  ← decoder
```

### Option B — export from checkpoint

```bash
# Clone the upstream repo and follow its export instructions:
git clone https://github.com/NVIDIA-AI-IOT/nanosam
cd nanosam
pip install -e .

# Export encoder (ResNet-18 distilled) and decoder (standard SAM):
python -m nanosam.tools.export_encoder \
    --output models/nano-sam/resnet18_image_encoder.onnx

python -m nanosam.tools.export_decoder \
    --output models/nano-sam/mobile_sam_mask_decoder.onnx
```

Exact commands may vary; check the upstream README for the current export workflow.

## I/O contract (verified)

### Encoder (`resnet18_image_encoder.onnx`)

- Input `image`: **NCHW float32 [1, 3, 1024, 1024]**, ImageNet-normalized.
  Go resizes the long side to 1024 (aspect-ratio preserving), zero-pads the shorter side
  (bottom/right), and applies: `pixel = (raw/255 - mean) / std` per channel.
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Output `image_embeddings`: **[1, 256, 64, 64]** — image embedding passed to decoder.
- `scale = 1024 / max(origW, origH)` maps original coordinates into the 1024 input space.

### Decoder (`mobile_sam_mask_decoder.onnx`)

Standard SAM decoder — **no `orig_im_size` input** (unlike MobileSAM's decoder):

| name | shape | notes |
|---|---|---|
| `image_embeddings` | [1, 256, 64, 64] | from encoder |
| `point_coords` | [1, N, 2] | prompt points in 1024 space (× scale) |
| `point_labels` | [1, N] | SAM labels: 2=box-TL, 3=box-BR, 1=fg, 0=bg |
| `mask_input` | [1, 1, 256, 256] | zeros when no previous mask |
| `has_mask_input` | [1] | 0 when no previous mask |

A box `[x,y,w,h]` → 2 points: top-left label 2, bottom-right label 3, in 1024 space.
Point prompts use label 1=fg, 0=bg.

Outputs:
- `low_res_masks`: **[1, M, 256, 256]** — low-resolution mask candidates (Go upsamples).
- `iou_predictions`: **[1, M]** — per-mask IoU score; highest is selected.

### Postprocess

- Mask threshold = **logit > 0** (SAM convention).
- Binary mask encoded as **column-major (Fortran-order) RLE**, COCO uncompressed style.
- Tight bbox and IoU-derived confidence attached per mask in the unified `Result` schema.

## License

NanoSAM is released under the **Apache-2.0 license** by NVIDIA.
The standard SAM decoder weights (Meta AI) are also **Apache-2.0**.
Both are permissive licenses; commercial and closed-source use is allowed.
