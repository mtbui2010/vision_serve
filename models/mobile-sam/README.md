# MobileSAM (segmentation) — weights

> ONNX weight files are **NOT committed** to git. Download/export them with the
> instructions below (`*.onnx` is `.gitignore`d).

MobileSAM is a lightweight Segment Anything model. The weights are **Apache-2.0**
(fully free and open-source) and the `samexporter` tooling used to produce the ONNX
files is **MIT** — both permissive, no AGPL. MobileSAM replaces SAM's heavy ViT-H image
encoder with a small **TinyViT** (that is the "Mobile" part).

## How it works (verified — no longer a stub)

MobileSAM is a **prompted, two-session model** (a `PipelineModel`, not a single-pass
detector). It requires a **PROMPT** (a box or a point) — SAM segments *around* a prompt.
It runs as two ONNX graphs, declared in `manifest.yaml` under `files:`:

```yaml
files:
  encoder: mobile_sam_encoder.onnx
  decoder: mobile_sam_decoder_single.onnx
```

1. **Image encoder** (TinyViT) — runs once per image, producing an image embedding that
   the decoder reuses for every prompt.
2. **Prompt encoder + mask decoder** — runs once per prompt set, turning the embedding +
   prompt into a mask.

The Go implementation lives in `internal/models/mobilesam/` and drives encoder → decoder
itself via a `Runner`. Inference (the ONNX calls) is handled by the engine/lifecycle;
the model package only does pre/postprocessing and prompt encoding.

## Get the two ONNX files

The two graphs follow the `samexporter` convention. MobileSAM weights are Apache-2.0;
`samexporter` is MIT.

```bash
# 1. Get the MobileSAM checkpoint (Apache-2.0): weights/mobile_sam.pt
#    https://github.com/ChaoningZhang/MobileSAM
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# 2. Export encoder + decoder ONNX with samexporter (MIT):
#    https://github.com/vietanhdev/samexporter
pip install samexporter
python -m samexporter.export \
    --checkpoint weights/mobile_sam.pt \
    --output models/mobile-sam/mobile_sam \
    --model-type vit_t \
    --quantize     # optional
# Produces mobile_sam_encoder.onnx and mobile_sam_decoder.onnx; rename the decoder to
# mobile_sam_decoder_single.onnx (single-output decoder) to match the manifest.
```

> Exact flag names vary by tool version — check the upstream README. **Do NOT commit**
> the weights to this repo; just place the two `.onnx` files next to `manifest.yaml` at
> runtime.

## I/O contract (verified)

Verified with `verify_sam.py` (dev-only spike — not part of the Go runtime; it inspects
tensors and runs an end-to-end sanity check) against the real ONNX files.

### Encoder

- Input `input_image`: raw **HWC float32, values 0..255**. Go resizes the **long side to
  1024** keeping aspect ratio; **normalization (SAM pixel mean/std) and pad-to-1024 are
  baked into the graph**, so Go does neither.
- Output `image_embeddings`: **[1, 256, 64, 64]**.
- `scale = 1024 / max(origW, origH)` maps original-image coordinates into the resized
  1024 space the decoder's point prompts use. Padding is bottom/right only, so no pad
  offset is applied to coordinates.

### Decoder

Inputs:

| name               | shape           | notes                                          |
|--------------------|-----------------|------------------------------------------------|
| `image_embeddings` | [1, 256, 64, 64]| from the encoder                               |
| `point_coords`     | [1, n, 2]       | prompt points in **1024 space** (× scale)      |
| `point_labels`     | [1, n]          | SAM labels (see below)                         |
| `mask_input`       | [1, 1, 256, 256]| zeros when no previous mask                    |
| `has_mask_input`   | [1]             | 0 when no previous mask                        |
| `orig_im_size`     | [2]             | `(H, W)` of the original image                 |

A **box** `[x0,y0,x1,y1]` is encoded as **2 points**: top-left with **label 2** and
bottom-right with **label 3**, in resized-1024 space. (Point prompts use label 1=fg,
0=bg; a (0,0) point with label -1 pads a points-only prompt.)

Outputs:

- `masks`: **[1, N, H, W]** — already **upsampled to the original image size** by the
  graph (using `orig_im_size`).
- `iou_predictions`: per-mask confidence; the highest-IoU channel is selected.

### Postprocess

- **Mask threshold = logit > 0** (the SAM convention). No NMS.
- The thresholded binary mask is encoded as **column-major (Fortran-order) RLE**, COCO
  uncompressed style: counts of alternating runs starting with a background (0) run,
  serialized as space-separated decimals.
- A tight bbox and the IoU-derived confidence are attached to each mask in the unified
  `Result` schema (`pkg/api/types.go`).

## Usage

SAM **requires a box or point prompt**. The CLI runs in-process (no server needed):

```bash
visionserve run mobile-sam img.jpg --box x,y,w,h --out mask.png
```

- `--box "x,y,w,h"` — box prompt (multiple boxes separated by `;`).
- `--point "x,y[,label]"` — point prompt (label 1=fg, 0=bg; multiple separated by `;`).
- `--out mask.png` — saves the image with the mask drawn on it.

Via the HTTP server, `POST /api/predict` with a `box` field (and `model: mobile-sam`):

```json
{ "model": "mobile-sam", "image_base64": "...", "box": "x,y,w,h" }
```

Running without any prompt is an error — MobileSAM has nothing to segment around.
