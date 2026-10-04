# Grounded-SAM (Apache-2.0)

Grounded-SAM is the composition of two models already in this repo:

1. **GroundingDINO** — open-vocabulary detection from a text prompt
   (`../grounding-dino/model.onnx`, Apache-2.0).
2. **MobileSAM** — box-prompted segmentation
   (`../mobile-sam/mobile_sam_encoder.onnx` + `mobile_sam_decoder_single.onnx`, Apache-2.0).

The pipeline runs GroundingDINO to get boxes + labels from the prompt, then feeds each box to
MobileSAM to get one mask per box. Output is the unified schema: `detections` plus `masks`
(index-aligned — `masks[i]` belongs to `detections[i]`).

This is a fully free community pipeline: only permissive (Apache-2.0/MIT) weights, all
inference through ONNX Runtime, no Python at runtime.

## Weights

This directory contains **no** ONNX files of its own. The `manifest.yaml` references the
weights of the sibling model directories via relative paths:

```
files:
  gdino:   ../grounding-dino/model.onnx
  encoder: ../mobile-sam/mobile_sam_encoder.onnx
  decoder: ../mobile-sam/mobile_sam_decoder_single.onnx
```

Pull the dependencies first, then grounded-sam:

```bash
visionserve pull grounding-dino
visionserve pull mobile-sam
visionserve pull grounded-sam   # creates the manifest — no extra download
```

`visionserve pull grounded-sam` checks that both dependencies are present before writing
the manifest, so the order above must be followed. The `grounded-sam` entry creates only
a `manifest.yaml` (no ONNX download) that points to the sibling model files.

## Run

```sh
visionserve run grounded-sam img.jpg --prompt "cat. remote." --out overlay.png
```

- `conf_threshold` (default 0.3): GroundingDINO box/query score filter.
- `text_threshold` (default 0.25): a second floor on the same score (labels are always one
  whole prompt phrase, so it never changes them).

Both can be **overridden per request** (not just in the manifest) via the
`box_threshold` / `text_threshold` HTTP form/JSON fields or the matching Python client
kwargs; precedence is per-request (>0) → manifest → default. A box is kept only when its
best-phrase score is above both, so with the defaults `box_threshold` decides; raise
`text_threshold` above it to be stricter:

```bash
curl -s -F model=grounded-sam -F image=@img.jpg -F prompt="canned coffee." \
  -F box_threshold=0.3 -F text_threshold=0.4 http://localhost:11435/api/predict
```

The overlay PNG draws each detection box + its mask.
