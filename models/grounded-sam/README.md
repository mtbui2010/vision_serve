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
- `text_threshold` (default 0.25): token→label assignment filter.

The overlay PNG draws each detection box + its mask.
