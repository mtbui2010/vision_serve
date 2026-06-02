# Manifest spec ("Modelfile for CV")

Each model is a subdirectory in the registry (`./models/<name>/`) containing a
`manifest.yaml` + the ONNX file(s) (not committed) + (optional) a labels file.

## Full example (single-session model)

```yaml
name: rf-detr                 # required — model identifier in the registry
task: detection               # required — detection | segmentation | open_vocab
license: Apache-2.0           # required — permissive ONLY (Apache-2.0/MIT/BSD); AGPL forbidden
architecture: rf-detr         # optional — factory key (default = name)
model_file: rf-detr-base.onnx # required (when no `files:`) — relative path to the .onnx

input:
  width: 560                  # required > 0
  height: 560                 # required > 0
  layout: NCHW                # NCHW | NHWC
  letterbox: true             # keep aspect ratio + pad
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

postprocess:
  type: detr                  # decode hint (detr/sam/...)
  box_format: cxcywh          # cxcywh | xyxy (normalized [0,1])
  conf_threshold: 0.5
  max_detections: 300

labels: coco.txt              # optional — one class per line

runtime:
  prefer: [tensorrt, cuda, cpu]   # EP fallback chain (CPU is always appended last)
  idle_unload_seconds: 300        # 0 = never auto-unload
```

## Multi-session model (the `files:` map)

Prompted / multi-session models (MobileSAM, Grounded-SAM) declare several ONNX graphs
keyed by **role** instead of a single `model_file`. The role keys must match the
model's `Roles()` (see [contributing-models.md](contributing-models.md)). When `files:`
is present, **`model_file` is optional**.

```yaml
name: mobile-sam
task: segmentation
license: Apache-2.0
architecture: mobile-sam

# role → relative .onnx path. Lifecycle loads one engine.Session per role.
files:
  encoder: mobile_sam_encoder.onnx
  decoder: mobile_sam_decoder_single.onnx

input:
  width: 1024
  height: 1024
  layout: NHWC
  letterbox: false

postprocess:
  type: sam        # mask threshold = logit > 0; mask encoded as column-major RLE

runtime:
  prefer: [tensorrt, cuda, cpu]
  idle_unload_seconds: 300
```

## Field reference

| Field | Meaning |
|-------|---------|
| `name` | required — registry identifier |
| `task` | `detection` / `segmentation` / `open_vocab` |
| `license` | required — must be in the permissive allowlist (below) |
| `architecture` | optional — factory key (default = `name`) |
| `model_file` | path to the .onnx — **optional when `files:` is present** |
| `files` | **map role → .onnx path** for multi-session models (e.g. SAM `encoder`/`decoder`). All listed files must exist on disk for the model to be `available`/loadable |
| `input.*` | width/height/layout/letterbox/normalize |
| `postprocess.type` | decode hint (`detr`, `sam`, ...) |
| `postprocess.box_format` | `cxcywh` / `xyxy` |
| `postprocess.conf_threshold` | confidence threshold (for GroundingDINO this is the **box** threshold) |
| `postprocess.text_threshold` | **GroundingDINO only** — threshold for assigning text tokens to a detected box (open-vocab label gating) |
| `postprocess.max_detections` | cap on returned detections |
| `labels` | optional labels file (one class per line) |
| `runtime.prefer` | EP fallback chain |
| `runtime.idle_unload_seconds` | idle auto-unload (0 = never) |

## Validation rules (the registry rejects violations)

| Field | Constraint |
|-------|------------|
| `name` | required, non-empty |
| `license` | must be ∈ {Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause}. **AGPL is strictly forbidden.** |
| `task` | ∈ {detection, segmentation, open_vocab} |
| `model_file` / `files` | at least one required — `model_file` OR a non-empty `files:` map |
| `input.width/height` | > 0 |
| `input.layout` | NCHW / NHWC (or empty) |
| `runtime.prefer` | each EP ∈ {tensorrt, cuda, cpu} |

A manifest that is invalid in **structure** is **skipped** during scan (collected into a
warning) and does not crash the server.

### Weights existing ≠ valid

The **existence of the `.onnx` file(s)** is NOT a structural validation condition. A
model with a valid manifest but no downloaded weights is still **listed**, with state:

| State | Meaning |
|-------|---------|
| `not_downloaded` (`list`: `missing`) | valid manifest, no `.onnx` file yet |
| `available` (`list`: `ready`) | weights present, ready to load |
| `loaded` | in memory |

For multi-session models, **all** files in `files:` must exist for the model to be
`available`. Missing weights only surface a clear error at **load/predict** time (like
Ollama: you see a model before you `pull` it).
