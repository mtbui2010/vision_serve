# Manifest format

Every model is a folder: weights (`.onnx`), an optional labels file, and a `manifest.yaml` that
tells VisionServe everything else. Think of it as a *Modelfile for computer vision*. This page is
the short version; the complete field-by-field specification is
[`docs/manifest-spec.md`](https://github.com/mtbui2010/vision_serve/blob/main/docs/manifest-spec.md).

```text
models/
└── rf-detr/
    ├── manifest.yaml
    ├── rf-detr-base.onnx
    └── coco.txt
```

## A single-model manifest

```yaml
name: rf-detr                  # folder name = model name
task: detection                # detection | segmentation | open_vocab | depth | classification | embed
license: Apache-2.0            # REQUIRED; only Apache-2.0, MIT, BSD-2/3-Clause are accepted
model_file: rf-detr-base.onnx
sha256: 9f86d0...              # optional: refuse to load if the file's bytes differ

preprocess:                    # how a photo becomes the input tensor
  resize: squash
  width: 560
  height: 560
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]

postprocess:
  conf_threshold: 0.5
  max_detections: 300

labels: coco.txt

runtime:
  prefer: [cuda, cpu]          # execution providers to try, in order (CPU always added)
  idle_unload_seconds: 300     # free memory after 5 idle minutes
```

## A multi-session manifest

Models that chain several networks (an image encoder and a mask decoder, for example) list one
file per **role**. The model's code asks for sessions by role name; the lifecycle manager loads
and owns them.

```yaml
name: mobile-sam
task: segmentation
license: Apache-2.0
files:
  encoder: mobile_sam_encoder.onnx
  decoder: mobile_sam_decoder_single.onnx
runtime:
  prefer: [cuda, cpu]
  threads: {decoder: 2}        # optional: CPU threads for one role's sessions
```

## What the registry checks

When the server starts it scans the model folder and **skips** (with a warning, never a crash)
any manifest that breaks a rule:

- `license` must be on the permissive allowlist — **AGPL and other copyleft licences are refused**,
  because a single AGPL model would force its licence on the whole project and its users.
- `task` and every execution provider must be known values.
- `model_file` or a non-empty `files:` map must be present.
- `preprocess:` fields must make sense for the chosen `resize` mode, and must agree with the older
  `input.*` spelling if both are given.
- `runtime.threads` keys must be roles from `files:`, values whole numbers ≥ 0.
- SHA-256 pins are checked when the model loads (the weights may not be downloaded at scan time).

!!! code "Where in the code"
    - [`internal/registry/manifest.go`](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/manifest.go) — parsing and validation.
    - [`internal/registry/preprocess.go`](https://github.com/mtbui2010/vision_serve/blob/main/internal/registry/preprocess.go) — the `preprocess:` block and its legacy aliases.
    - [`internal/catalog/catalog.go`](https://github.com/mtbui2010/vision_serve/blob/main/internal/catalog/catalog.go) — manifests generated for models installed with `visionserve pull`.
