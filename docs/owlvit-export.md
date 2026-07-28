# OWL-ViT / OWLv2 — ONNX Export Guide

OWL-ViT (OWLv2) performs **image-conditioned one-shot object detection**: given a scene image
and one or more template images, it returns bounding boxes of scene regions that visually
resemble the templates.  No text prompt is needed — the query _is_ an image.

License: **Apache-2.0** (Google Research).  OWLv2 checkpoints on HuggingFace are Apache-2.0;
verify before downloading any variant (`google/owlv2-*` family).

---

## PyTorch → ONNX export

```python
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch

# Load the model (Apache-2.0 checkpoint).
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
model.eval()

# Thin wrapper that calls image_guided_detection and returns only the two
# output tensors VisionServe expects.
class OWLv2ImageQuery(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_pixel_values, query_image_features):
        # query_pixel_values:   [1, 3, 960, 960]  — scene
        # query_image_features: [N, 3, 960, 960]  — N template images
        outputs = self.model.image_guided_detection(
            pixel_values=query_pixel_values,
            query_pixel_values=query_image_features,
        )
        # logits:    [1, num_patches, N]
        # pred_boxes: [1, num_patches, 4]  cxcywh normalized [0,1]
        return outputs.logits, outputs.target_pred_boxes

wrapper = OWLv2ImageQuery(model)
wrapper.eval()

dummy_query     = torch.zeros(1, 3, 960, 960)
dummy_templates = torch.zeros(2, 3, 960, 960)  # 2 templates — dynamic axis

torch.onnx.export(
    wrapper,
    (dummy_query, dummy_templates),
    "owlv2_base_patch16.onnx",
    input_names=["query_pixel_values", "query_image_features"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes={
        "query_pixel_values":   {0: "batch"},
        "query_image_features": {0: "num_templates"},
        "logits":               {0: "batch", 2: "num_templates"},
        "pred_boxes":           {0: "batch"},
    },
    opset_version=17,
)
print("Exported owlv2_base_patch16.onnx")
```

---

## Verify shapes before using

Always verify the actual ONNX I/O shapes before writing or trusting postprocess logic
(CLAUDE.md rule: "Do NOT guess a model's output format").

```bash
visionserve inspect owlv2_base_patch16.onnx
```

Verified output for `owlv2-base-patch16-ensemble` at 960×960 input:

```
inputs:
  query_pixel_values   [1, 3, 960, 960]  float32
  query_image_features [N, 3, 960, 960]  float32   (N = num_templates, dynamic)
outputs:
  logits               [N, 3600, 1]      float32   (N=num_templates, 3600=60×60 patches)
  pred_boxes           [1, 3600, 4]      float32   (cxcywh, normalized)
```

Note: `logits` is `[N, P, 1]`, **not** `[1, P, N]` — the template dimension is the batch
dimension (dim 0), not the last dim.  The postprocess in `internal/models/owlvit/owlvit.go`
accesses `logits[n, p, 0]` as `logits.Data[n*P + p]`.

### Patch count formula

```
num_patches = (input_H / patch_size) * (input_W / patch_size)
```

| Variant                | input H×W   | patch_size | num_patches |
|------------------------|-------------|------------|-------------|
| owlv2-base-patch16     | 960×960     | 16         | 3 600       |
| owlv2-base-patch32     | 960×960     | 32         | 900         |
| owlv2-large-patch14    | 1008×1008   | 14         | 5 184       |

---

## Manifest example

Place this file as `models/owlvit-base/manifest.yaml` and put the exported ONNX file in
the same directory.

```yaml
name: owlvit-base
task: instance_detection
license: Apache-2.0
architecture: owlvit

# Multi-session PipelineModel: a single "model" role.
files:
  model: owlv2_base_patch16.onnx

input:
  width: 960
  height: 960
  layout: NCHW
  letterbox: false
  normalize:
    mean: [0.48145466, 0.4578275, 0.40821073]   # CLIP normalization
    std:  [0.26862954, 0.26130258, 0.27577711]

postprocess:
  conf_threshold: 0.1    # minimum sigmoid(logit) to emit a detection
  max_detections: 10

instance:
  patch_size: 16         # must match the exported variant (base-patch16 → 16)
  sim_threshold: 0.1     # same semantics as conf_threshold; conf_threshold wins if set
  max_templates: 5       # cap how many template images are used per call (0 = no limit)

runtime:
  prefer: [cuda, cpu]
  idle_unload_seconds: 300
```

---

## Inference flow (VisionServe)

1. Register template images via `POST /api/templates` (template store, `internal/templates/`).
2. Call `POST /api/predict` with `{"model": "owlvit-base", "template": "<name>", ...}`.
3. The lifecycle manager resolves the template name → `[]image.Image` and puts them in
   `models.Prompt.TemplateImages` before calling `owlVIT.Infer`.
4. The model preprocesses the scene + templates with CLIP normalization, stacks templates
   into `[N, 3, H, W]`, runs the ONNX session, and maps `[cx, cy, w, h]` boxes back to
   original-image pixel coordinates.

---

## Normalization

OWL-ViT uses **CLIP normalization**, not ImageNet:

| Channel | mean         | std          |
|---------|--------------|--------------|
| R       | 0.48145466   | 0.26862954   |
| G       | 0.4578275    | 0.26130258   |
| B       | 0.40821073   | 0.27577711   |

These are the defaults in `owlvit.go`.  Override them per-manifest via `input.normalize`.
