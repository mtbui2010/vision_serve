# rfdetr-small-etri-qf — RF-DETR small (ETRI 22 classes) + `query_feats`

Same trained weights as [`../rfdetr-small-etri`](../rfdetr-small-etri), re-exported with one
extra ONNX output:

| output | shape | note |
|---|---|---|
| `dets` | `[1, 300, 4]` | unchanged |
| `labels` | `[1, 300, 23]` | unchanged |
| `cross_attn_weights` | `[3, 1, 16, 300, 1024]` | unchanged (drives `/api/explain`) |
| **`query_feats`** | **`[1, 300, 256]`** | **new** — decoder features feeding the class head |

## Why

An open-vocabulary ("text-aligned") head scores a detection as
`cosine(P · f_i, text_embedding(class))`, where `f_i` is the 256-d feature of object query
`i`. The shipped export emits only boxes and class logits, so there is no `f_i` to attach a
head to. This export exposes it without touching anything else.

## Equivalence — verified, not assumed

On real ETRI images, against `../rfdetr-small-etri/model.onnx`:

- `max|Δdets| = max|Δlabels| = max|Δcross_attn_weights| = **0.0**` (exactly zero)
- `labels == query_feats @ class_embed.weight.T + class_embed.bias`, max error **0.0**
- End-to-end through the Go server: `/api/predict` JSON identical on 5 images,
  `/api/explain` returns an identical PNG

So it is a drop-in superset. Treat it as such: anything that works on `rfdetr-small-etri`
works here.

## Output order is load-bearing

[`internal/models/rfdetr/postprocess.go`](../../internal/models/rfdetr/postprocess.go) picks
the first output whose last dimension is 4 as the boxes, and the **first remaining one** as
the class logits. `query_feats` must stay **after** `labels`. Emitted before it, it would be
silently taken for the logits — no error, just wrong results. Preserve the order
`dets, labels, cross_attn_weights, query_feats` in any future re-export.

## Provenance

- Training job `3735246b-6917-49fa-95ca-ea248134a855` (`rfdetr-small-etri_simple`,
  150 epochs, finished 2026-07-28), recorded in VisionForge's job database.
- Checkpoint pulled from the VisionForge R2 bucket; the zero-delta check above confirms it
  is exactly the checkpoint behind the currently deployed ONNX.
- Export script: `vision_forge/scripts/export_onnx_rfdetr_queryfeats.py`. It captures `hs`
  (final decoder layer after norm) with a `forward_pre_hook` on `net.class_embed` and
  appends it last in `output_names`. No new weights — the file grows by ~35 KB.

## Publishing so `visionserve pull` can fetch it

The weights are **not committed** (114 MB). To make this model pullable:

1. Upload `model.onnx` + `labels.txt` to a public HuggingFace repo.
2. Add `source_url` to `manifest.yaml` (the `sha256` is already filled in and pins these
   exact bytes).
3. Add an entry to [`internal/catalog/catalog.go`](../../internal/catalog/catalog.go):

```go
{
    Name:         "rfdetr-small-etri-qf",
    Task:         "detection",
    License:      "Apache-2.0",
    Architecture: "rf-detr",
    Description:  "RF-DETR small, ETRI 22-class, exposes query_feats for open-vocab heads",
    HFRepo:       "<your-org>/rfdetr-small-etri-qf",
    Files: []File{
        {Role: "model", HFFilename: "model.onnx", LocalFilename: "model.onnx"},
        {Role: "labels", HFFilename: "labels.txt", LocalFilename: "labels.txt"},
    },
    InputWidth: 512, InputHeight: 512, InputLayout: "NCHW", Letterbox: true,
    Normalize:       &Normalize{Mean: [3]float64{0.485, 0.456, 0.406}, Std: [3]float64{0.229, 0.224, 0.225}},
    PostprocessType: "detr", BoxFormat: "cxcywh", ConfThreshold: 0.5, MaxDetections: 300,
    LabelsFile:      "labels.txt",
    RuntimePrefer:   []string{"tensorrt", "cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,
},
```

The catalog entry is deliberately **not** added yet: `visionserve list` would advertise the
model as "available to pull" and every `pull` would fail until the repo exists.
