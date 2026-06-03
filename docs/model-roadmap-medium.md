# Model Roadmap — Medium Scope Feasibility Analysis

**Date:** 2026-06-03  
**Scope:** Three model groups requiring API schema extensions to `pkg/api/types.go`.  
**Base:** VisionServe Apache-2.0 Go binary, ONNX Runtime, no Python at runtime.

---

## Implementation status

`internal/models/depth/` (Depth Anything V2 + MiDaS) and
`internal/models/classification/` (EfficientNet-B0 + MobileNetV3) are now **fully
implemented** — the analysis below was the design doc used to guide those implementations.
`internal/models/pose/` (RTMPose) does **not** yet exist; that section remains a
forward-looking design.

Schema additions (`TaskDepth`, `TaskClassification`, `DepthMap`/`Classifications` fields
on `api.Result`) are live in `pkg/api/types.go`.

---

## Group 1: Depth Estimation — Depth Anything V2 + MiDaS

### A. API Impact

Two new items must be added to `pkg/api/types.go`:

```go
TaskDepth Task = "depth"

// DepthMap is a single-channel float32 depth map, row-major, H×W.
// Values are relative disparity (not metric depth) unless the model is metric-depth.
type DepthMap struct {
    Width  int       `json:"width"`
    Height int       `json:"height"`
    Data   []float32 `json:"data"`          // len == Width*Height, row-major
}
```

And `Result` gains one field:

```go
DepthMap *DepthMap `json:"depth_map,omitempty"`
```

**Backward compatibility:** fully additive. Existing `Detections`/`Masks` consumers receive
`depth_map: null` (omitempty), which is a no-op in any JSON parser.

The HTTP server (`internal/server/handlers.go`, `response.go`) needs **no changes** — it
calls `writeJSON(w, http.StatusOK, res)` generically; the new `DepthMap` field is included
automatically by `encoding/json`.

The **manifest validator** in `internal/registry/manifest.go` must add `"depth"` to
`validTasks`. Likewise, `internal/models/model.go` must export `TaskDepth = api.TaskDepth`.

**Wire size concern:** A 518×518 float32 depth map is 518×518×4 ≈ 1.04 MB of raw JSON
(float32 formatted as decimal text is even larger). The analysis below addresses this.

**Base64-encoded 16-bit PNG alternative:** `[]float32` flat JSON is simple to produce in Go
but bulky (each float32 is ~12 bytes as text ≈ 3 MB for a 518×518 image). A better wire
format is a base64-encoded 16-bit grayscale PNG (≈50–150 KB after PNG compression). This
requires:
- Go: `image/png` + a tiny `image.Gray16` helper to quantize float32 → uint16 (normalize
  to `[0, 65535]`).
- Python client: `base64.b64decode` + `PIL.Image.open(io.BytesIO(...)).point(lambda x:
  x/65535.0)` → numpy float32.

Recommendation: add **both** fields — `data []float32` for correctness + an optional
`png_b64 string` for efficient transport — and let the client pick. Or make the endpoint
accept a `?format=png` query param. The flat `[]float32` approach is simpler for Phase 1;
the PNG encoding can be added as a follow-up without breaking the schema.

**Python client changes required:**
- Add `DepthMap` dataclass to `types.py` with a `to_ndarray()` helper that reshapes
  `data` into a `(height, width)` numpy array.
- Add `depth_map: Optional[DepthMap]` to `Result`.
- The `from_json` factory must populate it.

### B. Implementation Complexity

**Depth Anything V2 (single-session plain Model):**

- Input: `[1, 3, H, W]` float32, NCHW. Standard ImageNet normalization
  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Typical sizes: 518×518 or 384×384.
  Resize to fixed input; no letterbox needed (depth maps are relative, no bounding boxes to
  inverse-transform).
- Output shape ambiguity is the key tricky part. Verified shapes from the official
  depth-anything-v2 ONNX exports:
  - `depth-anything-v2-small.onnx`: output `predicted_depth` shape `[1, H, W]` float32.
  - Some community exports wrap it as `[1, 1, H, W]`.
  - The postprocess should squeeze all leading size-1 dims and assert the result is 2D.
- Postprocess: squeeze → normalize to `[0, 1]` (divide by `max`, or keep raw for
  applications that want absolute disparity). Populate `Result.DepthMap`.
- No coordinate inverse-transform needed (depth maps are pixel-registered to the resized
  input, not to detected objects). However, the client may want to resize back to original
  dimensions — this is a client-side concern.
- Implements: plain **`Model`** interface. Single ONNX session via the existing
  `lifecycle.Manager` path.
- Tests: preprocess shape, postprocess squeeze + normalization edge cases.

**MiDaS (single-session plain Model):**

- Input: `[1, 3, 384, 384]` or `[1, 3, 256, 256]` depending on variant, NCHW. Same
  ImageNet normalization.
- Output: `[1, 384, 384]` float32 (relative inverse depth, not disparity-corrected).
- Implementation is nearly identical to Depth Anything V2 — the same model package can
  handle both by parameterizing the postprocess type in the manifest
  (`postprocess.type: depth-anything` or `midas`).
- Implements: plain **`Model`** interface.

**Tricky parts:**
1. Output shape variation across export tools — must squeeze robustly, not assume rank.
2. The `DepthMap` field in `Result` sits alongside `Detections`/`Masks`; the manifest
   validator must block depth models from accidentally returning detections (or accept that
   `Detections` is just `nil` — the latter is simpler, keep it).
3. `Config` has no depth-specific fields yet; postprocess only needs `Width`, `Height`, and
   a normalization flag — all already present.

### C. ONNX Availability (Apache-2.0/MIT/BSD only)

| Model | License | Best HF Source | Verified |
|---|---|---|---|
| Depth Anything V2 Small | **Apache-2.0** | `depth-anything/Depth-Anything-V2-Small` (official org; ONNX exports present) | Yes |
| Depth Anything V2 Base | **Apache-2.0** | same repo | Yes |
| MiDaS v2.1 small | **MIT** | `Intel/dpt-hybrid-midas` (ONNX via transformers export) or `isl-org/MiDaS` (GitHub, MIT) | Yes |

Depth Anything V2 ships official ONNX exports directly from the `depth-anything` HF
organization (`onnx/depth_anything_v2_vits.onnx`, ~100 MB for Small).
MiDaS small ONNX can be exported from the official PyTorch checkpoint using
`torch.onnx.export` with a one-time Python script (no Python at runtime; just an export
step). ONNX file size: MiDaS small ≈ 80 MB; Depth Anything V2 Small ≈ 100 MB.

**License verdict:** both are clean for VisionServe.

### D. Integration with Existing Patterns

Fits cleanly into the plain **`Model`** interface — same path as `rf-detr` / RT-DETR. No
lifecycle changes. The only structural changes are:
1. `pkg/api/types.go` — add `TaskDepth`, `DepthMap` struct, `Result.DepthMap` field.
2. `internal/registry/manifest.go` — add `"depth"` to `validTasks`.
3. `internal/models/model.go` — re-export `TaskDepth`.
4. `internal/catalog/catalog.go` — add two entries.
5. New package `internal/models/depth/` implementing `Model`.
6. Python client `types.py` — add `DepthMap` + update `Result`.

No server handler changes. No lifecycle changes.

### E. Verdict

**Status: IMPLEMENTED**

`internal/models/depth/` is complete. Depth Anything V2 (architecture `depth-anything-v2`,
518×518) and MiDaS (architecture `midas`, 256×256) are both in the catalog. The flat
`[]float32` wire format is shipped; PNG base64 encoding remains a future option.

---

## Group 2: Image Classification — EfficientNet-B0 + MobileNetV3

### A. API Impact

Two new items must be added to `pkg/api/types.go`:

```go
TaskClassification Task = "classification"

// Classification is a single ranked prediction (top-K).
type Classification struct {
    Class string  `json:"class"`
    Conf  float64 `json:"conf"`
}
```

And `Result` gains one field:

```go
Classifications []Classification `json:"classifications,omitempty"`
```

**Backward compatibility:** fully additive (same reasoning as depth). Existing consumers
see `classifications: null`.

The HTTP server needs **no changes**. Manifest validator and `models/model.go` must add
`"classification"` to `validTasks` and re-export `TaskClassification`.

**Labels file:** ImageNet-1k has 1000 classes. A `imagenet1k.txt` (one synset label per
line, ~8 KB) must be embedded in `internal/catalog/labels/` with `//go:embed` — exactly
the same pattern as `coco91.txt`. This file must be sourced from a permissively licensed
location (the standard `imagenet_classes.txt` used by PyTorch/torchvision is itself a
list of WordNet synset names and is not separately copyrighted; numerous Apache-2.0 repos
bundle it, e.g. `onnxruntime-examples`).

`MaxDetections` in the manifest controls K (top-K results). When `max_detections == 0`,
default to top-5.

**Python client changes required:**
- Add `Classification` dataclass to `types.py`.
- Add `classifications: List[Classification]` to `Result`.
- Update `from_json`.

### B. Implementation Complexity

**EfficientNet-B0 (single-session plain Model):**

- Input: `[1, 3, 224, 224]` float32 NCHW. ImageNet normalization
  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
- Output: `[1, 1000]` float32.
- **Softmax vs logits:** The standard HF `EfficientNet-B0` ONNX export already applies
  softmax (the `logits` output name is misleading in some exports — the actual values sum
  to 1.0). This must be confirmed against the real tensor before writing postprocess.
  If the export outputs raw logits, a softmax step is needed:
  `exp(x_i) / sum(exp(x))`. Pure Go implementation is trivial (no cgo). Guard against
  NaN by clamping or using the log-sum-exp trick.
- Postprocess: sort `[1000]` scores descending, take top-K, map index → label string.
- Implements: plain **`Model`** interface.
- Tests: preprocess shape, softmax correctness (sum-to-1), top-K extraction, label lookup.

**MobileNetV3 (single-session plain Model):**

- Input: `[1, 3, 224, 224]` float32 NCHW (MobileNetV3-Small: 224×224, sometimes 320×320
  depending on export). Same ImageNet normalization.
- Output: `[1, 1000]` float32.
- Implementation is identical to EfficientNet-B0 — the same factory handles both models.
  The manifest's `postprocess.type: classification` routes to the shared factory.
- Implements: plain **`Model`** interface.

**Tricky parts:**
1. Softmax vs logits ambiguity — must be resolved by inspecting the real ONNX export.
   The postprocess should check: if `sum(scores) ≈ 1.0` treat as probabilities; otherwise
   apply softmax. This heuristic is not perfect; better to note in the manifest
   (`postprocess.type: classification-softmax` vs `classification-logits`) and handle
   both explicitly.
2. `Config` currently holds no `TopK` field. `MaxDet` can be repurposed: `max_detections`
   in the manifest already exists and `Config.MaxDet` is already parsed. For classification,
   `MaxDet == 0` → default to 5; `MaxDet == 1` → top-1. No new config field needed.
3. The `PreprocessMeta` coordinate machinery (scale/pad) is irrelevant for classification
   (no spatial outputs), but the interface mandates returning it. Return a zero-value
   `PreprocessMeta{}` — the postprocess ignores it.

### C. ONNX Availability (Apache-2.0/MIT/BSD only)

| Model | License | Best HF Source | File Size |
|---|---|---|---|
| EfficientNet-B0 | **Apache-2.0** | `google/efficientnet-b0` (HF; ONNX via `optimum`) | ~21 MB |
| MobileNetV3-Small | **Apache-2.0** | `google/mobilenet_v3_small_100_224` (HF) OR `onnx/mobilenet_v3_small` | ~10 MB |

The `google/` HF organization ships EfficientNet-B0 under Apache-2.0. The ONNX export is
available via HF `optimum` exporter (`optimum-cli export onnx --model google/efficientnet-b0
--task image-classification ./eff-b0-onnx`) — a one-time offline Python step to produce
the ONNX artifact, not a runtime dependency.

MobileNetV3 ONNX is available on several HF repos; the `onnx/mobilenet_v3_small` repo
provides a pre-exported file. License: Apache-2.0.

**One concern:** HF `optimum` exports of EfficientNet-B0 sometimes include a preprocessing
op inside the ONNX graph (resize + normalize), which would make the Go preprocess step a
no-op. Inspect the ONNX graph first with `netron` before coding preprocess.

**License verdict:** both models are clean.

### D. Integration with Existing Patterns

Fits cleanly into the plain **`Model`** interface. Structural changes:
1. `pkg/api/types.go` — add `TaskClassification`, `Classification` struct, `Result.Classifications`.
2. `internal/registry/manifest.go` — add `"classification"` to `validTasks`.
3. `internal/models/model.go` — re-export `TaskClassification`.
4. `internal/catalog/labels/imagenet1k.txt` — embed (Apache-2.0 / public domain).
5. `internal/catalog/catalog.go` — add two entries.
6. New package `internal/models/classification/` implementing `Model`.
7. Python client `types.py` — add `Classification` + update `Result`.

No server handler changes. No lifecycle changes.

### E. Verdict

**Status: IMPLEMENTED**

`internal/models/classification/` is complete. EfficientNet-B0 (architecture `efficientnet`,
224×224) and MobileNetV3-Small (architecture `mobilenet-v3`, 224×224) share the same
factory. `MaxDet` controls top-K (default 5).

---

## Group 3: Pose Estimation — RTMPose

> **Status: Planned — NOT YET IMPLEMENTED.**
> RTMPose is a medium-priority future feature. No Go code exists for it.
> Before implementation begins: (1) verify the ONNX output format (SimCC vs heatmap) against
> the real `onnx-community/RTMPose-s` file; (2) land the API schema additions
> (`TaskPose`, `Keypoint`, `PersonPose`, `Result.Poses []PersonPose`) as a standalone
> prerequisite commit.

### A. API Impact

Two new items must be added to `pkg/api/types.go`:

```go
TaskPose Task = "pose"

// Keypoint is a single body keypoint.
type Keypoint struct {
    X    float64 `json:"x"`    // original image x coordinate
    Y    float64 `json:"y"`    // original image y coordinate
    Conf float64 `json:"conf"` // keypoint confidence (visibility score)
}

// PersonPose is the keypoints for one detected person.
type PersonPose struct {
    BBox      [4]float64 `json:"bbox"`      // person bounding box [x,y,w,h] in original coords
    Conf      float64    `json:"conf"`      // person detection confidence
    Keypoints []Keypoint `json:"keypoints"` // 17 COCO keypoints (or model-specific count)
}
```

And `Result` gains:

```go
Poses []PersonPose `json:"poses,omitempty"`
```

**Backward compatibility:** fully additive.

The manifest validator and `models/model.go` must add `"pose"` to `validTasks`.

**Python client changes required:**
- Add `Keypoint` and `PersonPose` dataclasses to `types.py`.
- Add `poses: List[PersonPose]` to `Result`.

### B. Implementation Complexity

RTMPose is a **top-down pose estimator**. The ONNX model takes a single person crop
(256×192) and returns 17 keypoints. This creates a fundamental architectural question.

**Top-down pipeline stages:**

1. **Stage 1 — Person detection:** Run a person detector (RF-DETR or RT-DETR) to get
   person bounding boxes.
2. **Stage 2 — Per-person crop:** For each person box, expand the box (standard top-down
   practice: expand by ~25% to include context), then crop + resize to 256×192.
3. **Stage 3 — Keypoint regression:** Run RTMPose on each 256×192 crop; the output is
   `[1, 17, 2]` (x,y coordinates in crop space, normalized [0,1]) + `[1, 17]` visibility
   scores, OR a heatmap `[1, 17, 64, 48]` depending on the export.
4. **Stage 4 — Inverse transform:** Map each keypoint from crop space → person box space
   → original image space.

**Three design options for chaining the detector + RTMPose:**

**Option A — PipelineModel bundling detector + pose (recommended).**
The RTMPose `PipelineModel` declares roles `["detector", "pose"]`. The detector role
uses an RF-DETR or RT-DETR ONNX session. On `Infer()`:
1. Run `Runner.Run("detector", ...)` to get person boxes.
2. For each box: crop + resize image, run `Runner.Run("pose", ...)`, decode keypoints,
   inverse-transform to original coordinates.
3. Return `Result{Poses: [...]PersonPose}`.

This matches the Grounded-SAM pattern exactly: GroundingDINO→MobileSAM becomes
detector→RTMPose. The lifecycle owns and keeps both ONNX sessions alive. The manifest
`files:` map declares `detector: rf-detr-base.onnx` and `pose: rtmpose-s.onnx`.

The wrinkle: the manifest currently has no concept of "reuse an existing loaded model's
ONNX session." The RTMPose pipeline would own its own copy of the detector ONNX session,
even if rf-detr is also separately loaded. This is a VRAM cost but not an architectural
blocker — it is the same tradeoff that Grounded-SAM already makes (grounded-sam loads its
own gdino session, separate from a standalone `grounding-dino` load). Acceptable for now;
session reuse can be added as a future optimization.

**Option B — User passes bboxes as a prompt (simpler, less magic).**
RTMPose implements `PipelineModel` with only a `["pose"]` role. The user (or client code)
first calls `/api/predict` with `rf-detr` to get person boxes, then calls `/api/predict`
with `rtmpose` and passes those boxes in the `box` prompt field. RTMPose's `Infer()` crops
per box, runs keypoints, returns `Result{Poses: [...]}`.

This avoids the two-session lifecycle complexity and reuses whatever detector the user
already has loaded. Downside: two HTTP round-trips, and the user must filter for person
class from the detector result. It is a valid design but shifts complexity to the client.

**Option C — New "pipeline" server concept.**
A `/api/pipeline` endpoint that chains two `/api/predict` calls server-side. This is
premature architecture for one use case and is ruled out per CLAUDE.md's scope guidance.

**Recommendation: Option A for a bundled "rtmpose" model; Option B as a simpler alternative
called "rtmpose-prompted".**

**RTMPose ONNX output format (key ambiguity):**

Exported RTMPose models exist in two flavors:
- **SimCC (coordinate classification):** output `simcc_x [1, 17, W_bins]` + `simcc_y [1, 17, H_bins]`. Decode via argmax on each → index → coordinate. No heatmap, fast.
- **Heatmap:** output `[1, 17, 64, 48]` float32 heatmaps. Decode via argmax on 2D + Gaussian offset. More complex.

The most common ONNX export from `open-mmlab/mmpose` uses SimCC. This must be verified
against the real ONNX tensor shapes **before writing postprocess** (CLAUDE.md: "Do NOT guess
a model's output format").

**Preprocessing per crop:**
1. Expand person bbox by a fixed factor (e.g. 1.25× height, 1.0× width — standard COCO
   top-down convention).
2. Crop from original image.
3. Resize crop to 256×192 (bilinear, pure Go via `disintegration/imaging`).
4. Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
5. NCHW float32 tensor.

**Inverse transform:**
Given a keypoint `(kp_x, kp_y)` in crop space [0, 256] × [0, 192]:
```
scale_x = box_w_expanded / 256
scale_y = box_h_expanded / 192
orig_x = box_x_expanded + kp_x * scale_x
orig_y = box_y_expanded + kp_y * scale_y
```

The transform is simpler than letterbox because the crop is direct (no padding), but the
box expansion step must be applied consistently between preprocessing and the inverse
transform.

**Tricky parts:**
1. SimCC vs heatmap decode — must be confirmed against real ONNX tensor shapes.
2. Per-person loop inside `Infer()`: if there are N persons, the pose session is called N
   times sequentially. The `Runner.Run()` interface supports this (it is a simple session
   call, not batched). For large crowds this may be slow; batching would need engine changes.
3. The detector built into the pipeline uses person class filtering — RF-DETR outputs all
   COCO classes; the `PipelineModel.Infer()` must filter for class `"person"` (COCO index 0).
   This requires the detector labels to be passed into the pose model's `Config`.
4. The `Config.Files` map currently uses string paths. The detector role path needs to point
   to a separate ONNX file from the pose model file. The manifest must declare both.

### C. ONNX Availability (Apache-2.0/MIT/BSD only)

| Model | License | Best HF Source | Notes |
|---|---|---|---|
| RTMPose-S (COCO 17kp) | **Apache-2.0** | `onnx-community/RTMPose-s` or exported from `open-mmlab/mmpose` | Verify: mmpose repo is Apache-2.0 |
| RT-DETR (for bundled detector) | **Apache-2.0** | `PekingU/rtdetr_r50vd` → ONNX via `optimum` | Confirmed Apache-2.0 |

**Critical license check:** The `open-mmlab/mmpose` GitHub repo is Apache-2.0. RTMPose is
a component of mmpose. The HF `onnx-community/RTMPose-s` mirrors the mmpose export and
carries the Apache-2.0 license on its model card. However, **this must be independently
verified** against the HF repo card before adding it to the catalog — do not take this
report's word for it. Specifically check that the HF repo card explicitly states Apache-2.0
(not "license: other" or absent).

ONNX file sizes: RTMPose-S ≈ 25–40 MB; RT-DETR-R50 ≈ 80 MB (or reuse rf-detr if it ships
as a separate ONNX file).

### D. Integration with Existing Patterns

**Option A (bundled)** fits the `PipelineModel` interface well. The Grounded-SAM code
(`internal/models/groundedsam/groundedsam.go`) is the direct template:
- Replace `gdino → detector` role.
- Replace `encoder+decoder → pose` role.
- Replace the SAM segment call with a per-box crop-and-pose loop.

However there are **two new lifecycle concerns** not present in Grounded-SAM:

1. **Per-inference loop calling `Runner.Run` N times:** This is already possible — `Runner`
   has no loop limit. But the detector session lock (if any) will be held across all N
   pose calls in `Infer()`. Currently `engine.Session.Run` takes a mutex per call
   (`session.go` line 89: `s.RunNamed(inputs)`). This is fine for single-threaded
   requests; it becomes a bottleneck only under concurrent requests. No change needed
   in Phase 1.

2. **Detector role in the pose model vs standalone rf-detr:** Two separate `engine.Session`
   instances will exist if the user loads both `rf-detr` and `rtmpose`. This is a known
   tradeoff (same as grounded-sam + grounding-dino) and is acceptable.

**Schema additions only in lifecycle (no new interface needed):** The `Session.Predict` path
and the `runner` type already support multi-role pipelines. No lifecycle code changes are
needed.

Structural changes:
1. `pkg/api/types.go` — add `TaskPose`, `Keypoint`, `PersonPose`, `Result.Poses`.
2. `internal/registry/manifest.go` — add `"pose"` to `validTasks`.
3. `internal/models/model.go` — re-export `TaskPose`.
4. `internal/catalog/catalog.go` — add RTMPose entry (bundled detector + pose files).
5. New package `internal/models/rtmpose/` implementing `PipelineModel`.
6. Python client `types.py` — add `Keypoint`, `PersonPose`, update `Result`.

### E. Verdict

**Feasibility: Feasible with caveats**

**Effort: L (7–10 days)**
- Day 1–2: Schema additions (types, validator, model aliases, Python client).
- Day 3: Investigate real RTMPose ONNX tensor shapes (SimCC vs heatmap), write stub with
  documented `TODO`s. Do NOT guess the decode format.
- Day 4–5: Preprocess (crop + expand bbox + resize to 256×192), postprocess (SimCC argmax
  or heatmap decode + inverse transform), unit tests.
- Day 6: PipelineModel wiring (detector role → person filter → per-box crop loop).
- Day 7–8: Integration test with real ONNX (or a dummy that mimics SimCC output shape).
- Day 9–10: Edge cases (no persons detected, overlapping boxes, image boundary crops).

**Blockers:**
1. **RTMPose ONNX output format must be verified before writing postprocess.** Use
   `netron` or the engine's `OutputNames()`/shape introspection on the real file. If the
   export is SimCC, the decode is clean. If it is heatmap-based, add ~50 lines of Gaussian
   refinement code.
2. **License must be independently verified** on the specific HF repo card used for the
   catalog entry. The mmpose GitHub is Apache-2.0, but a mirror HF repo might not have
   set the license field correctly.
3. **Person class filtering in the bundled detector** requires hardcoding (or configuring
   via manifest) the COCO person class label — currently `"person"` from `coco91.txt`. If
   the detector labels file is not available in the pose model's `Config.Labels`, filtering
   is impossible. Solution: pass the labels file path in the manifest, or hardcode
   `class == "person"` (fragile) or use class index 0 (COCO) in the postprocess.

**Recommendation:**
- Implement Option A (bundled PipelineModel) as the primary `rtmpose` model.
- Write a clear `TODO: verify SimCC vs heatmap output before this postprocess` comment.
- Do NOT implement RTMPose until Depth and Classification are shipped (lower-risk items
  first; RTMPose has genuine open questions).

---

## Cross-Cutting Concerns

### 1. Schema extension pattern (applies to all three groups)

Every new task follows the same three-file change:
- `pkg/api/types.go` — new `Task` const + new output struct + new `Result` field.
- `internal/registry/manifest.go` — add to `validTasks`.
- `internal/models/model.go` — add alias constant.

These three edits are small, contained, and additive. They can be batched into a single
commit ("Add TaskDepth, TaskClassification, TaskPose to schema").

### 2. Manifest `postprocess.type` field

Currently used by existing models (`detr`, `sam`, `grounding-dino`). New values to add:
`depth-anything`, `midas`, `classification`, `rtmpose`. These are purely documentation
strings — the factory selection is by `architecture:`, not `postprocess.type`. The
`postprocess.type` field is currently unused in Go code for dispatch (it is stored in
`Config.PostType` but no switch statement routes on it). This is fine; the architecture
name (`rf-detr`, `classification`, etc.) already routes to the correct factory.

### 3. The `PreprocessMeta` return for non-spatial models

Both `Depth` and `Classification` implement the plain `Model` interface, which requires
`Preprocess` to return a `PreprocessMeta`. For depth: the meta is meaningful only if the
client wants to resize the output map back to original dimensions (ScaleX/ScaleY are
useful). For classification: the meta is meaningless but the interface demands it — return
`PreprocessMeta{OrigWidth: img.Bounds().Dx(), OrigHeight: img.Bounds().Dy()}` and ignore
it in postprocess.

### 4. Imagenet1k labels file

Must be embedded in `internal/catalog/labels/imagenet1k.txt` with `//go:embed`. The
standard 1000-line ImageNet class list (one synset label per line, e.g. "tench",
"goldfish", ...) is widely redistributed under Apache-2.0 by the PyTorch/torchvision
project. Embed it the same way `coco91` is embedded in `catalog.go`.

### 5. Python client Result.from_json symmetry

When the server returns a `Result` with `depth_map`, `classifications`, or `poses`, the
Python client's `Result.from_json` must ignore unknown fields gracefully (Python dataclasses
do not have `ignore_unknown`; the `from_json` pattern already picks fields explicitly via
`d.get(...)`, so new optional fields default to `None`/`[]` when absent). **No breakage to
existing clients** when new fields are added to the server.

### 6. Wire size for depth maps

For production use, the flat `[]float32` JSON encoding will be impractical (≈3 MB for
518×518). Consider adding a `depth_png_b64 string` field alongside `data []float32` in the
`DepthMap` struct. The Go encoder writes a 16-bit grayscale PNG (`image.Gray16` from
stdlib) and base64-encodes it. This reduces the wire size to ≈50–150 KB. Implement in Phase
1 as an option; make it the default in Phase 2.

---

## Implementation Order Summary

| Priority | Group | Status | Notes |
|---|---|---|---|
| 1 | Classification (EfficientNet-B0 + MobileNetV3) | **Done** | Shipped with `TaskClassification` + `Classifications` field. |
| 2 | Depth Estimation (Depth Anything V2 + MiDaS) | **Done** | Shipped with `TaskDepth` + `DepthMap`/`DepthWidth`/`DepthHeight` fields. |
| 3 | RTMPose | Planned | Highest complexity (two-stage pipeline, crop loop, SimCC/heatmap ambiguity). Requires real ONNX tensor shape investigation before implementation begins. |

---

*Analysis based on codebase read at `/home/trung/trung_workdir/vision_serve` on 2026-06-03.
No implementation files were created by this analysis — see above for which files need to
be created/modified per group.*
