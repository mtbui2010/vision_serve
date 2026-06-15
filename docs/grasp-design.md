# Planar Grasp Detection — Design Note

> Design-only document. No code is changed by this note. It records the synthesis of four
> evaluated grasp-detection methods and the recommended integration path for VisionServe.

## 1. Scope & gating constraints

Three hard gates from `CLAUDE.md` govern everything below:

1. **License allowlist — Apache-2.0 / MIT / BSD only.** AGPL (Ultralytics YOLO, FastSAM,
   YOLO-World) and any non-commercial / "research-only" license are **REJECTED**. "It's on
   HuggingFace" says nothing about the license — each model's actual license must be checked.
2. **Inference is ONNX-Runtime only.** No custom inference engine, no custom ops, **no Python
   in the runtime** (lean Go binary for edge devices).
3. **One unified `Result` schema across all tasks** (`pkg/api/types.go`). Do **not** invent a
   per-model output schema.

**In scope: planar (2D, image-plane) grasp detection.** The output is a parallel-jaw grasp
rectangle in image-plane coordinates: center `(x, y)`, in-plane angle `θ`, jaw width, and a
quality score.

**Out of scope: 6-DoF / point-cloud grasping** (GraspNet, Contact-GraspNet). These are
explicitly excluded because they:

- rely on ONNX-hostile custom ops (point-cloud grouping/sampling kernels);
- take a **point cloud** as input rather than an image;
- emit an **SE(3)** pose that does not fit the unified `Detection` + `Mask` schema;
- carry **NVIDIA non-commercial license risk** (Contact-GraspNet weights).

## 2. Methods evaluated

| Method | License (SPDX + repo) | ONNX export | Input | Verdict |
| --- | --- | --- | --- | --- |
| **GG-CNN / GG-CNN2** | BSD-3-Clause (`dougsm/ggcnn`) — **PASS** | **CLEAN** — fully-conv, ~62k params, standard ops | depth `1×300×300` | **Do first** (lowest friction learned model) |
| **GR-ConvNet** | BSD-3-Clause (`skumra/robotic-grasping`) — **PASS** (confirmed; even inherits Morrison's 2018 copyright) | **CLEAN** — conv / residual / transposed-conv, standard ops | RGB-D `4×224×224` | **OK**, but needs an RGB-D (RGB+depth, 2-input) path |
| **GQ-CNN / FC-GQ-CNN (Dex-Net)** | UC Berkeley "educational, research, and not-for-profit purposes" (IPIRA) — **REJECT** | n/a | depth patch | **REJECT** (non-commercial bar, same as AGPL; also needs external sampler + CEM loop) |
| **User's `mask2grasps`** | analytic, no weights — n/a | **no ONNX needed** | a segmentation mask | **Cleanest fit** — pure-Go postprocess |

Notes:

- **GQ-CNN / FC-GQ-CNN** fails on **two** counts. (a) The Berkeley IPIRA license restricts use
  to "educational, research, and not-for-profit purposes" — this is **not** Apache/MIT/BSD and
  imposes the same commercial-use bar that makes AGPL unacceptable here. (b) Architecturally,
  GQ-CNN is a grasp-quality *classifier*: it requires an **external antipodal grasp sampler**
  plus a **Cross-Entropy-Method (CEM) optimization loop** running outside the network, which
  breaks the clean "single ONNX session in → grasps out" contract. **REJECT.**
- **GG-CNN** and **GR-ConvNet** share the same decoder convention: the network emits four dense
  maps — quality `Q`, `cos2θ`, `sin2θ`, and `width` — from which grasps are decoded.

## 3. Key insight — the user's method is analytic, not a learned model

The user's `mask2grasps` (`pyinterfaces/pyinterfaces/instances.py:22-122`, with `GraspGroup`
in `grasppose.py`) is a **classical antipodal force-closure search over a segmentation mask's
boundary normals** — **not** a neural network. It has **no weights and no ONNX**.

Algorithm:

1. **Mask → 2D normal map** via a Sobel-like convolution
   (`utils.py:578-593`, `mask2normalmap`).
2. **Decimate boundary points** onto a grid (angular `deg = 5°`, spatial `stride = 5px`)
   (`utils.py:595-614`, sampling).
3. **Enumerate finger combinations** over the decimated boundary points
   (`torch.combinations`).
4. For each candidate, compute the **pairwise distance `Ds`** (gripper width) and the **force
   directions `Fs`** pointing toward the finger centroid.
5. **Force-closure test:** every contact must satisfy the **friction-cone** condition
   `F·n > cos(atan(friction_coef))` **and** the width bound `dmin < Ds < dmax`.
6. **Weighted score** combining force-closure quality, contact term, object score, normalized
   min distance, and center distance — weights `[0.4, 0.3, 0.15, 0.05, 0.05, 0]`.
7. **Sort descending** by score.

Output: `[K, nfingers*2 + 1]` — per grasp the 2D pixel contact points plus the score; for the
2-finger case this is `[x0, y0, x1, y1, score]`. Angle and width are derived later in
`GraspGroup`: `θ = arctan2` of the finger vector (`twopoints2theta`), `width = ` pixel distance
between contacts.

The search is purely **2D / pixel-space**. The optional **3D lift** (`centers_3d` /
`finger_3d`) needs depth + camera intrinsics and is not part of the core path. The `torch`
dependency is **gratuitous** — every operation is plain array math, so it **ports 1:1 to Go**.

**Implication:** `mask2grasps` is **not** a `Model` or `PipelineModel` that owns an ONNX
session. It is a **pure-Go postprocess on a segmentation `Result`**.

**Performance — the dominant stage is the boundary-normal pass.** In the Go port
(`collectBoundary` in `internal/grasp/grasp.go`) the boundary-normal map dominates cost; the
combinatorial antipodal search is **not** the bottleneck (the decimated boundary-point count is
bounded by the `(deg, stride)` polar grid). The boundary pass is computed with an **integral
image (summed-area table)** for the separable Sobel-like kernel, so each per-pixel ±1 kernel sum
reduces to a few O(1) rectangle counts, plus a **uniform-window fast-skip** (a `(2r+1)²` window
that is all-set or all-background has a zero normal and is skipped — this is the solid interior
and surrounding background, leaving only the thin boundary band to run the full sums). The result
is **BIT-EXACT** — identical grasps to the direct convolution — but ~**2–6× faster** on this
stage: cost drops from `O(area·kernel)` to ~`O(area)` with a small constant, scaling with object
area far better.

## 4. Schema extension (shared, do once)

Per `CLAUDE.md` ("unified schema, no per-model schema"), extend `pkg/api/types.go` exactly the
way `DepthMap` / `Embeddings` were added — a new struct, a new `omitempty` field on `Result`,
and a new `Task` constant.

```go
type Grasp struct {
    X, Y    float64 `json:"x"`
    Theta   float64 `json:"theta"`   // in-plane gripper-closing angle (rad)
    Width   float64 `json:"width"`   // jaw opening, original-image px
    Quality float64 `json:"quality"` // [0,1]
}
// Result += Grasps []Grasp `json:"grasps,omitempty"`
// new const TaskGrasp Task = "grasp"
```

Why this form: **parallel-jaw `(center, θ, width, quality)`** is the normalized representation
that **both** families share — learned models (GG-CNN / GR-ConvNet) emit it directly, and the
analytic method's 2-finger contacts reduce to it (`θ` and `width` from the contact pair). Two
deliberate non-goals kept as future extensions:

- **n-finger / general grippers** — add `Contacts [][2]float64` later (do not add now).
- **3D pose** — `Z` and a metric 3D width are an **opt-in future field**, needing depth +
  camera intrinsics. Keep the core field set 2D so the analytic path needs neither.

## 5. Integration path A — user's `mask2grasps` (priority)

A new **pure-Go `internal/grasp` package** holding the ported antipodal search:

```go
// internal/grasp/grasp.go
func FromMask(mask Bitmap, p Params) []api.Grasp
```

`FromMask` ports the full pipeline of §3:

- **Normals:** Sobel → 2D normal map via a hand-rolled small convolution replacing
  `cv2.filter2D` (no OpenCV / cgo).
- **Boundary decimation** on the `(deg=5°, stride=5px)` grid.
- **Finger-combination enumeration** over the decimated boundary points.
- **Friction-cone + width test** (`F·n > cos(atan(friction_coef))`, `dmin < Ds < dmax`).
- **Weighted scoring** (weights `[0.4, 0.3, 0.15, 0.05, 0.05, 0]`) and **descending sort**.

This is **fully unit-testable offline** — no ONNX, no depth — which satisfies the `CLAUDE.md`
requirement to test the pre/postprocess of every model.

It is then wired as a **`PipelineModel` that REUSES the existing SAM sessions** — exactly like
Grounded-SAM chains GroundingDINO → SAM, except the **final stage is Go analytic code instead
of another ONNX session**. `Roles()` returns SAM's roles; `Infer` runs SAM to obtain masks
(already column-major RLE in the unified schema), decodes each, calls `grasp.FromMask`, appends
to `Result.Grasps`, and sets `Task = TaskGrasp`. **`lifecycle.Manager` still owns the SAM
sessions (VRAM-safe).** No new ONNX session, no depth, no core changes.

```go
// internal/models/graspsam/graspsam.go  (sketch)
const (
    roleEncoder = "encoder"
    roleDecoder = "decoder"
)

func (m *graspSAM) Roles() []string { return []string{roleEncoder, roleDecoder} }

func (m *graspSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
    // 1) Segment with the existing MobileSAM sessions (box/point prompt → masks).
    encRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleEncoder, in) }
    decRun := func(in map[string]engine.Tensor) ([]engine.Tensor, error) { return r.Run(roleDecoder, in) }
    encIn := firstName(r.InputNames(roleEncoder), "input_image")
    masks, err := mobilesam.Segment(img, prompt.Boxes, encRun, decRun, encIn, r.OutputNames(roleDecoder))
    if err != nil {
        return models.Result{}, err
    }

    // 2) Analytic grasp search per mask — pure Go, no ONNX, no depth.
    var grasps []api.Grasp
    for _, mk := range masks {
        bmp := grasp.DecodeRLE(mk.RLE, /* w,h from meta */) // column-major RLE → bitmap
        grasps = append(grasps, grasp.FromMask(bmp, m.params)...)
    }

    return models.Result{Task: models.TaskGrasp, Masks: masks, Grasps: grasps}, nil
}
```

### 5a. Class-agnostic vs. box-prompted fast path

The shipped `grasp` model (`internal/models/grasp/grasp.go`) has an OPTIONAL detector. With a
detector (e.g. `grasp-gd` = GroundingDINO) it runs **detect → segment-per-box → grasp-per-mask**
for **class-aware** grasps. **Without** a detector the model is **class-agnostic**, and the
incoming `Prompt` selects between two paths:

- **No `box` prompt** — whole-image **automask**, then `mask2grasp` on **every** mask. Cost
  scales with the number of objects in the *scene*.
- **`box` prompt present (fast path)** — the class-agnostic `grasp` model honors the boxes and
  segments **ONLY** those boxes (MobileSAM box-prompted), running the analytic `mask2grasp` on
  those masks alone. Cost scales with the number of *targets*, not the scene — one segmentation
  + one `FromMask` per box instead of automask + `FromMask` over everything, which is several×
  faster for a single object.

This enables a **"select the target client-side, then grasp just it"** flow:
`grounding-dino` → `select_target_object(...)` → `predict("grasp", img, box=target_bbox)`.

> NOTE: this fast path is the **plain `grasp` model only**. On `grasp-gd` the built-in
> GroundingDINO detector always runs and the incoming `box` is **ignored** — to target one
> object, use the two-step flow (GroundingDINO boxes selected client-side → plain `grasp` with
> `box=...`) rather than `grasp-gd`.

### 5b. Per-request GroundingDINO thresholds (grasp-gd)

The GroundingDINO detector stage inside `grasp-gd` honors per-request threshold overrides:
`box_threshold` / `text_threshold` carried on the `Prompt` (>0) take precedence over the
manifest/default thresholds (`defaultBoxThresh = 0.3`, `defaultTextThresh = 0.25`) for that
request, exactly as they do for the standalone `grounding-dino` model.

## 6. Integration path B — GG-CNN learned (later)

A plain **`Model`** (single ONNX session) implementing the standard
`Preprocess` / `Postprocess` seam.

- **Preprocess:** depth image (16-bit PNG → `image.Gray16`) → inpaint holes → crop/resize to
  `300×300` → normalize.
- **Postprocess:** `outs = [Q, cos2θ, sin2θ, width]`, each `1×1×300×300`. **The output order
  MUST be verified on the real exported `.onnx` — do not trust this order blindly.** Then:
  Gaussian-blur `Q` → `peak_local_max(Q, min_distance=20, threshold_abs=0.2, num_peaks=MaxDet)`
  → per peak `θ = ½·atan2(sin, cos)`, `width = width · ~150px` → map `(x, y, w)` back to
  **ORIGINAL** image coordinates via `PreprocessMeta` → `api.Grasp`.
- **Manifest:** `license: BSD-3-Clause`, `task: grasp`, `files: { model: ggcnn.onnx }`.

**GR-ConvNet** uses the *same* decoder but needs an **RGB-D 2-input** path (`4×224×224`), so
**defer** it until the depth-input plumbing exists.

```go
// internal/models/ggcnn/ggcnn.go  (sketch)
func (m *ggcnn) Task() models.Task    { return models.TaskGrasp }
func (m *ggcnn) OutputNames() []string { return []string{"q", "cos", "sin", "width"} } // VERIFY on real .onnx

func (m *ggcnn) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
    q, cos, sin, w := outs[0], outs[1], outs[2], outs[3] // order MUST match the real export
    qb := gaussianBlur(q)
    peaks := peakLocalMax(qb, /*min_distance*/ 20, /*threshold_abs*/ 0.2, m.maxDet)
    grasps := make([]api.Grasp, 0, len(peaks))
    for _, p := range peaks {
        theta := 0.5 * math.Atan2(sin.At(p), cos.At(p))
        widthPx := w.At(p) * 150.0
        x, y, wpx := mapToOriginal(p.X, p.Y, widthPx, meta) // back to ORIGINAL coords
        grasps = append(grasps, api.Grasp{X: x, Y: y, Theta: theta, Width: wpx, Quality: qb.At(p)})
    }
    return models.Result{Task: models.TaskGrasp, Grasps: grasps}, nil
}
```

## 7. Recommended sequencing

- **B0 — Schema.** Add `Grasp`, `Result.Grasps`, `TaskGrasp` to `pkg/api/types.go`
  (small, shared, unblocks both paths).
- **B1 — `internal/grasp` core + tests.** Port the user's analytic method (zero external
  dependency, lowest risk), with offline unit tests on synthetic masks.
- **B1 — `PipelineModel` on SAM.** Wire `internal/grasp` behind the existing MobileSAM sessions
  (no new ONNX, no depth).
- **B2 — GG-CNN (later).** Add the learned plain `Model` behind a depth-image input path.

## 8. Verify-before-ship (do NOT fabricate)

1. **Output tensors.** GG-CNN / GR-ConvNet output-tensor **names AND order** must be confirmed
   on the **actual exported `.onnx`** before writing decode logic
   (`CLAUDE.md`: verify real tensor shapes, never fabricate). The `[Q, cos2θ, sin2θ, width]`
   order in §6 is the *expected* convention, not a verified fact.
2. **Weights license vs. code license.** The **code** is BSD, but the **weights** are trained on
   the **Cornell** / **Jacquard** grasp datasets. **Jacquard** in particular is
   research/registration-gated. Confirm the weight/dataset terms before **redistributing**
   weights. This affects **path B only** — **path A uses no weights**.
