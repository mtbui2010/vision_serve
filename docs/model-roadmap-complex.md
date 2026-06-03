# Complex Model Feasibility Roadmap

 Analysis of candidate models for future implementation.
>
> **Current status:** NanoSAM, CLIP, SCRFD, and PaddleOCR are **complete** (catalog verified, ONNX confirmed, Go implementation done). NanoSAM requires manual ONNX download (no HuggingFace source). ArcFace, ByteTrack, and RTMPose remain planned. The analysis below was written as design documentation; implementation status is noted per section.

---

## 1. PaddleOCR PP-OCRv4

### A. Technical Requirements

**Sessions (3):**
- `det` — DBNet++ text-region detector: outputs shrunk binary maps (float32, `[1,1,H,W]`);
  contour extraction + Vatti polygon expansion produces rotated quads (not axis-aligned
  rectangles).
- `cls` — lightweight direction classifier (optional): 2-class softmax; decides whether
  to rotate the crop 180° before recognition.
- `rec` — SVTR/CTC recognizer: input is a 3×48×W crop, output is `[T, charset+1]` logits;
  greedy CTC decoding collapses blank/repeated tokens → UTF-8 string.

**New Go algorithms required:**

1. **Contour extraction from a probability map** — find connected components / contours in
   the DBNet binary map. The standard `image` package has no contour-finding. Options:
   - Implement a pure-Go connected-component labelling (CCL) and convex-hull/polygon
     approximation: feasible but ~300–400 lines, non-trivial numerics.
   - Pull in `golang.org/x/image` bitmap utilities + write a custom chain-code tracer.
   - Use CGO/OpenCV `findContours` (violates rule #4 unless strictly justified).
   **This is the hardest single piece of new code in the whole roadmap.**

2. **Perspective (warp affine) crop** — once a rotated quad is found, the recognizer needs a
   straightened crop of that quad. The standard `image` and `disintegration/imaging` packages
   support only axis-aligned resize/crop. A four-point perspective warp requires a 3×3 homography
   and bilinear resampling. This is ~100–150 lines of pure-Go math (no CGO needed) but it is
   non-trivial to get correct (and to test).

3. **CTC greedy decode** — straightforward: argmax per frame, collapse consecutive identical
   labels and blanks, look up each id in a character dictionary text file. ~30–40 lines of Go.

4. **Result schema extension** — `api.Result` currently has `Detections` (bboxes) and `Masks`
   (RLE). OCR needs a new top-level field like `Texts []TextRegion` where `TextRegion` carries
   `BBox [4][2]float64` (four corner points of the rotated quad) + `Text string` + `Conf float64`.
   Adding this field to `api.Result` is a **breaking schema change** visible to all API clients.
   The manifest `validTasks` enum also needs a new `"ocr"` task.

5. **Direction classifier** — in practice nearly all PP-OCRv4 deployments skip the cls stage
   (texts in most documents are upright). Skipping it is acceptable for v1 and removes one
   session.

**CGO requirement assessment:**  
Contour extraction is the only realistic CGO/OpenCV candidate. It is not *strictly* required
if a pure-Go CCL + polygon simplification is implemented, but implementing that correctly is
the largest engineering risk in this model. Perspective warp can be done in pure Go.

### B. License Verification

- **Original repo:** `PaddlePaddle/PaddleOCR` — Apache-2.0. ✅
- **PP-OCRv4 ONNX on HuggingFace:** `paddlepaddle/PP-OCRv4` — Apache-2.0. ✅
- **No AGPL components.** PaddlePaddle framework is Apache-2.0; the exported ONNX graphs
  have no runtime PaddlePaddle dependency.

### C. Architecture Fit

**PipelineModel** with roles `det`, `rec` (and optionally `cls`). Fits the existing
`PipelineModel` interface cleanly; lifecycle loads all sessions, the model orchestrates
them through the `Runner`.

**Server changes needed:**
- New `api.Task = "ocr"` constant.
- New `TextRegion` struct + `Texts []TextRegion` field on `api.Result`.
- `registry.validTasks` must accept `"ocr"`.
- `api.Result` schema change is a minor breaking change for existing API clients who
  parse the JSON strictly.

### D. Implementation Effort

**Effort: XL (10–15 engineering days)**

| Sub-task | Days |
|----------|------|
| Pure-Go contour extraction + polygon approximation | 3–5 |
| Perspective warp crop (pure Go, with tests) | 2 |
| CTC decode + charset loader | 0.5 |
| Pre/postprocess for all three sessions | 2 |
| Schema additions (api.Result, manifest task enum) | 0.5 |
| Integration + end-to-end tests | 2 |

**Key risks / unknowns:**
- DBNet contour quality vs. a real OpenCV `findContours` — this is the most error-prone step
  and the only one where a pure-Go re-implementation may diverge from reference Python results.
- Character sets: PP-OCRv4 ships separate charset files per language (Chinese 6625 chars,
  English, etc.); the model must load the correct one from the manifest `Dir`.
- Rotated-quad BBox representation differs from `[x,y,w,h]` — requires documenting a new
  convention in the schema.

**Recommendation: defer to v2.**  
The schema change (adding `Texts` to `api.Result`) should be deliberately designed rather than
rushed. The contour extraction code is a significant, isolated effort that is better tackled
after the simpler models are stable. OCR is a high-value capability but it is best delivered
with proper schema design and tested contour logic.

---

## 2. SCRFD (Face Detection)

### A. Technical Requirements

**Sessions: 1** (the `det_10g_bnkps.onnx` or similar InsightFace export).

SCRFD is an anchor-based, multi-scale face detector with three FPN strides (8, 16, 32).
The ONNX export exists in two variants:
- **Per-stride outputs** (3 × 3 = 9 tensors: score, bbox, kps per stride): more common
  in the InsightFace repo's direct export.
- **Concatenated outputs** (3 tensors: all scores, all bboxes, all kps): produced by some
  community re-exports.

Either way, postprocess needs:
1. **Anchor generation per stride** — for each stride S, tile anchor centres over the
   `(H/S) × (W/S)` grid; typically 2 anchors per location. ~40 lines of Go.
2. **Decode distance-to-box** (SCRFD uses `dist2bbox` not classic `(x-xa)/wa` offsets) —
   ~20 lines.
3. **NMS** — already implemented in `internal/imageproc/nms.go`. Ready to use. ✅
4. **Keypoint decode** (5 facial landmarks) — straightforward once anchor centres are known.

**Output 5 face keypoints** — not currently representable in `api.Detection` (which has
only `BBox`, `Class`, `Conf`). Options:
- Add `Keypoints [][2]float64` to `api.Detection` — small schema addition, backward-compatible
  (omitempty), visible to all models.
- Return keypoints as a separate slice on `api.Result` — less clean.
- Omit keypoints in v1 (just return face boxes without landmarks) — simplest, works for most
  use cases.

### B. License Verification

- **InsightFace repo** (`deepinsight/insightface`): **MIT** license. ✅
- **SCRFD ONNX on HuggingFace** (`deepinsight/insightface` model zoo): MIT. ✅
- No AGPL. InsightFace's own training code has additional terms for commercial use, but the
  **ONNX weights in the public model zoo are MIT**. Verify the specific file's README before
  including in a manifest.

### C. Architecture Fit

**Plain `Model`** — single ONNX session, no prompt needed for face detection. Fits the
standard `Model` interface exactly: `Preprocess` + `Postprocess`.

**NMS is already present** in `imageproc/nms.go` — SCRFD postprocess can call it directly.

No server changes needed for the core detection capability. Keypoints would require a small
`api.Detection` schema addition (backward-compatible `omitempty` field).

### D. Implementation Effort

**Effort: M (3–4 engineering days)**

| Sub-task | Days |
|----------|------|
| Anchor generation + dist2bbox decode | 1 |
| Postprocess (handle per-stride or concatenated ONNX variants) | 1 |
| Preprocess (letterbox, normalize — mostly reuse `imageproc`) | 0.5 |
| Tests (anchor grid, bbox decode, NMS integration) | 1 |
| Manifest + README | 0.5 |

**Key unknowns:**
- Need to inspect the actual ONNX output tensor names/shapes from the InsightFace model zoo
  export to know whether it is per-stride or concatenated. Do not guess; run `engine.Inspect`
  first.
- If keypoints are included, a small `api.Detection.Keypoints` field should be added — needs
  a deliberate schema decision.

**Recommendation: implement in v1 (after NanoSAM and CLIP).**  
SCRFD alone is well-scoped. The NMS utility already exists. Main ambiguity is the exact ONNX
export format, which a single `engine.Inspect` call resolves. Keypoints can be deferred to a
follow-up PR.

---

## 3. ArcFace (Face Recognition Embedding)

### A. Technical Requirements

**Sessions: 1** (a ResNet-50 or MobileNet backbone trained with ArcFace loss, e.g.
InsightFace `w600k_r50.onnx`).

**Input:** aligned 112×112 face crop (3-channel, NCHW, BGR or RGB depending on export).  
**Output:** 512-d float32 embedding (L2-normalized by the model graph in most exports).

**The core challenge: face alignment.**  
ArcFace was trained on images aligned via a 5-keypoint similarity transform. Without
alignment, accuracy drops significantly (same person may not match above threshold across
different poses/distances). Alignment requires:
1. **5 keypoints from SCRFD** (or another face detector that returns landmarks).
2. **Similarity transform estimation** (2D: scale + rotation, no perspective) from 5
   landmark pairs. Standard implementation: mean-square optimal rigid transform; ~50 lines
   of pure-Go linear algebra (no CGO needed; it is 2×2 SVD or closed-form).
3. **Affine crop + resize to 112×112** — the warp affine transform applied to the detected
   face bounding box; bilinear interpolation; pure-Go ~80 lines (same as the OCR perspective
   warp but simpler — similarity not full perspective).

**Without alignment:** ArcFace still produces embeddings and cosine similarity still
works for easy cases (frontal faces, consistent scale). For a v1 "useful but imperfect"
implementation, one can skip alignment and just crop + resize the axis-aligned box. This
may be acceptable for some use cases while being clearly documented.

**New Result field:** a face recognition endpoint needs to return an embedding vector, not
a detection. `api.Result` has no embedding field. This requires either:
- Adding `Embedding []float32` to `api.Result` (lossy approach — only one embedding
  per call) or `Embeddings [][]float32`.
- A dedicated `/api/embed` endpoint (cleaner API separation).
This is a schema design decision, not just an implementation detail.

### B. License Verification

- **InsightFace ArcFace weights:** model zoo is MIT. ✅
- The specific `w600k_r50.onnx` file (trained on WebFace600K) on HuggingFace — MIT. ✅
- Verify: ArcFace loss is a training trick, not a runtime component; no license issue.

### C. Architecture Fit

**Plain `Model`** for the embedding inference itself (single session, no prompt).

However ArcFace is most useful *chained with* SCRFD: detect faces → get keypoints →
align → embed. This chain is analogous to Grounded-SAM (DINO → SAM) and would naturally
be a `PipelineModel` wrapping SCRFD + ArcFace. That pipeline also needs a new "embed"
task type and new Result fields.

**Server changes needed:**
- New `api.Task = "embed"` or `"face_recognition"`.
- `Embeddings [][]float32` on `api.Result`, or a new `/api/embed` endpoint.

### D. Implementation Effort

**Effort: L (5–7 engineering days, with alignment)**  
**Effort: M (3 days, without alignment — just crop + embed)**

| Sub-task | Days |
|----------|------|
| ArcFace plain `Model` (preprocess 112×112, postprocess L2-norm) | 0.5 |
| Similarity transform + affine warp (pure Go, with tests) | 2 |
| SCRFD+ArcFace pipeline model | 1.5 |
| Schema additions (embed task, Embeddings field or new endpoint) | 1 |
| Tests | 1 |

**Key unknowns:**
- The embedding use case (similarity search, 1:1 verification, 1:N identification) requires
  client-side cosine similarity — VisionServe only returns the vector. This is fine for the
  API but should be documented.
- BGR vs RGB normalization varies between ArcFace exports; must verify the specific file.
- Without SCRFD keypoints, alignment is impossible. ArcFace is therefore best implemented
  as a dependent follow-on to SCRFD.

**Recommendation: defer to v2 (depends on SCRFD shipping first).**  
ArcFace without SCRFD is a partial solution. The schema question (how to return embeddings)
also deserves careful design. Implement as a follow-on to SCRFD with a dedicated `/api/embed`
endpoint.

---

## 4. ByteTrack (Multi-Object Tracking)

### A. Technical Requirements

ByteTrack is a **pure algorithm**, not an ONNX model. It combines:
- **IoU matching** (Hungarian algorithm) — ~100 lines of Go; Hungarian is O(n³) but with
  n < 100 objects it is fast enough.
- **Kalman filter** for motion prediction — requires a 8-state Kalman filter (position +
  velocity in `[cx, cy, ar, h]` space); ~150 lines of Go linear algebra (2D matrix ops,
  no external dependency needed for this scale).
- **Two-stage data association** (high-confidence detections first, then low-confidence
  detections to "byte"-match occluded objects) — the actual ByteTrack innovation.

**State is inherently per-stream and per-frame.** A tracker struct holds:
- Active tracks: each with ID, Kalman state, age, hit streak.
- Lost tracks: temporarily occluded objects.
- Removed tracks: expired.

**This cannot be a stateless per-call operation.** The server would need to maintain tracker
state *between* HTTP requests, keyed by `(model_name, stream_id)` or similar.

**Infrastructure changes required (significant):**

1. **New `Tracker` interface** — not `Model` or `PipelineModel`. Something like:
   ```go
   type Tracker interface {
       Update(detections []api.Detection, frameID int) []Track
   }
   ```
   where `Track` carries `ID int`, `BBox [4]float64`, `Conf float64`, `Class string`.

2. **Server-side tracker state map** — the server (or lifecycle manager) needs a concurrent
   map `streamID → *TrackerState` with the same idle-unload semantics as model sessions.
   This is new infrastructure; the current `lifecycle.Manager` manages ONNX sessions only.

3. **New HTTP endpoint** — `POST /api/track` accepting `{model, stream_id, image, ...}` or
   frame+detections, returning `[]Track` with track IDs. Or a new `/api/track/{stream_id}`
   streaming endpoint.

4. **New Result schema** — `api.Track` with `TrackID int`; existing `api.Result` does not
   have this concept.

5. **New manifest task** — `"tracking"`.

**An alternative simpler design:** accept pre-computed detections as input (skip ONNX
entirely for ByteTrack itself), so ByteTrack is a pure algorithmic service. This avoids
coupling to a specific detector, but still requires all the server-side state machinery.

### B. License Verification

- **ByteTrack paper / reference code** (`ifzhang/ByteTrack`): **MIT**. ✅
- No ONNX weights to verify — it is pure algorithm code.
- The Go reimplementation of the algorithm carries no license inheritance from the Python
  reference; a clean-room Go port is MIT-compatible.

### C. Architecture Fit

ByteTrack **does not fit** either `Model` or `PipelineModel`. It needs:
- A new `Tracker` interface.
- Server-side stateful session management (not ONNX lifecycle, but tracker state lifecycle).
- A new HTTP endpoint (not `/api/predict`).
- A new `api.Task` and new `api.Track` schema.

This is the most architecturally disruptive item in the roadmap. It touches
`internal/server/server.go` (new route), `pkg/api/types.go` (new types), and requires a
new lifecycle analog for tracker state management. **It is not a "no core change" model addition.**

### D. Implementation Effort

**Effort: XL (12–18 engineering days)**

| Sub-task | Days |
|----------|------|
| Kalman filter (pure Go, tested) | 2 |
| Hungarian assignment | 1.5 |
| ByteTrack two-stage association logic | 2 |
| Tracker state lifecycle (in-memory map, idle cleanup) | 2 |
| New `POST /api/track` endpoint + request/response types | 1.5 |
| Schema additions (`Track`, `TaskTracking`) | 1 |
| Integration tests (multi-frame synthetic scenario) | 3 |

**Key risks:**
- Kalman filter numerical correctness is subtle and hard to unit test without reference data.
- Tracker state management across concurrent requests (multiple streams in parallel) needs
  careful locking design analogous to but separate from `lifecycle.Manager`.
- ByteTrack is normally used in video pipelines (frames arrive in order). HTTP is stateless
  by design; the `stream_id` keying and state lifetime management add operational complexity
  for users.

**Recommendation: out of scope for v1; design carefully for v2.**  
ByteTrack requires the most new infrastructure of any item here. It should be designed
holistically (tracker lifecycle, streaming API, schema) rather than bolted on. The core
server is currently 100% stateless beyond ONNX session caching; introducing per-stream
stateful tracker sessions is a significant architectural commitment.

---

## 5. CLIP (Vision-Language Embedding)

### A. Technical Requirements

**Sessions: 2** (image encoder + text encoder — they are independent, often served
separately).

- **Image encoder** (`visual.onnx`): `[1,3,224,224]` NCHW float32 → `[1,512]` float32
  embedding. Standard ImageNet normalize (mean `[0.481, 0.457, 0.408]`, std `[0.268, 0.261,
  0.276]`). Architecturally identical to any ViT/CNN classification model. Preprocessing is
  a standard letterbox/resize + normalize + NCHW, all already available in `imageproc`.

- **Text encoder** (`textual.onnx`): tokenized int32/int64 `input_ids [1,77]` + optionally
  `attention_mask [1,77]` → `[1,512]` float32 embedding.

**The key challenge: CLIP BPE tokenizer.**

CLIP uses a custom BPE (Byte-Pair Encoding) tokenizer — specifically a variant of GPT-2 BPE
applied at byte level, with a 49408-token vocabulary. This is **fundamentally different** from
GroundingDINO's BERT WordPiece tokenizer already implemented in Go.

Options:
1. **Implement CLIP BPE in Go** — requires: byte-level vocabulary loading (from `vocab.json`
   + `merges.txt`), BPE merge iteration, and the Unicode/byte-level encoding mapping. This is
   ~200–300 lines of Go and non-trivial to get exactly right (especially for non-ASCII input).
   The existing `groundingdino/tokenizer.go` is a useful structural template.
2. **Export the text encoder with pre-tokenized inputs** — ONNX models can accept
   `input_ids` as `int64[1,77]` directly. If the caller pre-tokenizes (e.g., using a Python
   script offline or the client), the Go runtime only needs to pass int64 arrays. This sidesteps
   the tokenizer problem for server-side Go, but shifts it to the client. For a "zero-shot
   classification" workflow (common CLIP use case) the prompts are fixed and can be pre-computed
   as embeddings offline.
3. **Ship a pre-tokenized prompt library** — for the most common use case (zero-shot
   classification with ImageNet labels or similar), pre-compute all text embeddings offline and
   ship them as a float32 binary blob. The server only runs the image encoder at inference time.

For a v1 VisionServe implementation, option 3 is the pragmatic path: the image encoder alone
is immediately useful (image → embedding for similarity search, retrieval), and text embeddings
for fixed label sets can be pre-computed. Full BPE tokenizer is a v2 enhancement.

**Result schema:** CLIP returns embeddings, not detections or masks. Same schema challenge
as ArcFace (see above). A new `api.Task = "embed"` and `Embeddings [][]float32` on
`api.Result` (or a `/api/embed` endpoint) are needed.

### B. License Verification

- **openai/CLIP repo:** **MIT** license. ✅
- **CLIP ONNX on HuggingFace:** several exports exist:
  - `openai/clip-vit-base-patch32` (official): MIT. ✅
  - Community ONNX exports (e.g. `Xenova/clip-vit-base-patch32`): derived from MIT original.
    Verify the specific HuggingFace repo's license card before adding to the manifest. Most
    list MIT or Apache-2.0.
- **No AGPL.** OpenAI released CLIP under MIT; the Transformers library exporting it is
  Apache-2.0.

### C. Architecture Fit

**PipelineModel** with roles `image` and `text` (two independent ONNX sessions). The model
orchestrates them through the `Runner` as needed.

For the image-only v1 use case, it could even be a plain `Model` (single session, no
prompt). Upgrading to a `PipelineModel` when the text encoder is added is a clean extension.

**Server changes needed:**
- New `api.Task = "embed"`.
- `Embeddings [][]float32` on `api.Result` (or `/api/embed`).
- `registry.validTasks` must accept `"embed"`.

These same changes are needed for ArcFace, so implementing CLIP and ArcFace embedding support
together is efficient.

### D. Implementation Effort

**Effort: M (3–4 days, image encoder only + pre-tokenized text)**  
**Effort: L (6–8 days, full BPE tokenizer + both encoders)**

| Sub-task | Days (image-only) | Days (full) |
|----------|-------------------|-------------|
| Image encoder `PipelineModel` (preprocess 224×224, L2-norm postprocess) | 1 | 1 |
| Schema: `embed` task + `Embeddings` field | 0.5 | 0.5 |
| Pre-tokenized text embedding support (load float32 blob from manifest dir) | 0.5 | 0.5 |
| CLIP BPE tokenizer in Go | — | 2–3 |
| Text encoder session wiring | — | 1 |
| Tests | 1 | 1.5 |

**Key unknowns:**
- Need to confirm that the chosen ONNX export accepts int64 `input_ids` (not pre-normalized
  float32). Most do; run `engine.Inspect` to verify.
- L2 normalization: some CLIP ONNX exports normalize in-graph, others return raw embeddings.
  Must check the specific export.
- BPE special characters and Unicode handling is the fragile part of option 1 (full tokenizer).

**Recommendation: implement image encoder in v1 (M effort). Ship full BPE tokenizer in v2.**  
The image encoder is immediately useful for retrieval / similarity search. The `embed` task
and `Embeddings` schema addition needed for CLIP should be designed alongside ArcFace to avoid
doing it twice.

---

## 6. NanoSAM (NVIDIA Jetson-Optimized SAM)

### A. Technical Requirements

**Sessions: 2** (image encoder + mask decoder — exactly the same role structure as MobileSAM).

NanoSAM uses a ResNet-18-based image encoder (replacing MobileSAM's TinyViT) optimized with
NVIDIA TensorRT. The mask decoder is identical to the original SAM decoder. The ONNX I/O
contract is:

- **Encoder:** `[1,3,1024,1024]` float32 → `[1,256,64,64]` float32 image embeddings.
- **Decoder:** identical to MobileSAM decoder — same inputs (`image_embeddings`,
  `point_coords`, `point_labels`, `mask_input`, `has_mask_input`, `orig_im_size`),
  same outputs (`masks`, `iou_predictions`).

This means **the existing `mobilesam.Segment` function is reusable as-is**. The encoder
preprocessing also matches (resize long side to 1024, raw 0–255 values, normalization baked
in).

**On non-Jetson hardware:** NanoSAM's ONNX files run normally on CPU and CUDA EP. The TRT
optimization advantage only materializes when using the TensorRT EP on Jetson. On a CPU host
the encoder is slower than MobileSAM's TinyViT, but it is functionally correct. This means
NanoSAM is NOT Jetson-only; it degrades gracefully.

**Implementation:** the simplest possible approach is a thin wrapper that reuses
`mobilesam.Segment`:

```go
package nanosam

import (
    "visionserve/internal/models"
    "visionserve/internal/models/mobilesam"
)

func init() { models.Register("nano-sam", New) }

type nanoSAM struct{ cfg models.Config }

func New(cfg models.Config) (models.Base, error) { return &nanoSAM{cfg: cfg}, nil }
func (m *nanoSAM) Name() string      { return m.cfg.Name }
func (m *nanoSAM) Task() models.Task { return models.TaskSegmentation }
func (m *nanoSAM) Roles() []string   { return []string{"encoder", "decoder"} }

func (m *nanoSAM) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
    // identical to mobilesam.Infer — can literally call mobilesam.Segment
}
```

In fact, because the I/O contract is identical to MobileSAM, NanoSAM can share the same
**architecture registration** by pointing the manifest's `architecture: mobile-sam`. No
new Go package is needed at all. The manifest just declares different `files:` paths and
`runtime.prefer: [tensorrt, cuda, cpu]`.

### B. License Verification

- **NVIDIA/nanosam repo** (`NVIDIA-AI-IOT/nanosam`): **MIT** license. ✅
- **NanoSAM ONNX on HuggingFace** (`nmitchko/nanosam-with-onnx`): appears to be MIT
  (derived from the NVIDIA repo). **Must verify the specific HuggingFace repo's license
  card** — "It's on HuggingFace" is not a license (CLAUDE.md rule #1).
- The SAM decoder weights come from Meta's SAM, which is **Apache-2.0**. ✅
- No AGPL components.

### C. Architecture Fit

**Reuses `PipelineModel` (mobile-sam architecture).** Can be served by the existing
`mobile-sam` factory with no new Go code at all — the manifest `architecture: mobile-sam`
makes lifecycle load and wire it identically to MobileSAM.

If a dedicated `nano-sam` package is desired (for clearer attribution and any future
NanoSAM-specific tuning), it is a ~40-line thin wrapper that delegates to `mobilesam.Segment`.

**No server changes needed.** No schema changes needed.

### D. Implementation Effort

**Effort: S (0.5–1 engineering day)**

| Sub-task | Days |
|----------|------|
| Verify ONNX I/O contract matches MobileSAM (run engine.Inspect) | 0.25 |
| Write manifest.yaml for nano-sam (architecture: mobile-sam, TRT preferred) | 0.25 |
| Optional: thin Go wrapper package (for attribution/clarity) | 0.5 |
| Tests (reuse MobileSAM preprocess tests, smoke test with dummy ONNX) | 0.25 |

**Key unknowns:**
- The exact ONNX export needs inspection to confirm the decoder I/O is byte-for-byte
  identical to MobileSAM. The encoder input size (1024×1024) matches, but must verify
  the embedding shape is `[1,256,64,64]` (not a different number of feature channels).
- The normalization baked into the encoder graph may differ (NanoSAM's ResNet-18 encoder
  vs MobileSAM's TinyViT). Must inspect or test; if normalization differs, a small
  preprocessing override is needed.

**Recommendation: implement immediately (S effort, nearly free).** This is the easiest
item in the entire roadmap. It directly benefits Jetson users (the primary edge target
in CLAUDE.md) and may require zero new Go code.

---

## Summary Table

| Model | Feasibility | Effort | Status |
|-------|-------------|--------|--------|
| NanoSAM | ✅ straightforward | S (1 day) | **Complete** |
| CLIP (image encoder only) | ✅ straightforward | M (3–4 days) | **Complete** |
| SCRFD (face detection) | ✅ straightforward | M (3–4 days) | **Complete** |
| PaddleOCR PP-OCRv4 | ⚠️ significant work | XL (10–15 days) | **Complete** |
| ArcFace | ⚠️ significant work | L (5–7 days) | Deferred — depends on SCRFD |
| CLIP (full BPE + text encoder) | ⚠️ significant work | L (6–8 days) | Deferred to v2 |
| ByteTrack | ❌ needs new architecture | XL (12–18 days) | Out of scope for v1 |

---

## Implementation Order Summary

### Phase 1 — Complete

**1. NanoSAM** — `internal/models/nanosam/` implemented; reuses the existing `mobile-sam`
architecture. Requires manual ONNX download (no HuggingFace source).

**2. CLIP image encoder** — `internal/models/clip/` implemented; `TaskEmbed` + `Embeddings`
schema live in `pkg/api/types.go`.

**3. SCRFD face detection** — `internal/models/scrfd/` implemented; anchor-based decoder
complete.

**4. PaddleOCR PP-OCRv4** — `internal/models/paddleocr/` implemented; det + rec pipeline
complete. Pure-Go contour extraction implemented.

### Phase 2 — Deferred

**5. ArcFace + SCRFD pipeline** — depends on SCRFD keypoints shipping first.

### Future (v2)

**6. ByteTrack** — stateful tracker lifecycle requires new server infrastructure.
Design carefully; not rushed into v1.

---

## Minimum Server Infrastructure Changes (for Phase 2)

The following changes were needed to support CLIP and ArcFace embeddings and are now
**complete** (landed as part of Phase 1):

1. **`pkg/api/types.go`**:
   - `TaskEmbed Task = "embed"` — done.
   - `Embeddings [][]float32 \`json:"embeddings,omitempty"\`` on `api.Result` — done.
   - `Keypoints [][2]float64 \`json:"keypoints,omitempty"\`` on `api.Detection`
     (for SCRFD landmarks; backward-compatible with `omitempty`) — done.

2. **`internal/registry/manifest.go`**:
   - `api.TaskEmbed` added to `validTasks` (along with `depth` and `classification`) — done.

3. **`cmd/visionserve/main.go`**:
   - Blank imports for `nanosam`, `clip`, `scrfd`, `paddleocr` packages — done.

No changes to `server.go`, `handlers.go`, `lifecycle/manager.go`, or `engine/` are
needed for Phase 1 and 2. ByteTrack (Phase 4) is the only item that requires
`server.go` route additions and a new lifecycle analog for tracker state.
