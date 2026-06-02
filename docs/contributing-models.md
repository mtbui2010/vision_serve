# Adding a new model to VisionServe

> Design goal: **adding a model = adding a package, with NO core changes**
> (`server` / `engine` / `lifecycle`).

VisionServe is fully free and open-source (Apache-2.0). Contributions of new
permissive models are welcome.

## Licensing constraint (MANDATORY)

Only **permissive-licensed** models are accepted: **Apache-2.0 / MIT / BSD**.
**NEVER** add an AGPL model (Ultralytics YOLO, FastSAM, YOLO-World). Your manifest
must declare `license` accurately — the registry rejects anything outside the allowlist.

Why this matters even though the project is free: AGPL is viral copyleft. Pulling one
AGPL model in would relicense the whole project — and every downstream deployer —
under AGPL, breaking the permissive promise that lets the entire community (including
commercial and closed deployments) use VisionServe freely. "It's on HuggingFace" is
**not** a license; verify the model's actual license. Being community-driven *requires*
this discipline.

## Two kinds of model

Pick the interface that matches your architecture (`internal/models/model.go`):

- **Plain `Model`** — a single ONNX graph driven by the engine as `pre → infer → post`.
  The model implements `Preprocess` / `Postprocess` only; it does **not** call the
  session itself. Example: RF-DETR.
- **`PipelineModel`** — a **prompted** and/or **multi-session** model that drives its
  own inference. It implements `Roles()` (session keys) + `Infer(img, prompt, Runner)`,
  and its manifest declares a `files:` map (role → ONNX path). Lifecycle loads and owns
  the sessions; the model orchestrates them via the `Runner`. Examples: MobileSAM
  (encoder + decoder), GroundingDINO (text-prompted), Grounded-SAM (GroundingDINO → MobileSAM).

Both produce the same unified `Result` (`Detections` and/or `Masks`, masks as
column-major RLE). Never invent a per-model schema.

## Steps

### 1. Create the package `internal/models/<name>/`

#### Plain `Model` (no core change):

```go
package mymodel

import (
    "image"
    "visionserve/internal/engine"
    "visionserve/internal/models"
)

func init() { models.Register("my-arch", New) }

type myModel struct{ cfg models.Config }

func New(cfg models.Config) (models.Base, error) { return &myModel{cfg: cfg}, nil }

func (m *myModel) Name() string          { return m.cfg.Name }
func (m *myModel) Task() models.Task     { return models.TaskDetection }
func (m *myModel) InputName() string     { return "" } // "" = let the engine probe the ONNX
func (m *myModel) OutputNames() []string { return nil }

func (m *myModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
    // resize/letterbox/normalize → tensor; record scale/pad in meta
}
func (m *myModel) Postprocess(outs []engine.Tensor, meta models.PreprocessMeta) (models.Result, error) {
    // decode output → Result; BBox MUST be mapped back to ORIGINAL image coords via meta
}
```

#### Prompted / multi-session `PipelineModel`:

```go
func init() { models.Register("my-pipeline", New) }

type myPipe struct{ cfg models.Config }

func New(cfg models.Config) (models.Base, error) { return &myPipe{cfg: cfg}, nil }

func (m *myPipe) Name() string      { return m.cfg.Name }
func (m *myPipe) Task() models.Task { return models.TaskSegmentation }

// Roles are the session keys to load; each MUST be a key in the manifest 'files' map.
func (m *myPipe) Roles() []string { return []string{"encoder", "decoder"} }

// Infer drives the full pipeline. The prompt carries Text / Boxes / Points (in original
// image coords). Call r.Run(role, inputs) per stage; lifecycle owns the sessions.
func (m *myPipe) Infer(img image.Image, prompt models.Prompt, r models.Runner) (models.Result, error) {
    // e.g. encode image, then decode with the box/point/text prompt → Result (Detections/Masks)
}
```

The prompt is shared across CLI flags and HTTP fields (`models.ParsePrompt`):
`--prompt`/`prompt` (text), `--box`/`box` (`x,y,w,h`), `--point`/`point` (`x,y[,label]`).

### 2. Register a blank import in `cmd/visionserve/main.go`

```go
import _ "visionserve/internal/models/mymodel"
```

(This is the ONLY core touch point — a single import line.)

### 3. Create the registry directory `models/<name>/`

- `manifest.yaml` (see [manifest-spec.md](manifest-spec.md)) — `architecture` must match
  the name you `Register`ed. Multi-session models declare a `files:` map instead of (or
  alongside) `model_file`.
- `README.md` explaining how to download/export the ONNX weights (**do not commit** large files).
- (optional) a labels file.

### 4. Write tests for pre/postprocess

This is the most error-prone part (CLAUDE.md). At minimum:
- letterbox/normalize produce the correct shape + sample values.
- postprocess maps boxes back to original-image coords correctly (test the padded case).
- for `PipelineModel`s, prompt parsing and stage chaining behave as expected.

## Important notes

- **Do NOT guess the ONNX output format.** Verify the real tensor shapes before writing
  postprocess. If unsure → write a stub + `TODO`, do not fabricate.
- RF-DETR is **NMS-free** — do not apply YOLO-style NMS. Anchor-based models may use
  `imageproc.NMS`.
- Running a session is **not** the model's job for plain `Model`s — engine + lifecycle
  handle it; the model does pre/post only. `PipelineModel`s orchestrate via `Runner`, but
  lifecycle still owns and frees the sessions.
