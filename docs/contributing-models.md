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
  session itself. Examples: RF-DETR, RT-DETR, Depth Anything V2, MiDaS,
  EfficientNet-B0, MobileNetV3, SCRFD, CLIP (the `clip` image encoder).
- **`PipelineModel`** — a **prompted** and/or **multi-session** model that drives its
  own inference. It implements `Roles()` (session keys) + `Infer(img, prompt, Runner)`,
  and its manifest declares a `files:` map (role → ONNX path). Lifecycle loads and owns
  the sessions; the model orchestrates them via the `Runner`. The PipelineModels today:
  MobileSAM, NanoSAM and EfficientSAM (encoder + decoder), SAM2 (multi-scale
  encoder + decoder), GroundingDINO (text-prompted), Grounded-SAM
  (GroundingDINO → MobileSAM), the hybrid routers (`rfdetr-gdino`, `gdino-siglip`),
  `rfdetr-textalign`, grasp, background, OWL-ViT, PaddleOCR (det + rec), and the
  one-session text/image towers `clip-text`, `siglip-text` and `siglip-image`.

All produce the same unified `Result` (`Detections`, `Masks`, `Grasps`, `Classifications`,
`DepthMap`, or `Embeddings` depending on the task; masks as column-major RLE).
Never invent a per-model schema.

## Steps

### 1. Create the package `internal/models/<name>/`

#### Plain `Model` (no core change):

```go
package mymodel

import (
    "image"
    "visionserve/internal/engine"
    "visionserve/internal/models"
    "visionserve/internal/vision/preprocess"
)

func init() { models.Register("my-arch", New) }

type myModel struct{ cfg models.Config }

func New(cfg models.Config) (models.Base, error) { return &myModel{cfg: cfg}, nil }

func (m *myModel) Name() string          { return m.cfg.Name }
func (m *myModel) Task() models.Task     { return models.TaskDetection }
func (m *myModel) InputName() string     { return "" } // "" = let the engine probe the ONNX
func (m *myModel) OutputNames() []string { return nil }

// The modes this architecture can map back (internal/vision/preprocess); the first is its default.
var arch = preprocess.Arch{Name: "my-arch", Modes: []preprocess.Mode{preprocess.Squash, preprocess.Letterbox}}

func (m *myModel) Preprocess(img image.Image) (engine.Tensor, models.PreprocessMeta, error) {
    // The manifest's preprocess: block (or legacy input.* fields), resolved for this architecture,
    // applied by the ONE implementation: resize/pad/normalize → tensor + scale/pad in meta.
    s, err := arch.Resolve(m.cfg.PreprocessSpec())
    if err != nil {
        return engine.Tensor{}, models.PreprocessMeta{}, err
    }
    return s.Apply(img)
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

### 5. Make it pullable (`visionserve pull <name>`)

A model is only really "added" once anyone can get the weights. If the weights already live
in a public, permissively-licensed HF repo, skip to (c) — you only need to publish when the
artifact is **yours** (a re-export, a fine-tune, a conversion).

#### (a) Store your HF token once

`huggingface_hub` reads the token from `$HF_HOME/token` (default `~/.cache/huggingface/token`),
or from `$HF_TOKEN`. Create a **write** token at <https://huggingface.co/settings/tokens>, then:

```bash
hf auth login            # writes ~/.cache/huggingface/token
# or, non-interactively:
export HF_TOKEN="hf_..."
```

Never commit the token. If you override `HF_HOME` (e.g. to keep the cache off a full root
disk), the token is looked up under the *new* `HF_HOME` — export `HF_TOKEN` explicitly in that
case.

#### (b) Publish the artifact

Upload the ONNX graph plus any side files (vocab, labels, tokenizer). Keep the repo **public**
— a catalog entry pointing at a gated repo is useless.

```python
from huggingface_hub import HfApi

api = HfApi()
repo = "<user>/<model>-ONNX"
api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)

for local, remote in [
    ("MODEL_CARD.md", "README.md"),
    ("models/<name>/vocab.txt", "vocab.txt"),
    ("models/<name>/model.onnx", "model.onnx"),
]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo)
```

`upload_file` streams from disk — it does not stage a copy in the HF cache — so publishing a
700 MB graph needs no extra free space.

The model card (`README.md` in the HF repo) **MUST** declare the license in its YAML
frontmatter, and it must be one VisionServe accepts:

```markdown
---
license: apache-2.0
base_model: <the upstream you derived from>
---
```

Also record, in prose: what the file is, the exact upstream it derives from, what you changed
and why, the verified input/output contract (names, shapes, dtypes), and any measured
equivalence/latency numbers — with the hardware and the caveats. Do not quote a number you did
not measure. `models/grounding-dino-fixed/HF_MODEL_CARD.md` is a worked example (it is kept in
this repo and uploaded as the HF `README.md`, so the two never drift).

Finally, record the digest and pin it in `models/<name>/manifest.yaml`:

```bash
sha256sum models/<name>/model.onnx
```

```yaml
source_url: https://huggingface.co/<the audited upstream>
sha256:
  model: <digest>
```

The pin binds the declared license to specific bytes — `registry.VerifyWeights` refuses to
load a file whose hash drifted (see [threat-model.md](threat-model.md)).

#### (c) Add the catalog entry

`internal/catalog/catalog.go` holds the built-in, curated list (no remote registry). Append an
`Entry` next to the most similar existing model and copy its field conventions:

```go
{
    Name:         "my-model",
    Task:         "detection",
    License:      "Apache-2.0",          // permissive only — the registry rejects the rest
    Architecture: "my-arch",             // must match models.Register()
    Description:  "One line shown by `visionserve list`.",
    HFRepo:       "<user>/<model>-ONNX",
    Files: []File{
        // ManifestRole set  → written into the manifest `files:` map (multi-session).
        // ManifestRole empty → side file (vocab/labels) or, for a lone file, model_file.
        {Role: "model", HFFilename: "model.onnx", LocalFilename: "model.onnx", ManifestRole: "model",
         SHA256: "…"},                       // see (d) — omit it and verified mode refuses the model
        {Role: "vocab", HFFilename: "vocab.txt", LocalFilename: "vocab.txt"},
    },
    InputWidth: 800, InputHeight: 800, InputLayout: "NCHW",
    Normalize:         &Normalize{Mean: []float32{0.485, 0.456, 0.406}, Std: []float32{0.229, 0.224, 0.225}},
    PostprocessType:   "my-arch",
    RuntimePrefer:     []string{"cuda", "cpu"},
    IdleUnloadSeconds: 300,
    Verified:          true,             // false ⇒ `pull` prints Note as a warning first
    Note:              "I/O contract, quirks, provenance — read by the next maintainer.",
}
```

Two rules that are easy to get wrong:

- **A pulled model directory must be self-contained.** If your model needs a file another
  model also uses, list it again in `Files` and download a second copy. `VirtualFiles` +
  `Dependencies` (`../sibling/model.onnx`) is only for models that genuinely reuse another
  model's session: *composed* ones like `grounded-sam` (no `Files` of their own), and *partly
  composed* ones like `rfdetr-textalign-dec1-siglip` (own detector in `Files`, the shared
  `../siglip-text/model.onnx` in `VirtualFiles`).
- `pull` never clobbers a `manifest.yaml` a user edited; it writes one when missing or with
  `--force`, and regenerates one it generated itself if that file is unedited (the
  `# Generated by` header line records a hash of the body; an edit no longer matches it, see
  `isUneditedGenerated` in `internal/catalog/manifest.go`). So the hand-written
  `models/<name>/manifest.yaml` and `Entry.RenderManifest()` should agree — check them
  side by side.

#### (d) Pin the bytes, and record the licence audit (the ledger)

The load-time gate has three layers. The permissive-licence allowlist is always on. The other
two only bite in **verified mode** (`registry.EnableVerifiedMode()`), and both need data you
supply here:

| layer | what it proves | where it comes from |
|---|---|---|
| licence allowlist | the declared licence is permissive | `Entry.License` |
| **sha256 content pin** | these are the exact audited bytes | `File.SHA256` → manifest `sha256:` |
| **provenance ledger** | a human read the upstream's LICENSE | `registry.LicenseLedger` |

In verified mode a model with **no** `sha256` is refused, and so is one whose `source_url` has
no ledger entry. Skip either and `visionserve pull` will succeed, then the model will fail to
load — after the user has already downloaded several hundred megabytes.
`TestAuditedEntriesArePinned` catches that at `go test` time; keep it green.

**Get the digest from the Hub, not from your local copy.** What you must pin is what the Hub
will serve to somebody else. For LFS files the Hub already stores it:

```bash
HF_TOKEN="$(cat ~/.cache/huggingface/token)" python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("<user>/<model>-ONNX", files_metadata=True)
for s in info.siblings:
    print(s.rfilename, s.lfs.sha256 if s.lfs else "(not LFS — hash it after download)")
PY
```

Small non-LFS side files (a `vocab.txt`) have no Hub-side digest: download once, `sha256sum`
it, and pin that.

**Then write the ledger entry.** `internal/registry/ledger.go` is maintainer-owned — contributors
edit manifests, maintainers edit this table. One entry is one claim: *a human opened this
upstream's LICENSE file and read this SPDX id.*

```go
{SourcePrefix: "https://huggingface.co/<user>/<model>-ONNX/",  // trailing slash: matched by prefix
    License:    "Apache-2.0",                                   // what you READ, not what you hope
    LicenseURL: "https://huggingface.co/<original-author>/<upstream>",  // the file you actually opened
    AuditedBy:  "<your handle>", AuditedDate: "2026-08-18",
    Note:       "why this upstream is what it claims to be"},
```

Four things to get right:

1. **`SourcePrefix` is the repo you serve FROM; `LicenseURL` is where the licence was READ.**
   For a re-export these differ: `mtbui2010/grounding-dino-tiny-fixedmask-ONNX/` is the source,
   while the licence evidence lives at `IDEA-Research/grounding-dino-tiny`. The manifest's
   `source_url` must match `SourcePrefix`, not the lineage URL.
2. **Matching is longest-prefix**, so a narrow entry can override a broad one. Keep prefixes at
   repo granularity.
3. **`License` must equal the manifest's `license:` exactly.** A mismatch is refused with a
   "may be mislabeled or relicensed" error — that is the control working, not a bug.
4. **Auditing a repo you control is weaker evidence than auditing a stranger's.** Say so in the
   `Note`. What a first-party entry certifies is the *weight lineage*; the sha256 pin is what
   then binds the claim to bytes.

Verify the whole chain against the real weights, without re-downloading:

```go
// throwaway test in internal/registry
EnableVerifiedMode(); defer DisableVerifiedMode()
m, _ := LoadManifest("../../models/<name>/manifest.yaml")
if err := m.VerifyWeights(); err != nil { t.Fatal(err) }   // hashes the file on disk
```

Then verify end to end **into a throwaway directory**, so a broken entry cannot overwrite your
working local setup:

```bash
make build
go vet ./... && go test ./internal/catalog/...
./bin/visionserve pull my-model --models /tmp/pulltest
sha256sum /tmp/pulltest/my-model/model.onnx      # must match the manifest pin
./bin/visionserve list --models /tmp/pulltest    # must show "ready"
```

## Important notes

- **Do NOT guess the ONNX output format.** Verify the real tensor shapes before writing
  postprocess. If unsure → write a stub + `TODO`, do not fabricate.
- RF-DETR is **NMS-free** — do not apply YOLO-style NMS. Anchor-based models may use
  `nms.Detections` (`internal/vision/nms`).
- Do not hand-write resize / pad / HWC→CHW / normalise loops: declare a
  `preprocess.Spec` (`internal/vision/preprocess`) — from the manifest via
  `cfg.PreprocessSpec()` + your `preprocess.Arch`, or as a fixed literal when the export dictates
  it (the SAM encoders) — and call `Apply`. Genuinely special geometry (per-detection crops) still
  ends in the shared `Spec.Tensor`. See [manifest-spec.md, Preprocessing](manifest-spec.md#preprocessing).
- Do not re-implement shared geometry/mask code in a model package: use
  `internal/vision/geom` (sigmoid, normalized box → input pixels, `meta.Affine()` to map
  back to the original image, clamp, IoU) and `internal/vision/mask` (threshold → bitmap +
  bbox, the one column-major RLE encoder, PyTorch-bilinear / nearest upsampling, min-max
  normalisation).
- **Client-side resizing is opt-in per architecture.** The SDKs shrink a large photo before
  uploading it only when `GET /api/models` gives the model a hint, and an architecture gets one
  only by calling `models.RegisterUsefulSide(arch, fn)` in `init()` (see
  `internal/models/usefulside.go`; a plain model with a fixed-size `preprocess.Arch` registers
  `models.ResolvedUsefulSide(arch)`). Register only when the result cannot depend on pixels
  beyond 2 × the model's input: never for models that return full-resolution masks, read text,
  align the photo with another input (depth), crop the ORIGINAL photo (crop namers) or compare it
  with templates. Not registering is always correct: photos then go up at full resolution. Add
  the model to `TestUsefulSideShippedManifests` (`internal/lifecycle/usefulside_test.go`).
- Running a session is **not** the model's job for plain `Model`s — engine + lifecycle
  handle it; the model does pre/post only. `PipelineModel`s orchestrate via `Runner`, but
  lifecycle still owns and frees the sessions.
