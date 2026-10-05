# Manifest spec ("Modelfile for CV")

Each model is a subdirectory in the registry (`./models/<name>/`) containing a
`manifest.yaml` + the ONNX file(s) (not committed) + (optional) a labels file.

## Full example (single-session model)

```yaml
name: rf-detr                 # required — model identifier in the registry
task: detection               # required — detection | segmentation | open_vocab | depth | classification | embed | grasp | instance_detection
license: Apache-2.0           # required — permissive ONLY (Apache-2.0/MIT/BSD, any casing); AGPL forbidden
architecture: rf-detr         # optional — factory key (default = name)
model_file: rf-detr-base.onnx # required (when no `files:`) — relative path to the .onnx

# License-policy hardening (both OPTIONAL — see "Threat model" below):
sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
                              # optional — hex SHA-256 of the weight file. When set,
                              # the file is hashed at load time and load is REFUSED on
                              # mismatch (binds the declared license to exact bytes).
source_url: https://huggingface.co/your-org/rf-detr   # optional — audited upstream
                              # the weights came from (provenance; optionally checked
                              # against a curated allowlist).

input:
  width: 560                  # required > 0 (unless a preprocess: block gives the size)
  height: 560                 # required > 0
  layout: NCHW                # NCHW | NHWC
  letterbox: false            # squash; true = keep aspect ratio + pad
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]

# Optional — the same preprocessing as data (see "Preprocessing" below). The input.* fields
# above are its legacy aliases: either may be used; where both speak they must agree.
# preprocess:
#   resize: squash
#   size: 560
#   mean: [0.485, 0.456, 0.406]
#   std:  [0.229, 0.224, 0.225]

postprocess:
  type: detr                  # decode hint (detr/sam/...)
  box_format: cxcywh          # cxcywh | xyxy (normalized [0,1])
  conf_threshold: 0.5
  max_detections: 300

labels: coco91.txt            # optional — one class per line (RF-DETR: 91 COCO ids, index 0 = N/A)

runtime:
  prefer: [cuda, cpu]   # EP fallback chain (CPU is always appended last); tensorrt is opt-in:
                                  # `serve --tensorrt` / VISIONSERVE_TENSORRT=1 inserts it before cuda
                                  # valid EPs: tensorrt, cuda, coreml, directml, openvino, cpu
  idle_unload_seconds: 300        # 0 = never auto-unload
  # max_useful_side: 0            # optional client-resize override: 0 = SDKs never shrink
                                  # images; N = shrink to a longer side of N px (absent = derived)
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

# Multi-file models pin per role (OPTIONAL — pin all, some, or none of the roles):
sha256:
  encoder: 2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae
  decoder: fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9

input:
  width: 1024
  height: 1024
  layout: NHWC
  letterbox: false

postprocess:
  type: sam        # mask threshold = logit > 0; mask encoded as column-major RLE

runtime:
  prefer: [cuda, cpu]
  idle_unload_seconds: 300
```

## Field reference

| Field | Meaning |
|-------|---------|
| `name` | required — registry identifier |
| `task` | `detection` / `segmentation` / `open_vocab` / `depth` / `classification` / `embed` / `grasp` / `instance_detection` |
| `license` | required — must be in the permissive allowlist (below) |
| `architecture` | optional — factory key (default = `name`) |
| `model_file` | path to the .onnx — **optional when `files:` is present** |
| `files` | **map role → .onnx path** for multi-session models (e.g. SAM `encoder`/`decoder`). All listed files must exist on disk for the model to be `available`/loadable |
| `sha256` | **optional** — hex SHA-256 content pin. A scalar (single-file model) **or** a role→digest map (multi-file). When present, the weight bytes are hashed at load time and load is **refused on mismatch**. Absent ⇒ no hash check (backward compatible). Per-role pinning is optional: unpinned roles are skipped |
| `sha256_files` | **optional** — `relative/path: digest` pins for files that are not ONNX sessions: external weight data (`model.onnx.data`), tokenizers (`tokenizer.json`, `vocab.txt`), labels. Checked at load like `sha256`. In verified mode an unpinned `<session>.data` next to a session file is refused, because that file holds most of the weights |
| `source_url` | **optional** — the audited upstream the weights came from (provenance). Purely informational unless a deployer enables the verified-source allowlist (see threat model) |
| `input.*` | width/height/layout/letterbox/crop/keep_aspect/multiple_of/normalize — the legacy spelling of the preprocessing (aliases of `preprocess:`) |
| `preprocess` | **optional** — the preprocessing as data: `resize`, `size`/`width`/`height`, `multiple_of`, `no_upscale`, `resample`, `mean`, `std`, `rescale`, `layout`, `pad` (see [Preprocessing](#preprocessing)) |
| `postprocess.type` | decode hint (`detr`, `sam`, ...) |
| `postprocess.box_format` | `cxcywh` / `xyxy` |
| `postprocess.conf_threshold` | confidence threshold (for GroundingDINO this is the **box** threshold). Not read by classification (`efficientnet`, `mobilenet-v3`): it always returns the top K. Refused by `owlvit` (a load error): its requests are all template-prompted, and their threshold is `instance.sim_threshold` |
| `postprocess.text_threshold` | **GroundingDINO only** — a second minimum on the box's best-phrase score (kept only above both thresholds). It does not change labels: VisionServe names a box with one whole prompt phrase |
| `postprocess.max_detections` | cap on returned detections (classification: K of the top-K, default 5) |
| `labels` | optional labels file (one class per line) |
| `detector` / `segmenter` | **`grasp` architecture only** — composition: `segmenter` picks the mask backbone (default `mobile-sam`); `detector` is optional (e.g. `rf-detr`, `grounding-dino`) for class-aware grasps, omitted for class-agnostic ones. Their graphs are referenced by role in `files:` |
| `grasp` | **`grasp` architecture only** — `gripper_min` / `gripper_max`: default jaw-opening bounds in original-image pixels (a request may override them) |
| `explain` | **optional** — enables `/api/explain` heatmaps: `type` (`attention` or `score_cam`), `outputs` (`attention` → attention output name, or `features` → feature-map output name), `role` (PipelineModel: which role's graph has them), `spatial_stride` (default 32), `top_channels` (Score-CAM, default 64). Absent ⇒ the model does not support explain |
| `instance` | **optional** — one-shot / template detection (OWL-ViT): `max_templates` (≥ 0, 0 = no limit), `sim_threshold`, `patch_size` (≥ 0, 0 = the model's default). Absent ⇒ the model accepts no template prompts |
| `instance.sim_threshold` | default minimum score of a template match, in [0, 1); a request's `box_threshold` overrides it. Absent or 0 ⇒ the model's default: **0.9** for `owlvit`, the threshold of Hugging Face's image-guided example (`post_process_image_guided_detection(threshold=0.9)`). An image query's score is a raw sigmoid, not calibrated: OWLv2's text-query value 0.1 keeps weak background matches |
| `runtime.prefer` | EP fallback chain (NVIDIA `tensorrt`/`cuda`, Apple `coreml`, Windows `directml`, Intel `openvino`, `cpu`). Write `[cuda, cpu]` for NVIDIA: the operator turns TensorRT on for the whole server with `--tensorrt` / `VISIONSERVE_TENSORRT=1`, which inserts `tensorrt` before `cuda`. List `tensorrt` here only if the model was measured under it. `VISIONSERVE_EP` replaces this chain |
| `runtime.idle_unload_seconds` | idle auto-unload (0 = never) |
| `runtime.max_useful_side` | **optional** — overrides the client-resize hint `GET /api/models` publishes for this model (`max_useful_side` / `max_useful_short_side`), which the Python and JS SDKs use to shrink large photos before uploading them. Absent ⇒ derived from the architecture's preprocessing: 2 × the larger input side, on the longer side for fit-inside modes (`letterbox`, `top_left_pad`, `long_side`, `long_side_pad`) and on the shorter side for fill modes (`squash`, `center_crop`); none for `keep_aspect`, `none`, and every architecture that has not registered one (masks, OCR, depth-aligned grasping, templates, crop namers — see `models.RegisterUsefulSide`). `N` > 0 ⇒ SDKs may shrink an image until its **longer** side is `N` px; `0` ⇒ never shrink (always full resolution). Set `0` for a model whose output needs the photo's full resolution, or a larger `N` for one that reads fine detail (small text, far objects) |
| `runtime.threads` | **optional** — map role → ONNX Runtime intra-op threads for that role's session(s), e.g. `threads: {head: 1}`. Overrides the default for that role only: ORT's own (one spinning thread per physical core) for a lone session, the pool cap (`NumCPU/(2n)`, 1 to 3, `VISIONSERVE_POOL_THREADS`) for each session of a pool. `0` = ORT's default, explicitly. A value above the host's logical CPUs is capped at load (logged once). Roles not listed keep the default. Results do not depend on it; use it for a small session that runs between a large one's calls, whose default thread pool otherwise competes with the large one on CPU |

## Validation rules (the registry rejects violations)

| Field | Constraint |
|-------|------------|
| `name` | required; matches `^[A-Za-z0-9][A-Za-z0-9._-]*$` and is at most 128 characters (it becomes a directory name, so `../x` or `/abs` is refused) |
| `license` | must be ∈ {Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause}, matched case-insensitively (`apache-2.0` from an HF model card passes); the canonical SPDX spelling is stored back on the manifest. **AGPL is strictly forbidden**, in any casing. |
| `task` | ∈ {detection, segmentation, open_vocab, depth, classification, embed, grasp, instance_detection} |
| `model_file` / `files` | at least one required — `model_file` OR a non-empty `files:` map |
| `input.width/height` | > 0 (may be omitted when `preprocess:` gives `size` or `width`/`height`). At **load** time the tensor the preprocessing produces must fit the ONNX graph's input: every dimension fixed in both (size, layout, channels) must be equal, or the load fails naming the field and both shapes. Dynamic graph dims, and dims that vary per image (`keep_aspect`, `long_side`, `none`), are not compared |
| `input.layout` | NCHW / NHWC (or empty) |
| `input.crop` / `letterbox` / `keep_aspect` | `crop` is `center` or omitted; at most one of the three |
| `preprocess` | optional; each field valid for its `resize` mode (table below), `mean`+`std` together with 3 values each; a field also set through its legacy `input.*` alias must agree with it — the error names both |
| `runtime.prefer` | each EP ∈ {tensorrt, cuda, coreml, directml, openvino, cpu} |
| `runtime.threads` | optional; needs a `files:` map, every key must be one of its roles, every value an integer >= 0 (a float such as `1.5` is refused, not truncated) |
| `runtime.max_useful_side` | optional; an integer >= 0 (`1.5` or a string is refused) |
| `instance` | optional; `sim_threshold` in [0, 1) (`NaN` refused), `max_templates` and `patch_size` >= 0 |
| `sha256` | optional; if present, must be a hex string or a role→hex map. Mismatch is rejected at **load** time, not scan time (weights may not be downloaded yet) |
| `sha256_files` | optional; paths must be relative and stay inside the model directory; digests are hex |
| `source_url` | optional; if the verified-source allowlist is enabled, must start with an audited prefix |

A manifest that is invalid in **structure** is **skipped** during scan (collected into a
warning) and does not crash the server.

A key that no field above reads (e.g. the typo `runtime.idle_unload_second`) is **not** an
error — third-party manifests may carry extra keys — but the parser ignores it, so the intended
setting keeps its default. The scan therefore loads the manifest and logs a warning naming each
such key with its line: `registry: manifest …/manifest.yaml: unknown key(s) ignored, check for a
typo: runtime.idle_unload_second (line 14)`. Keys *inside* maps (`files:` roles, `sha256:` roles,
`sha256_files:` paths, `explain.outputs`, `runtime.threads`) are data and never reported.

## Preprocessing

How an image becomes the model's input is **data, not code**: one implementation
(`internal/vision/preprocess`) turns a *spec* into the tensor and the mapping back to the
original image, for every model whose preprocessing it can express. `/api/preprocess` returns
exactly what it feeds, and the converter's reference preprocessing
(`clients/python/visionserve/convert/spec.py`) implements the same spec, so its tier B1 compares
the server with the declared spec directly.

A manifest declares the spec in an optional `preprocess:` block, or through the legacy `input.*`
fields, which are its aliases:

```yaml
preprocess:
  resize: letterbox        # see the modes below (default: the architecture's)
  size: 640                # width = height = 640; or width: / height:
  multiple_of: 14          # keep_aspect: round sides to it; long_side_pad: pad sides up to it
  no_upscale: true         # long_side / long_side_pad: never enlarge
  resample: bilinear       # bilinear | bicubic (default: the mode's)
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  rescale: true            # false: keep 0..255 (no x/255); mean/std are then in 0..255 units
  layout: NCHW             # NCHW [1,3,H,W] | NHWC [1,H,W,3] | HWC [H,W,3]
  pad: 0                   # letterbox/top_left_pad: pixel gray level; long_side_pad: tensor value
```

Values: `v = (p/255 - mean[c]) / std[c]` per RGB channel (`p` = 0..255; no mean/std: `p/255`).
With `rescale: false`: `v = p`, or `v = (p - mean[c]) / std[c]` with mean/std in 0..255 units.

| `resize` | Geometry (exactly the upstream recipe) | Tensor size | Default resample |
|---|---|---|---|
| `squash` | resize to width×height, aspect not kept (RF-DETR, MiDaS, EfficientNet, SAM2) | width×height | bilinear |
| `letterbox` | fit inside width×height keeping the aspect (scale `min(W/w, H/h)`, sides rounded half up), centred, padded with the **pixel** `pad` (default black) before normalisation | width×height | bilinear |
| `center_crop` | short side → its target (long side truncated), centred crop — HF CLIPImageProcessor | width×height | bicubic |
| `keep_aspect` | HF DPTImageProcessor `keep_aspect_ratio`: both axes scaled by whichever of W/w, H/h is closer to 1, sides rounded (half to even) to `multiple_of`; no crop, no pad | varies per image (dynamic H/W graph) | bicubic |
| `long_side` | scale `min(W/w, H/h)` (long side → target), sides rounded half away from zero, no pad (MobileSAM: the graph pads) | varies per image | bilinear |
| `long_side_pad` | `long_side`, then the **normalised** tensor is padded bottom/right with `pad` (SAM: normalise, then zero-pad) to width×height, or up to multiples of `multiple_of` | width×height or multiples | bilinear |
| `top_left_pad` | InsightFace SCRFD: fit by the aspect ratios (new size truncated), pasted at the top-left of a **pixel** `pad` canvas. Boxes map back by each axis's own scale (new_w/w, new_h/h), not upstream's single `det_scale` | width×height | bilinear |
| `none` | the original image size (EfficientSAM: the graph resizes itself) | the image's | — |

`multiple_of` is only valid with `keep_aspect` / `long_side_pad`, `no_upscale` only with
`long_side` / `long_side_pad`, a non-zero `pad` only with the padding modes (pixel pads in
[0, 255]), `resample` not with `none`.

### Legacy aliases and precedence

| Legacy field | Alias of |
|---|---|
| `input.width` / `input.height` | `preprocess.width` / `height` (`size` sets both) |
| `input.letterbox: true` | "keep the aspect ratio and pad": agrees with `letterbox`, `top_left_pad`, `long_side_pad`; `false` agrees with every other mode. On its own it means `letterbox` |
| `input.crop: center` | `resize: center_crop` |
| `input.keep_aspect: true` | `resize: keep_aspect` (`false` rules it out) |
| `input.multiple_of` | `preprocess.multiple_of` (carried for `keep_aspect` / `long_side_pad`) |
| `input.normalize.mean` / `.std` | `preprocess.mean` / `std` |
| `input.layout` | `preprocess.layout` |

- **No block:** the legacy fields are the spec (none of letterbox / crop / keep_aspect = `squash`),
  read the way each architecture always read them (below).
- **Block:** every field it sets wins; every field it omits comes from its legacy alias; a field
  set on **both** sides must agree, or the manifest is refused with an error naming both
  (`input.letterbox: false conflicts with preprocess.resize: letterbox`). An *explicit*
  `letterbox: false` is a declaration; an absent key is not. Without `resize` (and no legacy
  geometry flag) the architecture's default mode applies.
- The registry writes what a block declares back into the `input.*` fields it leaves empty, so
  code reading those (and `visionserve list`) sees the same values.
- **Old servers** ignore `preprocess:`. To stay loadable by them, keep the legacy fields next to
  the block (they must agree); the converter writes only the legacy fields unless asked
  (`Bundle.render_manifest(preprocess_block=True)`).

### What each architecture accepts

| Architecture | Modes (first = default) | Notes |
|---|---|---|
| `rf-detr`, `rt-detr` (and the RF-DETR stage of composites) | `squash`, `letterbox` | boxes map back through the meta; crops / variable sizes are refused |
| `efficientnet`, `mobilenet-v3` | `squash` | |
| `midas`, `depth-anything-v2` | `squash`, `keep_aspect` | `keep_aspect` needs dynamic H/W in the graph (checked at load) |
| `clip` | `squash`, `center_crop` | CLIP mean/std (and 224) when not declared |
| `scrfd` | `top_left_pad` | its legacy `letterbox: true` always meant this; its legacy `normalize` is in 0..255 units (a block says `rescale: false`) |
| `mobile-sam`, `nano-sam`, `sam2`, `efficient-sam`, `paddle-ocr` (det) | fixed by the export | `long_side` raw HWC / `long_side_pad` / `squash` / `none` / `long_side_pad` to multiples of 32 with `no_upscale` (`input.width` = PaddleOCR's max side); the `input` block is reference only — also when grasp, background or grounded-sam build MobileSAM from their own manifest. A `preprocess:` block in their own manifest is a load error (it would be ignored) |
| `grounding-dino`, `owlvit`, `siglip-*`, PaddleOCR rec crops | model-specific | special geometry (text-conditioned, padding + blur, per-detection crops); the normalise step is the shared one |

A mode an architecture does not list is a **load error** when declared in `preprocess:`; when it
only comes from a legacy flag the architecture never read (e.g. `crop: center` on `rf-detr`), it
is ignored as it always was, and `input.layout` is never read for these architectures (they feed
NCHW).

## License-safety: threat model & hardening

VisionServe enforces a **permissive-license allowlist at model-load time** (Apache-2.0 /
MIT / BSD only; AGPL strictly rejected — see CLAUDE.md). This is **load-time license policy
enforcement, hardened by content hashing + a verified source**, *not* a cryptographic or
machine-checkable proof of the license itself.

**What the base check does:** it rejects any manifest whose declared `license` is not on the
permissive allowlist, before the model can be served.

**The relabeling hole (and the fix).** A bare allowlist only checks *the string the author
typed* — an AGPL model could be relabeled `Apache-2.0` and pass. The optional `sha256` field
closes this hole at the **byte** level: when a digest is declared, the weight file is hashed
with `crypto/sha256` (pure Go, streamed, no RAM blow-up) after it is located/downloaded, and
**load is refused on mismatch**. This *binds the declared license to specific, audited weight
bytes*: the license claim now refers to an exact artifact, so swapping in different weights
under the same Apache-2.0 manifest is caught.

**Verified-source allowlist (opt-in).** `source_url` records the audited upstream the bytes
came from. By default it is informational. A deployer can populate
`registry.VerifiedSourcePrefixes` (e.g. the official RF-DETR / MobileSAM / GroundingDINO
repos) to additionally **require** every loaded model's `source_url` to begin with a curated,
human-audited prefix. This is a local string-prefix policy gate — it performs **no network
fetch** and keeps VisionServe local-first; it is empty (a no-op) unless explicitly configured.

**What it does NOT guarantee.** It still *trusts the declared license string* for the audited
artifact — it does not parse the upstream LICENSE file, does not consult HuggingFace model-card
metadata, and is not a signed/transparency-logged attestation. It guarantees: *these exact
bytes* (sha256) *came from this declared origin* (source_url), and *the human who curated the
allowlist / declared the digest vouched that the origin is Apache-2.0/MIT/BSD.* Stronger,
optional layers (HF model-card cross-check, Sigstore/OpenSSF model signing, SPDX/CycloneDX
AI-BOM + a real license scanner) are future work and are not required to load a model.

**Backward compatibility.** Both fields are **optional**. A manifest that declares neither
behaves exactly as before: the license string is allowlist-checked, and weights load without a
hash or source check.

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

## Installing a local model — `pull <folder>` (the "Modelfile" path)

To register your **own** model (e.g. a fine-tuned RF-DETR) without editing the
built-in catalog, point `pull` at a **local folder** instead of a catalog name —
the CV equivalent of `ollama create`:

```bash
# the folder must contain manifest.yaml + the .onnx file(s) (+ labels, if any)
visionserve pull ./rf-detr-mycustom

# inside a running container: copy the folder in first, then pull it
docker cp ./rf-detr-mycustom visionserve:/tmp/rf-detr-mycustom
docker exec -it visionserve visionserve pull /tmp/rf-detr-mycustom
```

`pull` **auto-detects** the argument: anything that is (or looks like) a directory
path is installed from disk; a bare catalog name is still downloaded from
HuggingFace. On success the folder is **copied** into `<modelsDir>/<manifest.name>/`,
so it lives alongside the other models and survives container restarts (it is part
of the mounted models volume). Use `--force` to overwrite an existing model of the
same name.

A fine-tuned model normally only changes the **weights and class list**, so it
**reuses an existing `architecture`** (e.g. `architecture: rf-detr`) and ships its
own `labels:` file (one class per line, in the training-index order).

### What is validated before install (static checks — no ONNX session is opened)

`pull <folder>` rejects the folder with a clear error if any of these fail:

| Check | Failure |
|-------|---------|
| `manifest.yaml` present | missing → rejected |
| Manifest structurally valid | bad `license` (AGPL → rejected), `task`, input dims, or EP chain |
| `architecture` is a built-in factory | unknown architecture → rejected (lists the valid ones) |
| Referenced `.onnx` / `labels` exist on disk | missing weights → rejected |
| Paths stay inside the folder | a `../escape.onnx` reference → rejected (must be self-contained) |
| Name not already taken | exists → rejected unless `--force` |

> Note: the **`sha256` content check runs at load/predict time**, not at install — install
> only performs the static checks above (no ONNX session and no hashing). A pinned digest that
> does not match the bytes is caught when the model is first loaded into memory.

> A custom architecture (different output shape / decode) that no built-in factory
> handles still requires a new Go package under `internal/models/<name>/` +
> `models.Register()` and a rebuild — see [contributing-models.md](contributing-models.md).
> `pull <folder>` only wires up models that reuse an existing architecture.
