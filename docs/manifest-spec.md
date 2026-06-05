# Manifest spec ("Modelfile for CV")

Each model is a subdirectory in the registry (`./models/<name>/`) containing a
`manifest.yaml` + the ONNX file(s) (not committed) + (optional) a labels file.

## Full example (single-session model)

```yaml
name: rf-detr                 # required — model identifier in the registry
task: detection               # required — detection | segmentation | open_vocab | depth | classification | embed
license: Apache-2.0           # required — permissive ONLY (Apache-2.0/MIT/BSD); AGPL forbidden
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
                                  # valid EPs: tensorrt, cuda, coreml, directml, openvino, cpu
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
  prefer: [tensorrt, cuda, cpu]
  idle_unload_seconds: 300
```

## Field reference

| Field | Meaning |
|-------|---------|
| `name` | required — registry identifier |
| `task` | `detection` / `segmentation` / `open_vocab` / `depth` / `classification` / `embed` |
| `license` | required — must be in the permissive allowlist (below) |
| `architecture` | optional — factory key (default = `name`) |
| `model_file` | path to the .onnx — **optional when `files:` is present** |
| `files` | **map role → .onnx path** for multi-session models (e.g. SAM `encoder`/`decoder`). All listed files must exist on disk for the model to be `available`/loadable |
| `sha256` | **optional** — hex SHA-256 content pin. A scalar (single-file model) **or** a role→digest map (multi-file). When present, the weight bytes are hashed at load time and load is **refused on mismatch**. Absent ⇒ no hash check (backward compatible). Per-role pinning is optional: unpinned roles are skipped |
| `source_url` | **optional** — the audited upstream the weights came from (provenance). Purely informational unless a deployer enables the verified-source allowlist (see threat model) |
| `input.*` | width/height/layout/letterbox/normalize |
| `postprocess.type` | decode hint (`detr`, `sam`, ...) |
| `postprocess.box_format` | `cxcywh` / `xyxy` |
| `postprocess.conf_threshold` | confidence threshold (for GroundingDINO this is the **box** threshold) |
| `postprocess.text_threshold` | **GroundingDINO only** — threshold for assigning text tokens to a detected box (open-vocab label gating) |
| `postprocess.max_detections` | cap on returned detections |
| `labels` | optional labels file (one class per line) |
| `runtime.prefer` | EP fallback chain (NVIDIA `tensorrt`/`cuda`, Apple `coreml`, Windows `directml`, Intel `openvino`, `cpu`) |
| `runtime.idle_unload_seconds` | idle auto-unload (0 = never) |

## Validation rules (the registry rejects violations)

| Field | Constraint |
|-------|------------|
| `name` | required, non-empty |
| `license` | must be ∈ {Apache-2.0, MIT, BSD-3-Clause, BSD-2-Clause}. **AGPL is strictly forbidden.** |
| `task` | ∈ {detection, segmentation, open_vocab, depth, classification, embed} |
| `model_file` / `files` | at least one required — `model_file` OR a non-empty `files:` map |
| `input.width/height` | > 0 |
| `input.layout` | NCHW / NHWC (or empty) |
| `runtime.prefer` | each EP ∈ {tensorrt, cuda, coreml, directml, openvino, cpu} |
| `sha256` | optional; if present, must be a hex string or a role→hex map. Mismatch is rejected at **load** time, not scan time (weights may not be downloaded yet) |
| `source_url` | optional; if the verified-source allowlist is enabled, must start with an audited prefix |

A manifest that is invalid in **structure** is **skipped** during scan (collected into a
warning) and does not crash the server.

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
