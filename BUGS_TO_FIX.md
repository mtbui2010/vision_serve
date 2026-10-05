# Bugs to fix in vision_serve

Handover for a future session. Written 2026-09-29 from the experiments in the sibling repo
`../ovd-edge`, which served its protocol through this binary and found the problems below.
Nothing in this file was discovered by reading code alone; every number comes from a served run
on the held-out-names protocol (247 tabletop images, 17 trained names and 5 held-out names).

Primary sources, read them before changing anything:

- `../ovd-edge/docs/FINDINGS.md` §2b, §7b, §15, §16, §17
- `../ovd-edge/docs/TOKENIZER_AUDIT.md`
- `../ovd-edge/tools/vision_serve_changes/` (README, `PREPROCESSING_MISMATCH.md`,
  `siglip_tokenizer_edit.md`, `tracked_files.patch`)

## Read this first: the working tree is not clean, and not all of it is yours

`git status` shows modified tracked files and many untracked ones (the whole `siglip` package,
`hybrid/rescore.go`, `hybrid/fastpath.go`, `promptens/`, several model directories). Some of that
is the ovd-edge work, some belongs to the repo owner and was never committed.

- **Do not `git add -A` and commit.** That bundles someone else's unfinished work into a commit
  that describes yours, and destroys the record of what was uncommitted.
- Stage only the files you changed for a given fix, by path.
- `tracked_files.patch` from ovd-edge is **already applied** (`git apply --check -R` passes).
  Do not apply it again.

## Status at a glance

| # | problem | cost | status |
|---|---|---:|---|
| 1 | RF-DETR served letterboxed, trained squashed | **7.35 mAP** (512 group), 1.91 (560 group); 3.16 on `rf-detr-nano`; 43.25 on `rt-detr` (stand-in weights); 2.27 on `grasp-rfdetr`'s detector | **fixed 2026-09-29** (served +7.13); `rf-detr-nano`, `rt-detr` and `grasp-rfdetr` **fixed 2026-10-05** |
| 2 | `cropTemp = 0.02` documented as the measured vertex; it is not | 1.67 mAP | **fixed 2026-09-29** (router default 0.05, +1.53 served) |
| 3 | TensorRT preferred on 4 manifests whose numbers were measured on CUDA | 6.91 mAP (rfdetr-gdino open branch); 6.83 on standalone GroundingDINO | **fixed 2026-09-29**: owner moved EVERY model to `[cuda, cpu]`; TensorRT is now opt-in per process (`--tensorrt` / `VISIONSERVE_TENSORRT=1`) |
| 4 | Fast-path head is a 59 M-MAC scalar Go loop | 19 ms per request | **fixed 2026-09-29** (head on ORT, ~36 ms/request saved) |
| 5 | SigLIP tokenizer padded with `<pad>` (id 0) instead of `</s>` (id 1) | 4.3 mAP | fixed, keep the test |
| 6 | Fast-path `conf_threshold` leaked into the shared RF-DETR postprocess | control moved +2.17 | fixed, keep it in code |
| 7 | `Fold` result multiplied by `Scale` a second time | 9.2 mAP | fixed, keep the test |

## Resolution log, 2026-09-29

All numbers below are SERVED (Go server, CUDA EP unless stated, held-out-names protocol, 247
images), harness `ovd-edge/experiments/{holdout_names_sweep,map_eval}` driven by a copy of
`ship_verify.evaluate`. Baseline reproduced first: letterbox 82.62 / 49.54, rescored 82.62 / 60.94.

**#1.** `letterbox: false` on every manifest serving a checkpoint trained through `rfdetr`
(its default `square_resize_div_64=True`; VisionForge's `train_rfdetr.py` and ovd-adapt's
`train_variant.py` do not override it): the 512 family (`*-dec1*`, `*-etri*`, `rfdetr-small*`,
`rfdetr-textalign-*`, `rfdetr-gdino-fastpath`) and the 560 family (`rf-detr`, `rfdetr-gdino{,-sam,-siglip}`),
plus the catalog entries `rf-detr`, `rfdetr-gdino{,-sam}` and every new one. **Not flipped**,
as instructed: `scrfd`, `rt-detr`, `grasp-rfdetr` — and `rf-detr-nano` (upstream nano, never
measured). `grasp-rfdetr` serves the same `rf-detr-base-real` checkpoint as `rf-detr`, so it is
the obvious next candidate once its grasp outputs are re-checked.

| served | 17-trained | 5-held-out |
|---|---:|---:|
| letterbox | 82.62 | 49.54 |
| squash | **89.75** | 49.54 (open branch unmoved) |

89.75 against the local 89.97: Go's bilinear resize vs PIL's. Box mapping under ScaleX != ScaleY:
`TestPostprocessSquashNonSquareRoundTrip` (400x100 image) and checked by eye on a 848x153 strip
through the pulled `rfdetr-gdino-siglip-etri`. The letterbox branch's `TODO(verify)` is replaced
by a comment saying why RF-DETR manifests do not use it. **560 group re-measured on all 5000
val2017 images** (`ovd-edge/experiments/preproc_coco.py --n 5000`, in-process ORT CUDA):
letterbox 40.88, squash 42.56, **+1.68** (the 500-image +1.91 was slightly high). Quote +1.68.

**#1, follow-up 2026-10-05: `rf-detr-nano`.** The same bug on the one RF-DETR detector left
letterboxed above ("upstream nano, never measured"). Measured with `visionserve check rf-detr-nano
--checkpoint ~/.roboflow/models/rf-detr-nano.pth --images <dir> --labels <json>`: the official COCO
checkpoint run by `rfdetr` itself (`RFDETR.predict`) as the reference, Go server on CPU (ORT 1.26),
the first 200 val2017 images by id (1460 boxes), both sides at conf 0.5:

| `rf-detr-nano` served | B1 (gray levels) | B2 boxes matched | C mAP served / rfdetr | verdict |
|---|---:|---:|---:|---|
| `letterbox: true` (as shipped until now) | 46.6 | 31 / 36 | 40.92 / 44.09 (**-3.16**) | FAIL |
| `letterbox: false` | 0.4 | 38 / 38 | 43.80 / 44.09 (-0.29) | PASS |

Flipped in `models/rf-detr-nano/manifest.yaml` and the catalog entry (`internal/catalog/catalog.go`).
Existing installs: a manifest an older `pull` generated and nobody edited is rewritten by a plain
`visionserve pull rf-detr-nano` ("updated manifest.yaml (generated by an older catalog)", checked
with the binaries before and after the fix; `TestRFDETRDetectorsSquashAndOldManifestsAreRegenerated`);
a hand-edited one is kept (`pull --force` regenerates it).

**#1, follow-up 2026-10-05: `rt-detr`, same class of bug, worse.** Its manifest letterboxed and
applied ImageNet mean/std; RT-DETR is trained and evaluated squashed to 640x640 and scaled to
[0, 1] with no mean/std (`RTDetrImageProcessor`: `do_normalize: false`; the original repos'
eval transforms: Resize [640, 640] + ToTensor, Paddle `keep_ratio: False`, mean 0 / std 1). The
weights the manifest names (onnx-community/RT-DETR-l-hf) are gone (401), so it was measured on
the only RT-DETR checkpoint available locally, PekingU/rtdetr_r50vd (exported with `visionserve
convert hf`, tier A 4.3e-5), placed under the shipped manifest, against transformers' own
pipeline, same 200 images:

| `rt-detr` served (rtdetr_r50vd weights) | B1 | B2 | C mAP served / transformers | verdict |
|---|---:|---:|---:|---|
| letterbox + ImageNet mean/std (as shipped) | 68.9 | 4 / 39 | 7.16 / 50.40 (**-43.25**) | FAIL |
| squash, mean 0 / std 1 (now) | 0.2 | 41 / 41 | 50.34 / 50.40 (-0.06) | PASS |

Flipped in `models/rt-detr/manifest.yaml` and the catalog entry (which cannot be pulled anyway).
Separate commit, so it can be dropped if the stand-in weights are not considered enough evidence.

**Audit 2026-10-05: `visionserve check` on every shipped detection / classification / depth /
embedding model with a reference available locally.** CPU server (ORT 1.26) on a scratch registry
holding the repo's manifests and weights; B1/B2 on 8 COCO val2017 CC BY photos (the ids in
`website/docs/assets/img/CREDITS.md`), C on the 200-image COCO subset above. B1 is the mean
difference in gray levels against the reference preprocessing (more than 8 costs accuracy).

| model | reference | B1 | B2 | C mAP served / ref | verdict | action |
|---|---|---:|---|---|---|---|
| `rf-detr-nano` (was letterbox) | `rf-detr-nano.pth` | 50.0 | 36/41 boxes | 40.92 / 44.09 | FAIL | **fixed** (squash) |
| `rf-detr-nano` (squash) | `rf-detr-nano.pth` | 0.38 | 42/42 | 43.80 / 44.09 | PASS | — |
| `rf-detr` | `rf-detr-base.pth` | 0.41 | 41/41 | 47.62 / 47.78 | PASS | — |
| `rfdetr-small` | `rf-detr-small.pth` | 0.40 | 44/44 | 47.45 / 47.85 | PASS | — |
| `rfdetr-small-qf` | `rf-detr-small.pth` | 0.40 | 44/44 | — | PASS | — |
| `rfdetr-small-etri`, `-etri-qf`, `-etri-probe` | rfdetr recipe (fine-tuned checkpoint not local) | 0.40 | — | — | PASS (B1 only) | — |
| `rt-detr` (was letterbox + ImageNet) | rtdetr_r50vd stand-in | 78.5 | 11/40 | 7.16 / 50.40 | FAIL | **fixed** (squash, no mean/std) |
| `rt-detr` (now) | rtdetr_r50vd stand-in | 0.23 | 43/43 | 50.34 / 50.40 | PASS | — |
| `depth-anything-v2` | Depth-Anything-V2-Small-hf | 0.27 | Pearson r ≥ 0.9994 | — | PASS | — |
| `clip` | `CLIPImageProcessor` (openai/clip-vit-base-patch32) | 0.24 | cosine ≥ 0.9990 | — | PASS | — |
| `siglip-image`, `-fp16` | `SiglipImageProcessor` (google/siglip-base-patch16-224) | 0.24 | cosine ≥ 0.9995 | — | PASS | — |
| `efficientnet-b0` | timm eval transform (crop_pct 0.875, bicubic) | **40.5** | top-1 agrees 4/8 | not measured | FAIL | reported, not fixed |
| `mobilenet-v3` | torchvision `IMAGENET1K_V1.transforms()` (resize 256, crop 224) | **39.9** | top-1 agrees 3/8 | not measured | FAIL | reported, not fixed |
| `midas` | manifest only (no reference) | 0.22 | — | — | server code only | — |
| `scrfd` | manifest only (no reference) | 22137 | — | — | false FAIL (check bug) | reported |
| `grasp-rfdetr` (was letterbox) | served detections vs COCO GT (`check` cannot run grasp) | — | — | 45.50 (`rf-detr` 47.77) | FAIL | **fixed** (squash) |
| `grasp-rfdetr` (squash) | same | — | — | 47.77 (= `rf-detr`) | PASS | — |

- **Classifiers.** The two ONNX files come from timm (`conv_stem`/`blocks` initializers) and
  torchvision (`features.N.block`); both eval transforms resize then centre-crop, the manifests
  squash the whole photo. The reference scripts load no weights (B2 runs the installed ONNX on
  both tensors), and there is no labelled ImageNet set here for C, so the cost is not measured:
  not fixed. Fix candidate: `crop: center` with the right resize ratio, after a top-1 run.
- **`scrfd`: a bug in `check`, not in the manifest.** `check.py` builds its manifest reference with
  the generic `spec_from_manifest`, which reads SCRFD's legacy `letterbox: true` as a centred
  letterbox and its 0..255 mean/std as 0..1 units; Go resolves both per architecture
  (`top_left_pad`, `NoRescale`). The report then tells the user to set a field that is already
  set. Fix: resolve the architecture's legacy rules on the Python side too (shared fixture).
- **Not run:** `paddle-ocr` (no reference), `owlvit-base`, `grounding-dino(-fixed)` and the
  open-vocab pipelines (`check --checkpoint` takes HF object detection only for task
  `detection`), the text towers, the segmentation models and the grasp models other than `grasp-rfdetr` (below).
- **`grasp-rfdetr`: fixed (squash).** It letterboxed its RF-DETR stage, which serves the same
  `rf-detr-base-real` weights as `rf-detr` (squash, PASS above). `check` cannot run a grasp model,
  so its detections were scored directly: served (CUDA, ORT 1.26), the same 200 images, conf 0.5:

  | `grasp-rfdetr` detector | mAP / mAP50 | detections | grasp centres on a GT mask of their class |
  |---|---:|---:|---:|
  | `letterbox: true` (as shipped) | 45.50 / 56.83 | 984 | 86.1% (16864 / 19577) |
  | `letterbox: false` (now) | **47.77 / 59.25** | 1003 | **87.6%** (17505 / 19986) |
  | `rf-detr` (reference, same weights) | 47.77 / 59.25 | 1003 | — |

  Squashed, its detections are `rf-detr`'s box for box (identical mAP and count). Grasp spot check,
  three photos: #139 (living room) keeps the same ten objects with near-identical boxes and top
  grasps, and adds a second person (GT IoU 0.87), the dining table (0.89) and one potted plant
  with no ground truth; #9448
  (person under an umbrella) loses two phantom `donut` detections (40 grasps on nothing);
  #3661 (banana, cup, keyboard) letterboxed found only the banana with a box half its true size
  (GT IoU 0.48), squashed finds the banana (IoU 1.00), the cup (0.94) and the keyboard (0.44), all
  with grasps. The grasp search itself is unchanged; better boxes give better masks and grasps.
  Manifest, catalog entry (re-pull regenerates an unedited manifest:
  `TestRFDETRDetectorsSquashAndOldManifestsAreRegenerated`), note in `reference/models.md`. No
  other shipped manifest embeds an RF-DETR detector with `letterbox: true` (`grasp-gd` uses
  GroundingDINO; `rf-detr-nano` only mentions it in a comment).

**#2.** Confirming sweep on the router after #1, rescored entry, `crop_temp` per request:

| T | 0.02 | 0.03 | 0.05 | 0.07 | 0.1 |
|---|---:|---:|---:|---:|---:|
| 5-held-out | 60.94 | 62.01 | **62.47** | 61.92 | 60.90 |

17-trained 89.75 at every T. `hybrid/rescore.go::cropTemp` is now 0.05 with the table in its
comment; the served default (no `crop_temp` field) was re-measured at 62.47. `textalign/crophead.go`
keeps 0.02 but no longer calls it a vertex: its sweeps were pre-#5 and that head was not re-swept.

**#3.** First decision: keep TensorRT and state its numbers; **superseded the same day** — the
owner moved every manifest in `models/` and every catalog entry to `prefer: [cuda, cpu]`.
For the record, the measurements behind both decisions: The four manifests now state what that
delivers instead of the CUDA figures: `[cuda, cpu]` 89.75 / 49.54 at 177.7 ms against
`[tensorrt, …]` 89.90 / 42.59 at 117.6 ms (−6.95 on the open branch, 1.5x faster). With the
rescorer, TensorRT gives 57.82 (T=0.02) / 59.20 (T=0.05) against CUDA's 62.47. The
`*-siglip*` entries stay on CUDA. Standalone `grounding-dino` (and its alias entry
`grounding-dino-fixed`), by contrast, was **switched to `[cuda, cpu]`** at the owner's request:
same weights, CUDA 45.25 / 44.61 (1959 boxes) against TensorRT 44.79 / 37.78 (1716 boxes).
Two different numbers, do not mix them up: the "6.8 mAP lower on GroundingDINO" quoted in
CLAUDE.md, the README and `visionserve --help` is this standalone figure (44.61 − 37.78 = 6.83,
153 vs 104 ms); the −6.91 in the open item below is the rfdetr-gdino open branch (FINDINGS §7b,
−6.95 after #1). The GroundingDINO manifest also records that TensorRT rebuilds its engine for
every new prompt length.

Since then TensorRT is opt-in for the whole process, not only per manifest: `serve --tensorrt` /
`run --tensorrt` or `VISIONSERVE_TENSORRT=1` inserts `tensorrt` before `cuda` in every chain
(`internal/engine/provider.go`), and the default stays CUDA → CPU.

**#4.** `files.head: head.onnx` (from `models/rfdetr-gdino-fastpath/export_head_onnx.py`) runs the
head through the Runner / `lifecycle.Manager` on the detector's provider; `head.bin` stays as the
fallback when the role is absent. Equivalence: ORT CPU vs the Go path 1.43e-6 max abs on the
shipped head (`TestHeadONNXMatchesGoORT`), plus a fake-Runner test that needs no ORT. Served,
letterbox, same run order: 82.62 / 44.39 at ~193 ms (Go) against 82.62 / 44.38 at ~157 ms (ORT).
Still open from §15: the fast-path manifest also loads GroundingDINO's session although the head
answers every unknown word, and the 22-box CPU crop stage is untouched.

## Open (as handed over; see the resolution log above)

### 1. Train/serve preprocessing mismatch on RF-DETR (largest single gain available)

**What is wrong.** RF-DETR manifests declare `letterbox: true`. The `dec1_holdout` detector was
trained on squashed square images: the fine-tune recipe sets `square_resize_div_64=True`, and
rfdetr's own `predict` does `F.resize(img, [res, res])` with no padding.

**Measured** (same weights, same postprocess, only the resize changed; FINDINGS §17):

| group | checkpoint | letterbox | squash | Δ |
|---|---|---:|---:|---:|
| 512: `*-dec1*`, `*-etri*` | `dec1_holdout` | 82.62 | 89.97 | **+7.35** |
| 560: `rfdetr-gdino{,-sam,-siglip}` | `rf-detr-base-real` (COCO-91) | 44.08 | 45.99 | +1.91 |

The letterbox column reproduces the served 82.62 exactly, which confirms the server really
letterboxes.

**The fix is config only.** Set `letterbox: false` in the RF-DETR manifests. No code change is
needed: `internal/models/rfdetr/preprocess.go` already has a squash branch
(`imageproc.Resize` + per-axis `ScaleX`/`ScaleY`), and `postprocess.go` already divides by each
axis separately. That branch has simply never been selected. The `TODO(verify)` about pad colour
on the letterbox branch can be resolved by this change: there should be no padding at all.

**Scope, carefully.** `grep -l "letterbox: true" models/*/manifest.yaml` returns 25 manifests,
but only the RF-DETR ones were measured. **Do not flip `scrfd`, `rt-detr`, or `grasp-rfdetr`**
on the strength of this note; their training preprocessing was never checked. For each manifest
you change, confirm which checkpoint it serves and that the checkpoint was trained squashed.

**Verify.**
1. The GroundingDINO open branch must not move (it does its own preprocessing): held-out stays at
   its previous value within ±0.5.
2. The 17-trained column on the 512 group should rise to about 89.97.
3. Every returned `Detection.BBox` must still land in original image coordinates. Under squash
   `ScaleX != ScaleY` for non-square images, which is the path that was never exercised. Test on a
   strongly non-square image and check the boxes by eye, not only by mAP.
4. Re-measure the 560 group on more than 500 val2017 images before quoting its number.

Harness: `../ovd-edge/experiments/ship_verify.py` (served) and `local_closed_squash.json` for the
expected values.

### 2. `cropTemp` comment claims a vertex that is not the vertex

`internal/models/textalign/crophead.go` sets `const cropTemp = 0.02` and calls it "a measured
vertex". Every sweep behind that comment ran text through the Go SigLIP tower **before** bug 5
was fixed, on embeddings 0.706 cosine away from the correct ones. Re-derived on the shipped router
with correct embeddings (FINDINGS §16):

```
crop_temp   0.005  0.01   0.02   0.03   0.05   0.1    0.2    1.0
held-out    56.58  59.72  62.23  63.48  63.90  62.64  59.85  54.20
```

The vertex is 0.05; 0.02 costs 1.67 mAP. The same constant is used by `hybrid/rescore.go`.

**Fix, in order.** First apply fix 1, then run one confirming sweep over {0.02, 0.03, 0.05},
because the two interact through the router's detection budget. Then either change the constant
to the measured vertex and rewrite the comment to cite that run, or keep 0.02 and delete the word
"vertex". Do not leave the current comment: its argument rests on the tokenizer bug.

### 3. TensorRT preferred where the quoted numbers were measured on CUDA

*Resolved 2026-09-29 (resolution log above): every manifest is `[cuda, cpu]`, and TensorRT is
opt-in with `--tensorrt` / `VISIONSERVE_TENSORRT=1`. The text below is the original hand-over.*

`models/rfdetr-gdino{,-etri,-sam,-sam-etri}/manifest.yaml` ship
`prefer: [tensorrt, cuda, cpu]` while their comments quote accuracy measured on CUDA. TensorRT
measured **−6.91 mAP** on the open branch (FINDINGS §7b). The newer `rfdetr-gdino-siglip`
manifests already prefer CUDA and explain why.

This changes the recommended default's execution provider, so it is **the repo owner's
decision**; the ovd-edge session only annotated these files. If the owner agrees, switch to
`[cuda, cpu]` and re-quote the numbers; if not, the quoted numbers must be re-measured under
TensorRT so the manifest stops claiming accuracy it does not deliver.

### 4. Fast-path head runs as scalar Go

`internal/models/hybrid/fastpath.go` loads `head.bin` and scores 300 queries in Go. Offline this
head is 0.063 ms on a GPU in torch; served it was 56 ms, and 19 ms after the `Fold` rewrite.
Scalar Go is the wrong home for a 300 × 768 × 256 kernel.

**Fix.** Export the head as ONNX and run it through ONNX Runtime on the same provider as the
detector (rule 2 of `CLAUDE.md`: all inference goes through ONNX Runtime). The second cost
identified in §15 is the crop stage at 22 boxes per image against the rescorer's 3.4, all
Catmull-Rom resized on the CPU.

**Context before spending time here.** As built the fast path is dominated on both axes
(44.39 mAP at 228.7 ms against the router's 60.94 at 207.7 ms). Fixing the latency does not fix
the 16.55 mAP gap. Treat this as lower priority than 1 and 2.

**Rule this item leaves:** a latency claim is not a claim until it is served. Time the whole
request, not the component.

## Fixed: keep these guards in place

### 5. SigLIP tokenizer padding (4.3 mAP, silent)

`internal/models/siglip/tokenizer.go` padded with the `<pad>` piece (id 0). The checkpoint's
`tokenizer_config.json` declares `pad_token = "</s>"` (id 1). SigLIP pools with attention, so it
reads the padding: embeddings came out 0.706 cosine from the reference, the argmax flipped on
9.9 % of crops, 29/849 boxes fell below the rejection floor. Nothing errored.

Now `padID = eosID`, and `TestEncodePaddedPadsWithEOSNotPadPiece` asserts the filler. It went
undetected because `testdata/tokens.json` stores **unpadded** ids. Do not weaken that test.

Consequences already propagated: the dual head `reg_final/map-ta-a` is 17.98 held-out, not
15.77. Any document still quoting 15.77 is stale.

The CLIP tokenizer was checked and is **not** affected: its tower is causally masked and pools
at the first EOS, so padding is inert (cosine 1.000000). For any new text tower, run
`../ovd-edge/experiments/pad_sensitivity.py` before assuming either way; whether padding matters
is a property of the model's pooling, not of the tokenizer.

### 6. Branch threshold leaking into shared postprocess

The fast-path head needs `conf_threshold ≈ 0.001`; putting it in the manifest made RF-DETR's
supervised head emit at 0.001 as well, and the 17-trained control rose 82.62 → 84.79. The
threshold now lives in code on the head's own sub-model (`ConfThresh = 0`), and the manifest keeps
0.35 for the closed branch. Do not move it back into the manifest. A control column that moves,
in either direction, is a bug.

### 7. `Scale` applied twice after the `Fold` optimisation

`textalign.Fold` already multiplies by `Scale` (`tk := t[k] * p.Scale`); the new fast-path code
multiplied again, so every logit was 10.28× too large and held-out fell 44.39 → 35.18. Fixed (see
the `NO Scale here` comment in `fastpath.go`), and `TestFoldedScoringMatchesDirect` now checks the
folded path against the direct one. An optimisation without an equivalence test is a rewrite.

## Also known, not a vision_serve bug

The residual 1.22 mAP between the served rescorer (60.94) and the Python reference (62.16) is
accounted for: about 40 near-ties resolve differently between the Go Catmull-Rom crop resize and
PIL bicubic (mean probability ratio 0.997, argmax agreement 96.2 %). `ship_verify.py` tolerates
±1.5 for that comparison. Do not chase it.
