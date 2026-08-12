# rfdetr-textalign-etri — text-aligned (open-vocabulary) head on a frozen RF-DETR

An open-vocabulary detector that costs almost nothing on top of a closed-set one. The frozen
[`../rfdetr-small-etri-qf`](../rfdetr-small-etri-qf) detector exposes the 256-d feature `f_i`
of every object query (`query_feats`). One trained matrix `P` (512×256) maps it into CLIP text
space, and a class score is a cosine against the CLIP text embedding of the class name:

```
logit_ic = a · ⟨ t̂_c , P f_i / ‖P f_i‖ ⟩ + b        conf = sigmoid(logit)
```

Because that is bilinear, once the vocabulary is known the CLIP text tower and `P` collapse
into ONE small matrix — exactly the shape and role of RF-DETR's original `class_embed`:

```
W = a · T̂ P        [C, 256]
logit_ic = ⟨W_c, f_i⟩ / ‖P f_i‖ + b
```

**Changing the vocabulary swaps a small matrix. There is no ONNX re-export, no second
detector, and no Python at runtime** (the CLIP BPE tokenizer is pure Go).

`proj.bin` here is a **trained** projection (`headB.npz`, 2026-08-11): 131 k parameters,
supervised by CLIP text anchors on 22 ETRI class names plus label-free CLIP region
distillation over the detector's own proposals.

## Use

```bash
visionserve run rfdetr-textalign-etri img.png --prompt "cup. cola can. unicorn."
curl -X POST localhost:11435/api/predict \
     -F model=rfdetr-textalign-etri -F image=@img.png \
     -F 'prompt=cup. cola can. unicorn.' -F method=exact
```

* `prompt` — the vocabulary, "."-separated (the CLIP / GroundingDINO convention). **Omit it**
  and the vocabulary is `labels.txt` of the frozen detector (22 ETRI classes).
* `method` — `exact` (default) keeps the `‖P f‖` normalisation, so `conf_threshold` keeps the
  meaning it had at training time. `folded` drops it, making the head a literal `nn.Linear`:
  ~30 ms cheaper per request, the SAME boxes and (measured) the same labels on 74099/74100
  queries, but a slightly different score calibration — see the fold section below.
* `/api/explain` works, via the `explain` role (the un-stripped export), loaded lazily.

The vocabulary really is free text, not an enum. On an image whose annotations say
`remote` / `towel` / `coffee`, `--prompt "television remote control. rolled towel. unicorn."`
returns `television remote control` (0.505) and two `rolled towel` — phrases that appear in no
labels file and were never trained on — and no `unicorn`.

## What it is worth — measured, on this dataset, through this server

RTX A6000 · ONNX Runtime 1.26 CUDA EP (no TensorRT) · ETRI `etri_simple`: 247 usable images /
1447 human boxes / 22 classes. **Accuracy is on the 62 images head B held out at training
time.** Latency is p50/p99 over 100 warm requests per condition, conditions interleaved.

| system | mAP@[.5:.95] | mAP@.5 | p50 | p99 | resident VRAM | open-vocab? |
|---|---|---|---|---|---|---|
| `rfdetr-small-etri` (closed-set, 22 classes) | **70.0** | **95.1** | **31.8 ms** | 69.1 ms | **794–796 MiB** | no |
| **`rfdetr-textalign-etri`, `folded`** | 54.2 | 74.9 | **37.0 ms** | 46.2 ms | 1866–1904 MiB | **yes** |
| `rfdetr-textalign-etri`, `exact` | 54.2 | 74.9 | 61.6 ms | 94.3 ms | 1866–1904 MiB | **yes** |
| `rfdetr-textalign-etri`, `exact`, 78-name vocabulary | 50.9 | 70.8 | 61.3 ms | 89.5 ms | 1866–1904 MiB | **yes** |
| `grounding-dino-fixed` (the teacher, one net) | 46.1 | 53.7 | 220.2 ms | 366.6 ms | 5404–5680 MiB | yes |
| `rfdetr-gdino` router, in-vocabulary prompt | 70.0 | 95.1 | 31.2 ms | 43.9 ms | 4654–5074 MiB | via gdino |
| `rfdetr-gdino` router, out-of-vocabulary prompt | — | — | 362.1 ms | 627.0 ms | 4654–5074 MiB | yes |

Latency is the **median of three full runs** taken across a 2 h window on a GPU shared with
another job; per-run values and wall-clock times are in the deployment scratch directory. The
spread is informative: the RF-DETR-based rows move by 2–8 ms between runs, GroundingDINO by
177 ms and the router's GroundingDINO branch by 253 ms — **the one-net head degrades gracefully
under contention and the two-net baseline does not.**

VRAM is the **process-resident** figure (`nvidia-smi` against that server's pid, one model
loaded, after warm requests) — it includes the CUDA context and ONNX Runtime's arena, which is
what a deployment budgets, not the weight file sizes (114 MB / 245 MB / 695 MB). Ranges are
three independent samples; ORT's arena growth is workload-dependent. `mAP` for `folded` and
`exact` is identical because the fold is a positive, class-independent rescale — it cannot
change class order, only calibration. The router row at 31.2 ms uses a `cross_attn_weights`-
stripped RF-DETR leg (see below); as shipped it is 77.9 ms.

Read it as three facts:

1. **Open vocabulary here costs 15.8 mAP, not a second network.** The head runs within ~5 ms
   of the bare detector (`folded`: 37.0 vs 31.8 ms) and needs one 512 KB matrix.
2. **It beats GroundingDINO on this data**: +8.1 mAP@[.5:.95], +21.2 mAP@.5, at ~6× lower
   latency and ~3× lower VRAM. GroundingDINO's boxes are the weak part (mAP@.5 53.7).
3. **The frozen 22-class head is still the best thing for its own 22 classes** (70.0 mAP) and
   it is still loaded. Use the text-aligned head when you need a name it was never trained on.
   Widening the vocabulary from 22 to 78 names costs 3.2 mAP and nothing in latency.

Box-conditioned label accuracy through this server (a detection matched to a human box at
IoU ≥ 0.5 must carry the right name), 40 images: **85.6 %** for the head vs **98.8 %** for the
frozen 22-class head. With the 78-name vocabulary: 79.7 %. The head's failures are structural
(near-synonym CLIP text collapse: `coffee`→`coffee can`, `beer can`→`sprite can`,
`bread`→`orange`), not random.

**Boxes are the frozen detector's boxes, exactly.** Verified through the server with both
models at `conf_threshold: 0` so all 300 object queries come back from each: **4500/4500
boxes over 15 images match at `max|Δ| = 0.0`** on all four components. The head changes the
naming and the ranking, never the geometry.

## Weights

Nothing third-party is duplicated: `manifest.yaml` references the sibling model directories by
relative path and pins their SHA-256.

| role | file | note |
|---|---|---|
| `rfdetr` | `detector-noattn.onnx` (114 MB, here) | the qf export **minus** `cross_attn_weights` |
| `text` | `../clip-text/model.onnx` (+ `vocab.json`, `merges.txt`) | that directory's README |
| `explain` | `../rfdetr-small-etri-qf/model.onnx` | lazy, only for `/api/explain` |
| — | `proj.bin` (512 KB, here) | the trained projection |

### Why a local 114 MB copy of the detector

`lifecycle.Manager` opens every `PipelineModel` role session with **all ONNX outputs bound** —
a plain `Model` can filter the explain tensors out via the manifest `explain:` block, a
pipeline role cannot. The qf export emits `cross_attn_weights [3,1,16,300,1024]` = **59 MB per
inference**, which this head never reads. Measured cost of that one unread tensor, same server,
interleaved, 100 requests each:

| | p50 | p99 |
|---|---|---|
| `files.rfdetr` = the un-stripped qf export | 138.2 ms | 200.2 ms |
| `files.rfdetr` = `detector-noattn.onnx` | **61.6 ms** | **94.3 ms** |

**−76.6 ms p50 / −105.9 ms p99** — larger than the head itself. `detector-noattn.onnx` is the
same graph with one entry removed from `graph.output`; the remaining three outputs are
bit-identical (checked through the server at `conf_threshold: 0`, i.e. all 300 object queries:
4500 detections, `max|Δ| = 0` on bbox and conf, 0 label differences).

The same trap catches the `rfdetr-gdino` router, which is also a `PipelineModel`: pointing its
`rfdetr` leg at the stripped graph took it from 77.9 ms to 31.2 ms p50.

**Tradeoff:** the fast detect path can no longer produce an attention map, so the manifest
declares a third role, `explain`, pointing at the un-stripped export. It is not in the model's
`Roles()`, so it gets **no session at load** — `lifecycle.Manager` opens it lazily on the first
`/api/explain` call. `/api/explain` therefore still works (verified, HTTP 200, 277 KB PNG) and
costs nothing until used; the price is 114 MB of disk and ~700 MB of VRAM while an explain
session is alive.

## `proj.bin` — the only trained tensor

Binary `VSTXALN1` container: a 32-byte header (`d_text=512`, `d_feat=256`, logit scale `a`,
logit bias `b`, flags) followed by `P` as row-major `[512][256]` float32 — i.e. exactly
`torch.nn.Linear(256, 512).weight` in C order. The format is specified in
[`internal/models/textalign/proj.go`](../../internal/models/textalign/proj.go) and written by
`buildProj()` in that package's tests.

The shipped file carries the **trained** calibration `a = 13.7738`, `b = −5.0417` and
`flags` bit 0 (trained with the `(‖P f‖−1)²` penalty).

### Swapping in a new P

One command, from the deployment scratch directory:

```bash
bash <scratchpad>/headb_deploy/run_all.sh /path/to/newP.npz
```

It installs `proj.bin` + `templates.txt`, re-strips the detector, runs `go build/vet/test`,
and re-measures every number in this README (mAP, threshold sweep, label agreement, box
identity, latency, VRAM). Then update `sha256.rfdetr` and `postprocess.conf_threshold` from
its output.

### `conf_threshold` is 0.28, and 0.5 would return nothing

This head's confidences are on **its own** scale: `sigmoid(a·cos + b)` with the trained
`a, b`, where the CLIP-distillation term deliberately keeps the cosine small. Over all 74 100
object queries of the 247 ETRI images the max-class confidence **never exceeds 0.4997**
(p50 0.083, p99 0.344). The frozen detector's 0.5 does not transfer — copying it across yields
an empty response, which looks exactly like a broken model.

F1 on the 62 held-out images (TP = IoU ≥ 0.5 **and** correct label):

| conf_threshold | 0.15 | 0.20 | 0.25 | **0.28** | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|---|---|---|
| precision | 0.490 | 0.609 | 0.663 | **0.720** | 0.752 | 0.869 | 0.955 |
| recall | 0.855 | 0.841 | 0.796 | **0.740** | 0.651 | 0.427 | 0.176 |
| F1 | 0.623 | 0.707 | 0.723 | **0.730** | 0.698 | 0.573 | 0.297 |

Flat optimum; anything in 0.22–0.30 is defensible. The frozen 22-class head at its own 0.5
scores P 0.865 / R 0.894 / F1 0.879 on the same images. **Re-run the sweep after installing a
new `P`** — `a` and `b` are retrained with it.

### `method=folded` — measured, not assumed

`folded` drops the `‖P f‖` divisor, which is what makes the head a literal `[C,256]` linear
layer. Measured on the shipped `P` over **all 74 100 queries of all 247 images**, through the
server:

* label flips exact→folded: **1 / 74 100 (0.0013 %)**;
* max |Δ conf|: 0.187 (it is largest at the decision boundary, by construction);
* at `conf_threshold = 0.28` on the held-out images: F1 **0.723** folded vs **0.730** exact
  (P/R 0.697/0.751 vs 0.720/0.740).

`‖P f‖` over those queries is 0.9953 ± 0.0083, p1 0.982, p99 1.036 — just outside the ±0.01
tolerance that would make `folded` a free default, which is exactly why it is opt-in and
`exact` is the default. If you can absorb ±0.02 of confidence, `folded` is ~25 ms cheaper and
identical in every other respect (same boxes, same mAP, same class ordering).

## Prompt templates

`templates.txt` here holds the **10 templates the head was trained with**. It is part of the
trained head's contract: `T̂` must be built in the same text space `P` was fitted in. One
template per line, `{}` where the class name goes, `#` for comments. The runtime L2-normalises
each template's embedding, averages, and re-normalises. Absent the file the runtime falls back
to the single `"a photo of a {}."`.

Compiled vocabularies (`W`) are cached in-process, keyed by a hash of (templates ‖ class names,
in order), 32 entries, FIFO. **One CLIP text-tower call per NEW vocabulary — never per
request.** A cold vocabulary costs one batched text call (~20 ms for 22 classes × 1 template,
~76 ms for 22 × 10).

## Known limitations

1. **One dataset, one room.** Every number above is in-domain ETRI tabletop. On COCO val2017
   novel categories this same `P` scores 3.7 % box-conditioned top-1 against a CLIP-crop
   baseline of 57.0 % — **it does not transfer across domains.** Treat it as an
   open-vocabulary head *for this scene*, not a general one.
2. **Near-synonym collapse.** `coffee` vs `coffee can` sit at cosine 0.926 in CLIP text space;
   the projection cannot manufacture a margin CLIP does not have. Four of 22 classes are at
   0 % for this reason.
3. **Train/deploy resampling gap.** `P` was fitted on features from Pillow's bilinear resize;
   the server uses `disintegration/imaging`'s. Median |Δconf| 0.0027, max 0.112. Uncorrected.
4. `mAP` here is the *frozen detector's* localization with this head's naming — the boxes are
   identical by construction, so the 15.8 mAP gap to the closed-set head is entirely naming
   and ranking.

## License

RF-DETR export Apache-2.0, CLIP ViT-B/32 text tower MIT — both permissive, both allowed by
the registry's allowlist. No AGPL component anywhere in this pipeline. `proj.bin` is this
repository's own trained artifact.
