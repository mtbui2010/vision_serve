# rfdetr-textalign-dec1 — open-vocabulary detection at the bare detector's latency

Head B (a text-aligned projection into CLIP space) on `dec1`: RF-DETR small fine-tuned on the
22 ETRI classes with the **backbone and box head frozen and only the last decoder layer left
trainable**.

Two things distinguish it from the sibling [`../rfdetr-textalign-etri`](../rfdetr-textalign-etri):
the backbone kept its generic objectness, and the deploy path is `gated` rather than
`exact`/`folded`.

## Why `gated` exists — the fold is exact for naming and wrong for ranking

Head B scores a query with `s_c = a⟨t̂_c, Pf⟩/‖Pf‖ + b`. Folding `W = a·T̂P` drops the
normalisation to get one small matmul. Writing `ρ = ‖Pf‖`, the folded score obeys an identity:

```
s_fold(c) − b  =  ρ · (s_exact(c) − b)        (verified to 4.8e-6, float32 noise)
```

`ρ > 0` is **one positive scalar shared by every class**, so it cannot reorder the classes of a
query — the fold is *provably* exact for naming, and measured identical on 100.0000% of 358
held-out ETRI rows. But `ρ` **differs between queries**, so it does reorder queries against each
other: Spearman 0.713, and only 19 of the top 100 queries survive. The threshold moves too
(`s > −b` becomes `s > −b/ρ`), which predicted all 382 ETRI disagreements exactly.

`internal/models/rfdetr`'s postprocess takes both the name and the score from one tensor. So
`gated` splits the job along the line the algebra draws: **the detector's own class head
supplies the score used for thresholding and ranking, head B supplies only the name.** The
cheap path is then also the correct one. See `internal/models/textalign/gated.go`.

A direct consequence, visible in any two requests: changing the vocabulary changes the labels
and leaves the confidences bit-identical, because selection never consults head B.

```
--prompt (none)                     remote 0.974 | cup 0.964 | towel 0.912
--prompt "cola can. keyboard. towel."  keyboard 0.974 | cola can 0.964 | towel 0.912
```

## Measured, end to end, through the Go server

62 held-out ETRI images, one server process, conditions interleaved. mAP is COCO-style via
pycocotools against human boxes, taken through the `-probe` copy (`conf_threshold` 0.001) so
the precision/recall curve is complete.

| system | val mAP | mAP50 | AR100 | p50 | p99 | VRAM |
|---|---|---|---|---|---|---|
| closed 22-class `rfdetr-small-etri` | **69.97** | 95.05 | 75.72 | 38.2 ms | 92.3 | 795 MiB |
| **this model, `gated`** | **56.10** | 63.42 | **72.56** | **38.2 ms** | 121.0 | 1588 MiB |
| `../rfdetr-textalign-etri` (COCO-91 backbone) | 54.18 | 74.94 | 63.28 | 71.7 ms | 132.0 | 1885 MiB |
| this model, `exact` | — | — | — | 72.2 ms | 161.0 | 1588 MiB |

**Open vocabulary is free at inference here.** `gated` has the same p50 as the bare closed
detector — 38.2 ms both — because the folded head is 5 632 MACs per query against the
detector's own cost, and the `exact` path's extra `P·f` is what doubles it to 72.2 ms.

**Against the previous open-vocabulary configuration** it gains 1.9 mAP and 9.3 points of
AR100 at half the latency, and the backbone underneath keeps **97.6** class-agnostic recall on
21861 COCO novel boxes where the full fine-tune keeps 68.1 (`paper/FINDINGS-2026-08.md` §5).

**The honest weak spot is mAP50: 63.42 against the sibling's 74.94**, while overall mAP is
higher. Higher AR100 with lower mAP50 says the boxes are there and the *naming* is what fails
at loose IoU — consistent with head B on this backbone being 12 points better on ETRI overall
but the four cans still sitting at 66.7%. Not yet diagnosed further.

`conf_threshold` is 0.35, tuned on these images (F1 0.768 versus the sibling's 0.730); see the
manifest for the sweep and for why the number is not transferable between methods.

## Files

| file | what |
|---|---|
| `detector.onnx` | `dec1` re-exported with `query_feats` appended **after** `labels` (not committed) |
| `proj.bin` | `P` [512,256] + `(a, b)` = 524 320 bytes. The only trained tensor |
| `labels.txt` | the detector's own labels, **in the detector's order** — `gated` indexes the class tensor by these positions to find the `N/A` background column |
| `templates.txt` | prompt ensemble for the text tower |
| `head.onnx` | `proj.bin` exported as an ONNX graph (`files.head`, with `runtime.threads: {head: 1}`): method `exact` on ONNX Runtime, same outputs as the Go head. **Generated locally, not committed, not on the HF catalog** — `python3 models/rfdetr-textalign-dec1-siglip/export_head_onnx.py --proj models/rfdetr-textalign-dec1/proj.bin`; re-export whenever `proj.bin` changes. See [`../rfdetr-textalign-dec1-siglip/README.md`](../rfdetr-textalign-dec1-siglip/README.md#headonnx--method-exact-on-onnx-runtime) |

`P`'s effective rank is 32.5 of 256 (top 64 singular directions carry 95.2% of the energy), so
there is room to compress it for smaller edge targets; not attempted.

## Reproduce

Head B: `scratchpad/dec1_headb/` — `extract_eval.py` → `ceiling_probe.py` → `choose_k.py` →
`etri_extract.py`/`etri_targets.py` → `run_extract.sh 0 30000 6` → `train_qf.py --n-coco 30000
--steps 6240 --lam-dec 4 --lam-norm 1`. One trap: `cocolib.py`'s `REAL_COLS` must be
`slice(0, 22)` for this detector (`N/A` is the **last** column, not column 0 as in the COCO-91
export) or the objectness ranking that selects which queries get distilled is silently wrong.

`proj.bin` is written from `heads/dec1_30k/headB.npz` with flags = 0: this `P` was **not**
trained with the `(‖Pf‖−1)²` penalty, and `ρ ∈ [0.773, 1.471]` — which is exactly why the
deploy path is `gated` and not `folded`.
