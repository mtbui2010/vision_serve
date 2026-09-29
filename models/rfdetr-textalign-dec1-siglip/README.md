# rfdetr-textalign-dec1-siglip — open-vocabulary detection distilled from SigLIP

Head B on the 17-class held-out `dec1` backbone, distilled from **SigLIP** on a **mixed** image
pool (in-domain ETRI + COCO). Served with `method: gated`.

Two changes from [`../rfdetr-textalign-dec1`](../rfdetr-textalign-dec1), neither of them to the
head's architecture, and both measured before being made
([`paper/FINDINGS-2026-08.md`](../../paper/FINDINGS-2026-08.md) §9–§10).

## 1. The teacher was the ceiling

Head B is a student of whatever produced its distillation targets and cannot beat that teacher on
words it has never seen. On five names held out of **every** trained component:

| | held-out 5 | base 17 | COCO macro |
|---|---|---|---|
| CLIP-crop | 65.3 | 72.4 | 66.4 |
| **SigLIP-crop** | **85.3** | 78.8 | **73.7** |

So `P` is 768×256 here rather than 512×256. **The serving code needed no change** — `d_text` is a
field of `proj.bin`, and `internal/models/textalign/dim_test.go` exercises Fold, ProjNorm and the
gated decode at 512, 768 and 1152. What did need writing was the tokenizer: SigLIP uses
SentencePiece Unigram where CLIP uses byte-level BPE (see `../siglip-text/README.md`).

## 2. A head is only good where it was distilled — in both directions

| head B distilled on | ETRI base 17 | ETRI unseen 5 | COCO top1 | COCO macro |
|---|---|---|---|---|
| COCO crops, CLIP teacher | 54.1 | 46.7 | 80.3 | 77.2 |
| in-domain crops only, SigLIP | 73.9 | **66.7** | 38.0 | **20.5** |
| **mixed, SigLIP (this model)** | **81.6** | 57.3 | **85.0** | **78.3** |
| *CLIP-crop, the baseline it replaces* | *72.4* | *65.3* | *57.7* | *66.4* |
| *SigLIP-crop, the teacher* | *78.8* | *85.3* | *58.5* | *73.7* |

Distilled only in-domain the head **collapses to 20.5 macro on COCO**. Mixing recovers it. This
model is above CLIP-crop on three of the four columns — and its COCO macro is above the SigLIP
teacher's own — at one matmul against a ViT per crop.

**The ratio matters more than the amount.** Proportional weighting at 10 000 COCO images lets
COCO outweigh ETRI 20:1 and unseen names fall to 48.0; at 2 000 it is the best row in the table.

### Two shipped projections

| file | recipe | when to prefer it |
|---|---|---|
| `proj.bin` | n_coco 2000, `prop` | the general default — the table above |
| `proj-openvocab.bin` | n_coco 5000, `balanced` | ETRI unseen 64.0, base 78.1, COCO macro 72.2 — when unseen-name accuracy matters more than COCO breadth |

Swap by renaming; the loader reads `proj.bin`.

## Measured end to end, through the Go server

| | p50 | p90 | p99 | VRAM |
|---|---|---|---|---|
| bare closed detector | 57.8 ms | 109.0 | 133.1 | — |
| CLIP head B, gated | 55.9 ms | 103.2 | 130.3 | 1588 MiB |
| **this model, gated** | **57.1 ms** | 97.5 | 116.5 | **2610 MiB** |

**The heavier text tower costs nothing on the hot path.** It runs only on a vocabulary cache
miss: a new vocabulary costs ~146 ms once, and every request afterwards is ~37 ms. (The absolute
numbers above are higher than the sibling's published 38.2 ms because three models shared one
server and one GPU with other tenants; the comparison between rows is the measurement.)

The `gated` invariant holds at 768-d exactly as at 512-d — same image, three different
vocabularies, **identical confidences**:

```
no prompt (17 base names)         cup 0.98 · remote 0.97
"towel. hat. bread. beer can."    beer can 0.98 · bread 0.97
"keyboard. cola can. laptop."     book 0.98 · keyboard 0.97
```

Selection never consults head B, so the vocabulary cannot move a box or a score. Note the second
line: offer five unrelated names and every kept box is forced onto the nearest one. That is the
documented closed-choice behaviour of the method, not a failure of this model.

## Files

| file | what |
|---|---|
| `detector.onnx` | the 17-class held-out fine-tune, `query_feats` appended **last** (not committed) |
| `proj.bin` / `proj-openvocab.bin` | `P` [768,256] + `(a, b)`, 786 464 bytes each |
| `labels.txt` | **the detector's own 17 names + `N/A`** — see below |
| `vocab-all22.txt` | all 22 names, for pasting into a prompt |
| `templates.txt` | the prompt ensemble the head was trained against |

`labels.txt` **must** be the detector's own file. `gated` indexes it to find the background
column *and* matches its length against the class tensor's width to locate that tensor at all; a
22-name file makes `classLogits` fail to find the head. That is how it was caught here. The
consequence is that the no-prompt fallback vocabulary is the 17 base names — the other five are
what a prompt is for.

## Known gaps

- `conf_threshold` is inherited from the sibling (0.35, tuned there for a 22-class detector) and
  has **not** been re-tuned for this one. Re-run `scratchpad/dec1_deploy/threshold_gated.py`.
- No end-to-end mAP for this model; the table above is naming accuracy on matched rows.
- The ETRI-unseen column rests on 75 rows throughout.
