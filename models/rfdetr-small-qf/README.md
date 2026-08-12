# rfdetr-small-qf — RF-DETR small (COCO-91) + `query_feats`

Same weights as [`../rfdetr-small`](../rfdetr-small), re-exported with one extra ONNX output:

| output | shape | note |
|---|---|---|
| `dets` | `[1, 300, 4]` | unchanged |
| `labels` | `[1, 300, 91]` | unchanged |
| `cross_attn_weights` | `[3, …]` | unchanged |
| **`query_feats`** | **`[1, 300, 256]`** | **new** — decoder features feeding the class head |

## Why this backbone and not the ETRI fine-tune

There is a sibling model, [`../rfdetr-small-etri-qf`](../rfdetr-small-etri-qf), built the same
way from the 22-class ETRI fine-tune. It is the wrong base for an open-vocabulary head, and the
comparison is unusually clean: `onnx.load` shows the two graphs are **node-for-node identical**
(1754 nodes, same op histogram, same `input [1,3,512,512]`), differing only in classifier width
(91 vs 23). So the numbers below isolate the fine-tune itself.

Class-agnostic query coverage — the fraction of human-GT boxes covered by at least one of the
300 object queries, class label ignored:

| | ETRI (955 out-of-vocabulary boxes) | COCO val2017 (21861 novel boxes) |
|---|---|---|
| **rfdetr-small (COCO-91, this model)** | 100 / **99.9** / 99.3 | 99.8 / **98.3** / 90.6 |
| rfdetr-small-etri (22-class) | 100 / **100** / 97.4 | 90.4 / **68.1** / 32.8 |

(@IoU 0.3 / 0.5 / 0.7. A shuffled-query null control sits at 15–24% @0.5, so none of this is an
artifact of having 300 boxes to choose from.)

Three things that follow, each measured:

- **The fine-tune cost 30.2 points** of IoU-0.5 recall on objects outside its vocabulary; 9.4 of
  those are objects it fails to cover at all, ~20.8 are objects it covers loosely (found at
  IoU 0.3, too loose for 0.5).
- **Its localization became class-conditional.** Restricted to boxes of 1–5% image area, the
  fine-tune scores 77.4% on its own 22 categories versus 55.3% on novel ones — a 22-point split
  by category alone. These COCO-91 weights split by 0.1 points. A box regressor is not supposed
  to know what it is looking at.
- **The features lost most of their content.** A supervised linear probe with COCO labels on the
  frozen `query_feats` reaches **89.0% macro** here versus **8.7%** on the fine-tune, on the
  identical held-out rows both models localize. For reference, the same probe on CLIP's own
  embedding of the same crops reaches 63–67% — these features are richer than CLIP's by 22
  points. (Caveat: COCO-91 was trained on these categories, so this shows the information is
  linearly decodable, not that it generalizes to unseen concepts.)

The price is paid in the other direction: a linear head on these frozen features classifies the
22 ETRI classes at 86.7% versus 98.6% for the fine-tune, and that gap is an **information** limit,
not a capacity one — an RBF-kernel SVM reaches 86.8% and a 9M-parameter MLP reaches 88.3%, while
all of them reach 97–98% on the fine-tuned features. The errors concentrate in the four cans
(`sprite can` ↔ `cola can`), a distinction COCO-91 never had reason to encode.

## Equivalence — verified, not assumed

Against `../rfdetr-small/model.onnx` on real images: `max|Δ|` is exactly **0.0** on `dets`,
`labels` and `cross_attn_weights`, and `labels == query_feats @ class_embed.weight + bias` to
**0.0** error. Produced by graph surgery, not retraining: the tensor was already internal
(`labels` is literally `class_embed.bias + MatMul(features, W[256,91])`), so promoting the MatMul
input to a graph output adds no compute and no weights.

## Output order is load-bearing

[`internal/models/rfdetr/postprocess.go`](../../internal/models/rfdetr/postprocess.go) takes the
first output whose last dimension is 4 as the boxes, and the **first remaining one** as the class
logits. `query_feats` must stay **after** `labels`. Emitted earlier it would be silently taken for
the logits — no error, just wrong results. Preserve the order
`dets, labels, cross_attn_weights, query_feats`.

Note the shared tensor feeds **both** the class head and the box head, so a script that looks for
"the first MatMul consuming it" will find the box head (a `[256,256]` weight) instead. Select the
`class_embed` node by name.

## Reproduce

`scripts/export_rfdetr_queryfeats.py` in the scratchpad recipe, or equivalently: load
`../rfdetr-small/model.onnx`, append an `Identity` from
`/model/transformer/decoder/norm/LayerNormalization_output_0` to a new output `query_feats`, and
save. Weights are not committed (110 MB, `*.onnx` is ignored); `manifest.yaml` pins the sha256.

## Publishing so `visionserve pull` can fetch it

Same recipe as [`../rfdetr-small-etri-qf/README.md`](../rfdetr-small-etri-qf/README.md): upload
`model.onnx` + `labels.txt` to a public HuggingFace repo, add `source_url` to `manifest.yaml`, and
add a catalog entry. Deliberately not added yet — `visionserve list` would advertise a pull that
fails until the repo exists.
