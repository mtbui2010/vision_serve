---
license: apache-2.0
library_name: onnx
pipeline_tag: zero-shot-object-detection
tags:
  - onnx
  - grounding-dino
  - open-vocabulary-detection
  - zero-shot-object-detection
  - visionserve
base_model: IDEA-Research/grounding-dino-tiny
---

# grounding-dino-tiny-fixedmask-ONNX

An ONNX re-export of [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
whose **block-diagonal text self-attention mask is rebuilt correctly for any number of class
phrases**, so a multi-class prompt such as `"chair. tv. vase."` is scored correctly in a
**single** forward pass.

License: **Apache-2.0** (same as the upstream weights — only the ONNX graph differs).

| file | size | notes |
|---|---|---|
| `model-fixedmask.onnx` | 694.8 MB (662.6 MiB), opset 17 | fp32, single graph, weights embedded |
| `vocab.txt` | 231 508 bytes | BERT-base-uncased tokenizer vocab (unchanged from upstream) |

`sha256(model-fixedmask.onnx) = ae9a0026953c6d5ce5a97b421af84c065d7f07173971cb105edb0071868fc180`

## Why this export exists

GroundingDINO's BERT text branch needs a **block-diagonal attention mask** so the tokens of
one class phrase cannot attend to another phrase's tokens. Hugging Face builds that mask
*inside* the model from `input_ids`
(`generate_masks_with_special_tokens_and_transfer_map`) — it is not a graph input, so any
ONNX export has to reproduce it.

In `transformers` 4.48 that function is a **Python `for` loop over
`torch.nonzero(special_tokens_mask)`**, and `torch.onnx.export` bakes the trip count of that
loop into the graph. The widely used community export
[`onnx-community/grounding-dino-tiny-ONNX`](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX)
was traced with a **one-class** prompt (`[CLS] … . [SEP]` = 3 special tokens), so its graph
contains exactly three hard-coded iterations. The signature is visible in the graph:

```
input_ids -> Equal(101) / Equal(102) / Equal(1012) / Equal(1029)
          -> Or  ->  NonZero  ->  Transpose
          -> Gather(axis 0, index 0), Gather(index 1), Gather(index 2)
          -> 6 ScatterND   (attention mask + position_ids, per iteration)
```

At runtime, only the **first** `.`-separated phrase ever receives its attention block. Every
later phrase keeps just the identity (`EyeLike`) row — each token attends to itself alone —
with `position_id` 0. The consequence is that **multi-class prompts silently drop classes and
are order-dependent**. Measured on COCO `000000000139.jpg`, HF PyTorch returns the same 15
detections for every reordering of a 12-class prompt, while a raw joint pass through the
community ONNX returns 4 / 0 / 0 / 3 detections depending on the ordering.

Nothing downstream can repair that graph: the only text inputs are `input_ids` /
`attention_mask` / `token_type_ids`, all shaped `[1, L]`, and `attention_mask` is consumed
purely as the padding mask — it can never carry a block-diagonal `[L, L]` mask. The only
workaround with the community weights is to run the session **once per phrase**, which is
correct but costs N passes for N classes.

## What was changed

1. **Vectorized, data-independent mask.** `generate_masks_with_special_tokens_and_transfer_map`
   is monkeypatched with an equivalent that has no Python-level loop over the token positions,
   so nothing gets baked in. `transformers >= 5` already vectorizes this function, but that
   version does not export — the legacy exporter rejects `aten::isin`, and ONNX has no
   `CumMax` / `CumMin`. The substitutions are:
   - `isin` -> an `Equal` / `Or` chain,
   - `cummax` / `cummin` -> an `L x L` compare plus `ReduceMax` / `ReduceMin`
     (`L <= max_text_len = 256`, so the cost is negligible),
   - `torch.eye` -> `(i == j)` (ORT has no `EyeLike` kernel for bool).
2. **Export settings.** `torch.onnx.export` (legacy exporter, `dynamo=False`), opset 17,
   `dynamic_axes` on `sequence_length` for the three text inputs, pixel inputs fixed at
   `[1,3,800,800]` / `[1,800,800]` to match the community graph, and
   `config.disable_custom_kernels = True` so deformable attention traces. The trace uses a
   **two-class** example precisely so a baked single-phrase loop could not slip through
   unnoticed; the resulting mask is then verified at L = 4, 6, 8, 12 and 26 tokens.
3. **float64 -> float32 demotion.** The legacy exporter type-promotes mixed int64/float32
   arithmetic to `DOUBLE`. In `GroundingDinoEncoder.get_reference_points`,
   `ref_y / (valid_ratios[...] * height)` has `height` as a traced int64 spatial-shape tensor,
   and the resulting double flows into the deformable-attention sampling grid. ONNX Runtime
   then refuses to load the model with *"Could not find an implementation for
   GridSample(16)"*. PyTorch itself runs this in float32, so demoting every `Cast(to=DOUBLE)`
   and every double constant is a faithful correction (15 cast attributes and 20 attribute
   tensors in this graph).

Input and output names are **identical** to the community export, so this file is a drop-in
replacement for existing loaders.

## I/O contract (verified)

```
inputs:
  pixel_values      [1, 3, 800, 800]  float32   ImageNet-normalized image
  pixel_mask        [1, 800, 800]     int64     1 for valid pixels
  input_ids         [1, L]            int64     BERT token IDs (dynamic L)
  attention_mask    [1, L]            int64     1 for real tokens, 0 for padding
  token_type_ids    [1, L]            int64     all zeros (single segment)

outputs:
  logits            [1, Q, L]         float32   per-query, per-token scores
  pred_boxes        [1, Q, 4]         float32   cxcywh, normalized to [0, 1]
```

`Q = num_queries = 900`; `L` is the tokenized prompt length (`max_text_len = 256`). The
output ranks are declared dynamically in the graph, so a loader must read them from the
returned tensor rather than assume them.

Post-process: `sigmoid(logits)`, box score = max over the valid text tokens, filter by a box
threshold (0.3 is a good default) and a text threshold (0.25) for assigning label words to a
box, then decode `cxcywh` to original-image `[x, y, w, h]`.

Preprocessing is the standard GroundingDINO one: resize to 800x800, ImageNet mean
`[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`, NCHW, no letterbox. The prompt is
lowercased and dot-separated (`"chair. tv. vase."`).

## Equivalence to HF PyTorch

On the same pixel tensor and the same `input_ids`, against `transformers` PyTorch:

- **identical detection lists on all five test prompts** (1, 2, 3, 6 and 12 classes),
- max `|Δlogit|` **7.3e-3**, max `|Δsigmoid|` **1.1e-4** over the valid token range for the
  12-class prompt.

For the 12-class COCO prompt this graph returns exactly HF PyTorch's 15 detections
(chair 7, clock 1, laptop 1, potted plant 2, tv 2, vase 2). The per-phrase workaround on the
community weights returns 17 — over-counting vases and missing `laptop`, because per-phrase
inference never lets the fusion layers see the other classes.

Permuting the class order changes scores by at most **3.529e-2** — and HF PyTorch drifts by
exactly the same 3.529e-2. That residual order sensitivity is inherent to GroundingDINO (the
fusion and decoder layers attend over all text tokens with only the padding mask), not an
artifact of this export. "Order-stable" here means *the same detections*, not bit-identical
coordinates.

## Speed

Measured through the VisionServe HTTP server (warm, model resident, median of 3), ONNX
Runtime 1.26.0 with the **CUDA execution provider, no TensorRT**, on an NVIDIA RTX A6000.
12-class prompt on COCO `000000000139.jpg`. **The GPU was shared with other tenants**, so
these are ranges rather than single figures:

| weights | passes | latency |
|---|---|---|
| `onnx-community/grounding-dino-tiny-ONNX` (per-phrase workaround) | 12 | 3860 – 4166 ms |
| `model-fixedmask.onnx` (joint, this repo) | **1** | **227 – 285 ms** |

Roughly **15x** end to end. A single-phrase pass costs ~320 ms on CUDA, which is also about
what the joint path costs for the *whole* prompt — the win is entirely in doing one pass
instead of N. TensorRT is expected to be faster still; no TensorRT number is quoted because it
was not available on the measurement host.

## Usage

### With VisionServe

[VisionServe](https://github.com/) is a single Go binary that serves CV models locally
(Apache-2.0, ONNX Runtime, no Python at runtime).

```bash
visionserve pull grounding-dino-fixed
visionserve run grounding-dino-fixed img.jpg --prompt "chair. tv. vase."
```

### With onnxruntime directly

```python
import numpy as np, onnxruntime as ort
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
sess = ort.InferenceSession("model-fixedmask.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

inputs = proc(images=image, text="chair. tv. vase.", return_tensors="np")
logits, boxes = sess.run(
    ["logits", "pred_boxes"],
    {
        "pixel_values": inputs["pixel_values"].astype(np.float32),
        "pixel_mask": inputs["pixel_mask"].astype(np.int64),
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": inputs["token_type_ids"].astype(np.int64),
    },
)
```

## License and attribution

**Apache-2.0.** The weights are unmodified from
[`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
(Apache-2.0); only the exported computation graph differs. Please cite the original
GroundingDINO work:

```bibtex
@article{liu2023grounding,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```
