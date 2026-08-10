# GroundingDINO (open-vocabulary detection) — weights

> ONNX weight files are **NOT committed** to git. Download them with the instructions
> below (`*.onnx` and `vocab.txt` are `.gitignore`d).

GroundingDINO is a text-prompted, zero-shot object detector. License: **Apache-2.0**
(NOT AGPL). It is fully free and open-source, usable in commercial and closed products.

Given a text prompt such as `"cat. remote."` it returns bounding boxes and labels for
every detected concept — no predefined class list needed.

## How it works

GroundingDINO is a **`PipelineModel`** (text-prompted, single ONNX session):

- Preprocess: resize the image (800 px long side, NCHW, ImageNet mean/std), tokenize the
  text with a BERT-style tokenizer (reads `vocab.txt` from this directory).
- Infer: single ONNX session (`model.onnx`), outputs box predictions + logit scores per
  token.
- Postprocess: filter by `conf_threshold` (box score) and `text_threshold` (token-to-label
  assignment), decode `cxcywh`-normalized boxes to original-image `[x, y, w, h]`.

## Get the ONNX weights

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=grounding-dino     # downloads model.onnx + vocab.txt into this directory
```

### Option B — download manually from Hugging Face

The weights used by VisionServe come from the
[`onnx-community/grounding-dino-tiny-ONNX`](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX)
repository on Hugging Face (Apache-2.0).

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os

dest = "models/grounding-dino"
os.makedirs(dest, exist_ok=True)

for filename in ["model.onnx", "vocab.txt"]:
    p = hf_hub_download(repo_id="onnx-community/grounding-dino-tiny-ONNX",
                        filename=filename)
    shutil.copy(p, os.path.join(dest, filename))
    print(f"Copied {filename}")
PY
```

Both `model.onnx` and `vocab.txt` must be present next to `manifest.yaml` for the model
to be usable.

## Verified I/O contract

```
inputs:
  pixel_values          [1, 3, H, W]   float32   ImageNet-normalized image (long side 800)
  input_ids             [1, L]         int64      BERT token IDs (text prompt)
  attention_mask        [1, L]         int64      1 for real tokens, 0 for padding
  token_type_ids        [1, L]         int64      all zeros (single-segment input)

outputs:
  logits                [1, Q, L]      float32   per-query, per-token scores
  pred_boxes            [1, Q, 4]      float32   cxcywh normalized [0, 1]
```

Post-process: `sigmoid(logits)` → box score = max over text tokens; filter by
`conf_threshold` (box) and `text_threshold` (label assignment).

## Known defect in this ONNX export (one pass per class phrase)

`onnx-community/grounding-dino-tiny-ONNX` **does not rebuild the block-diagonal text
self-attention mask for more than one class phrase**, so a joint multi-class prompt is
order-dependent and silently drops classes.

GroundingDINO's BERT branch needs a block-diagonal mask so the tokens of one phrase cannot
attend to another phrase's tokens. HF builds it inside the model from `input_ids`
(`generate_masks_with_special_tokens_and_transfer_map`) — it is not a graph input, so the
export has to reproduce it. In transformers 4.48 (the version this file was traced with)
that function is a **Python `for` loop over `torch.nonzero(special_tokens_mask)`**, and
`torch.onnx.export` baked the trip count in. In the graph, `input_ids` feeds
`Equal(101)/Equal(102)/Equal(1012)/Equal(1029)` → `Or` chain → `NonZero` → `Transpose` →
`Gather(axis 0, index 0)`, `Gather(index 1)`, `Gather(index 2)` — exactly **three hard-coded
iterations** driving 6 `ScatterND` nodes (mask + `position_ids` per iteration). The export
prompt held one class (`[CLS] … . [SEP]` = 3 special tokens).

At runtime only the **first** phrase gets its attention block; every later phrase keeps just
the `EyeLike` identity (each token attends to itself alone) with `position_id` 0. Measured on
`demo/images/000000000139.jpg`, HF PyTorch returns the same 15 detections for every
reordering of a 12-class prompt, while a raw joint ONNX pass returns 4 / 0 / 0 / 3.

Nothing on the Go side can repair that graph: the only text inputs are `input_ids` /
`attention_mask` / `token_type_ids`, all `[1, L]`, and `attention_mask` is used solely as the
padding mask (`Cast` → `Not` → `Tile` into the fusion and decoder attention) — it can never
encode a block-diagonal `[L, L]` mask.

**Fallback for these weights:** `groundingdino.Detect` runs the session **once per
`.`-separated phrase** — the single-phrase regime this export handles exactly right (for
`"chair."` the ONNX mask is bit-identical to HF's and the scores match). Results are
order-stable, at the cost of N passes for N classes.

## FIXED weights: `model-fixedmask.onnx`

A re-export with a correct, dynamic mask lives at **`model-fixedmask.onnx`** (694.8 MB,
opset 17), registered as the model **`grounding-dino-fixed`** (see
`models/grounding-dino-fixed/manifest.yaml`). VisionServe **probes the weights at load**
(`groundingdino.SupportsJointTextPass`, which looks for the `NonZero` op that only the
baked-loop graph contains) and picks the regime automatically, so both manifests are safe to
use — old weights keep the per-phrase path, new weights get a single joint pass.

Measured on `demo/images/000000000139.jpg`, 12-class prompt, warm server, RTX A6000 CUDA EP,
median of 3 (model resident):

| weights | passes | latency | detections |
|---|---|---|---|
| `model.onnx` (community, per-phrase) | 12 | 1779.7 ms | 17 |
| `model-fixedmask.onnx` (joint) | **1** | **149.3 ms** | **15** |

**11.9× faster**, and the 15 detections are exactly HF PyTorch's
(chair 7, clock 1, laptop 1, potted plant 2, tv 2, vase 2). The old path's 17 over-counted
vases and missed `laptop`, because per-phrase inference cannot let the fusion layers see the
other classes.

Accuracy vs HF PyTorch on the same pixel tensor and `input_ids`: identical detection lists on
all five prompts, max `|Δlogit|` 7.3e-3 and max `|Δsigmoid|` 1.1e-4 over the valid token range
for the 12-class prompt. Permuting the class order changes scores by at most 3.529e-2 — and HF
PyTorch drifts by exactly the same 3.529e-2, i.e. the residual order sensitivity is inherent
to GroundingDINO (the fusion and decoder layers attend over all text tokens with only the
padding mask), not an artifact of the export. "Order-stable" therefore means the same
detections, not bit-identical coordinates.

### How it was produced

1. Monkeypatch `generate_masks_with_special_tokens_and_transfer_map` with a vectorized,
   data-independent equivalent. transformers ≥5 already vectorizes it, but that version does
   not export: the legacy exporter rejects `aten::isin` and ONNX has no `CumMax`/`CumMin`.
   The substitutions are `isin` → `Equal`/`Or` chain, `cummax`/`cummin` → an `L×L` compare
   plus `ReduceMax`/`ReduceMin` (`L ≤ max_text_len = 256`, so this is negligible), and
   `torch.eye` → `(i == j)` (`EyeLike` on bool has no ORT kernel).
2. `torch.onnx.export` (legacy, `dynamo=False`), opset 17, `dynamic_axes` on
   `sequence_length` for the three text inputs, pixel inputs fixed at `[1,3,800,800]` /
   `[1,800,800]` to match the old graph, and `config.disable_custom_kernels = True` so the
   deformable attention traces. Trace with a **two-class** example so a baked single-phrase
   loop could not pass unnoticed; the mask is then verified at L = 4, 6, 8, 12 and 26 tokens.
3. **Demote float64 → float32.** The legacy exporter type-promotes mixed int64/float32
   arithmetic to `DOUBLE`; in `GroundingDinoEncoder.get_reference_points`,
   `ref_y / (valid_ratios[...] * height)` has `height` as a traced int64 spatial-shape tensor,
   and the resulting double flows into the deformable-attention sampling grid. ORT then
   refuses to load the model with *"Could not find an implementation for GridSample(16)"*.
   PyTorch itself runs this in float32, so demoting every `Cast(to=DOUBLE)` and double
   constant is a faithful correction (15 cast attributes, 20 attribute tensors here).

Both weight files keep the same 5 input names and 2 output names, so
`internal/models/groundingdino` binds them identically.

## Usage

GroundingDINO requires a text prompt. Queries are lowercased and dot-separated:

```bash
# CLI (in-process, no server needed)
visionserve run grounding-dino img.jpg --prompt "cat. remote." --out boxes.png

# Server
make serve
curl -s -F model=grounding-dino -F image=@img.jpg -F prompt="cat. remote." \
  http://localhost:11435/api/predict
```

Thresholds (adjustable in `manifest.yaml`):
- `conf_threshold` (default 0.3): minimum box/query score to keep a detection.
- `text_threshold` (default 0.25): minimum token score for assigning a label to a box.

Both thresholds can also be **overridden per request** instead of editing the manifest —
pass `box_threshold` / `text_threshold` as HTTP form/JSON fields, or as
`box_threshold=` / `text_threshold=` kwargs on the Python client. Precedence is
per-request (>0) → manifest → built-in default. Lowering `text_threshold` keeps more
prompt words in each label — e.g. the prompt `"canned coffee"` yields the label
`"canned coffee"` instead of just `"coffee"`.

```bash
# server: tighter boxes, richer labels
curl -s -F model=grounding-dino -F image=@img.jpg -F prompt="canned coffee." \
  -F box_threshold=0.4 -F text_threshold=0.15 http://localhost:11435/api/predict
```

## Performance

Measured on NVIDIA RTX A6000 via VisionServe HTTP server (warm, `duration_ms`):

| Device | p50 latency | Notes |
|--------|-------------|-------|
| GPU + TensorRT EP (`gpu:0+trt`) | **~70 ms** | requires `libnvinfer.so.10` |
| GPU CUDA EP only (`gpu:0`) | ~6 000 ms | same as CPU — deformable attention ops fall back to CPU |
| CPU | ~6 000 ms | standard ORT |

> **Why CUDA EP ≈ CPU:** GroundingDINO uses deformable multi-scale attention and
> BERT-style cross-attention ops that have no CUDA kernels in ORT's standard build.
> TRT compiles the full graph to GPU, eliminating the fallback entirely.

VisionServe auto-detects TRT at startup. Check status with `visionserve version` — the
response also includes a `hint` field when TRT is absent.

## License

Apache-2.0. See [Hugging Face](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
for the upstream model card. "It's on HuggingFace" is not a license check — verify the
actual license field before adding any model to VisionServe.
