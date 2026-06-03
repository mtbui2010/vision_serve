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

## Performance

Measured on NVIDIA RTX A6000 via VisionServe HTTP server (warm):

| Device | p50 latency | Notes |
|--------|-------------|-------|
| GPU (CUDA EP) | ~325–340 ms | 719 MB ONNX, 4.4 GB VRAM |
| CPU | ~2 500 ms | standard `onnxruntime` |

Use GPU for practical throughput (~2.9 RPS). TensorRT EP would give a further 2–4× speedup.

## License

Apache-2.0. See [Hugging Face](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
for the upstream model card. "It's on HuggingFace" is not a license check — verify the
actual license field before adding any model to VisionServe.
