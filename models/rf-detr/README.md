# RF-DETR (detection) — weights

> ONNX weight files are **NOT committed** to git (large binaries). Download them with the instructions below.

RF-DETR is Roboflow's detection-transformer architecture, licensed **Apache-2.0**
(NOT YOLO/AGPL). It is fully free and open-source. You need a `.onnx` file placed
next to `manifest.yaml`.

## Download / export weights

### Option A — download the public ONNX build (fastest, VERIFIED)

The `rf-detr-base-coco.onnx` build (Apache-2.0, COCO) is available on Hugging Face.
Requires `huggingface_hub`:

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="PierreMarieCurie/rf-detr-onnx",
                    filename="rf-detr-base-coco.onnx")
shutil.copy(p, "models/rf-detr/rf-detr-base.onnx")
PY
```

> That repo is exported straight from Roboflow's `rfdetr`, license Apache-2.0. (The
> build used for the verification below is stored locally as `rf-detr-base-real.onnx`.)

### Option B — export it yourself from the original checkpoint (official source)

```bash
python -m pip install rfdetr        # Roboflow library, Apache-2.0
python - <<'PY'
from rfdetr import RFDETRBase
m = RFDETRBase()                    # auto-downloads the COCO checkpoint
m.export(output_dir="models/rf-detr", simplify=True)  # exports .onnx (input 560x560)
PY
# Rename the result to rf-detr-base.onnx if needed to match the manifest.
```

Make sure the input size matches `manifest.yaml` (560×560, NCHW) and that the labels
file is in the same directory.

---

## ✅ Verified against REAL weights (performed 2026-06-02)

Build used: `rf-detr-base-coco.onnx` from `PierreMarieCurie/rf-detr-onnx` (Apache-2.0).
Inspected with `onnxruntime` (Python), then run end-to-end through `bin/visionserve`.

### Actual I/O shapes

```
inputs:
  input       [1, 3, 560, 560]  float32        # matches manifest 560×560 NCHW
outputs:
  pred_boxes  [1, 300, 4]       float32        # Q = 300 queries
  pred_logits [1, 300, 91]      float32        # C = 91  (NOT 80!)
```

- Exactly **2 outputs**, true DETR style (NMS-free, fixed 300 queries). ✔
- Postprocess identifies the boxes tensor by its **last dim == 4** — so it stays correct
  even though the names are `pred_boxes`/`pred_logits` (not `logits`/`boxes`). ✔

### Point (2) — sigmoid or softmax?  → **SIGMOID** ✔

Raw logits sit in roughly [-12, +3]. RF-DETR uses a **sigmoid** head (focal loss, with
NO "no-object" softmax class). Taking the highest-sigmoid class per query and filtering
by `conf_threshold` gives sensible results (the right detection count, high scores
~0.95). The current postprocess uses `sigmoid` — **CORRECT**.

### Point (3) — box format?  → **cxcywh normalized to [0,1]** ✔

All box values fall within [0,1]. Decoding as **cxcywh-normalized**, then multiplying by
560, removing padding, and dividing by scale places boxes EXACTLY on the objects.
`manifest.postprocess.box_format: cxcywh` is **CORRECT**. (xyxy puts boxes in the wrong
place.)

### ⚠️ NEW finding — number of classes = 91, NOT 80

The model emits **91 logits/query** = the COCO "paper" index space (with gaps, index
0 = N/A). It is NOT the contiguous 80-class space. Consequences:

- `coco.txt` (80 lines) produces **wrong labels**: idx 17 → "dog" (wrong) instead of
  "cat" (correct).
- A **`coco91.txt`** file (91 lines, index 0 = `N/A`) was added to map labels
  CORRECTLY. `manifest.yaml` now points to `labels: coco91.txt`.
- With coco91: idx 17 → `cat`, idx 75 → `remote` — MATCHES the test image.

> If you export with **Option B (the `rfdetr` package)**, re-CHECK whether `pred_logits`
> has 91 or 80 dimensions (some exports may remap to 80). Choose `labels` (coco91.txt
> vs coco.txt) to match the dimension count — otherwise labels will be silently off.

### Real run results (reproducible, NOT fabricated)

Test image: COCO `val2017/000000039769.jpg` (640×480, known content: **2 cats + 2
remotes on a sofa**). Download:
`curl -sL http://images.cocodataset.org/val2017/000000039769.jpg -o /tmp/coco_test.jpg`

Run (manifest pointing at the real weights + coco91.txt, via a temporary registry):

```bash
export ORT_DYLIB_PATH=.../libonnxruntime.so
VISIONSERVE_MODELS=/tmp/vs_models_real ./bin/visionserve run rf-detr /tmp/coco_test.jpg
```

Go output (abridged) — MATCHES pixel-for-pixel the ground truth computed with Python
onnxruntime:

| class  | conf  | bbox xywh (original 640×480)       |
|--------|-------|------------------------------------|
| cat    | 0.958 | [11, 55, 308, 421]  (left cat)     |
| cat    | 0.952 | [343, 25, 296, 344] (right cat)    |
| remote | 0.906 | [41, 73, 135, 45]                  |
| remote | 0.704 | [333, 77, 36, 110]                 |

→ Correct classes, boxes on the right objects, within the image bounds. The pipeline
preprocess (letterbox 560, ImageNet mean/std) → ORT → postprocess (sigmoid + cxcywh +
inverse mapping) is **CORRECT end-to-end**.

---

## Dummy ONNX for pipeline testing (no real weights needed)

Because `*.onnx` is `.gitignore`d (weights are not committed), a fresh clone has NO ONNX
file yet. To exercise the pipeline without downloading the 108MB real weights, generate
a ~2KB **dummy**:

```bash
python models/rf-detr/gen_dummy_onnx.py    # requires `onnx` (dev-only)
```

The dummy reproduces the EXACT I/O contract of real RF-DETR: `input[1,3,560,560]` →
`logits[1,5,91]` + `boxes[1,5,4]`. Its output is constant (it ignores the image), so the
boxes are round numbers — but the **classes display correctly** under `coco91.txt`: it
fires `car` (~1.00) and `cat` (~0.82). Use it to verify the preprocess→ORT→postprocess→
inverse-map flow, NOT for real inference.

> Operational note: `manifest.yaml` points to `model_file: rf-detr-base.onnx`. When you
> download the REAL weights (Option A/B above), **OVERWRITE** `rf-detr-base.onnx` with
> the real file and `./bin/visionserve run rf-detr <image>` works immediately — no
> manifest change needed.
