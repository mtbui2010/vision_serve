# RT-DETR

**RT-DETR** (Real-Time Detection Transformer) is a real-time, NMS-free object detector
from Baidu Research. It achieves competitive accuracy/speed tradeoffs without the
non-maximum suppression post-processing step required by YOLO-family models.

**License:** Apache-2.0

## Model variant

This directory targets **RT-DETR-l** (large), COCO-pretrained, from the HuggingFace export:
`onnx-community/RT-DETR-l-hf`.

## Expected I/O shapes

| Tensor         | Shape          | Notes                                        |
|----------------|----------------|----------------------------------------------|
| `pixel_values` | `[1,3,640,640]`| NCHW, ImageNet-normalized, letterboxed       |
| `pred_logits`  | `[1,300,80]`   | Raw logits, COCO-80, sigmoid applied at decode|
| `pred_boxes`   | `[1,300,4]`    | cxcywh, normalized [0,1] to input image size |

Q=300 queries (NMS-free; all 300 decoded, filtered by conf_threshold).

## Class labels

Uses COCO-80 (contiguous indices 0-79, **no N/A gap**). See `coco80.txt` in this directory.
This is different from RF-DETR which uses COCO-91 (index 0 = N/A placeholder).

## Manual download

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="onnx-community/RT-DETR-l-hf",
    filename="onnx/model.onnx",
    local_dir="models/rt-detr",
)
EOF

mv models/rt-detr/onnx/model.onnx models/rt-detr/model.onnx
rmdir models/rt-detr/onnx
```

## Running

```bash
# In-process (no server needed)
make run MODEL=rt-detr IMAGE=photo.jpg

# Via the HTTP server
make serve
curl -s -F model=rt-detr -F image=@photo.jpg \
  http://localhost:11435/api/predict | jq .
```

## Architecture note

RT-DETR is NMS-free (DETR-style set prediction). Do NOT apply post-hoc NMS — the
postprocess in `internal/models/rtdetr/postprocess.go` intentionally omits it. Only
confidence filtering + top-K sorting are applied.
