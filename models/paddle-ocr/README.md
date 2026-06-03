# PP-OCRv4 — paddle-ocr

**License:** Apache-2.0 (PaddlePaddle project)
**Task:** Text detection + recognition (OCR)

## What it is

PP-OCRv4 is a two-stage OCR pipeline from Baidu PaddleOCR:

1. **det (DBNet++)** — detects text regions and produces a binary probability map.
2. **rec (SVTR-tiny CTC)** — recognizes the text string within each detected region crop.

Both models are released under Apache-2.0. This implementation uses ONNX Runtime — no
PaddlePaddle dependency at runtime, fully self-contained in the Go binary.

## How to get the ONNX files

### Option A — pull with VisionServe (recommended)

```bash
make pull MODEL=paddle-ocr    # downloads det + rec ONNX + keys file from HuggingFace
```

### Option B — download manually from HuggingFace

HuggingFace repo: [`webnn/PP-OCRv4-ONNX`](https://huggingface.co/webnn/PP-OCRv4-ONNX)
(Apache-2.0 — PaddlePaddle project)

```bash
MODEL_DIR=models/paddle-ocr
hf download webnn/PP-OCRv4-ONNX \
    ch_PP-OCRv4_det.onnx ch_PP-OCRv4_rec.onnx ch_PP-OCR_keys_v1.txt \
    --local-dir $MODEL_DIR
mv $MODEL_DIR/ch_PP-OCRv4_det.onnx  $MODEL_DIR/det_model.onnx
mv $MODEL_DIR/ch_PP-OCRv4_rec.onnx  $MODEL_DIR/rec_model.onnx
mv $MODEL_DIR/ch_PP-OCR_keys_v1.txt $MODEL_DIR/ppocr_keys_v1.txt
```

File mapping:

| Local file in `models/paddle-ocr/` | HF filename | Notes |
|------------------------------------|-------------|-------|
| `det_model.onnx` | `ch_PP-OCRv4_det.onnx` | DBNet++ text detector |
| `rec_model.onnx` | `ch_PP-OCRv4_rec.onnx` | SVTR-tiny CTC recognizer |
| `ppocr_keys_v1.txt` | `ch_PP-OCR_keys_v1.txt` | ~6625-class character set |

The character set file `ppocr_keys_v1.txt` is required at startup. It covers Chinese
characters, digits, Latin letters, and common symbols (~6623 classes).

## Verified I/O

| Session | Input | Output |
|---------|-------|--------|
| det | `x` dynamic `[1, 3, H, W]` | `sigmoid_0.tmp_0` `[1, 1, H, W]` probability map |
| rec | `x` `[batch, 3, 48, W]` (height fixed at 48) | `softmax_11.tmp_0` `[batch, T, 6625]` CTC logits |

## Running

```bash
visionserve run paddle-ocr image.jpg
```

No text prompt is needed (unlike GroundingDINO). The pipeline detects all text regions
automatically.

## Result format

Each detected text region is returned as a `Detection` entry:

```json
{
  "bbox": [x, y, w, h],
  "class": "recognized text string",
  "conf": 0.92
}
```

- `bbox` — axis-aligned text region in original image coordinates `[x, y, w, h]`.
- `class` — the recognized text string (Chinese, English, digits, symbols).
- `conf` — average CTC token confidence (0..1) for the recognition result.

## V1 limitations

- **Axis-aligned boxes only**: The detection stage uses BFS flood-fill on the binary
  probability map to extract rectangular bounding boxes. Full polygon extraction
  (DBNet++'s contour output) is not implemented in v1, so rotated or perspective-distorted
  text will have slightly larger bounding boxes than the tight polygon would.
- **No direction classifier**: The optional PP-OCRv4 `cls` model (text direction
  classifier) is skipped. Upside-down text may not be recognized correctly.
- **Chinese + English**: The default `ppocr_keys_v1.txt` covers Chinese characters,
  digits, Latin letters, and common symbols (~6623 classes). Other scripts require a
  different character set and model.

## Performance

Measured on NVIDIA RTX A6000 (48 GB VRAM), VisionServe Go HTTP server, 20 warm requests.

| Metric | Value |
|--------|-------|
| p50 latency (end-to-end HTTP) | 54 ms |
| p95 latency | 78 ms |
| Inference only (srv p50) | 34 ms |
| Throughput | 17.3 RPS |
| VRAM (GPU) | 462 MB |
| ONNX size | 15 MB |
| Cold-start | 4.7 s |

det + rec pipeline in series. Size = det (4.6 MB) + rec (11 MB) combined.
