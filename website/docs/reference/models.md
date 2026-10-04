# Models and performance

Every model VisionServe can pull, how to choose between them, and what they cost on a GPU.
Real outputs are in the [model gallery](../gallery.md); the live list on your machine is
`visionserve list`.

## Supported models

| Task | Model | License | Source | Architecture key | Input | Status |
|------|-------|---------|--------|-----------------|-------|--------|
| Detection | RF-DETR | Apache-2.0 | [PierreMarieCurie/rf-detr-onnx](https://huggingface.co/PierreMarieCurie/rf-detr-onnx) | `rf-detr` | 560×560 | working |
| Detection | RT-DETR | Apache-2.0 | [onnx-community/RT-DETR-l-hf](https://huggingface.co/onnx-community/RT-DETR-l-hf) | `rt-detr` | 640×640 | code works (NMS-free, COCO-80); **cannot be pulled**: upstream returns 401 |
| Segmentation | MobileSAM | Apache-2.0 | [Acly/MobileSAM](https://huggingface.co/Acly/MobileSAM) | `mobile-sam` | 1024×1024 | working — box/point prompt, or no prompt → segment everything (AMG) |
| Segmentation | EfficientSAM | Apache-2.0 | [yunyangx/EfficientSAM](https://huggingface.co/yunyangx/EfficientSAM) | `efficient-sam` | 1024×1024 | working — box/point prompt |
| Segmentation | SAM2-Tiny | Apache-2.0 | [SharpAI/sam2-hiera-tiny-onnx](https://huggingface.co/SharpAI/sam2-hiera-tiny-onnx) | `sam2` | 1024×1024 | working — multi-scale encoder |
| Segmentation | NanoSAM | Apache-2.0 | [NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam) (manual) | `nano-sam` | 1024×1024 | implemented — manual download |
| Open-vocab detection | GroundingDINO | Apache-2.0 | [onnx-community/grounding-dino-tiny-ONNX](https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX) | `grounding-dino` | 800×… | working — text → boxes |
| Open-vocab segmentation | Grounded-SAM | Apache-2.0 | composite: GroundingDINO + MobileSAM | `grounded-sam` | — | working — text → boxes → masks |
| Open-vocab detection | RF-DETR + text-aligned head (`rfdetr-textalign-dec1`, `-dec1-siglip`, `-dec1-siglip-prod`, `-etri`) | Apache-2.0 | [mtbui2010/rfdetr-textalign-ONNX](https://huggingface.co/mtbui2010/rfdetr-textalign-ONNX) | `rfdetr-textalign` | 512×512 | working — tabletop fine-tunes; text → boxes at about the bare detector's cost; pull also fetches `clip-text` or `siglip-text` |
| Text embedding | CLIP ViT-B/32 text tower | MIT | [mtbui2010/rfdetr-textalign-ONNX](https://huggingface.co/mtbui2010/rfdetr-textalign-ONNX) (`clip-text/`) | `clip-text` | 77 tokens | working — 512-d, same space as `clip` |
| Depth estimation | Depth Anything V2 | Apache-2.0 | [onnx-community/depth-anything-v2-small-hf](https://huggingface.co/onnx-community/depth-anything-v2-small-hf) | `depth-anything-v2` | 518×518 | working |
| Depth estimation | MiDaS | MIT | [Heliosoph/midas-small-onnx](https://huggingface.co/Heliosoph/midas-small-onnx) | `midas` | 256×256 | working |
| Classification | EfficientNet-B0 | Apache-2.0 | [onnxmodelzoo/efficientnet_b0_Opset17](https://huggingface.co/onnxmodelzoo/efficientnet_b0_Opset17) | `efficientnet` | 224×224 | working — top-K ImageNet |
| Classification | MobileNetV3-Small | Apache-2.0 | [onnxmodelzoo/mobilenet_v3_small_Opset17](https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17) | `mobilenet-v3` | 224×224 | working — top-K ImageNet |
| Image embedding | CLIP | MIT | [khasinski/clip-ViT-B-32-onnx](https://huggingface.co/khasinski/clip-ViT-B-32-onnx) | `clip` | 224×224 | working — 512-d embeddings |
| Face detection | SCRFD | MIT | [cromsc/scrfd-10g](https://huggingface.co/cromsc/scrfd-10g) | `scrfd` | 640×640 | working |
| OCR | PaddleOCR | Apache-2.0 | [webnn/PP-OCRv4-ONNX](https://huggingface.co/webnn/PP-OCRv4-ONNX) | `paddle-ocr` | variable | working — text det + rec |
| Pose estimation | RTMPose | Apache-2.0 | planned | `rtmpose` | 256×192 | **planned** — 17 COCO keypoints |

**All models are permissive-licensed (Apache-2.0 / MIT).** This is a deliberate,
load-bearing constraint — not a limitation we work around. See [Why it is built this way](../index.md#why-it-is-built-this-way).

## Model selection guide

Quick reference for choosing the right model. All models are free (Apache-2.0 / MIT).

### Object detection

| Scenario | Model | Why |
|----------|-------|-----|
| Max speed — edge / real-time | `rf-detr-nano` | ~23 ms GPU, near YOLO speed, 384×384 |
| Best COCO accuracy | `rf-detr` | 53.4 AP, NMS-free, 560×560 |
| Balanced accuracy + speed | `rt-detr` | 53.0 AP, NMS-free, COCO-80, 640×640 (upstream weights gone: only if you already have them) |
| No fixed class list (text query) | `grounding-dino` | zero-shot: `"cat. remote."` → boxes |
| **Mix of known + novel classes** | `rfdetr-gdino` | hybrid router — known words go to RF-DETR (fast), unknown words to GroundingDINO (open-vocab); pays GroundingDINO's cost only when a request actually needs it |
| Face detection | `scrfd` | WiderFace-tuned, returns 5 keypoints |

### Segmentation

| Scenario | Model | Why |
|----------|-------|-----|
| Fastest SAM on CPU/GPU | `mobile-sam` | TinyViT encoder; no prompt → segment everything (AMG) |
| Lightweight SAM alternative | `efficient-sam` | ViT-Tiny SAMI, similar quality; box/point prompt |
| Best mask quality | `sam2` | Multi-scale encoder, Meta AI SAM2-Tiny |
| NVIDIA Jetson / TensorRT | `nano-sam` | ResNet-18 encoder, optimized for TRT |
| Text → masks (zero-shot) | `grounded-sam` | GroundingDINO + MobileSAM chained |

### Depth estimation

| Scenario | Model | Why |
|----------|-------|-----|
| Speed-first | `midas` | 256×256, lightweight, MIT |
| Accuracy-first | `depth-anything-v2` | 518×518, state-of-the-art |

### Classification and embeddings

| Scenario | Model | Why |
|----------|-------|-----|
| ImageNet top-K, standard | `efficientnet-b0` | 77.1% top-1, solid baseline |
| Ultra-lightweight (edge) | `mobilenet-v3` | 67.4% top-1, ~8 MB ONNX, very fast |
| Zero-shot / visual search / retrieval | `clip` | 512-d L2 embeddings, cosine similarity |
| OCR — Chinese + English | `paddle-ocr` | PP-OCRv4 DBNet++ det + SVTR-tiny rec |

> **Size filtering tip:** add `--min-size N` / `--max-size N` (% of image area, 0 = no limit)
> to any detection or segmentation run to drop noise or oversized objects. Example:
> `--min-size 0.5` drops anything covering less than 0.5% of the image. Works for every
> model — server-side, no extra overhead.

## Performance

Measured on a single **NVIDIA RTX A6000 (48 GB VRAM)**, 48-core CPU, 251 GB RAM.
Latency = median of 30 warm requests via the HTTP server (model already loaded).
Cold-start = wall-clock time from server launch to first response (includes model load +
ONNX session creation + first inference). Scripts live in [`benchmarks/`](https://github.com/mtbui2010/vision_serve/tree/main/benchmarks).

### Latency — all models (VisionServe Go HTTP, GPU)

Measured on **NVIDIA RTX A6000 (48 GB)**, 20 warm requests via the HTTP server (model already loaded). `srv p50` = server-side inference only; the gap to `p50` is Go preprocess + HTTP overhead.

| Model | Task | Size MB | p50 ms | p95 ms | RPS | srv p50 | VRAM MB | Cold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **clip** | embed | 335 | **33** | 69 | 27.9 | 12 | 810 | 5.8 s |
| **mobilenet-v3** | classification | 10 | **38** | 56 | 26.1 | 9 | 308 | 2.9 s |
| **efficientnet-b0** | classification | 20 | **40** | 58 | 24.2 | 11 | 356 | 3.0 s |
| **scrfd** | face detection | 16 | **45** | 69 | 22.4 | 23 | 420 | 3.9 s |
| **paddle-ocr** | OCR | 15 | **54** | 78 | 17.3 | 34 | 462 | 4.7 s |
| **rf-detr-nano** | detection | 103 | **57** | 90 | 16.9 | 37 | 548 | 4.6 s |
| **midas** | depth | 63 | **65** | 98 | 14.6 | 13 | 420 | 4.1 s |
| **rf-detr** | detection | 103 | **78** | 105 | 12.6 | 55 | 804 | 4.9 s |
| **mobile-sam** | segmentation | 58 | **161** | 185 | 6.4 | 136 | 966 | 7.3 s |
| **efficient-sam** | segmentation | 39 | **181** | 247 | 5.5 | 158 | 1628 | 5.8 s |
| **sam2** | segmentation | 148 | **242** | 544 | 3.9 | 222 | 2508 | 5.3 s |
| **grounding-dino** | open_vocab | 686 | **570** | 647 | 1.8 | 550 | 4392 | 12.3 s |

> rt-detr, depth-anything-v2, nano-sam, grounded-sam not yet measured — see [Reproducing](#reproducing) to run `bench_all_models.py`.

### Comparison with YOLO

```
YOLOv8n  GPU (PyTorch):       ~18 ms   CNN, 6 MB, AGPL-3.0 ✗
RF-DETR-nano  GPU (VisionServe): 57 ms    transformer, 103 MB, Apache-2.0 ✓  (srv-only: 37 ms)
RF-DETR-base  GPU (VisionServe): 78 ms    transformer, 103 MB, Apache-2.0 ✓  (srv-only: 55 ms)
YOLOv8m  GPU (PyTorch):       ~45 ms   CNN, 52 MB, AGPL-3.0 ✗
GroundingDINO GPU CUDA (VisionServe): ~153 ms  open-vocab (text query), 686 MB, Apache-2.0 ✓  (default)
GroundingDINO GPU+TRT  (VisionServe): ~104 ms  --tensorrt opt-in: 6.8 mAP lower on held-out names
```

RF-DETR-nano at 57 ms (srv-only 37 ms) is **competitive with YOLOv8n** at the server level. The gap for RF-DETR-base comes from:
1. **DETR transformer architecture** — global cross-attention on 300 queries is more expensive
   than YOLO's local grid predictions, but NMS-free and more accurate on dense/occluded scenes.
2. **CUDA EP, not TensorRT** — VisionServe runs CUDA → CPU by default. TensorRT compiles the
   whole graph and is opt-in (`--tensorrt` / `VISIONSERVE_TENSORRT=1`, needs `libnvinfer.so.10`);
   on GroundingDINO it measured ~1.5x faster but 6.8 mAP lower (BUGS_TO_FIX.md #3).
   `visionserve version` and the server log print the chain in effect.
3. **Go preprocess + HTTP** — adds ~20 ms overhead on top of inference.

**YOLO (Ultralytics) is forbidden** in VisionServe by design — it is AGPL-3.0 copyleft,
which would virally relicense the entire project and every downstream user. RF-DETR and
GroundingDINO are both Apache-2.0 and can be used freely in commercial and closed products.

To get RF-DETR-nano pull it from the catalog:
```bash
make pull MODEL=rf-detr-nano        # ~103 MB, 384×384 input, 57 ms GPU (srv-only 37 ms)
```

### Key takeaways

- **Fastest models (GPU):** CLIP (33 ms), MobileNetV3 (38 ms), EfficientNet-B0 (40 ms) — lightweight tasks.
- **Face detection:** SCRFD at 45 ms, only 16 MB ONNX, 420 MB VRAM — very efficient.
- **Detection:** RF-DETR-nano at 57 ms (srv 37 ms), RF-DETR at 78 ms (srv 55 ms). Add `--min-size`/`--max-size` to filter noise.
- **Depth:** MiDaS at 65 ms (srv 13 ms) — Go preprocess dominates (52 ms overhead). Depth Anything V2 not yet measured.
- **Segmentation:** MobileSAM box/point prompt: ~160 ms (table above). AMG (no prompt, 256 calls): ~7 s (TRT+pool) / ~27 s (CUDA EP). SAM2 p95 is 544 ms — multi-scale encoder is VRAM-heavy (2.5 GB).
- **OCR:** PaddleOCR at 54 ms total, 34 ms inference.
- **Open-vocab:** GroundingDINO ~153 ms on the CUDA EP (the default) / ~104 ms with the TensorRT opt-in, which scores held-out names 6.8 mAP lower (served, 29 Sep 2026; the table above is an older run).
- **Go HTTP overhead:** typically 10–55 ms on top of pure inference. Bottleneck is always ORT, not the server.
- **Cold-start** ranges from 2.9 s (MobileNetV3) to 12.3 s (GroundingDINO). Use `make serve` for production.
- **TensorRT EP** is off by default: every shipped model lists `[cuda, cpu]` in `runtime.prefer`. Turn it on with `--tensorrt` or `VISIONSERVE_TENSORRT=1` (needs `libnvinfer.so.10`, falls back to CUDA without it). On GroundingDINO it measured ~1.5x faster than CUDA but 6.8 mAP lower, and it rebuilds its engine for every new prompt length. See [Model files and ONNX Runtime](../concepts/onnx.md#cpu-or-gpu-execution-providers).

### Accuracy reference (from papers / official repos)

> Numbers from original papers / official repos. Measured on standard public benchmarks —
> **not** by VisionServe. Actual values may vary slightly depending on the ONNX export
> and preprocessing pipeline. All models are permissive-licensed.

| Task | Model | Metric | Score | Benchmark |
|------|-------|--------|------:|-----------|
| Detection | RF-DETR | AP | 53.4 | COCO val2017 |
| Detection | RF-DETR-nano | AP | 48.0 | COCO val2017 |
| Detection | RT-DETR-l | AP | 53.0 | COCO val2017 |
| Face detection | SCRFD-10GF | AP Easy | 95.2 % | WiderFace val |
| Segmentation | MobileSAM | mIoU | 75.7 | SA-23B (zero-shot) |
| Segmentation | EfficientSAM-Ti | mIoU | 74.0 | SA-23B (zero-shot) |
| Segmentation | SAM2-Tiny | J&F | 75.0 | DAVIS 2017 video |
| Open-vocab det | GroundingDINO-T | AP | 48.4 | COCO zero-shot |
| Depth | Depth Anything V2-S | AbsRel | 0.076 | NYUv2 |
| Depth | MiDaS v2.1-small | AbsRel | ≈0.083 | NYUv2 (approx) |
| Classification | EfficientNet-B0 | Top-1 | 77.1 % | ImageNet-1k |
| Classification | MobileNetV3-Small | Top-1 | 67.4 % | ImageNet-1k |
| Image embedding | CLIP ViT-B/32 | Zero-shot Top-1 | 63.4 % | ImageNet |
| OCR | PP-OCRv4 | Rec Acc | 79.0 % | Chinese OCR benchmark |

### Reproducing

```bash
# Run all baselines (rf-detr + grounding-dino, GPU + CPU)
python3 benchmarks/bench.py

# Run all 16 models in parallel background processes
python3 benchmarks/bench_all_models.py --device gpu
python3 benchmarks/bench_all_models.py --device cpu --workers 4

# GPU benchmark only, specific models
source scripts/gpu-env.sh
python3 benchmarks/bench_all_models.py --device gpu --models rf-detr mobile-sam grounding-dino
```
