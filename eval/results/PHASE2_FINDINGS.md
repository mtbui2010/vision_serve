# Phase 2 measured findings (W1 / W4 / W6)

> Host: 1× node, 4× RTX A6000 (48 GB), CUDA 12.8 driver, onnxruntime-gpu 1.26.0 (CUDA EP).
> VisionServe (Go) on GPU 2, FastAPI+onnxruntime baseline on GPU 3, **same .onnx** each.
> Load client: pure-Python closed-loop (`eval/loadgen/sweep_openloop.py`), HdrHistogram percentiles.
> Date: 2026-06-05. **No fabricated numbers — every value below is from a run in this dir.**

## W6 — ImageNet-val Top-1 measured THROUGH the running server (5000-image evanarlian/imagenet_1k_resized_256 `val` subset, label order verified against `internal/catalog/labels/imagenet1k.txt`)

| Model (served ONNX) | Served Top-1 | Commonly-cited Top-1 | Gap |
|---|---|---|---|
| MobileNetV3-Small (`onnxmodelzoo/mobilenet_v3_small_Opset17`) | **0.6810** | ~0.677 (torchvision) | +0.4% |
| EfficientNet-B0 | **0.7746** | ~0.777 (torchvision) | −0.2% |

**Claim supported:** the ONNX export + VisionServe's served preprocess/postprocess reproduce
published ImageNet Top-1 within <0.5% (well inside 5k-subset sampling noise). Note VisionServe
uses `letterbox:false` resize-to-224 (not 256→center-crop), so this is the *served* accuracy a
user actually gets, not a notebook number. (Raw JSON + provenance: `w6_top1_*.json[.meta.json]`.)

## W6 Part A — export parity (PyTorch vs ONNX, `eval/accuracy/parity_torchvision.py`)

Identical random inputs through torchvision IMAGENET1K_V1 and the shipped ONNX (n=16, CPU EP):

| Model (torchvision ref) | max abs err | cosine sim | argmax match | rtol=1e-3,atol=1e-4 |
|---|---|---|---|---|
| MobileNetV3-Small | 1.6e-5 | 0.999999999999 | 100% | **PASS** |
| EfficientNet-B0 | 4.0e-6 | 0.999999999998 | 100% | **PASS** |

The shipped ONNX exports are numerically identical to the torchvision pretrained weights — the
export is faithful (this also confirms the W6 Top-1 reference is torchvision IMAGENET1K_V1).

## W6 Part B (detection) — COCO val2017 mAP for RF-DETR, through the server (1000 imgs)

| Setting | mAP@[.5:.95] | AP50 |
|---|---|---|
| Served default (manifest `conf_threshold=0.5`) | **0.427** | 0.550 |
| Eval-standard (`conf_threshold=0.001`) | **0.523** | 0.714 |

The eval-standard 52.3 is within ~1–2 points of published RF-DETR-base (~54), i.e. export+serve
preserves detection accuracy too (1000-img subset; full val may shift slightly). **Note:** COCO mAP
*must* use a low conf threshold (it integrates over recall); the default 0.5 — correct for clean
user-facing detection — chops low-confidence true positives and understates AP by ~10 points. Both
numbers are honest: 42.7 is what a user gets out-of-box, 52.3 is the model's true accuracy.
(coco91.txt line-index = COCO category_id, names matched exactly → task_eval mapping was direct.)
WiderFace AP for SCRFD still TODO (needs the official WiderFace eval toolkit).

## Input-format sensitivity — JPEG vs PNG decode (relevant to ndarray clients)

| Encoded input the server decodes | pure-Go decode cost |
|---|---|
| JPEG (sample.jpg, 810×1080) | ~19–25 ms |
| **PNG (same pixels)** | **~112 ms** (4.5× slower; zlib inflate, 7.8 MB/4772 allocs) |

VisionServe's HTTP API only accepts an *encoded* image (`image_base64` / multipart file) and
decodes server-side; it has **no raw-tensor input path**. The Python client PNG-encodes a
`numpy.ndarray`/PIL image before sending, so an "ndarray input" arrives as PNG and pays the
~112 ms PNG decode — far worse than the JPEG path the benchmarks used. The benchmarks therefore
used the *favorable* (JPEG) decode; a tensor/ndarray client over the current API is *slower*, not
faster, at low load. A future zero-copy tensor endpoint (or shared memory) would remove decode
entirely and is the right comparison point against Triton's tensor-in numbers.

## W1 / W4 — throughput & latency vs concurrency (closed-loop, 20 s/cell)

mobilenet-v3 — measured completed-request rate (rps):

| C | VisionServe rps | FastAPI+ORT rps |
|---|---|---|
| 1 | 12.7 | 66.0 |
| 2 | 36.6 | 74.9 |
| 4 | 74.6 | 76.4 |
| 8 | 202 | 87.1 |
| 16 | 405 | 77.4 |
| 32 | 652 | 84.1 |
| 64 | **695** | 84.6 |

efficientnet-b0 follows the same shape (VisionServe 510 vs baseline 77 rps at C=64).

**Honest, nuanced reading:**
- At low concurrency the **Python baseline is faster per request** (C=1 mean ~15 ms vs VisionServe ~78 ms).
  The GPU inference is <1 ms (A6000), so VisionServe's per-request cost is preprocess (pure-Go JPEG
  decode + resize) + HTTP/JSON, **not** the model. This must be explained, not hidden (→ W8 breakdown).
- VisionServe **scales to ~8× the baseline's peak throughput** (695 vs 85 rps) because Go goroutines
  parallelise across cores while the FastAPI baseline (single uvicorn worker, SESSION_POOL=1) is
  GIL/single-session bound and plateaus at ~85 rps regardless of C.
- **Validity:** the *same* 64-thread client drove VisionServe to 695 rps, so the baseline's ~85 rps
  plateau is a real server-side limit, not a client bottleneck. (VisionServe's own ceiling may be
  higher still under a native load generator — these rps are therefore conservative for VisionServe.)

## W8 — latency decomposition (faithful, no core change; `internal/models/classification` benchmark)

Pure-Go per-request CPU cost on this host (sample.jpg, **810×1080**), via
`go test ./internal/models/classification -bench 'Decode|Preprocess' -benchmem`:

| Stage | Cost | Where it runs |
|---|---|---|
| JPEG decode (`image/jpeg`) | **~19.2 ms** | HTTP handler — **outside** server `duration_ms` |
| Preprocess (resize→224 + ImageNet normalize → NCHW) | **~3.6 ms** | inside `duration_ms` |
| ORT Run (CUDA, A6000) | **<1 ms** (measured separately) | inside `duration_ms` |

**This explains the W1 anomaly:** VisionServe's ~25 ms C=1 *min* latency ≈ the ~23 ms of pure-Go
decode+preprocess; **the GPU is not the bottleneck** — pure-Go `image/jpeg` decode is. The FastAPI
baseline is faster *per request* because Pillow uses libjpeg-turbo (decode in a few ms), but it is
GIL-serialized; VisionServe's pure-Go decode parallelises across cores, which is exactly why it
reaches ~8× the baseline's peak throughput. This is the CV-specific instance of "Beyond Inference"
(DAC'24): non-inference overhead, not the model, governs throughput for small CV nets on fast GPUs.
Fair-engineering note (Exp 5): a build-tagged libjpeg-turbo decode would quantify the pure-Go tax;
we keep pure-Go for single-binary portability and report the Δ.

## W4b — open-loop constant-rate (Pareto / tail), mobilenet-v3

Offered-rate sweep (open-loop, CO-correct, pure-Python client; `eval/loadgen/rate_sweep.py`).
**Read the `errors` column carefully** — beyond saturation the client's "goodput" is meaningless
(fast-failing connections count as completed). Only `errors=0` rows with bounded latency are valid:

| Offered rps | VisionServe goodput / p50 / errors | FastAPI+ORT goodput / p50 / errors |
|---|---|---|
| 25  | 24.9 / 35 ms / **0** | 24.8 / 18 ms / **0** |
| 50  | 49.7 / 30 ms / **0** | 49.7 / 19 ms / **0** |
| 100 | 99.3 / 28 ms / **0** | 62.0 / **6.3 s** / 0  ← saturated |
| 200 | 193.7 / 31 ms / **0** | 61.6 / 17.6 s / 0 |
| 400 | 390 / 805 ms / 0 ← degrading | 73 / 32 s / 0 |
| 800–1400 | ~390 / multi-s / **0** (client-capped) | ✗ **100% errors** (conn refused) — invalid |

**Honest reading (SLO-based, the only defensible cut):**
- **FastAPI+ORT** holds ~50 rps at p50 $<$20 ms, then **saturates hard at offered 100** (p50 jumps
  to 6.3 s); past offered ~600 it **refuses connections** (errors climb to 100%). Real sustainable
  rate $\approx$ 50 rps.
- **VisionServe** holds ~200 rps at p50 $\sim$31 ms with **zero errors**, degrading past 400.
- So under a p50 $<$50 ms SLO, **VisionServe sustains $\sim$4$\times$ the baseline's load (≈200 vs
  ≈50 rps) with no errors where the baseline starts dropping connections.**
**Caveat:** the pure-Python client itself caps measurement near ~390 rps (GIL+CO scheduling), so
VisionServe's true open-loop ceiling is understated (closed-loop reached 695). The baseline's
saturation is well below that cap, so it is a real server limit. Re-measure headline open-loop
p99.9 with wrk2/vegeta. **Do NOT cite the offered≥800 baseline goodput — it is 100% failed requests.**

## Tensor-in regime — VisionServe vs Triton, the FAIR no-decode comparison

VisionServe gained a `POST /api/infer_tensor` endpoint (raw NCHW float32, skips decode+preprocess).
Benchmarked head-to-head with Triton's tensor-in path (`eval/baselines/vs_tensor_bench.py`),
mobilenet-v3, closed-loop:

| C | **VS tensor-in** rps / p50 | VS JPEG-in rps / p50 | Triton tensor-in rps / p50 |
|---|---|---|---|
| 1 | 118 / 8.1 ms (min 3.4 ms) | 12.7 / 57 ms | 214 / 4.0 ms |
| 16 | 458 / 34 ms | 405 / 38 ms | 549 / 28 ms |
| 64 | **563** / 109 ms | 695 / 92 ms | 524 / 116 ms |

**Key results:**
- Removing server-side decode lifts VS C=1 from 12.7→**118 rps** and cuts p50 57→**8 ms** (min 3.4 ms)
  — i.e. the per-request penalty that came from pure-Go JPEG decode (W8) **disappears** with a
  tensor input, putting VS's min latency (3.4 ms) on par with Triton's (2.9 ms).
- In the fair no-decode regime, **VS tensor-in ≈ Triton** (peak 563 vs 549 rps) — VisionServe matches
  a default Triton's inference-serving ceiling in a single 11 MB license-gated binary.
- Subtle, honest twist: **VS JPEG-in peak (695) > VS tensor-in peak (563)**. The heavy JPEG decode is
  CPU work that Go parallelises across 48 cores, pushing aggregate throughput higher; tensor-in has
  almost no CPU work so it is GPU-mutex/HTTP-bound (~563), the same regime that caps Triton (~549).
  So decode hurts *latency* but, parallelised, *raises* peak throughput.

## Client input-format fix — ndarray PNG→JPEG (`clients/python`)

The Python client now JPEG-encodes a numpy/RGB frame (was PNG). Round-trip through the server,
60 reqs (`eval/client_format_bench.py`):

| ndarray encoding | payload | p50 round-trip | min |
|---|---|---|---|
| PNG (old) | 1327 KB | 136 ms | 85 ms |
| **JPEG-q92 (new)** | **263 KB** | **64 ms** | **34 ms** |

5× smaller payload, 2.1× faster. (Still slower than tensor-in's 8 ms — encode+decode remains; for
in-memory frames the `infer_tensor` endpoint is the right path, no encode/decode at all.)

## W1 (N-way) — Triton raw inference-serving ceiling (tensor-in)

Triton Inference Server (ONNX-Runtime backend, same `model.onnx`, GPU) benchmarked with a closed-
loop concurrency sweep that sends a **pre-made FP32 tensor** (`eval/baselines/triton_bench.py`).
This measures Triton's raw inference ceiling with **NO server-side JPEG decode/preprocess** — an
upper bound, not like-for-like with VisionServe/FastAPI (which decode a JPEG in-server).

| C | Triton (tensor-in) rps / p50 | VisionServe (JPEG-in) rps | FastAPI+ORT (JPEG-in) rps |
|---|---|---|---|
| 1  | 214 / 4 ms  | 12.7 | 66 |
| 4  | 488 / 7.6 ms | 74.6 | 76 |
| 16 | 549 / 28 ms | 405 | 77 |
| 64 | 524 / 116 ms | **695** | 85 |

**Honest reading:** at C=1 Triton is fastest (C++/no-GIL **and no server-side decode**: p50 4 ms).
But VisionServe's *full-pipeline* peak (695 rps, JPEG-in) is in the same ballpark as / exceeds
this **default-config** Triton's tensor-in peak (~549 rps at C=16) for this small model — because
the GPU work is sub-ms, VisionServe parallelises decode+preprocess across the host's cores, while
this Triton runs **HTTP, a single `instance_group` (count 1), no dynamic batching, and a 602 KB
FP32 tensor per request** (larger on the wire than the 137 KB JPEG). **Caveat:** Triton is
under-tuned here; with gRPC + shared memory + more instances + dynamic batching it would go
substantially higher. We therefore do NOT claim "VisionServe beats Triton"; the honest claim is
that a single 11 MB license-gated binary serving the full decode→infer pipeline reaches throughput
comparable to a default Triton deployment that needs a multi-GB container and pre-made tensors.

### Caveats / threats to validity (state these in the paper)
- Pure-Python client is GIL-bound at very high RPS; fair for the VS-vs-baseline comparison under the
  *same* client, but absolute p99.9 should be re-measured with wrk2/vegeta for headline tail numbers.
- Single baseline so far (FastAPI+ORT). Add Triton / TorchServe / BentoML on the same ONNX (W1).
- No SessionPool>1 sweep yet (VisionServe pool size is hardcoded per model; baseline has SESSION_POOL knob).
- Resized-256 val mirror + squash-resize preprocess; re-run on full ILSVRC val if exact parity needed.
