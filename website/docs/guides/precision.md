# Reduced precision: FP16, INT8, INT4, mixed per layer, and per-layer sensitivity

!!! info "This is a deep dive"
    For a Jetson, [Make it smaller and faster for Jetson](jetson.md) does this for you with one
    command (`visionserve optimize`). This page describes the converter flags behind it, how a
    reduced model is judged, and what each format cost on a real model.

`visionserve convert` can shrink a model after exporting it. Everything here is **opt-in**: with
no `--precision` flag the output is the FP32 ONNX file it always was.

```bash
# FP16: half the file, float32 inputs and outputs (the manifest does not change)
visionserve convert rfdetr ckpt.pth --name my-det-fp16 --precision fp16 --calib ./photos --eval val.json

# INT8 (static, QDQ form: per-channel int8 weights, symmetric int8 activations)
visionserve convert rfdetr ckpt.pth --name my-det-int8 --precision int8 --calib ./photos --eval val.json

# Which layers hurt? Writes <model>/sensitivity.json and prints the worst ones
visionserve convert rfdetr ckpt.pth --name my-det --precision int8 --calib ./photos --sensitivity --dry-run

# Mixed precision: keep the fewest sensitive layers in float32 that keep the error under 0.05
visionserve convert rfdetr ckpt.pth --name my-det-mixed --precision int8 --calib ./photos \
    --max-output-err 0.05 --eval val.json
```

| Flag | Meaning |
|---|---|
| `--precision fp16\|int8\|int4\|mixed` | Reduce the converted ONNX. `int4` is weight-only 4-bit (`MatMulNBits`, activations stay float). `mixed` gives each layer its own format (see below). Single-session image models only for now (RF-DETR, and the generic classification / detection / depth / embed formats) |
| `--formats LIST` | With `mixed`: the formats a layer may take, from `int4,int8,fp16` (default `int8,fp16`). `fp32` is always the fallback |
| `--sens-formats LIST` | Formats `--sensitivity` measures per layer, from `int4,int8,fp16,fp8,fp4`. `fp8` and `fp4` are simulated, see below |
| `--int4-algo rtn\|hqq`, `--int4-block N` | INT4 weight quantizer (`rtn` needs nothing; `hqq` needs torch) and its block size along K (default 32) |
| `--calib DIR` | Images for INT8 ranges and for `--sensitivity`, preprocessed exactly as the server will. Use photos like the ones the model will see (default: `--images`) |
| `--calib-method` | `minmax`, `entropy` or `percentile` (default; clips rare outliers) |
| `--sensitivity` | Score every MatMul / Gemm / Conv (see below) |
| `--keep-fp-top N` | Keep the N most sensitive layers in float32 |
| `--max-output-err E` | With `int8` / `fp16` / `int4`: find the smallest N layers to keep in float so the output error is ≤ E. With `mixed`: required, the target the per-layer assignment is searched against. Each step rebuilds the model |
| `--keep-fp-node NAME`, `--keep-fp-op OP` | Keep a node, or every node of an op type (`Softmax`, `LayerNormalization`), in float32 |

## How the result is judged

Three checks, in this order of authority:

1. **Tier C (`--eval`)**: mAP or top-1 on your labelled data, reference vs the served reduced model.
   This is the only check that says the model is still good.
2. **Tiers B1 / B2**: the installed model's preprocessing and outputs against the original framework.
3. **Tier P**: the reduced ONNX against the FP32 ONNX on the calibration images. It uses a distance
   that is `0` for the same detections and `1` for nothing in common (`1 - matched IoU`, same-class,
   as a *set*). For a DETR head the queries are not compared row by row, because the top-K
   selection reshuffles them on the slightest numeric change. Raw relative L2 on RF-DETR said
   0.28 for a harmless FP16 conversion and ≈0.38 for every single layer.

Without `--eval` the report carries a WARN: the accuracy of a reduced model was not measured.
A P row above the FAIL threshold uninstalls the model like any failed tier
(`--threshold p_err_fail=0.4` to change it).

## A format per layer: `--precision mixed`

```bash
visionserve convert rfdetr ckpt.pth --name my-det-mixed --precision mixed \
    --formats int4,int8,fp16 --calib ./photos --max-output-err 0.05 --eval val.json
```

One model can hold INT4 (`MatMulNBits`), INT8 (QDQ), FP16 and FP32 layers. The converter

1. measures, for every layer alone, the output error under each format in `--formats` (the sensitivity
   matrix below),
2. gives each layer the most aggressive format whose own error is ≤ τ (a layer none fits stays FP32),
3. searches τ by bisection, **building and measuring the whole model at each step**, and keeps the
   largest τ whose model error is ≤ `--max-output-err`. The result is a good assignment, not a proven
   optimum: errors of many layers do not add up simply and the measured error is noisy.

The chosen format of every layer is written to `sensitivity.json` (`assignment`). Layers that no format
fits stay FP32; every op that is not a MatMul / Gemm / Conv follows FP16 when `fp16` is in `--formats`,
FP32 otherwise. INT4 applies to a MatMul with a 2-D constant weight, which is most of a transformer.

## Sensitivity

For each MatMul / Gemm / Conv alone, the weight is replaced by its quantize-dequantize round trip,
a QuantizeLinear / DequantizeLinear (or an fp16 Cast) is put on its activation inputs, and the
model's outputs are compared with FP32. The table also shows the weight's own SQNR and the
activation **outlier ratio** (largest value over the 99.9th percentile): a large ratio means a few
values dominate the range, so INT8 resolution is spent on them.

`--sens-formats int4,int8,fp16,fp8,fp4` measures several formats per layer in one run and writes every
number to `sensitivity.json` (`errs`). The formats differ in what they touch: `int8` and `fp8` round the
weight and the activation inputs, `fp16` casts both, `int4` and `fp4` are weight-only and blockwise
(32 and 16 values along K) and exist only for a MatMul with a 2-D constant weight.

**FP8 and FP4 are simulated.** The round trips are exact (checked against ONNX Runtime's own Cast to
float8), so the numbers say what those formats would cost in accuracy. Nothing is built: ONNX Runtime 1.26
cannot run a full-model FP8 QDQ graph (its QDQ fusion produces a `QLinearConv` of float8 type that fails
to load) and has no FP4 path. FP8 needs Ada, Hopper or Blackwell GPUs under TensorRT; FP4 needs Blackwell.
The Ampere GPUs in the converter's target machines (A6000, Jetson Orin) have neither.

It is a proxy, one layer at a time. Errors of 120 layers do not add up simply, and a layer that only
moves queries nobody keeps can rank high. Use it to decide *what to try keeping in float*; tier C
decides whether the result is acceptable. Cost is one ONNX Runtime session per layer.

## What we measured

RF-DETR small fine-tuned on tabletop-22 (22 classes, 512 px, letterbox), ONNX Runtime 1.26 on CPU.
Calibration and sensitivity use 32 images; mAP is measured on the other **215 images** with
pycocotools, at the served threshold (confidence 0.5) and at 0.05. The model was fine-tuned on all of
tabletop-22, so the absolute mAP is flattering; only the differences between rows mean something.
"Error" is the P row's output distance to FP32 on the calibration images.

| Model | Size | Error | mAP @ conf 0.5 | mAP @ conf 0.05 |
|---|---|---|---|---|
| FP32 | 114.4 MB | 0 | 63.48 | 68.30 |
| FP16 | 57.6 MB | 0.016 | 63.38 | 68.33 |
| FP16, Softmax and LayerNormalization kept in float32 | 57.7 MB | 0.014 | 63.65 | 68.42 |
| INT8, every MatMul / Gemm / Conv | 31.5 MB | 0.241 | 58.67 | 63.66 |
| INT8, `--max-output-err 0.10` (keep sensitive layers in float) | 95.7 MB | 0.098 | 62.05 | 67.30 |
| INT4 RTN, every MatMul | 26.7 MB | 0.266 | 54.77 | 59.86 |
| INT4 HQQ, every MatMul | 30.0 MB | 0.208 | 60.84 | 65.41 |
| **mixed** `int8,fp16`, `--max-output-err 0.1` | 47.7 MB | 0.097 | 62.74 | 67.74 |
| **mixed** `int4,fp16` (HQQ), `--max-output-err 0.1` | 42.9 MB | 0.095 | 64.15 | 69.23 |
| **mixed** `int4,int8,fp16` (HQQ), `--max-output-err 0.1` | 44.9 MB | 0.076 | 62.79 | 68.09 |

What to take from it:

- **FP16 is free** here (within ±0.3 mAP) and halves the file.
- **One format for every layer is costly.** INT8 everywhere lost 4.8 points, INT4 RTN 8.7. INT4 HQQ is much
  better than RTN (−2.6) for 5 s more, so use `--int4-algo hqq` for INT4 when torch is present.
- **A format per layer is the useful result.** The mixed models are 2.4 to 2.7 times smaller than FP32
  and lose between −0.7 and +0.7 mAP. The INT8-only mix is the same size as INT8 with the sensitive layers
  kept in float32 (47.7 vs 95.7 MB) and loses half as much.
- **The P error predicts mAP only roughly.** Errors of 0.076 to 0.098 gave −1.4 to +0.7 points. The
  differences between the three mixed rows are within noise of a 215-image test set. Check the winner
  with `--eval` on your own data; do not pick it by the P number alone.
- **The assignment search is noisy.** Adding a single INT8 layer moved the model error from 0.076 to 0.162
  in one step: the layers' single-layer errors do not add up, and percentile calibration is itself
  noisy. The bisection finds a good assignment, not the best one, and each step costs a full rebuild
  (30 to 170 s on this model, on a shared CPU).
- **Single-layer sensitivity differs by format.** Mean single-layer error: INT8 0.020, FP8 0.020, FP4 0.026,
  INT4 0.034, FP16 ~0. FP8 costs the same as INT8 on this model; FP4 (blockwise 16) beats INT4 (blockwise 32)
  as a weight-only format. They are simulations (see above), not something to deploy on ONNX Runtime.

On RF-DETR base (COCO, 560 px, 5 photos, agreement with FP32 only) the same pattern held: FP16 0.018,
INT8 on every layer 0.36, and the layers that mattered most were the attention `MatMul_1` (softmax times V,
activation outlier ratio 13 to 28) and the second MLP layer of backbone block 2. That agrees with the
published account of transformer INT8: a few activation outliers, not the weights, set the error.

## Limits you should know

- The ONNX Runtime CUDA execution provider does not speed up QDQ INT8; INT8 pays off on CPU and
  under TensorRT. FP16 is the CUDA win. The converter image is CPU-only, so tiers B/C run FP16
  and INT8 on CPU; a GPU converter image is not built yet.
- A `MatMul` whose constant operand is not a 2-D matrix (a folded `[1, heads, Q, D]` tensor in an
  exported DETR decoder) is left in float: ONNX Runtime's `QLinearMatMul` cannot run a per-channel
  zero point on it.
- Calibration ranges from 5 images are unreliable (the converter warns). Use 32 or more.
- `--precision mixed` is the way to combine formats. Plain `--precision int8` keeps the layers it leaves out in FP32.
- Build order inside `mixed` matters and is fixed: INT8, then FP16, then INT4. The other orders failed to load or to quantize (an FP16 bias Add reading a `MatMulNBits` output; ORT's quantizer on a graph that already mixes FP16 and FP32).
- INT4 and FP4 simulation need ONNX Runtime >= 1.18 with `MatMulNBitsQuantizer` and `ml_dtypes` >= 0.5. The converter image's PyTorch venv has them; its TensorFlow venv (tensorflow, keras, tflite formats) does not and refuses those options up front.
- TensorRT engines are not built here. They are tied to the GPU, TensorRT and ONNX Runtime
  versions, so they belong on the machine that runs them (`--tensorrt` at serve time).
