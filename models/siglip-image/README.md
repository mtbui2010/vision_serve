# siglip-image — SigLIP image tower (Apache-2.0)

The vision branch of `google/siglip-base-patch16-224`. Sibling of `models/siglip-text/`: same
checkpoint, same licence, the other tower. It exists to be the **crop namer** of the two-head
open-vocabulary detector — RF-DETR proposes boxes, head A names what it was trained on, and for
everything else we crop the box, embed the crop here and compare against text embeddings.

SigLIP rather than CLIP on measured evidence (identical 75 held-out rows, identical crops,
identical prompt ensemble — `paper/FINDINGS-2026-08.md` §9–§10):

| namer | held-out 5 | base 17 | all 22 |
|---|---|---|---|
| CLIP ViT-B/32-crop | 65.3 | 72.4 | 70.9 |
| **SigLIP-crop** | **85.3** | 78.8 | **80.2** |

## The cost question, answered

`base-patch16` at 224² is 196 visual tokens per crop against CLIP ViT-B/**32**'s 49 — roughly 4×
the attention work — and the whole detection path is only ~20 ms, so "is this affordable?" had to
be measured before anything was built on it.

RTX A6000 (sm86, **shared with other jobs**, 16–63 % busy during the run), ONNX Runtime 1.26 GPU,
fp32, warm, median of 100 reps. Times are wall-clock through `sess.run` with NumPy inputs, i.e.
they include the host→device copy, which is what a Go caller will also pay.

| batch | TensorRT (opt=8) | ms/crop | TensorRT (opt=1) | ms/crop | CUDA EP | ms/crop |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.35 ms | 4.35 | **2.33 ms** | **2.33** | 3.66 ms | 3.66 |
| 4 | 6.93 ms | 1.73 | 7.52 ms | 1.88 | 10.78 ms | 2.69 |
| 8 | 11.12 ms | 1.39 | 14.00 ms | 1.75 | 19.65 ms | 2.46 |
| 16 | **21.02 ms** | **1.31** | 26.07 ms | 1.63 | 38.79 ms | 2.42 |

CPU EP on the same host (48 cores, busy), median of 7: **80 ms** for one crop, 43 ms/crop at
batch 16. GPU-only feature in practice; on a CPU-only edge box the crop namer must be optional.

**Answer to the question that mattered:** batched, a crop costs **1.3–1.7 ms**, so naming *all*
detections in a typical 3–15 box scene costs **~7–21 ms** — of the same order as the ~20 ms
detector, not 4× it. Unbatched it costs 2.3–4.4 ms per crop and a 15-box scene would spend
35–65 ms naming, i.e. it becomes the dominant cost somewhere around **8–9 crops**. **Batch the
crops.** With batching the design stays affordable to ~15 crops (≈2× the detector budget at
worst); past ~30 crops per image it is the detector that becomes the cheap part.

Two things a serving path must get right, both learned here:

- **The TensorRT profile is not free.** The engine is built for an explicit shape range and a
  batch outside `[min, max]` is a **hard error**, not a slow path (this bit during the first run
  at batch 32). And the `opt` batch matters a lot: the engine built with `opt=1` is 1.9× faster
  at batch 1 but 1.24× slower at batch 16 than the one built with `opt=8`. Pick the profile for
  the batch you will actually serve, or pad every request to a fixed batch.
- **First TensorRT load builds the engine: ~30 s**, cached to disk afterwards (373 MB engine,
  `_sm86` — GPU-architecture specific), after which the session loads in ~3.7 s. Excluded from
  every median above. The CUDA EP has no build step (1.4 s load) and is the better choice if
  sessions are short-lived.

## Numerical verification

Exported graph vs the HF PyTorch model on **32 real crops** (ground-truth boxes from
`ovd-adapt/data/tabletop-22`, i.e. the crops the serving path will actually see):

| execution provider | max abs delta | worst cosine over 32 crops |
|---|---|---|
| CPU (ORT) | **1.29e-05** | **0.99999988** |
| CUDA | 4.78e-03 | 0.99999869 |
| TensorRT fp32 | 3.68e-03 – 5.22e-03 | 0.99999809 |

CPU is the honest export check and it is exact to fp32 rounding. The GPU deltas are ~400× larger
because Ampere runs fp32 matmuls in **TF32** by default; cosine stays ≥ 0.999998, which is far
below the resolution of an argmax over class names, but it is worth knowing that "the ONNX is
bit-exact" is only true on CPU.

Dynamic batch was verified at N = 1, 4, 8, 16, 32 — the exported graph reports
`pixel_values ['batch', 3, 224, 224]`, so **batching is available**, which is what makes the
latency table above reachable.

## Quality smoke test (a smoke test, not a benchmark)

32 crops embedded here, the 22 class names embedded with the text tower under the 10-template
prompt ensemble from `models/rfdetr-textalign-dec1-siglip/templates.txt`, argmax:
**25/32 correct**. All seven misses are one class: `coffee` — predicted `coffee can` (4×) or
`cup` (2×) — plus one `water bottle` → `towel`. That is a vocabulary-ambiguity result, not a
wiring bug, and it is exactly the shape of confusion you would expect from a working embedder.
It says the preprocessing and the text/image spaces line up; it says **nothing** about accuracy
— these are ground-truth boxes drawn from train and val alike. The real number is the 85.3/80.2
above.

## Reproducing the export

Weights are not committed (`*.onnx`, `*.onnx.data`). The pair is 113 KB of graph + 372 MB of
external data.

```bash
# export + numerical verification + smoke test  (env: conda `label` — torch, transformers, ORT-CPU)
HF_HOME=/mnt/nas/huggingface HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TMPDIR=/tmp/uv-cache \
CUDA_VISIBLE_DEVICES= /home/trung/miniconda3/envs/label/bin/python \
  /home/trung/trung_workdir/ovd-adapt/experiments/07_teacher_and_pool/export_siglip_image.py

# latency  (env: conda `vseval` — onnxruntime-gpu 1.26 + TensorRT 10.13, deliberately no torch)
SP=/home/trung/miniconda3/envs/vseval/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/tensorrt_libs:$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:\
$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /home/trung/miniconda3/envs/vseval/bin/python \
  /home/trung/trung_workdir/ovd-adapt/experiments/07_teacher_and_pool/bench_siglip_image.py \
  --ep tensorrt --opt 8 --reps 100
```

The export script **refuses to write a graph it cannot verify** (rel error > 1e-4 or cosine
< 0.9999 against PyTorch). It also dumps `crops_224.npz` — the preprocessed crops and their
PyTorch reference embeddings — so the latency harness can check correctness on the GPU without
needing torch in that env.

Notes on reproducing elsewhere:

- `HF_HOME=/mnt/nas/huggingface` reuses the existing 812 MB snapshot instead of re-downloading it
  onto a root filesystem with 3.8 GB free. `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` is not optional
  here: a stale token in `~/.cache/huggingface/token` makes the Hub answer **401
  RepositoryNotFound** for this public repo, which reads like "model gone" and is not.
- The NAS snapshot only had `config.json` + `model.safetensors`; `preprocessor_config.json`,
  `spiece.model`, `tokenizer_config.json` and `special_tokens_map.json` were fetched to complete
  it. The image processor is loaded from that file rather than hand-rolled, deliberately.

## Status

**Exported, verified and benchmarked. No Go side yet.** There is no `siglip-image` architecture
registered in `internal/models/`, so `visionserve` cannot serve this manifest today. What a Go
implementation needs:

1. A crop preprocessor: box → clamp to image → crop → resize 224×224 bicubic (squash) →
   `(v/255 - 0.5) / 0.5` → NCHW float32. `disintegration/imaging` covers the resize.
2. A batched embed call: shape `[N, 3, 224, 224]` in one session run, N = number of boxes to
   name. **Do not loop one crop at a time** — that is 2.3–4.4 ms each instead of 1.3.
3. Reuse of `internal/models/siglip`'s tokenizer + the `siglip-text` tower for the text side, and
   the existing L2/cosine path. Both towers are the same checkpoint, so their embedding spaces
   already agree; neither tower normalises internally.
