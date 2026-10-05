# Make it smaller and faster for Jetson

## When you need this

You will run the model on an NVIDIA Jetson (Orin, or Thor) and want a smaller file, or a faster
one, without losing accuracy. A model can be stored with less precise numbers: *FP16* (16-bit
floats, half the size of the usual 32-bit *FP32*), *INT8* (8-bit integers, a quarter) or a *mix*
of formats, layer by layer. Some layers tolerate it; some lose accuracy. `optimize` builds each
variant, measures it, and recommends one.

## The command

On your PC:

```console
$ visionserve optimize my-detector --target jetson-orin --images ./photos \
      --labels ./val200/instances.json --gpu --install
```

- `--target`: `jetson-orin`, `jetson-thor`, `cuda` (a PC GPU) or `cpu`. It picks the variants
  worth trying for that hardware.
- `--images`: 8 to 32 photos like the ones the model will see. They set the INT8 ranges.
- `--labels`: your labelled photos (a COCO json), so it can measure accuracy. Without them it
  can only say how far each variant is from FP32, not how accurate it is.
- `--gpu`: measure on this PC's GPU (the same kind of execution provider as the Jetson's).
- `--install`: install the recommended variant as `my-detector-<format>` (nothing, when the
  recommendation is to keep FP32). `--install-format fp16` installs that variant instead.

**Accuracy can be measured on your PC; speed only on the device.** The same ONNX file computes
almost the same numbers on a PC and on a Jetson, so a variant that keeps its accuracy here keeps
it there. Its speed does not carry over. So the last step is always on the Jetson: copy the
variant's folder there and time it next to FP32 (not run for this page: we had no Jetson):

```console
$ scp -r ~/.visionserve/models/my-detector-fp16 jetson:~/.visionserve/models/
$ ssh jetson visionserve bench my-detector-fp16 --images ./photos --ep cuda
$ ssh jetson visionserve bench my-detector --images ./photos --ep cuda
```

Optional, before `optimize`: `visionserve sensitivity my-detector --images ./photos --save sens.json`
tells you which layers do not tolerate INT8 (below), and `optimize --sensitivity sens.json`
reuses the scores instead of measuring them again.

## Reading the result

A real run (100 labelled COCO photos for accuracy, 8 photos for calibration, the PC's RTX A6000
standing in for the Jetson's GPU):

```console
$ visionserve optimize my-detector --target jetson-orin --images ./photos \
      --labels ./val200/instances.json --label-images ./val200/images --sensitivity sens.json --gpu --install
WARN: keep FP32 for speed: no reduced variant is as fast on this host's GPU (my-detector-fp16 is 2.0× smaller but 1.8× slower here); use it only if size matters, and bench both on the device

  model                    my-detector                          detection, rf-detr, model.onnx (107.8 MB)
  target                   jetson-orin                          Jetson AGX Orin / Orin NX / Orin Nano: Ampere GPU (sm_87), FP16 and INT8 tensor cores, no FP8
  recommended              keep FP32                            the installed model
  budget                   output error <= 0.05, mAP drop <= 1
  latency measured on      ketiserver (x86_64), gpu:0           this host, not the jetson-orin
...
Variants for jetson-orin
  variant  file size               output error mean / max  mAP (Δ vs FP32)  latency p50 / p95 (THIS host)  status
  ✓ fp32   107.8 MB                0 / 0                    47.03            19.9 / 29.8 ms (gpu:0)         recommended ✓
  fp16     54.2 MB (2.0× smaller)  0.0325 / 0.129           47.11 (+0.07)    36.3 / 55.7 ms (gpu:0)         within budget
  int8     29.2 MB (3.7× smaller)  0.241 / 0.433            43.01 (-4.02)    22.5 / 38.0 ms (gpu:0)         over budget: error 0.241 > 0.05; mAP −4.02 > 1
  mixed    56.1 MB (1.9× smaller)  0.0335 / 0.13            47.32 (+0.29)    33.9 / 41.9 ms (gpu:0)         within budget
...
Next steps
  If size matters more than speed: re-run with --install --install-format fp16 (it becomes my-detector-fp16).
  On the device: `visionserve bench my-detector-fp16 --images DIR` and `visionserve bench my-detector --images DIR` — only that comparison says how much faster it is there.
```

Read the verdict first: **keep FP32**. Then the table, one row per variant:

| Column | In plain words |
|---|---|
| **file size** | FP16 and *mixed* halve the file; INT8 makes it 3.7 times smaller. |
| **output error** | How far its answers are from FP32's on your photos: 0 is the same boxes, 1 nothing in common. The budget is 0.05. |
| **mAP (Δ vs FP32)** | Accuracy on your labelled photos (0 to 100, higher is better) and the change against FP32. FP16 and mixed lose nothing (+0.07, +0.29: noise); INT8 everywhere loses 4 points. |
| **latency (THIS host)** | Time per photo **on this PC**, not on the Jetson. |
| **status** | `within budget` (good enough), `over budget` (changes the answers too much) and the `recommended ✓` row. |

!!! warning "Honest notes: what this does and does not tell you"
    - **The latency column is this PC's.** A PC GPU, an Orin and a Thor differ by an order of
      magnitude, and even the ranking of FP16 against FP32 can change. Only `bench` on the
      device answers "how fast is it there".
    - **Smaller is not faster on ONNX Runtime's CUDA support.** For this model, FP16 ran slower
      than FP32 (36.3 against 19.9 ms here). Measured directly on an otherwise idle GPU, one run
      of the FP16 graph of this RF-DETR Nano export took 16.1 ms against 4.6 ms for FP32. That is
      why `optimize` lets FP32 compete instead of assuming FP16 wins. INT8 is a size win on the
      CUDA path, not a speed win; it pays off in speed on the CPU and under TensorRT.
    - **The Orin preset ran on an RTX A6000** (Ampere, like Orin, with the same CUDA execution
      provider), not on an Orin. **The Thor preset is untested on hardware.**
    - Accuracy carries over to the device much better than speed, but check it there too for
      anything you ship (`visionserve check` against the Jetson's server).

### The same comparison on the PC

After `optimize ... --install --install-format fp16` installed
`my-detector-fp16`, shows why this step matters: `bench my-detector-fp16` gave
`PASS: p50 40.4 ms, 24.0 req/s on gpu:0`, `bench my-detector` gave `PASS: p50 18.5 ms, 49.7 req/s
on gpu:0`. Half the file, twice the time, on this PC. The Jetson may rank them differently.
The installed folder's `manifest.yaml` records how it was made:

```yaml
# Generated by `visionserve optimize` from my-detector on 2026-10-05.
# precision: fp16 (FP16 weights and activations, float32 inputs/outputs)
# target preset: jetson-orin
# output error vs FP32: mean 0.0325, max 0.129 on 8 image(s) of photos
# mAP 47.11 vs FP32 47.03 on 100 labelled image(s)
# Same preprocessing, labels and postprocessing as my-detector; only the ONNX file differs.
name: my-detector-fp16
```

## If it says WARN or FAIL

| Message | What to do |
|---|---|
| `WARN: keep FP32 for speed: no reduced variant is as fast on this host's GPU (... is 2.0× smaller but ...× slower here)` | Normal on ONNX Runtime's CUDA execution provider (see the note above). Keep FP32, unless the file size matters more: then install the smaller one with `--install --install-format fp16` and `bench` both on the Jetson. |
| `WARN: ... accuracy NOT measured (pass --labels)` | Add `--labels` with 100 or more labelled photos. Without them, "close to FP32" is all it knows. |
| `WARN: ... latency here was noisy, so the speed ranking may be noise` | Something else used the GPU. Run again when it is idle, or trust only the size and accuracy columns. |
| `no variant stays inside the budget ...: keep FP32` | Every reduced variant lost too much. Keep FP32, or allow more loss with `--max-drop 2` if that is acceptable for you. |
| `over budget: error 0.241 > 0.05; mAP −... > 1` (a row) | That variant changes the answers too much. It is not recommended; nothing to do. |
| `FAIL: no variant could be built: ...` | `optimize` works on single-model image models (RF-DETR, classifiers, depth). SAM, GroundingDINO and the pipelines are refused. |
| `FAIL: the variants could not be measured: ...` | The temporary server could not start; read the message (often a GPU or ONNX Runtime setup issue, see [Measure speed](measure-speed.md#if-it-says-warn-or-fail)). |
| sensitivity: `WARN: 29 of 140 layers are sensitive to INT8` | Normal for a transformer. `optimize`'s `mixed` variant keeps those layers in a more precise format. |
| bench on the Jetson: `WARN: CUDA was requested but the model ran on the CPU` | The Jetson's ONNX Runtime has no CUDA support. Use the Jetson image (`deploy/Dockerfile.edge`, below) or a Jetson build of ONNX Runtime. |

### Running on the Jetson

For JetPack 6 (Orin), `deploy/Dockerfile.edge` builds a server image with CUDA and TensorRT
support:

```console
$ docker buildx build --platform linux/arm64 -f deploy/Dockerfile.edge --build-arg ORT_SOURCE=jetson -t visionserve:jetson --load .
```

It builds on an ordinary x86 PC (checked on 5 October 2026), but has not run on an Orin yet.
Jetson Thor (JetPack 7) has its own `deploy/Dockerfile.thor`. Details and the build options:
[deploy/README.md](https://github.com/mtbui2010/vision_serve/blob/main/deploy/README.md#jetson--arm64).
On the device, set the power mode before you measure (`sudo nvpmodel -q`; `sudo jetson_clocks`
for stable clocks) and write down which one you used.

## Want the details?

??? note "Which layers break at INT8? (`visionserve sensitivity`)"
    A real run on the same model (8 photos, the PC's CPU, 13 minutes; trimmed):

    ```console
    $ visionserve sensitivity my-detector --images ./photos --formats int8,fp16 --save sens.json
    WARN: 29 of 140 layers are sensitive to INT8; the worst is …/encoder/encoder/encoder/layer.2/mlp/fc2/MatMul (MatMul, error 0.146) — keep it in FP32

      model              my-detector                                                       detection, rf-detr, model.onnx
      layers measured    140                                                               MatMul / Gemm / Conv, each reduced alone
      formats            INT8, FP16
      sensitive to INT8  29 of 140                                                         error alone > 0.05; mean 0.0311
      sensitive to FP16  0 of 140                                                          error alone > 0.05; mean 0.000113
    ...
    ```

    Each layer is reduced **alone** and the answers compared with FP32. FP16 hurts no layer; INT8
    hurts 29 of 140, mostly the attention and MLP layers of the backbone. The mixed variant keeps
    such layers in FP16 or FP32. A layer's score is a hint (errors of many layers do not simply
    add up); only accuracy on labelled photos decides. `--formats int8,fp8,fp4` also *simulates*
    the formats Thor adds (nothing is built: ONNX Runtime cannot run them).

- The presets (which variants each target tries, and why), more measurements, TensorRT rows,
  and the Thor checklist: [Jetson details](edge.md).
- The converter flags behind it (`--precision`, `--max-output-err`, INT4, per-layer formats) and a
  full table of what each format cost: [Reduced precision](precision.md).
- Measuring speed and reading every line of `bench`: [Measure speed](measure-speed.md).

<small>Run on 5 October 2026: RTX A6000 shared with other jobs (CUDA, ONNX Runtime 1.26) and its
48-core CPU; `sensitivity` and `optimize` with `--python` and the converter package from
`clients/python`; calibration on 8 COCO val2017 photos, accuracy on the first 100 images of COCO
val2017. Nothing on this page ran on a Jetson.</small>
