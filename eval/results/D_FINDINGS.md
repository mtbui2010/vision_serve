# D1 / D2 findings — multi-worker baseline and heavier-model crossover

Measured 2026-06-09 on the A6000 host (VisionServe on GPU 2, FastAPI+ORT baseline on GPU 3),
same closed-loop harness (`eval/loadgen/sweep_openloop.py`), same ONNX + CUDA EP, JPEG input
(`test/testdata/sample.jpg`), 15 s/level, `errors=0` on every row. These answer two reviewer
points: the single-worker baseline is a strawman (D1), and the crossover was only shown on a tiny
classifier (D2).

## D1 — mobilenet-v3: does a MULTI-worker baseline erase the ~8× advantage? (Yes, mostly.)

Peak goodput (rps), across C ∈ {1,8,32,64}:

| System | peak rps | VisionServe ÷ baseline |
|---|---|---|
| VisionServe (1 Go process) | **651.9** (C=64) | 1.0× |
| FastAPI+ORT, uvicorn `--workers 1` (deployed default) | 86.1 (C=32) | **7.6×** |
| FastAPI+ORT, `--workers 2` | 169.2 (C=32) | 3.85× |
| FastAPI+ORT, `--workers 4` | 309.1 (C=32) | 2.11× |
| FastAPI+ORT, `--workers 8` | 585.1 (C=64) | **1.11×** |

**Honest finding.** VisionServe's ~8× peak advantage holds **only against the single-worker
GIL-bound default**. A FastAPI baseline scaled to 8 uvicorn worker *processes* reaches 585 rps —
within ~11% of VisionServe's 652 — i.e. rough parity. The correct claim is therefore **not** "8×
faster than Python"; it is: *a single, untuned Go binary matches an 8-process Python deployment*,
without the ~8× process-memory cost or the need to pick a worker count. The 8× number must be
explicitly scoped to the deployed default in the paper.

## D2 — rf-detr-nano (heavier detector): does the crossover generalize beyond a tiny classifier?

Peak goodput (rps), across C ∈ {1,2,4,8,16,32,64}:

| System | peak rps | note |
|---|---|---|
| VisionServe (1 Go process) | 189.5 (C=32) | 3.14× the 1-worker baseline |
| FastAPI+ORT `--workers 1` | 60.4 (C=8) | — |
| FastAPI+ORT `--workers 4` | **212.8** (C=32) | **slightly exceeds VisionServe** |

**Honest finding.** For the heavier detector the crossover still holds versus the single-worker
default (VisionServe 3.1×), but a 4-worker baseline (213 rps) **edges past** VisionServe (190 rps).
This is consistent with the C2 thesis: VisionServe's throughput edge comes from parallel pure-Go
*decode*, whose relative weight shrinks as GPU *compute* grows. The heavier the model, the more
both systems become GPU-bound and the smaller the serving-layer advantage — so the single-binary
win is largest for light, decode-bound models and narrows for compute-heavy ones.

## Takeaway for the paper

Reframe the headline from "≈8× throughput" to: **a single Go binary matches a tuned multi-process
Python deployment** (8 workers for the light model; 4 workers already match/beat it for the heavier
one), at a fraction of the process memory and with no worker-count tuning. The 8× figure is real
but is strictly versus the single-worker `SESSION_POOL=1` default that most teams actually ship.
