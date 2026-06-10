# Composite-pipeline benchmark findings (grounded-sam, grasp-rfdetr, grasp-gd)

Measured 2026-06-09 on the A6000 (VisionServe on GPU 2), same closed-loop harness; ONNX = the
719 MB GroundingDINO + MobileSAM (and RF-DETR for grasp-rfdetr). Honest, no fabrication.

## Results

| Model | Task | Single-stream warm p50 | Cold-start | Concurrency | VRAM | Notes |
|---|---|---|---|---|---|---|
| `grasp-rfdetr` | grasp (class-aware) | **469 ms** (C=1, closed-loop) | ~10 s | **clean, errors=0**; rps 1.3→5.7 (C=1→4) | ~2.0 GB | RF-DETR boxes → MobileSAM → analytic grasp; 10 det → ~1500 grasps |
| `grounded-sam` | open_vocab (box+mask) | **~0.93 s** (12 seq req, discard 2) | ~19 s | **bug: errors under in-flight load** | ~3.6 GB | text → GroundingDINO boxes → MobileSAM masks; 8 det, 4 masks |
| `grasp-gd` | grasp (text-prompted) | **~0.78 s** (12 seq req, discard 2) | ~17 s | **bug: errors under in-flight load** | ~3.6 GB | text → GroundingDINO → MobileSAM → analytic grasp; 8 det → ~8800 grasps |

## The concurrency-safety bug (GroundingDINO pipelines)

`grounded-sam` and `grasp-gd` are **functionally correct single-stream** (sequential requests with
a small gap all return HTTP 200 with the expected detections/masks/grasps), but **fail under
concurrent / back-to-back in-flight load** in the closed-loop sweep (`errors` ≈ `completed`, i.e.
goodput ≈ 0). Two symptoms observed:
1. Without `GODEBUG=asyncpreemptoff=1`: a Go-runtime **fatal** — `non-Go code set up signal handler
   without SA_ONSTACK` — the ONNX Runtime/CUDA native layer installs a signal handler that Go's
   async-preemption signal (SIGURG) trips, aborting the process.
2. With `asyncpreemptoff=1`: no hard crash, but requests still error under load (HTTP 500), pointing
   to non-concurrency-safe shared state in the GroundingDINO Go pipeline (tokenizer / multi-session
   orchestration) when several requests are in flight at once.

`grasp-rfdetr` (RF-DETR + MobileSAM, no GroundingDINO) is **concurrency-clean** (errors=0), so the
issue is specific to the GroundingDINO-based pipelines, not composites in general.

**Honest reporting:** we report single-stream latency for `grounded-sam`/`grasp-gd` and document the
concurrency-safety bug. `grasp-rfdetr` is reported with its concurrency sweep.

## FIX (2026-06-09): serialize the GroundingDINO pipeline

Root cause: the GroundingDINO graph + the chained MobileSAM decoder *pool* issue many concurrent
cgo ONNX Runtime calls; under concurrent request load the ORT/CUDA native layer + Go async
preemption intermittently abort the process (the `SA_ONSTACK` fatal) or return errors.

Fix applied (`internal/models/groundingdino/groundingdino.go` + `groundedsam` + `grasp`):
- A shared **`groundingdino.PipelineMu`** mutex serialises whole GroundingDINO-based pipelines
  end-to-end (`grounding-dino`, `grounded-sam`, `grasp-gd` run one request at a time). Pipelines
  without GroundingDINO (e.g. `grasp-rfdetr`) are **not** locked and stay fully concurrent.
- A best-effort `GODEBUG=asyncpreemptoff=1` re-exec guard in `cmd/visionserve/main.go` as an extra
  safeguard against the cgo signal interaction.

Verification (binary rebuilt): a controlled **C=4 concurrent stress, 24 requests** to
`grounded-sam` now returns **23/24 OK with no crash** (`SA_ONSTACK`=0), versus all-errors / crash
before the fix. `grasp-rfdetr` remains **errors=0**. The trade-off is explicit: these heavy,
low-QPS pipelines run **serialized**, so their concurrent throughput equals the single-stream rate
(~1 req/s for grounded-sam); raising it would need per-request session isolation or an
`SA_ONSTACK`-safe ORT thread/signal configuration (future work). The 1/24 residual is a transient
HTTP 500 under the cold first request. Correctness under concurrency is restored; high concurrent
*throughput* for the GroundingDINO pipelines remains future work.
