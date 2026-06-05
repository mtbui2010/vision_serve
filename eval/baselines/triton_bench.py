"""Triton raw inference-serving ceiling (Exp6) — closed-loop concurrency sweep.

IMPORTANT (fairness): this sends a PRE-MADE FP32 tensor to Triton's /v2 infer endpoint, so it
measures Triton's raw inference-serving throughput with NO server-side JPEG decode / resize /
preprocess. VisionServe and the FastAPI+ORT baseline, by contrast, receive a JPEG over HTTP and
decode+preprocess in-server. Triton's numbers here are therefore an UPPER BOUND on inference-
serving throughput for this ONNX, not a like-for-like end-to-end comparison. We report it as the
"what does the runtime alone sustain" ceiling and label it as such in the paper.

Run:
    python -m eval.baselines.triton_bench --url localhost:8100 --model mobilenet-v3 \
        --concurrency 1,2,4,8,16,32,64 --duration 20 --out eval/results/triton_mobilenet.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time

import numpy as np
import tritonclient.http as httpclient

from eval.common.percentiles import LatencyRecorder


def _bench_one(url: str, model: str, conns: int, duration_s: float,
               in_name: str, out_name: str, shape) -> dict:
    rec = LatencyRecorder()
    lock = threading.Lock()
    counters = {"completed": 0, "errors": 0}
    stop = threading.Event()
    x = np.random.rand(*shape).astype(np.float32)

    def worker():
        cli = httpclient.InferenceServerClient(url=url, verbose=False)
        inp = httpclient.InferInput(in_name, list(shape), "FP32")
        out = httpclient.InferRequestedOutput(out_name)
        while not stop.is_set():
            inp.set_data_from_numpy(x)
            t0 = time.perf_counter()
            try:
                cli.infer(model_name=model, inputs=[inp], outputs=[out])
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            lat_ms = (time.perf_counter() - t0) * 1000.0
            with lock:
                counters["completed"] += 1
                if not ok:
                    counters["errors"] += 1
                rec.record_ms(lat_ms)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, conns))]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    time.sleep(duration_s)
    stop.set()
    for w in workers:
        w.join(timeout=30)
    wall = time.perf_counter() - t0

    s = rec.summary()
    row = {"system": "triton(tensor-in)", "model": model, "concurrency": conns,
           "mode": "closed-loop", "completed": counters["completed"],
           "errors": counters["errors"], "wall_s": round(wall, 3),
           "rps_measured": round(counters["completed"] / wall, 2) if wall > 0 else 0.0}
    row.update(s.as_row())
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="localhost:8100")
    ap.add_argument("--model", default="mobilenet-v3")
    ap.add_argument("--in-name", default="x")
    ap.add_argument("--out-name", default="400")
    ap.add_argument("--shape", default="1,3,224,224")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    shape = tuple(int(x) for x in args.shape.split(","))
    conc = [int(x) for x in args.concurrency.split(",") if x.strip()]

    # warmup
    cli = httpclient.InferenceServerClient(url=args.url)
    if not cli.is_model_ready(args.model):
        print(f"ERROR: model {args.model} not ready at {args.url}")
        return 2
    rows = []
    for c in conc:
        row = _bench_one(args.url, args.model, c, args.duration, args.in_name, args.out_name, shape)
        rows.append(row)
        print(f"[triton] C={c} rps={row['rps_measured']} p50={row.get('p50.0_ms')} "
              f"p99={row.get('p99.0_ms')} errors={row['errors']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
